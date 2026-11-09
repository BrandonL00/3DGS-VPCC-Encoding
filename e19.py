#!/usr/bin/env python3
"""Portable E19 command line. The launcher itself needs only Python's stdlib."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'pipeline/tmc2_adapter/scripts'


def configuration():
    cfg = ROOT / 'e19.local.json'
    values = json.loads(cfg.read_text()) if cfg.exists() else {}
    defaults = {
        'ffsplat_root': str(Path.home() / '.local/share/e19/ffsplat'),
        'tmc2_root': str(Path.home() / 'tools/mpeg-pcc-tmc2'),
    }
    result = {**defaults, **values}
    for key, env in [('ffsplat_root', 'FFSPLAT_ROOT'), ('tmc2_root', 'TMC2_ROOT'),
                     ('python', 'E19_PYTHON'), ('adapter_binary', 'E19_ADAPTER_BINARY')]:
        if os.environ.get(env):
            result[key] = os.environ[env]
    result.setdefault('python', str(Path(result['ffsplat_root']) / '.venv/bin/python'))
    target = Path(os.environ.get('CARGO_TARGET_DIR', str(ROOT / 'target')))
    result.setdefault('adapter_binary', str(target / 'release/tmc2_adapter'))
    # Resolving a venv's python symlink selects the system interpreter and loses
    # its site-packages. Preserve the interpreter entry path.
    return {key: str(Path(value).expanduser().absolute() if key == 'python'
                     else Path(value).expanduser().resolve()) for key, value in result.items()}


def doctor(config):
    checks = []
    def add(name, ok, detail):
        checks.append({'check': name, 'ok': ok, 'detail': detail})
    add('platform', sys.platform == 'linux', 'Linux / WSL2 required for this release candidate')
    for key in ('python', 'adapter_binary'):
        add(key, Path(config[key]).is_file() and os.access(config[key], os.X_OK), config[key])
    tmc2 = Path(config['tmc2_root'])
    for relative in ('bin/PccAppEncoder', 'bin/PccAppDecoder',
                     'dependencies/HM-16.20+SCM-8.8-app/bin/TAppEncoderStatic',
                     'dependencies/HM-16.20+SCM-8.8-app/bin/TAppDecoderStatic',
                     'cfg/common/ctc-common-lossless-geometry.cfg'):
        add(relative, (tmc2 / relative).is_file(), str(tmc2 / relative))
    if Path(config['python']).is_file():
        probe = '''import json, torch, cupy
from ffsplat.coding.scene_encoder import EncodingParams
from ffsplat.coding.scene_decoder import decode_gaussians
from PIL import Image
import pillow_heif
assert torch.cuda.is_available(), 'PyTorch cannot access an NVIDIA GPU'
x=torch.ones(8,device='cuda'); assert x.sum().item()==8
y=cupy.ones(8); assert float(y.sum())==8
EncodingParams.from_template_yaml('SOG-web')
print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'capability':torch.cuda.get_device_capability(0),'architectures':torch.cuda.get_arch_list()}))
'''
        try:
            process = subprocess.run([config['python'], '-c', probe], text=True,
                                     capture_output=True, timeout=120, cwd=config['ffsplat_root'])
            add('CUDA and SOG profile', process.returncode == 0,
                (process.stdout + process.stderr).strip())
        except (OSError, subprocess.TimeoutExpired) as error:
            add('CUDA and SOG profile', False, str(error))
    print(json.dumps({'ok': all(c['ok'] for c in checks), 'configuration': config, 'checks': checks}, indent=2))
    return 0 if all(c['ok'] for c in checks) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version='E19 release candidate 1')
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('doctor', help='Check the configured GPU, Python environment and codec binaries')
    enc = commands.add_parser('encode', help='Encode a 3DGS PLY and validate immediate decoding')
    enc.add_argument('source', type=Path)
    enc.add_argument('output', type=Path)
    enc.add_argument('--preset', choices=('compact', 'quality'), default='compact')
    enc.add_argument('--threads', type=int, default=8)
    enc.add_argument('--compaction-version', default='none', help='Lineage label for an already compacted input; does not run compaction')
    enc.add_argument('--sh-clusters', type=int)
    enc.add_argument('--sh-levels', type=int)
    dec = commands.add_parser('decode', help='Decode using only package files and installed dependencies')
    dec.add_argument('package', type=Path)
    dec.add_argument('output', type=Path)
    dec.add_argument('--threads', type=int, default=8)
    args = parser.parse_args(argv)
    config = configuration()
    if args.command == 'doctor':
        return doctor(config)
    if sys.platform != 'linux':
        parser.error('Run inside Linux or WSL2, not Windows Python.')
    if not Path(config['python']).is_file():
        parser.error('Python environment missing. Run python3 setup.py or configure E19_PYTHON.')
    if args.output.exists():
        parser.error(f'Output already exists: {args.output}')
    if args.threads < 1:
        parser.error('--threads must be positive')
    if args.command == 'encode':
        command = [config['python'], str(SCRIPTS / 'run_v19_encoder.py'),
                   str(args.source.resolve()), str(args.output.resolve()),
                   '--ffsplat-root', config['ffsplat_root'], '--tmc2-root', config['tmc2_root'],
                   '--adapter-binary', config['adapter_binary'], '--preset', args.preset,
                   '--compaction-version', args.compaction_version, '--threads', str(args.threads)]
        for name in ('sh_clusters', 'sh_levels'):
            if getattr(args, name) is not None:
                command.extend(['--' + name.replace('_', '-'), str(getattr(args, name))])
    else:
        command = [config['python'], str(SCRIPTS / 'decode_v18_hybrid.py'),
                   str(args.package.resolve()), str(args.output.resolve()),
                   '--tmc2-root', config['tmc2_root'], '--threads', str(args.threads)]
    env = os.environ.copy()
    # Release commands send notifications only when the user explicitly opts in.
    env.setdefault('VPCC_DISCORD_DISABLED', '1')
    return subprocess.run(command, cwd=ROOT, env=env).returncode


if __name__ == '__main__':
    raise SystemExit(main())
