#!/usr/bin/env python3
"""Set up a pinned E19 environment, or reuse explicitly supplied dependencies."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(command, **kwargs):
    print('+', ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reuse-existing', action='store_true', help='Do not download or change ffsplat/TMC2; reuse the given installations')
    parser.add_argument('--ffsplat-root', type=Path, default=Path.home() / '.local/share/e19/ffsplat')
    parser.add_argument('--tmc2-root', type=Path, default=Path.home() / 'tools/mpeg-pcc-tmc2')
    parser.add_argument('--python', type=Path, help='Existing ffsplat Python interpreter, for reuse mode')
    parser.add_argument('--adapter-binary', type=Path, help='Reuse an explicitly supplied adapter instead of building')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run setup inside Linux / WSL2.')
    if (ROOT / 'e19.local.json').exists():
        parser.error('e19.local.json already exists; preserve or rename it before reconfiguring.')
    metadata = ROOT / 'dependencies'
    tested = json.loads((metadata / 'tested-environment.json').read_text())
    ffsplat, tmc2 = args.ffsplat_root.expanduser().resolve(), args.tmc2_root.expanduser().resolve()
    if not args.reuse_existing:
        uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
        if not Path(uv).is_file():
            parser.error('Install uv first: https://docs.astral.sh/uv/getting-started/installation/')
        missing = [x for x in ('git', 'cmake', 'ninja', 'make', 'g++') if not shutil.which(x)]
        if missing:
            parser.error('Missing build tools: ' + ', '.join(missing) + '. See README prerequisites.')
        # Do not mutate a pre-existing third-party checkout in installation mode.
        if ffsplat.exists() or tmc2.exists():
            parser.error('Dependency destination exists. Use --reuse-existing or supply new empty destinations.')
        run(['nvidia-smi'])
        ffsplat.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', tested['ffsplat_url'], ffsplat])
        run(['git', '-C', ffsplat, 'checkout', '--detach', tested['ffsplat_commit']])
        for name in ('pyproject.toml', 'uv.lock'):
            shutil.copyfile(metadata / 'ffsplat' / name, ffsplat / name)
        run([uv, 'sync', '--frozen', '--no-dev', '--python', '3.12'], cwd=ffsplat)
        env = {**os.environ, 'TMC2_ROOT': str(tmc2)}
        run(['bash', ROOT / 'tools/tmc2/bootstrap.sh'], cwd=ROOT, env=env)
    python = (args.python or ffsplat / '.venv/bin/python').expanduser().absolute()
    if not python.is_file():
        parser.error('No Python at ' + str(python))
    adapter = args.adapter_binary
    if adapter is None:
        cargo = shutil.which('cargo') or str(Path.home() / '.cargo/bin/cargo')
        run([cargo, 'build', '--locked', '--release', '-p', 'tmc2_adapter'], cwd=ROOT)
        target = Path(os.environ.get('CARGO_TARGET_DIR', str(ROOT / 'target'))).resolve()
        adapter = target / 'release/tmc2_adapter'
    config = {'ffsplat_root': str(ffsplat), 'tmc2_root': str(tmc2),
              'python': str(python), 'adapter_binary': str(adapter.expanduser().resolve())}
    # Check before persisting the configuration, so a failed setup can be retried.
    env = {**os.environ, 'FFSPLAT_ROOT': config['ffsplat_root'], 'TMC2_ROOT': config['tmc2_root'],
           'E19_PYTHON': config['python'], 'E19_ADAPTER_BINARY': config['adapter_binary']}
    run([sys.executable, ROOT / 'e19.py', 'doctor'], env=env)
    (ROOT / 'e19.local.json').write_text(json.dumps(config, indent=2) + '\n')
    print('Ready: python3 e19.py encode input.ply outputs/example')


if __name__ == '__main__':
    main()
