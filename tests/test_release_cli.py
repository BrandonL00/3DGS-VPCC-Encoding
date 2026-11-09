"""Check the configuration boundary without requiring CUDA or third-party code."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('e19_cli', ROOT / 'e19.py')
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


class ReleaseCLITest(unittest.TestCase):
    def test_explicit_dependencies_with_spaces_are_one_argument(self):
        with tempfile.TemporaryDirectory(prefix='e19 test ') as tmp:
            root = Path(tmp)
            python = root / 'python'
            python.touch()
            config = {'python': str(python), 'ffsplat_root': str(root / 'my ffsplat'),
                      'tmc2_root': str(root / 'my tmc2'), 'adapter_binary': str(root / 'adapter')}
            with patch.object(cli, 'configuration', return_value=config), \
                 patch.object(cli.sys, 'platform', 'linux'), \
                 patch.object(cli.subprocess, 'run') as run:
                run.return_value.returncode = 0
                self.assertEqual(cli.main(['encode', str(root / 'input.ply'), str(root / 'out')]), 0)
                command = run.call_args.args[0]
                self.assertIn(str(root / 'my ffsplat'), command)
                self.assertIn(str(root / 'my tmc2'), command)
                self.assertEqual(run.call_args.kwargs['env']['VPCC_DISCORD_DISABLED'], '1')

    def test_environment_override_beats_local_configuration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'e19.local.json').write_text('{"ffsplat_root":"/local/ffsplat"}')
            with patch.object(cli, 'ROOT', root), patch.dict(os.environ, {'FFSPLAT_ROOT': '/override/ffsplat'}, clear=True):
                config = cli.configuration()
                self.assertEqual(config['ffsplat_root'], '/override/ffsplat')
                self.assertEqual(config['python'], '/override/ffsplat/.venv/bin/python')

    def test_decode_rejects_overwrite_before_running(self):
        with tempfile.TemporaryDirectory() as tmp:
            existing = Path(tmp) / 'existing.ply'
            existing.touch()
            with patch.object(cli.sys, 'platform', 'linux'), \
                 patch.object(cli, 'configuration', return_value={'python': str(existing)}), \
                 patch.object(cli.subprocess, 'run') as run:
                with self.assertRaises(SystemExit):
                    cli.main(['decode', 'package', str(existing)])
                run.assert_not_called()

    def test_venv_interpreter_symlink_is_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            interpreter = root / 'venv/bin/python'
            interpreter.parent.mkdir(parents=True)
            interpreter.symlink_to('/usr/bin/python3')
            with patch.object(cli, 'ROOT', root), patch.dict(os.environ, {'E19_PYTHON': str(interpreter)}, clear=True):
                self.assertEqual(cli.configuration()['python'], str(interpreter))


if __name__ == '__main__':
    unittest.main()
