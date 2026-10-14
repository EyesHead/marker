"""Compatibility wrapper checks; cache/output/key contracts live in tests/cloud."""

import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which("sh"), "POSIX shell wrappers require sh")
class LauncherWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="marker wrapper ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "Папка с пробелами"
        self.folder.mkdir()
        for name in ("ScriptOllama.py", "ScriptOllama.sh", "ScriptMacOS-Ollama.sh"):
            source = ROOT / name
            self.assertTrue(source.is_file(), f"Missing portable launcher file: {name}")
            shutil.copy2(source, self.folder / name)
        self.env = {k: v for k, v in os.environ.items() if k not in {"OLLAMA_API_KEY", "OPENAI_API_KEY"}}

    def run_wrapper(self, wrapper, *args):
        return subprocess.run(["sh", str(self.folder / wrapper), *args], cwd=self.root,
                              env=self.env, capture_output=True, text=True)

    def test_compatible_mac_entry_uses_pdf_and_output_beside_launcher(self):
        result = self.run_wrapper("ScriptMacOS-Ollama.sh")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((self.folder / "PDF").is_dir())
        self.assertTrue((self.folder / "Output").is_dir())
        self.assertFalse((self.root / "PDF").exists())
        self.assertIn("No pending PDF", result.stdout)

    def test_generic_shell_wrapper_forwards_help_without_installing(self):
        result = self.run_wrapper("ScriptOllama.sh", "--help")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("--context-tokens", result.stdout)
        self.assertFalse((self.folder / "PDF").exists())

    def test_shell_wrapper_forwards_a_unicode_directory_argument(self):
        target = self.root / "Другая папка с пробелами"
        result = self.run_wrapper("ScriptMacOS-Ollama.sh", "--directory", str(target))
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((target / "PDF").is_dir())
        self.assertTrue((target / "Output").is_dir())
        self.assertFalse((self.folder / "PDF").exists())

    def test_missing_python_companion_reports_an_actionable_error(self):
        (self.folder / "ScriptOllama.py").unlink()
        result = self.run_wrapper("ScriptMacOS-Ollama.sh")
        self.assertEqual(result.returncode, 1)
        self.assertIn("ScriptOllama.py", result.stderr)
        self.assertIn("download", result.stderr.lower())


if __name__ == "__main__":
    unittest.main()
