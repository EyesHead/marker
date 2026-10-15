"""Offline launcher contracts; real Git/filesystem, deterministic converter process."""

import contextlib
import builtins
import importlib.util
import io
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from unittest.mock import patch


LAUNCHER = Path(__file__).resolve().parents[2] / "ScriptOllama.py"
RUN = subprocess.run
POPEN = subprocess.Popen
KEY = "DISTINCTIVE_DUMMY_OLLAMA_KEY_9472_keep_in_memory"


def load_launcher():
    spec = importlib.util.spec_from_file_location("portable_ollama_launcher", LAUNCHER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git(*args):
    env = {k: v for k, v in os.environ.items() if k not in {"OLLAMA_API_KEY", "OPENAI_API_KEY"}}
    result = RUN(["git", *map(str, args)], capture_output=True, text=True, env=env)
    if result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip()


class OfflineProcesses:
    """Stub only venv/pip/model execution, while running local Git commands."""

    def __init__(self):
        self.calls = []
        self.status = "complete"
        self.reasons = []
        self.returncode = 0
        self.metadata = True
        self.markdown = True
        self.fail_install = False
        self.emit_secret = False
        self.write_secret = False
        self.publishable = None

    def __call__(self, args, **kwargs):
        args = list(map(str, args))
        child_env = dict(kwargs.get("env", {}))
        self.calls.append((args, child_env))
        if "venv" in args and "-m" in args:
            venv = Path(args[-1])
            python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            python.parent.mkdir(parents=True)
            python.write_text("offline interpreter placeholder", encoding="utf-8")
            python.chmod(0o755)
            return subprocess.CompletedProcess(args, 0, "", "")
        if "pip" in args and "-m" in args:
            return subprocess.CompletedProcess(args, 1 if self.fail_install else 0, "", "install failure" if self.fail_install else "")
        if "marker.scripts.convert_single" in args:
            pdf = Path(args[args.index("marker.scripts.convert_single") + 1])
            output = Path(args[args.index("--output_dir") + 1]) / pdf.stem
            output.mkdir(parents=True, exist_ok=True)
            if self.markdown:
                (output / (pdf.stem + ".md")).write_text(
                    '<table><tr><td><img src="cell.png"></td><td>$\\varphi$</td></tr></table>', encoding="utf-8"
                )
            (output / "cell.png").write_bytes(b"unchanged native image pixels")
            if self.write_secret:
                (output / "cell.png").write_bytes(b"accidental credential echo: " + KEY.encode())
            if self.metadata:
                state={"status":self.status,"reasons":self.reasons,"requests":2}
                if self.publishable is not None:
                    state['publishable']=self.publishable
                    state['work_dir']=str(output/'.marker-work')
                (output / (pdf.stem + "_meta.json")).write_text(json.dumps({"cloud_conversion": {
                    **state
                }}), encoding="utf-8")
            stdout = "converter " + KEY if self.emit_secret else "converter finished"
            stderr = "diagnostic " + KEY if self.emit_secret else ""
            return subprocess.CompletedProcess(args, self.returncode, stdout, stderr)
        return RUN(args, **kwargs)

    def conversions(self):
        return [call for call in self.calls if "marker.scripts.convert_single" in call[0]]

    def installs(self):
        return [call for call in self.calls if "pip" in call[0]]

    def popen(self, args, **kwargs):
        if "marker.scripts.convert_single" not in args:
            return POPEN(args, **kwargs)
        result = self(args, env=kwargs.get("env"), cwd=kwargs.get("cwd"))

        class ConversionProcess:
            def __init__(self):
                self.stdout = io.BytesIO((result.stdout + result.stderr).encode("utf-8"))
                self.returncode = result.returncode

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.stdout.close()

            def wait(self):
                return self.returncode

        return ConversionProcess()


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(LAUNCHER.is_file(), "Missing portable ScriptOllama.py launcher")
        self.launcher = load_launcher()
        self.tmp = tempfile.TemporaryDirectory(prefix="launcher spaces ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.folder = self.root / "Обсидиан папка"
        (self.folder / "PDF").mkdir(parents=True)
        self.pdf = self.folder / "PDF" / "СП текст с пробелами.PDF"
        self.pdf.write_bytes(b"untouched PDF source")
        self.cache = self.root / "cache with spaces"
        self.remote = self.root / "offline remote"
        self.remote.mkdir()
        git("init", "-b", "adaptive", self.remote)
        git("-C", self.remote, "config", "user.email", "offline@example.invalid")
        git("-C", self.remote, "config", "user.name", "Offline test")
        converter = self.remote / "marker/converters/ollama.py"
        converter.parent.mkdir(parents=True)
        converter.write_text("# adaptive converter capability", encoding="utf-8")
        (converter.parent/'ollama_document.py').write_text('# staged whole-object converter',encoding='utf-8')
        native = self.remote / "marker/processors/llm/llm_native.py"
        native.parent.mkdir(parents=True)
        native.write_text("# native whole-object processor", encoding="utf-8")
        defaults = self.remote / "marker/cloud/defaults.py"
        defaults.parent.mkdir(parents=True)
        defaults.write_text("# cloud model defaults", encoding="utf-8")
        (self.remote / "pyproject.toml").write_text("# installation fixture", encoding="utf-8")
        git("-C", self.remote, "add", ".")
        git("-C", self.remote, "commit", "-m", "Adaptive source")
        self.revision = git("-C", self.remote, "rev-parse", "HEAD")
        self.process = OfflineProcesses()

    def run_launcher(self, extra=(), *, env=None, prompt=None):
        arguments = ["--directory", str(self.folder), "--cache-dir", str(self.cache),
                     "--repo", str(self.remote), "--ref", "adaptive", *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        environment = {"OLLAMA_API_KEY": KEY}
        if env is not None:
            environment.update(env)
        with patch.dict(os.environ, environment), patch.object(self.launcher.subprocess, "run", self.process), \
                patch.object(self.launcher.subprocess, "Popen", self.process.popen), \
                contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if prompt is None:
                status = self.launcher.main(arguments)
            else:
                with patch.object(self.launcher, "request_api_key", return_value=prompt):
                    status = self.launcher.main(arguments)
        return status, stdout.getvalue(), stderr.getvalue()

    @property
    def note(self):
        return self.folder / "Output" / self.pdf.stem

    def test_first_install_uses_isolated_python_and_preserves_unicode_paths(self):
        status, stdout, stderr = self.run_launcher()
        self.assertEqual(status, 0, stdout + stderr)
        self.assertEqual((self.cache / "installed-revision").read_text().strip(), self.revision)
        self.assertEqual(git("-C", self.cache / "source", "rev-parse", "HEAD"), self.revision)
        self.assertEqual(len(self.process.installs()), 1)
        args, env = self.process.conversions()[0]
        relative = "Scripts/python.exe" if os.name == "nt" else "bin/python"
        self.assertEqual(Path(args[0]), self.cache / "venv" / relative)
        self.assertEqual(args[1:4], ["-m", "marker.scripts.convert_single", str(self.pdf)])
        self.assertEqual(env["PYTHONPATH"], str(self.cache / "source"))
        self.assertEqual((self.note / "cell.png").read_bytes(), b"unchanged native image pixels")
        self.assertIn('<img src="cell.png">', (self.note / (self.pdf.stem + ".md")).read_text())
        self.assertEqual(self.pdf.read_bytes(), b"untouched PDF source")

    def test_real_launcher_prints_unicode_paths_under_an_ascii_stdio_locale(self):
        self.note.mkdir(parents=True)
        (self.note / (self.pdf.stem + ".md")).write_text("complete output", encoding="utf-8")
        (self.note / (self.pdf.stem + "_meta.json")).write_text(json.dumps({"cloud_conversion": {
            "status": "complete", "reasons": [], "requests": 0
        }}), encoding="utf-8")
        env = {k: v for k, v in os.environ.items() if k not in {"OLLAMA_API_KEY", "OPENAI_API_KEY"}}
        env.update(PYTHONIOENCODING="ascii", PYTHONUTF8="0")
        result = RUN([sys.executable, str(LAUNCHER), "--directory", str(self.folder)], env=env,
                     capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(self.pdf.name, result.stdout)

    def test_normal_second_conversion_works_when_remote_is_unavailable(self):
        self.assertEqual(self.run_launcher()[0], 0)
        self.remote.rename(self.root / "unavailable remote")
        self.process.calls.clear()
        status, stdout, stderr = self.run_launcher(["--force"])
        self.assertEqual(status, 0, stdout + stderr)
        self.assertFalse(self.process.installs())
        self.assertEqual(len(self.process.conversions()), 1)
        self.assertFalse(any("fetch" in args or "clone" in args for args, _ in self.process.calls))

    def test_complete_output_skips_install_and_credentials(self):
        self.assertEqual(self.run_launcher()[0], 0)
        self.process.calls.clear()
        self.remote.rename(self.root / "unavailable remote")
        with patch.object(self.launcher, "request_api_key", side_effect=AssertionError("unexpected prompt")):
            status, stdout, stderr = self.run_launcher(env={"OLLAMA_API_KEY": ""})
        self.assertEqual(status, 0, stdout + stderr)
        self.assertFalse(self.process.calls)

    def test_legacy_cache_migrates_once_then_stays_offline(self):
        self.check_cache_migration(legacy_json=True)

    def test_previous_html_cache_migrates_once_then_stays_offline(self):
        self.check_cache_migration(legacy_json=False)

    def check_cache_migration(self, *, legacy_json):
        source = self.cache / "source"
        source.parent.mkdir()
        git("clone", self.remote, source)
        git("-C", source, "config", "user.email", "offline@example.invalid")
        git("-C", source, "config", "user.name", "Offline test")
        (source / "marker/converters/ollama.py").unlink()
        legacy = source / "marker/processors/llm/llm_table_json.py"
        if legacy_json:
            legacy.parent.mkdir(parents=True, exist_ok=True)
            legacy.write_text("# old JSON pipeline", encoding="utf-8")
        git("-C", source, "add", ".")
        git("-C", source, "commit", "-m", "Legacy installation")
        (self.cache / "installed-revision").write_text(git("-C", source, "rev-parse", "HEAD"))
        status, stdout, stderr = self.run_launcher()
        self.assertEqual(status, 0, stdout + stderr)
        self.assertFalse(legacy.exists())
        self.assertEqual(git("-C", source, "rev-parse", "HEAD"), self.revision)
        self.assertEqual(len(self.process.installs()), 1)
        self.remote.rename(self.root / "unavailable remote")
        self.process.calls.clear()
        self.assertEqual(self.run_launcher(["--force"])[0], 0)
        self.assertFalse(any("fetch" in args for args, _ in self.process.calls))
        self.assertFalse(self.process.installs())

    def test_dirty_cache_is_preserved_during_requested_update(self):
        self.assertEqual(self.run_launcher()[0], 0)
        custom = self.cache / "source" / "custom.txt"
        custom.write_text("local cache change", encoding="utf-8")
        self.process.calls.clear()
        status, stdout, stderr = self.run_launcher(["--force", "--update-marker"])
        self.assertEqual(status, 1)
        self.assertEqual(custom.read_text(), "local cache change")
        self.assertFalse(any("fetch" in args for args, _ in self.process.calls))
        self.assertFalse(self.process.conversions())

    def test_explicit_update_reinstalls_only_when_revision_changes(self):
        self.assertEqual(self.run_launcher()[0], 0)
        (self.remote / "new-capability.txt").write_text("new source revision")
        git("-C", self.remote, "add", ".")
        git("-C", self.remote, "commit", "-m", "Updated source")
        self.process.calls.clear()
        status, stdout, stderr = self.run_launcher(env={"FORCE": "1", "UPDATE_MARKER": "1"})
        self.assertEqual(status, 0, stdout + stderr)
        self.assertEqual(len(self.process.installs()), 1)
        self.assertEqual((self.cache / "installed-revision").read_text().strip(), git("-C", self.remote, "rev-parse", "HEAD"))

    def test_failed_install_is_retried_without_a_second_clone(self):
        self.process.fail_install = True
        self.assertEqual(self.run_launcher()[0], 1)
        self.assertFalse((self.cache / "installed-revision").exists())
        self.process.fail_install = False
        self.process.calls.clear()
        self.assertEqual(self.run_launcher()[0], 0)
        self.assertEqual(len(self.process.installs()), 1)
        self.assertFalse(any("clone" in args for args, _ in self.process.calls))

    def test_cloud_cli_defaults_and_ocr_controls(self):
        status, stdout, stderr = self.run_launcher(env={"DISABLE_OCR": "1"})
        self.assertEqual(status, 0, stdout + stderr)
        args = self.process.conversions()[0][0]
        for flag, value in {"--converter_cls": "marker.converters.pdf.PdfConverter",
                            "--output_format": "markdown", "--llm_service": "marker.services.ollama.OllamaService",
                            "--cloud_base_url": "https://ollama.com/v1", "--cloud_model": "kimi-k3",
                            "--cloud_context_tokens": "245760", "--cloud_max_output_tokens": "16384",
                            "--cloud_max_images": "16", "--cloud_crop_dpi": "600",
                            "--cloud_max_attempts": "3", "--cloud_max_requests": "0",
                            "--page_batch_size": "30"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertIn("--use_llm", args)
        self.assertIn("--disable_ocr", args)
        self.assertIn("--redo_inline_math", args)
        self.assertNotIn("--openai_api_key", args)

    def test_environment_and_arguments_configure_cloud_budgets(self):
        status, stdout, stderr = self.run_launcher(
            ["--model", "alternate-model", "--context-tokens", "4096", "--max-images", "2", "--crop-dpi", "450", "--max-requests", "10"],
            env={"CLOUD_MAX_OUTPUT_TOKENS": "1024", "CLOUD_MAX_ATTEMPTS": "2"}
        )
        self.assertEqual(status, 0, stdout + stderr)
        args = self.process.conversions()[0][0]
        for flag, value in {"--cloud_model": "alternate-model", "--cloud_context_tokens": "4096",
                            "--cloud_max_output_tokens": "1024", "--cloud_max_images": "2",
                            "--cloud_crop_dpi": "450", "--cloud_max_requests": "10",
                            "--cloud_max_attempts": "2"}.items():
            self.assertEqual(args[args.index(flag) + 1], value)

    def test_cancelled_credentials_do_not_install_or_convert(self):
        with patch.object(self.launcher, "request_api_key", return_value=None):
            status, stdout, stderr = self.run_launcher(env={"OLLAMA_API_KEY": ""})
        self.assertEqual(status, 1)
        self.assertFalse(self.process.calls)
        self.assertFalse(self.cache.exists())

    def test_api_key_never_enters_output_logs_files_or_process_arguments(self):
        self.process.emit_secret = True
        logs = io.StringIO()
        handler = logging.StreamHandler(logs)
        logging.getLogger().addHandler(handler)
        try:
            status, stdout, stderr = self.run_launcher()
        finally:
            logging.getLogger().removeHandler(handler)
        self.assertEqual(status, 0, stdout + stderr)
        for output in (stdout, stderr, logs.getvalue()):
            self.assertNotIn(KEY, output)
        self.assertIn("[redacted]", stdout + stderr)
        for args, env in self.process.calls:
            self.assertNotIn(KEY, " ".join(args))
            if "marker.scripts.convert_single" in args:
                self.assertEqual(env.get("OLLAMA_API_KEY"), KEY)
                self.assertNotIn("OPENAI_API_KEY", env)
            else:
                self.assertNotIn("OLLAMA_API_KEY", env)
                self.assertNotIn("OPENAI_API_KEY", env)
        for path in self.root.rglob("*"):
            if path.is_file():
                self.assertNotIn(KEY.encode(), path.read_bytes(), str(path))

    def test_incomplete_result_is_published_and_retried(self):
        self.process.status = "incomplete"
        self.process.reasons = ["unreadable formula on page 2"]
        self.process.returncode = 1
        status, stdout, stderr = self.run_launcher()
        self.assertEqual(status, 1)
        self.assertIn("incomplete", stderr.lower())
        self.assertIn("unreadable formula", stderr)
        self.assertTrue((self.note / (self.pdf.stem + ".md")).exists())
        self.assertTrue((self.note / "cell.png").exists())
        metadata = json.loads((self.note / (self.pdf.stem + "_meta.json")).read_text())
        self.assertEqual(metadata["cloud_conversion"]["status"], "incomplete")
        self.process.status, self.process.reasons, self.process.returncode = "complete", [], 0
        self.process.calls.clear()
        self.assertEqual(self.run_launcher()[0], 0)
        self.assertEqual(len(self.process.conversions()), 1)

    def test_unpublishable_staged_result_preserves_existing_note(self):
        self.note.mkdir(parents=True)
        existing=self.note/(self.pdf.stem+'.md')
        existing.write_text('Previous readable result',encoding='utf-8')
        self.process.status='incomplete'
        self.process.publishable=False
        status,stdout,stderr=self.run_launcher(['--force'])
        self.assertEqual(status,1)
        self.assertEqual(existing.read_text(),'Previous readable result')
        self.assertIn('draft',stderr.lower())

    def test_generated_resource_credential_echo_is_rejected_before_publication(self):
        self.assertEqual(self.run_launcher()[0], 0)
        target = self.note / (self.pdf.stem + ".md")
        target.write_text("preserved previous result", encoding="utf-8")
        original_image = (self.note / "cell.png").read_bytes()
        self.process.write_secret = True
        status, stdout, stderr = self.run_launcher(["--force"])
        self.assertEqual(status, 1)
        self.assertIn("credential", stderr.lower())
        self.assertNotIn(KEY, stdout + stderr)
        self.assertEqual(target.read_text(encoding="utf-8"), "preserved previous result")
        self.assertEqual((self.note / "cell.png").read_bytes(), original_image)
        for path in self.note.rglob("*"):
            if path.is_file():
                self.assertNotIn(KEY.encode(), path.read_bytes(), str(path))

    def test_missing_completion_metadata_cannot_replace_existing_note(self):
        self.assertEqual(self.run_launcher()[0], 0)
        target = self.note / (self.pdf.stem + ".md")
        target.write_text("preserved previous result", encoding="utf-8")
        self.process.metadata = False
        status, stdout, stderr = self.run_launcher(["--force"])
        self.assertEqual(status, 1)
        self.assertEqual(target.read_text(), "preserved previous result")
        self.assertIn("metadata", stderr.lower())

    def test_invalid_completion_metadata_is_not_success(self):
        self.process.status = "unknown"
        status, stdout, stderr = self.run_launcher()
        self.assertEqual(status, 1)
        self.assertFalse(self.note.exists())
        self.assertIn("metadata", stderr.lower())

    def test_missing_markdown_is_failure_and_keeps_previous_output(self):
        self.assertEqual(self.run_launcher()[0], 0)
        target = self.note / (self.pdf.stem + ".md")
        target.write_text("preserved previous result", encoding="utf-8")
        self.process.markdown = False
        status, stdout, stderr = self.run_launcher(["--force"])
        self.assertEqual(status, 1)
        self.assertEqual(target.read_text(), "preserved previous result")

    def test_publication_and_rollback_failure_keeps_previous_result_on_disk(self):
        self.assertEqual(self.run_launcher()[0], 0)
        target = self.note / (self.pdf.stem + ".md")
        target.write_text("preserved previous result", encoding="utf-8")
        original_rename = Path.rename
        def failing_publication(path, destination):
            if path.name in {"result", "previous"} or path.name.startswith(".marker-previous-"):
                raise OSError("publication and rollback unavailable")
            return original_rename(path, destination)
        with patch.object(Path, "rename", failing_publication):
            status, stdout, stderr = self.run_launcher(["--force"])
        self.assertEqual(status, 1)
        previous = [path for path in (self.folder / "Output").rglob("*.md")
                    if path.read_text(encoding="utf-8") == "preserved previous result"]
        self.assertTrue(previous, "Publication/rollback failure deleted the previous result")

    def test_full_publication_retires_generated_pngs_to_hidden_backup(self):
        old_images = {
            "formula-p0-0123456789abcdef.png": b"previous formula evidence",
            "unresolved-table-p1-fedcba9876543210.png": b"previous unresolved table",
            "source-p2-0011223344556677.png": b"previous source page",
        }
        user_images = {
            "мой рисунок.png": b"user image",
            "formula-not-generated.png": b"user image with a similar name",
        }
        figure = "figure-p3-abcdef0123456789.png"
        markdown_name = self.pdf.stem + ".md"
        metadata_name = self.pdf.stem + "_meta.json"
        for inventoried in (False, True):
            with self.subTest(inventoried=inventoried):
                parent = self.root / ("inventoried" if inventoried else "legacy")
                note = parent / self.pdf.stem
                note.mkdir(parents=True)
                previous_images = dict(old_images)
                state = {"status": "incomplete", "reasons": ["old unresolved images"], "requests": 4}
                if inventoried:
                    previous_images["tracked-old-resource.png"] = b"inventoried generated image"
                    state["generated_assets"] = list(previous_images)
                old_markdown = "Previous result\n" + "\n".join(
                    f"![previous]({name})" for name in previous_images
                )
                (note / markdown_name).write_text(old_markdown, encoding="utf-8")
                (note / metadata_name).write_text(json.dumps({"cloud_conversion": state}), encoding="utf-8")
                for name, data in {**previous_images, **user_images}.items():
                    (note / name).write_bytes(data)

                generated = parent / "generated"
                generated.mkdir()
                new_markdown = f"Complete formula: $x^2$\n\n![Figure]({figure})\n"
                (generated / markdown_name).write_text(new_markdown, encoding="utf-8")
                (generated / figure).write_bytes(b"new referenced figure")
                (generated / metadata_name).write_text(json.dumps({"cloud_conversion": {
                    "status": "complete", "reasons": [], "requests": 2,
                    "pipeline": "whole-object-v1", "generated_assets": [figure],
                }}), encoding="utf-8")

                backup = self.launcher.publish_output(
                    generated, note, self.pdf.stem, KEY, replace_generated=True
                )

                self.assertIsNotNone(backup)
                self.assertEqual(backup.parent, note.parent)
                self.assertTrue(backup.name.startswith(".marker-previous-"))
                self.assertEqual((backup / markdown_name).read_text(encoding="utf-8"), old_markdown)
                self.assertEqual((note / markdown_name).read_text(encoding="utf-8"), new_markdown)
                for name, data in previous_images.items():
                    self.assertFalse((note / name).exists(), name)
                    self.assertEqual((backup / name).read_bytes(), data)
                for name, data in user_images.items():
                    self.assertEqual((note / name).read_bytes(), data)
                    self.assertEqual((backup / name).read_bytes(), data)
                self.assertEqual((note / figure).read_bytes(), b"new referenced figure")
                self.assertEqual(
                    self.launcher.completion_metadata(note, self.pdf.stem)["status"], "complete"
                )

    def test_legacy_table_json_forces_reconversion_then_is_removed(self):
        self.assertEqual(self.run_launcher()[0], 0)
        legacy = self.note / (self.pdf.stem + "_tables.json")
        legacy.write_text('{"status":"rejected"}')
        self.process.calls.clear()
        self.assertEqual(self.run_launcher()[0], 0)
        self.assertEqual(len(self.process.conversions()), 1)
        self.assertFalse(legacy.exists())

    def test_windows_venv_python_uses_scripts_directory(self):
        python = self.launcher.venv_python(PureWindowsPath(r"C:\Users\Алексей\marker cache\venv"), "win32")
        self.assertEqual(str(python), r"C:\Users\Алексей\marker cache\venv\Scripts\python.exe")

    def test_platform_cache_paths_reuse_existing_mac_installation(self):
        home = self.root / "home"
        home.mkdir()
        self.assertEqual(self.launcher.default_cache_dir("linux", {"XDG_CACHE_HOME": str(home / "xdg")}, home), home / "xdg/eyeshead-marker")
        self.assertEqual(self.launcher.default_cache_dir("win32", {"LOCALAPPDATA": str(home / "local")}, home), home / "local/eyeshead-marker")
        self.assertEqual(self.launcher.default_cache_dir("darwin", {}, home), home / "Library/Caches/eyeshead-marker")
        legacy = home / ".cache/eyeshead-marker"
        legacy.mkdir(parents=True)
        self.assertEqual(self.launcher.default_cache_dir("darwin", {}, home), legacy)

    def test_unavailable_mac_gui_falls_back_to_hidden_terminal_input(self):
        with patch.object(self.launcher.sys, "platform", "darwin"), \
                patch.object(self.launcher.shutil, "which", return_value="/usr/bin/osascript"), \
                patch.object(self.launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "GUI unavailable")), \
                patch.object(self.launcher.sys.stdin, "isatty", return_value=True), \
                patch.object(self.launcher.getpass, "getpass", return_value=KEY):
            self.assertEqual(self.launcher.request_api_key(), KEY)

    def test_cancelled_mac_gui_does_not_reprompt_in_terminal(self):
        with patch.object(self.launcher.sys, "platform", "darwin"), \
                patch.object(self.launcher.shutil, "which", return_value="/usr/bin/osascript"), \
                patch.object(self.launcher.subprocess, "run", return_value=subprocess.CompletedProcess([], 1, "", "User canceled. (-128)")), \
                patch.object(self.launcher.getpass, "getpass", side_effect=AssertionError("unexpected reprompt")):
            self.assertIsNone(self.launcher.request_api_key())

    def test_terminal_echo_warning_cannot_expose_a_credential(self):
        def insecure_prompt(*args, **kwargs):
            warnings.warn("Cannot control terminal echo", self.launcher.getpass.GetPassWarning)
            raise AssertionError("echoing fallback must never run")
        with patch.object(self.launcher.sys, "platform", "linux"), \
                patch.object(self.launcher.sys.stdin, "isatty", return_value=True), \
                patch.object(self.launcher.getpass, "getpass", side_effect=insecure_prompt):
            self.assertIsNone(self.launcher.request_api_key())

    def test_unavailable_tk_gui_without_terminal_returns_cancelled(self):
        original_import = builtins.__import__
        def no_tk(name, *args, **kwargs):
            if name == "tkinter":
                raise ImportError("tkinter unavailable")
            return original_import(name, *args, **kwargs)
        with patch.object(self.launcher.sys, "platform", "linux"), \
                patch.object(self.launcher.sys.stdin, "isatty", return_value=False), \
                patch.object(builtins, "__import__", side_effect=no_tk):
            try:
                result = self.launcher.request_api_key()
            except Exception as error:
                self.fail(f"Unavailable GUI raised {type(error).__name__} instead of cancelling")
            self.assertIsNone(result)


@unittest.skipUnless(os.name == "nt", "Native Windows wrapper checks run only on Windows")
class WindowsWrapperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="marker wrapper ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / "Папка с пробелами"
        self.folder.mkdir()
        for name in ("ScriptOllama.py", "ScriptOllama.cmd", "ScriptOllama.ps1"):
            shutil.copy2(LAUNCHER.parent / name, self.folder / name)
        self.env = {k: v for k, v in os.environ.items() if k not in {"OLLAMA_API_KEY", "OPENAI_API_KEY"}}
        self.env["PYTHONIOENCODING"] = "utf-8"

    def test_cmd_wrapper_forwards_unicode_directory_and_exit_status(self):
        wrapper = self.folder / "ScriptOllama.cmd"
        target = self.root / "Другая папка с пробелами"
        comspec = os.environ.get("COMSPEC", "cmd.exe")
        command = f'"{comspec}" /d /s /c ""{wrapper}" --directory "{target}""'
        result = RUN(command, cwd=self.root, env=self.env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((target / "PDF").is_dir())
        self.assertTrue((target / "Output").is_dir())
        self.assertFalse((self.folder / "PDF").exists())

    def test_powershell_wrapper_forwards_unicode_directory(self):
        powershell = shutil.which("powershell") or shutil.which("pwsh")
        self.assertIsNotNone(powershell, "Windows portability job requires PowerShell")
        target = self.root / "Другая папка с пробелами"
        result = RUN([powershell, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                      str(self.folder / "ScriptOllama.ps1"), "--directory", str(target)],
                     cwd=self.root, env=self.env, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue((target / "PDF").is_dir())
        self.assertTrue((target / "Output").is_dir())


if __name__ == "__main__":
    unittest.main()
