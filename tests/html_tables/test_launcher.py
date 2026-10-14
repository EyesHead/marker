"""Exercise the launcher's API preflight without network or real credentials."""

import contextlib
import io
import json
import os
import ssl
import sys
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError


class LauncherTests(unittest.TestCase):
    def test_html_conversion_publishes_resources_without_table_json(self):
        script = (Path(__file__).parents[2] / "ScriptMacOS-Ollama.sh").read_text()
        pipeline = "failed=0" + script.split("\nfailed=0", 1)[1]
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            pdf = root / "Документ.pdf"
            pdf.write_bytes(b"untouched source PDF")
            marker = root / "marker_single"
            marker.write_text(
                f"#!{sys.executable}\n"
                "import json, os, sys\n"
                "from pathlib import Path\n"
                "args = sys.argv[1:]\n"
                "Path(os.environ['ARGS_PATH']).write_text(json.dumps(args))\n"
                "name = Path(args[0]).stem\n"
                "out = Path(args[args.index('--output_dir') + 1]) / name\n"
                "out.mkdir(parents=True)\n"
                "(out / (name + '.md')).write_text('<table><tr><td><img src=\"cell.png\"></td><td>$\\\\varphi$</td></tr></table>')\n"
                "(out / 'cell.png').write_bytes(b'table cell image')\n"
                "(out / (name + '_meta.json')).write_text('{}')\n"
            )
            marker.chmod(0o755)
            output = root / "Output"
            note = output / pdf.stem
            note.mkdir(parents=True)
            # An old rejected JSON archive must not gate the new HTML result.
            (note / (pdf.stem + "_tables.json")).write_text('{"validation":[{"status":"rejected"}]}')
            work = root / "work"
            work.mkdir()
            env = dict(os.environ, TEST_PDF=str(pdf), MARKER_BIN=str(marker),
                       MARKER_VENV=str(root / "unused-venv"), OUTPUT_DIR=str(output),
                       WORK_ROOT=str(work), ARGS_PATH=str(root / "args.json"),
                       api_key="offline-placeholder", OLLAMA_BASE_URL="https://ollama.com/v1",
                       OLLAMA_MODEL="gemma4:31b", PAGE_BATCH_SIZE="30", REDO_INLINE_MATH="1",
                       MODE="", FORCE_OCR="0", DISABLE_OCR="1")
            result = subprocess.run(["bash", "-c", 'set -u; shopt -s nullglob; pending=("$TEST_PDF");\n' + pipeline],
                                    env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            markdown = (note / (pdf.stem + ".md")).read_text()
            self.assertIn('<img src="cell.png">', markdown)
            self.assertEqual((note / "cell.png").read_bytes(), b"table cell image")
            self.assertFalse((note / (pdf.stem + "_tables.json")).exists())
            self.assertEqual(pdf.read_bytes(), b"untouched source PDF")
            args = json.loads((root / "args.json").read_text())
            self.assertIn("--disable_ocr", args)
            self.assertIn("--redo_inline_math", args)
            self.assertNotIn("--table_json_reconstruction", args)
            self.assertNotIn("offline-placeholder", args)

    def test_json_cache_migrates_to_html_once(self):
        self.check_cache_migration(legacy_json=True)

    def test_previous_html_cache_migrates_to_geometry_once(self):
        self.check_cache_migration(legacy_json=False)

    def check_cache_migration(self, legacy_json):
        script = (Path(__file__).parents[2] / "ScriptMacOS-Ollama.sh").read_text()
        cache_code = "repo_origin=" + script.split("\nrepo_origin=", 1)[1].split("\nWORK_ROOT=", 1)[0]
        def git(*args):
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            remote = root / "remote"
            remote.mkdir()
            git("init", "-b", "html-tables", str(remote))
            git("-C", str(remote), "config", "user.email", "offline@example.invalid")
            git("-C", str(remote), "config", "user.name", "Offline test")
            table = remote / "marker/processors/llm/llm_table.py"
            table.parent.mkdir(parents=True)
            table.write_text('data-bbox')
            (remote / "marker/processors/table_geometry.py").write_text('# physical grid extraction')
            converter = remote / "marker/converters/pdf.py"
            converter.parent.mkdir(parents=True)
            converter.write_text('page_batch_size')
            git("-C", str(remote), "add", ".")
            git("-C", str(remote), "commit", "-m", "HTML workflow")
            html_revision = git("-C", str(remote), "rev-parse", "HEAD")
            cache = root / "cache"
            source = cache / "source"
            git("clone", str(remote), str(source))
            git("-C", str(source), "config", "user.email", "offline@example.invalid")
            git("-C", str(source), "config", "user.name", "Offline test")
            legacy = source / "marker/processors/llm/llm_table_json.py"
            if legacy_json:
                legacy.write_text('# legacy JSON reconstruction')
            (source / 'marker/processors/table_geometry.py').unlink()
            git("-C", str(source), "add", ".")
            git("-C", str(source), "commit", "-m", "Legacy JSON workflow")
            (cache / "installed-revision").write_text(git("-C", str(source), "rev-parse", "HEAD"))
            venv = cache / "venv"
            (venv / "bin").mkdir(parents=True)
            python = venv / "bin/python"
            python.write_text('#!/bin/bash\nprintf "install\\n" >> "$INSTALL_LOG"\n')
            python.chmod(0o755)
            (venv / "bin/marker_single").touch()
            (venv / "bin/marker_single").chmod(0o755)
            env = dict(os.environ, MARKER_REPO=str(remote), MARKER_REF="html-tables",
                       MARKER_CACHE=str(cache), MARKER_ROOT=str(source), MARKER_VENV=str(venv),
                       INSTALL_LOG=str(root / "installs.log"))
            result = subprocess.run(["bash", "-c", "set -u;\n" + cache_code], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertFalse(legacy.exists(), "Old JSON cache was reused instead of migrating")
            self.assertEqual(git("-C", str(source), "rev-parse", "HEAD"), html_revision)
            self.assertEqual((cache / "installed-revision").read_text().strip(), html_revision)
            self.assertEqual((root / "installs.log").read_text().splitlines(), ["install"])
            # Remove the remote: an ordinary second run must work offline.
            remote.rename(root / "unavailable-remote")
            result = subprocess.run(["bash", "-c", "set -u;\n" + cache_code], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "installs.log").read_text().splitlines(), ["install"])

    @staticmethod
    def success_response():
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"choices":[{"message":{"content":"OK"}}]}'
        return response

    def run_preflight(self, responses):
        script = (Path(__file__).parents[2] / "ScriptMacOS-Ollama.sh").read_text()
        code = script.split("<<'PY'\n", 1)[1].split("\nPY\nthen", 1)[0]
        stderr = io.StringIO()
        status = 0
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-placeholder"}), \
                patch("sys.argv", ["preflight", "https://ollama.com/v1", "gemma4:31b"]), \
                patch("urllib.request.urlopen", side_effect=responses) as request, \
                patch("time.sleep"), contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(io.StringIO()):
            try:
                exec(compile(code, "launcher-preflight", "exec"), {})
            except SystemExit as error:
                status = error.code
        self.last_request = request.call_args.args[0]
        self.last_context = request.call_args.kwargs.get("context")
        return status, request.call_count, stderr.getvalue()

    def test_preflight_loads_ca_certificates_and_keeps_https_verification(self):
        status, _, _ = self.run_preflight([self.success_response()])
        self.assertEqual(status, 0)
        self.assertIsNotNone(self.last_context)
        self.assertTrue(self.last_context.check_hostname)
        self.assertEqual(self.last_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(self.last_context.get_ca_certs())

    def test_preflight_checks_model_access_with_minimal_chat_request(self):
        status, _, _ = self.run_preflight([self.success_response()])
        self.assertEqual(status, 0)
        self.assertEqual(self.last_request.get_method(), "POST")
        self.assertEqual(self.last_request.full_url, "https://ollama.com/v1/chat/completions")
        payload = json.loads(self.last_request.data)
        self.assertEqual(payload["model"], "gemma4:31b")
        self.assertEqual(payload["max_tokens"], 1)
        self.assertNotIn("offline-placeholder", self.last_request.data.decode())

    def test_transient_network_error_is_retried(self):
        status, calls, _ = self.run_preflight([URLError(ConnectionResetError("connection reset")), self.success_response()])
        self.assertEqual(status, 0)
        self.assertEqual(calls, 2)

    def test_persistent_error_reports_reason_without_key(self):
        status, calls, message = self.run_preflight([URLError(OSError("offline-placeholder connection failed"))] * 3)
        self.assertEqual(status, 1)
        self.assertEqual(calls, 3)
        self.assertIn("OSError", message)
        self.assertNotIn("offline-placeholder", message)

    def test_invalid_key_http_error_is_not_retried(self):
        error = HTTPError("https://ollama.com/v1/models", 401, "unauthorized", {}, None)
        status, calls, message = self.run_preflight([error])
        self.assertEqual(status, 1)
        self.assertEqual(calls, 1)
        self.assertIn("401", message)


if __name__ == "__main__":
    unittest.main()
