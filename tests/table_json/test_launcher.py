"""Exercise the launcher's API preflight without network or real credentials."""

import contextlib
import io
import json
import os
import ssl
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError, URLError


class LauncherTests(unittest.TestCase):
    def run_table_check(self, validation):
        script = (Path(__file__).parents[2] / "ScriptMacOS-Ollama.sh").read_text()
        code = script.split("<<'TABLECHECK'", 1)[1].split("\n", 1)[1].split("\nTABLECHECK", 1)[0]
        status = 0
        with tempfile.TemporaryDirectory() as root:
            archive = Path(root, "tables.json")
            archive.write_text(json.dumps({"validation": validation, "reconstruction": {"tables": []}}))
            with patch("sys.argv", ["table-check", str(archive)]), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    exec(compile(code, "launcher-table-check", "exec"), {})
                except SystemExit as error:
                    status = error.code
        return status

    def test_rejected_table_makes_conversion_require_review(self):
        self.assertEqual(self.run_table_check([{"status": "rejected"}]), 2)

    def test_valid_tables_complete_successfully(self):
        self.assertEqual(self.run_table_check([{"status": "structurally_valid"}]), 0)

    def test_text_layer_mode_disables_local_ocr_and_keeps_cloud_tables(self):
        script = (Path(__file__).parents[2] / "ScriptMacOS-Ollama.sh").read_text()
        code = "args=(" + script.split("    args=(", 1)[1].split('    echo "--------------------------------------------------"', 1)[0]
        env = dict(os.environ, run_dir="/tmp/offline-test", OLLAMA_BASE_URL="https://ollama.com/v1",
                   OLLAMA_MODEL="gemma4:31b", PAGE_BATCH_SIZE="30", REDO_INLINE_MATH="1",
                   MODE="", FORCE_OCR="0", DISABLE_OCR="1")
        result = subprocess.run(["bash", "-c", code + '\nprintf "%s\\0" "${args[@]}"'],
                                env=env, capture_output=True, check=True)
        args = result.stdout.decode().strip("\0").split("\0")
        self.assertIn("--disable_ocr", args)
        self.assertIn("--table_json_reconstruction", args)
        self.assertIn("--redo_inline_math", args)
        self.assertNotIn("--force_ocr", args)

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
