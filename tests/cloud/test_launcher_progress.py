"""Streaming progress is prompt and never exposes a split credential."""

import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_launcher import load_launcher


class StreamingRedactionTests(unittest.TestCase):
    def test_every_chunk_boundary_and_overlapping_key_prefix_is_redacted(self):
        launcher = load_launcher()
        for key in ("abcabc", "a", "ключ🔒", "first\nsecond"):
            source = "prefix " + key + key + " suffix " + key[:max(0, len(key) - 1)]
            expected = source.replace(key, "[redacted]")
            for size in range(1, len(source) + 1):
                redactor = launcher._SecretRedactor(key)
                output = "".join(redactor.feed(source[offset:offset + size]) for offset in range(0, len(source), size))
                output += redactor.feed("", final=True)
                self.assertEqual(output, expected, (key, size))

    def test_regular_progress_line_is_not_held_until_the_next_message(self):
        launcher = load_launcher()
        redactor = launcher._SecretRedactor("long_dummy_credential_for_chunk_boundary_checks")
        self.assertEqual(redactor.feed("Page 1/2\n"), "Page 1/2\n")

    def test_child_progress_arrives_before_exit_and_key_split_across_streams_is_redacted(self):
        launcher = load_launcher()
        key = "DUMMY_ключ_credential_9472"
        with tempfile.TemporaryDirectory(prefix="launcher progress ") as temporary:
            gate = Path(temporary) / "continue"

            class ProgressOutput(io.StringIO):
                def write(self, value):
                    result = super().write(value)
                    if "Страница 1/2" in self.getvalue():
                        gate.touch()
                    return result

            child = (
                "import os, pathlib, sys, time\n"
                "sys.stdout.write('Страница 1/2\\n'); sys.stdout.flush()\n"
                "gate = pathlib.Path(sys.argv[1])\n"
                "for _ in range(200):\n"
                " if gate.exists(): break\n"
                " time.sleep(0.01)\n"
                "else: sys.exit(7)\n"
                "key = os.environ['OLLAMA_API_KEY']; cut = len(key)//2\n"
                "sys.stdout.write('token=' + key[:cut]); sys.stdout.flush()\n"
                "sys.stderr.write(key[cut:] + '\\n'); sys.stderr.flush()\n"
                "sys.stdout.write('Done\\n'); sys.stdout.flush()\n"
            )
            output = ProgressOutput()
            process = launcher.Processes(key)
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                result = process.run([sys.executable, "-c", child, gate], cwd=temporary, conversion=True)
            self.assertEqual(result.returncode, 0, output.getvalue())
            self.assertTrue(gate.is_file())
            self.assertIn("Страница 1/2", output.getvalue())
            self.assertIn("token=[redacted]", output.getvalue())
            self.assertNotIn(key, output.getvalue() + result.stdout + result.stderr)

    def test_utf8_characters_split_between_reads_are_preserved(self):
        launcher = load_launcher()

        class OneByteOutput(io.BytesIO):
            def read1(self, _size):
                return super().read(1)

        class FakeChild:
            def __init__(self):
                self.stdout = OneByteOutput("Страница 🔒\nDUMMY_KEY\n".encode("utf-8"))
                self.returncode = 0

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                self.stdout.close()

            def wait(self):
                return self.returncode

        output = io.StringIO()
        with patch.object(launcher.subprocess, "Popen", return_value=FakeChild()) as popen, contextlib.redirect_stdout(output):
            result = launcher.Processes("DUMMY_KEY").run(["offline-python"], cwd=".", conversion=True)
        self.assertEqual(output.getvalue(), "Страница 🔒\n[redacted]\n")
        self.assertEqual(result.stdout, output.getvalue())
        self.assertEqual(popen.call_args.kwargs["stderr"], subprocess.STDOUT)
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONUNBUFFERED"], "1")


if __name__ == "__main__":
    unittest.main()
