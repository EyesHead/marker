#!/usr/bin/env python3
"""Portable, standard-library launcher for adaptive Ollama Cloud conversion."""

import argparse
import codecs
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import warnings


REPOSITORY = "https://github.com/EyesHead/marker.git"
REF = "codex/my-change"
CONVERTER = "marker.converters.ollama.OllamaPdfConverter"


class LauncherError(Exception):
    pass


def default_cache_dir(platform=None, environ=None, home=None):
    platform = sys.platform if platform is None else platform
    environ = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    if environ.get("MARKER_CACHE"):
        return Path(environ["MARKER_CACHE"]).expanduser()
    if platform == "win32":
        return Path(environ.get("LOCALAPPDATA", home / "AppData/Local")) / "eyeshead-marker"
    if platform == "darwin":
        legacy = Path(environ.get("XDG_CACHE_HOME", home / ".cache")) / "eyeshead-marker"
        # Existing virtual environments contain absolute paths; reuse, do not move.
        return legacy if legacy.exists() else home / "Library/Caches/eyeshead-marker"
    return Path(environ.get("XDG_CACHE_HOME", home / ".cache")) / "eyeshead-marker"


def venv_python(venv, platform=None):
    platform = sys.platform if platform is None else platform
    return venv / ("Scripts/python.exe" if platform == "win32" else "bin/python")


def nonsecret_environment():
    return {k: v for k, v in os.environ.items() if k not in {"OLLAMA_API_KEY", "OPENAI_API_KEY"}}


def request_api_key():
    """Return an in-memory credential, or None on cancellation/unavailable input."""
    if sys.platform == "darwin" and shutil.which("osascript"):
        script = ('text returned of (display dialog "Ollama Cloud API key:" '
                  'default answer "" with hidden answer buttons {"Cancel", "Continue"} '
                  'default button "Continue" with title "Marker / Ollama")')
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True,
                                    text=True, env=nonsecret_environment())
            if result.returncode == 0:
                return result.stdout.strip() or None
            if "(-128)" in result.stderr:
                return None
        except OSError:
            pass
    if sys.stdin.isatty():
        try:
            with warnings.catch_warnings():
                # getpass's echoing fallback must never read the credential.
                warnings.simplefilter("error", getpass.GetPassWarning)
                return getpass.getpass("Ollama Cloud API key: ").strip() or None
        except (EOFError, KeyboardInterrupt, getpass.GetPassWarning):
            return None
    try:
        import tkinter
        from tkinter import simpledialog
    except ImportError:
        return None
    try:
        root = tkinter.Tk()
        root.withdraw()
        try:
            value = simpledialog.askstring("Marker / Ollama", "Ollama Cloud API key:",
                                           show="*", parent=root)
            return value.strip() if value and value.strip() else None
        finally:
            root.destroy()
    except (RuntimeError, tkinter.TclError):
        return None


class _SecretRedactor:
    """Redact complete credentials even when reads split their characters."""

    def __init__(self, secret):
        self.secret = secret
        self.pending = ""

    def feed(self, text, *, final=False):
        text = self.pending + text
        self.pending = ""
        if not self.secret:
            return text
        pieces, offset = [], 0
        while True:
            match = text.find(self.secret, offset)
            if match < 0:
                break
            pieces.extend((text[offset:match], "[redacted]"))
            offset = match + len(self.secret)
        tail = text[offset:]
        keep = 0 if final else min(len(tail), len(self.secret) - 1)
        while keep and not tail.endswith(self.secret[:keep]):
            keep -= 1
        if keep:
            self.pending = tail[-keep:]
            tail = tail[:-keep]
        pieces.append(tail)
        return "".join(pieces)


class Processes:
    def __init__(self, secret):
        self.secret = secret
        self.env = nonsecret_environment()
        self.env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")

    def message(self, text, *, error=False):
        text = str(text).replace(self.secret, "[redacted]") if self.secret else str(text)
        print(text, file=sys.stderr if error else sys.stdout, flush=True)

    def _stream_conversion(self, args, *, cwd, env):
        # One pipe preserves ordering when a credential is split between the
        # child's stdout/stderr. Decode incrementally before redacting so UTF-8
        # characters split by an OS pipe read remain intact.
        redactor = _SecretRedactor(self.secret)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        output = []

        def emit(text):
            if text:
                output.append(text)
                sys.stdout.write(text)
                sys.stdout.flush()

        with subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT) as process:
            while True:
                chunk = process.stdout.read1(4096)
                if not chunk:
                    break
                emit(redactor.feed(decoder.decode(chunk)))
            emit(redactor.feed(decoder.decode(b"", final=True), final=True))
            return subprocess.CompletedProcess(args, process.wait(), "".join(output), "")

    def run(self, args, *, cwd=None, conversion=False, quiet=False):
        env = dict(self.env)
        if conversion:
            env["OLLAMA_API_KEY"] = self.secret
            env["PYTHONPATH"] = str(cwd)
        try:
            if conversion:
                return self._stream_conversion(list(map(str, args)), cwd=cwd, env=env)
            result = subprocess.run(list(map(str, args)), cwd=cwd, env=env,
                                    capture_output=True, text=True, encoding="utf-8", errors="replace")
        except OSError as error:
            raise LauncherError(f"Cannot start {Path(args[0]).name}: {type(error).__name__}") from None
        if not quiet or result.returncode:
            if result.stdout.strip():
                self.message(result.stdout.rstrip())
            if result.stderr.strip():
                self.message(result.stderr.rstrip(), error=True)
        return result

    def checked(self, args, *, cwd=None, quiet=False):
        result = self.run(args, cwd=cwd, quiet=quiet)
        if result.returncode:
            raise LauncherError(f"{Path(args[0]).name} failed (exit {result.returncode}).")
        return result.stdout.strip()


def adaptive_source(source):
    return ((source / "marker/converters/ollama.py").is_file()
            and not (source / "marker/processors/llm/llm_table_json.py").exists())


def ensure_install(args, processes):
    cache = args.cache_dir
    source, venv = cache / "source", cache / "venv"
    cache.mkdir(parents=True, exist_ok=True)
    if not (source / ".git").is_dir():
        if source.exists():
            raise LauncherError(f"Cache source is not a Git repository: {source}")
        processes.message(f"Installing Marker from {args.repo} ({args.ref})...")
        with tempfile.TemporaryDirectory(prefix="source.", dir=cache) as temporary:
            clone = Path(temporary) / "source"
            processes.checked(["git", "clone", "--depth", "1", "--branch", args.ref, args.repo, clone])
            if not adaptive_source(clone):
                raise LauncherError(f"Ref {args.ref} does not contain the adaptive cloud converter.")
            clone.rename(source)
    origin = processes.checked(["git", "-C", source, "remote", "get-url", "origin"], quiet=True)
    if origin != args.repo:
        raise LauncherError(f"Cache belongs to another repository: {source}")
    if args.update_marker or not adaptive_source(source):
        dirty = processes.checked(["git", "-C", source, "status", "--porcelain"], quiet=True)
        if dirty:
            raise LauncherError(f"Cache contains local changes; update it manually: {source}")
        processes.message("Updating Marker cache for adaptive cloud conversion...")
        processes.checked(["git", "-C", source, "fetch", "--depth", "1", "origin", args.ref])
        processes.checked(["git", "-C", source, "checkout", "--detach", "FETCH_HEAD"])
        if not adaptive_source(source):
            raise LauncherError(f"Ref {args.ref} does not contain the adaptive cloud converter.")
    revision = processes.checked(["git", "-C", source, "rev-parse", "HEAD"], quiet=True)
    stamp = cache / "installed-revision"
    installed = stamp.read_text(encoding="utf-8").strip() if stamp.is_file() else ""
    python = venv_python(venv)
    if not python.is_file() or not os.access(python, os.X_OK) or installed != revision:
        processes.message(f"Installing isolated Python dependencies in {venv}...")
        if not python.is_file():
            processes.checked([sys.executable, "-m", "venv", venv])
        processes.checked([python, "-m", "pip", "install", "--editable", source])
        stamp.write_text(revision + "\n", encoding="utf-8")
    return python, source, revision


def completion_metadata(folder, name):
    try:
        metadata = json.loads((folder / (name + "_meta.json")).read_text(encoding="utf-8"))
        status = metadata["cloud_conversion"]
        valid = (isinstance(status, dict) and status.get("status") in {"complete", "incomplete"}
                 and isinstance(status.get("reasons"), list)
                 and all(isinstance(reason, str) for reason in status["reasons"])
                 and type(status.get("requests")) is int and status["requests"] >= 0)
        if valid:
            return status
    except (OSError, ValueError, TypeError, KeyError):
        pass
    raise LauncherError(f"Missing or invalid cloud completion metadata for {name}.")


def complete_output(folder, name):
    if not (folder / (name + ".md")).is_file() or (folder / (name + "_tables.json")).exists():
        return False
    try:
        return completion_metadata(folder, name)["status"] == "complete"
    except LauncherError:
        return False


def conversion_arguments(args, python, pdf, output):
    command = [python, "-m", "marker.scripts.convert_single", pdf,
               "--converter_cls", CONVERTER, "--output_dir", output,
               "--output_format", "markdown", "--use_llm",
               "--llm_service", "marker.services.openai.OpenAIService",
               "--openai_base_url", args.base_url, "--openai_model", args.model,
               "--openai_image_format", "png", "--max_concurrency", "1", "--timeout", "180",
               "--page_batch_size", args.page_batch_size, "--redo_inline_math"]
    for field in ("context_tokens", "max_output_tokens", "max_images", "crop_dpi", "max_attempts", "max_requests", "max_concurrency"):
        command.extend(["--cloud_" + field, getattr(args, field)])
    if args.disable_ocr:
        command.append("--disable_ocr")
    if args.force_ocr:
        command.append("--force_ocr")
    return list(map(str, command))


def publish_output(generated, note, name, secret):
    # Reject an accidental credential echo before it can reach published files.
    if secret and any(secret.encode("utf-8") in path.read_bytes()
                      for path in generated.rglob("*") if path.is_file()):
        raise LauncherError("Converter output contains a credential; result was not published.")
    with tempfile.TemporaryDirectory(prefix=".marker-publish-", dir=note.parent) as temporary:
        stage, backup = Path(temporary) / "result", None
        if note.exists():
            shutil.copytree(note, stage)
        else:
            stage.mkdir()
        shutil.copytree(generated, stage, dirs_exist_ok=True)
        (stage / (name + "_tables.json")).unlink(missing_ok=True)
        if note.exists():
            # Keep the backup outside staging cleanup in case rollback also fails.
            backup = Path(tempfile.mkdtemp(prefix=".marker-previous-", dir=note.parent))
            backup.rmdir()
            note.rename(backup)
        try:
            stage.rename(note)
        except OSError:
            if backup is not None:
                try:
                    backup.rename(note)
                except OSError:
                    raise LauncherError(f"Publication failed; previous result retained at {backup}.") from None
            raise
        if backup is not None:
            shutil.rmtree(backup)


def parse_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent,
                        help="Folder containing PDF and Output (default: beside this launcher).")
    parser.add_argument("--cache-dir", type=Path, default=default_cache_dir())
    parser.add_argument("--repo", default=os.environ.get("MARKER_REPO", REPOSITORY))
    parser.add_argument("--ref", default=os.environ.get("MARKER_REF", REF))
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "gemma4:31b"))
    parser.add_argument("--base-url", default=os.environ.get("OLLAMA_BASE_URL", "https://ollama.com/v1"))
    for name, default in (("context_tokens", 8192), ("max_output_tokens", 2048),
                          ("max_images", 3), ("crop_dpi", 600), ("max_attempts", 3), ("max_requests", 0), ("max_concurrency", 2)):
        parser.add_argument("--" + name.replace("_", "-"), type=int,
                            default=os.environ.get("CLOUD_" + name.upper(), default))
    parser.add_argument("--page-batch-size", type=int, default=os.environ.get("PAGE_BATCH_SIZE", 30))
    for name in ("force", "update_marker", "disable_ocr", "force_ocr"):
        parser.add_argument("--" + name.replace("_", "-"), action="store_true",
                            default=os.environ.get(name.upper()) == "1")
    args = parser.parse_args(argv)
    for name in ("context_tokens", "max_output_tokens", "max_images", "crop_dpi", "max_attempts", "page_batch_size"):
        if getattr(args, name) < 1:
            parser.error(name.replace("_", "-") + " must be positive")
    if args.max_requests < 0 or args.context_tokens <= args.max_output_tokens:
        parser.error("max-requests must be >= 0; context-tokens must exceed max-output-tokens")
    if not 1 <= args.max_concurrency <= 4:
        parser.error("max-concurrency must be between 1 and 4")
    if args.disable_ocr and args.force_ocr:
        parser.error("DISABLE_OCR=1 and FORCE_OCR=1 cannot be combined")
    args.directory = args.directory.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    return args


def main(argv=None) -> int:
    secret = ""
    processes = Processes(secret)
    try:
        for stream in (sys.stdout, sys.stderr):
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        if sys.version_info < (3, 10):
            raise LauncherError("Python 3.10 or newer is required.")
        try:
            args = parse_arguments(argv)
        except SystemExit as error:
            return int(error.code)
        pdf_dir, output = args.directory / "PDF", args.directory / "Output"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        pending = []
        for pdf in sorted(pdf_dir.iterdir()):
            if pdf.is_file() and pdf.suffix.lower() == ".pdf":
                if not args.force and complete_output(output / pdf.stem, pdf.stem):
                    processes.message(f"Skipping complete PDF: {pdf.name}")
                else:
                    pending.append(pdf)
        if not pending:
            processes.message("No pending PDF files.")
            return 0
        secret = os.environ.get("OLLAMA_API_KEY", "").strip() or request_api_key()
        if not secret:
            raise LauncherError("Credential entry cancelled or unavailable; use a hidden prompt or OLLAMA_API_KEY.")
        processes = Processes(secret)
        python, source, revision = ensure_install(args, processes)
        processes.message(f"Marker: {source} ({revision[:12]}); Ollama Cloud: {args.model}")
        failures = 0
        with tempfile.TemporaryDirectory(prefix="marker-ollama-") as temporary:
            for index, pdf in enumerate(pending):
                run = Path(temporary) / str(index)
                processes.message(f"Converting: {pdf.name}")
                try:
                    result = processes.run(conversion_arguments(args, python, pdf, run),
                                           cwd=source, conversion=True)
                    generated = run / pdf.stem
                    if not (generated / (pdf.stem + ".md")).is_file():
                        raise LauncherError(f"No Markdown produced for {pdf.name} (exit {result.returncode}).")
                    status = completion_metadata(generated, pdf.stem)
                    if result.returncode and status["status"] != "incomplete":
                        raise LauncherError(f"Conversion failed for {pdf.name} (exit {result.returncode}).")
                    note = output / pdf.stem
                    publish_output(generated, note, pdf.stem, secret)
                    if status["status"] == "incomplete":
                        failures += 1
                        processes.message(f"Incomplete: {pdf.name}; partial output saved to {note}", error=True)
                        for reason in status["reasons"]:
                            processes.message("  " + reason, error=True)
                    else:
                        processes.message(f"Complete: {note / (pdf.stem + '.md')}")
                except (LauncherError, OSError) as error:
                    failures += 1
                    processes.message(f"Error: {error}", error=True)
        processes.message(f"Finished: {len(pending) - failures} complete, {failures} incomplete/failed PDF(s).",
                          error=bool(failures))
        return 1 if failures else 0
    except (LauncherError, OSError) as error:
        processes.message(f"Error: {error}", error=True)
        return 1
    except KeyboardInterrupt:
        processes.message("Cancelled.", error=True)
        return 130
    finally:
        processes.secret = ""
        secret = ""


if __name__ == "__main__":
    sys.exit(main())
