#!/bin/sh
# Keep this wrapper and ScriptOllama.py beside the PDF/Output folders.
set -eu
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -f "$script_dir/ScriptOllama.py" ]; then
    printf '%s\n' 'Missing ScriptOllama.py. Download it beside this wrapper (see README).' >&2
    exit 1
fi
if command -v python3 >/dev/null 2>&1; then
    exec python3 "$script_dir/ScriptOllama.py" "$@"
elif command -v python >/dev/null 2>&1; then
    exec python "$script_dir/ScriptOllama.py" "$@"
fi
printf '%s\n' 'Python 3.10 or newer is required. Install Python and run this wrapper again.' >&2
exit 1
