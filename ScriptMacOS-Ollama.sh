#!/bin/sh
# Compatible macOS entry point for the shared portable launcher.
set -eu
script_dir="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -f "$script_dir/ScriptOllama.sh" ]; then
    printf '%s\n' 'Missing ScriptOllama.sh. Download it and ScriptOllama.py beside this wrapper (see README).' >&2
    exit 1
fi
exec sh "$script_dir/ScriptOllama.sh" "$@"
