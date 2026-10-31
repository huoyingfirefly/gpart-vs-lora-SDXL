#!/usr/bin/env sh
# Launch GPart Studio. Paths inside app.py are relative to the repo root.
cd "$(dirname "$0")/.." || exit 1
if [ -x ".venv/bin/python" ]; then
    exec .venv/bin/python gui/app.py "$@"
fi
exec python gui/app.py "$@"
