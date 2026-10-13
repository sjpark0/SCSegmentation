#!/usr/bin/env bash
# Torch tests run inside the scsam3 image (the host venv has no torch).
#   scripts/test_torch.sh [pytest args]     (default: the whole tests_torch/)
set -euo pipefail
if [ $# -eq 0 ]; then set -- tests_torch; fi
exec "$(dirname "$0")/docker_run.sh" scsam3 python -m pytest -q "$@"
