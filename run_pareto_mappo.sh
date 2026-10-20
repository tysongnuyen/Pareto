#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PARETO_PYTHON:-python}" -B -u "$script_dir/train_pareto_mappo.py" "$@"
