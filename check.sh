#!/usr/bin/env bash
# Run the acceptance tests for one drill (or all of them).
#   ./check.sh t2               # test your mine/t2_*.py
#   ./check.sh c4 -x            # extra args go to pytest
#   ./check.sh all              # every drill
#   ./check.sh t2 --solutions   # run the reference solution instead
#   ./check.sh t2 --bench       # benchmark (needs a CUDA GPU)
set -euo pipefail
cd "$(dirname "$0")"

id="${1:-all}"
shift || true
args=()
bench=0
for a in "$@"; do
  case "$a" in
    --solutions) export DRILL_IMPL=solutions ;;
    --bench) bench=1 ;;
    *) args+=("$a") ;;
  esac
done

PY="${PYTHON:-python}"
# X track (MLX / Metal) runs in its own arm64 venv
if [[ "$id" == x* && -x .venv-mlx/bin/python ]]; then PY=.venv-mlx/bin/python; fi
if (( bench )); then
  exec "$PY" bench.py "$id"
fi

if [[ "$id" == "all" ]]; then
  targets=(tests)
else
  shopt -s nullglob
  targets=(tests/test_"${id}"_*.py)
  if (( ${#targets[@]} == 0 )); then
    echo "no test file matches tests/test_${id}_*.py" >&2
    exit 1
  fi
fi

exec "$PY" -m pytest "${targets[@]}" ${args[@]+"${args[@]}"}
