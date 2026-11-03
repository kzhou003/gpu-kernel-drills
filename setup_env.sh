#!/usr/bin/env bash
# One-time setup on a GPU machine: a venv for the drills + a pip-installed CUDA toolkit for the
# CUDA C++ drills (no system-wide CUDA toolkit or root needed).
#
#   ./setup_env.sh [/path/to/python-that-has-torch]     # default: python3 on PATH
#   source env.sh                                       # every new shell
#
# The venv uses --system-site-packages, so it reuses that Python's torch and triton instead of
# downloading them again. nvcc / headers are pinned to torch's own CUDA version (torch.version.cuda),
# because nvcc, NVVM and ptxas from different CUDA releases do not mix.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
BASE_PY=${1:-python3}

"$BASE_PY" -c "import torch, triton" || { echo "need a python with torch + triton: ./setup_env.sh /path/to/python" >&2; exit 1; }
CUDA_VER=$("$BASE_PY" -c "import torch; print(torch.version.cuda or '')")
[[ -n "$CUDA_VER" ]] || { echo "this torch is a CPU build" >&2; exit 1; }
MAJOR=${CUDA_VER%%.*}
echo "torch CUDA $CUDA_VER"

[[ -d .venv ]] || "$BASE_PY" -m venv --system-site-packages .venv
PIP=".venv/bin/pip"
$PIP install -q --upgrade pip >/dev/null
$PIP install -q pytest ninja markdown-it-py

if command -v nvcc >/dev/null && [[ -z "${FORCE_PIP_CUDA:-}" ]]; then
  echo "using system nvcc: $(command -v nvcc)"
  CUDA_HOME_DIR=$(dirname "$(dirname "$(command -v nvcc)")")
else
  # The nvcc wheel pulls in the newest nvvm/crt; pin them all to torch's CUDA version.
  $PIP install -q "nvidia-cuda-nvcc==${CUDA_VER}.*" "nvidia-nvvm==${CUDA_VER}.*" \
                  "nvidia-cuda-crt==${CUDA_VER}.*" "nvidia-cuda-cccl==${CUDA_VER}.*" \
                  "nvidia-cuda-runtime==${CUDA_VER}.*"
  SITE=$(.venv/bin/python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  BASE_SITE=$("$BASE_PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  # cu13 wheels install under nvidia/cu13/, cu12 wheels under nvidia/cuda_*/ — collect both.
  CUDA_HOME_DIR=$ROOT/.cuda_home
  rm -rf "$CUDA_HOME_DIR"; mkdir -p "$CUDA_HOME_DIR/include" "$CUDA_HOME_DIR/lib64"
  for site in "$BASE_SITE" "$SITE"; do
    for pkg in "$site"/nvidia/cu"$MAJOR" "$site"/nvidia/cuda_*; do
      [[ -d "$pkg" ]] || continue
      [[ -d "$pkg/bin" && ! -e "$CUDA_HOME_DIR/bin" && -x "$pkg/bin/nvcc" ]] && ln -s "$pkg/bin" "$CUDA_HOME_DIR/bin"
      [[ -d "$pkg/nvvm" && ! -e "$CUDA_HOME_DIR/nvvm" ]] && ln -s "$pkg/nvvm" "$CUDA_HOME_DIR/nvvm"
      for f in "$pkg"/include/*; do [[ -e "$f" ]] && ln -sfn "$f" "$CUDA_HOME_DIR/include/$(basename "$f")"; done
      for f in "$pkg"/lib/*; do [[ -e "$f" ]] && ln -sfn "$f" "$CUDA_HOME_DIR/lib64/$(basename "$f")"; done
    done
  done
  [[ -x "$CUDA_HOME_DIR/bin/nvcc" ]] || { echo "nvcc not found after install" >&2; exit 1; }
  if [[ ! -e "$CUDA_HOME_DIR/lib64/libcudart.so" ]]; then
    so=$(ls "$CUDA_HOME_DIR"/lib64/libcudart.so.* | head -1); ln -s "$(basename "$so")" "$CUDA_HOME_DIR/lib64/libcudart.so"
  fi
fi

ARCH=$("$BASE_PY" -c "import torch; c = torch.cuda.get_device_capability(); print(f'{c[0]}.{c[1]}')" 2>/dev/null || echo "")
cat > env.sh <<EOF
# source env.sh — written by setup_env.sh
export CUDA_HOME=$CUDA_HOME_DIR
export PATH=\$CUDA_HOME/bin:$ROOT/.venv/bin:\$PATH
export LD_LIBRARY_PATH=\$CUDA_HOME/lib64:\${LD_LIBRARY_PATH:-}
export PYTHON=$ROOT/.venv/bin/python
${ARCH:+export TORCH_CUDA_ARCH_LIST=$ARCH   # only build for this GPU: faster compiles}
EOF
echo "nvcc: $("$CUDA_HOME_DIR/bin/nvcc" --version | tail -1)"
echo "done. next: source env.sh && ./check.sh all --solutions"
