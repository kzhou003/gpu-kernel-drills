#!/usr/bin/env bash
# Make "Go to Definition" work in VS Code on a machine without a GPU (e.g. a Mac).
#
#   ./setup_ide.sh          # idempotent; re-run after changing build/sources/pins.txt
#
# What it sets up (all inside the repo, all gitignored):
#   .venv-ide/          Python for the editor: torch (CPU), mlx, numpy, pytest
#   .refs/<repo>/       triton, flashinfer, vllm sources at the commits pinned in build/sources/pins.txt
#                       (sparse checkout: only the parts the drills import)
#   .refs/cuda/         NVIDIA's CUDA headers, taken from the official pip wheels (headers only)
#   .clangd             clangd flags for *.cu / *.cuh: CUDA mode + torch / CUDA / FlashInfer includes
#   .vscode/settings.json   interpreter, Pylance extraPaths, clangd instead of cpptools IntelliSense
# This is only for reading code. Running the CUDA / Triton drills still needs the GPU machine (setup_env.sh).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD
CUDA_VER=${CUDA_VER:-12.8}          # headers only; any recent 12.x works for navigation

# ---------------------------------------------------------------- 1. editor Python
BASE_PY=$(python3 -c 'import sys; print(sys._base_executable)')
[[ -x .venv-ide/bin/python ]] || "$BASE_PY" -m venv .venv-ide
.venv-ide/bin/pip install -q --upgrade pip
.venv-ide/bin/pip install -q torch numpy pytest markdown-it-py mlx
TORCH_DIR=$(.venv-ide/bin/python -c 'import torch, os; print(os.path.dirname(torch.__file__))')
PY_INC=$(.venv-ide/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')
echo "editor python: .venv-ide (torch at $TORCH_DIR)"

# ---------------------------------------------------------------- 2. pinned library sources
pin() { awk -v a="$1" '$1 == a {print $2, $3}' build/sources/pins.txt; }
fetch_src() {                       # fetch_src <alias> <sparse paths...>
  local alias=$1; shift
  read -r repo sha < <(pin "$alias")
  local dir=.refs/$alias
  if [[ ! -d $dir/.git ]]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "https://github.com/$repo.git"
  fi
  git -C "$dir" sparse-checkout set --no-cone "$@"
  if [[ "$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null)" != "$sha" ]]; then
    git -C "$dir" fetch -q --depth 1 --filter=blob:none origin "$sha"
    git -C "$dir" checkout -q --detach "$sha"
  fi
  echo "  $alias  $repo @ ${sha:0:7}"
}
echo "library sources:"
fetch_src triton     /python/triton/ /python/triton_kernels/
fetch_src flashinfer /flashinfer/ "!/flashinfer/experimental/" /include/ "/csrc/*" "!/csrc/*/"   # skip generated kernel trees
fetch_src vllm       /vllm/ /csrc/

# ---------------------------------------------------------------- 3. CUDA headers
CUDA=.refs/cuda
if [[ ! -f $CUDA/include/cuda_runtime.h || ! -f $CUDA/include/curand_mtgp32_kernel.h ]]; then
  echo "CUDA $CUDA_VER headers (from NVIDIA's pip wheels):"
  rm -rf "$CUDA"; mkdir -p "$CUDA/wheels" "$CUDA/include" "$CUDA/bin" "$CUDA/lib64"
  .venv-ide/bin/pip download -q --no-deps --only-binary=:all: --platform manylinux2014_x86_64 \
      --python-version 3.12 -d "$CUDA/wheels" \
      "nvidia-cuda-runtime-cu12==$CUDA_VER.*" "nvidia-cuda-nvcc-cu12==$CUDA_VER.*" "nvidia-cuda-cccl-cu12==$CUDA_VER.*" "nvidia-curand-cu12"   # curand: clang's CUDA wrapper includes it
  for w in "$CUDA"/wheels/*.whl; do
    tmp=$(mktemp -d); unzip -q "$w" 'nvidia/*/include/*' -d "$tmp" || true
    for inc in "$tmp"/nvidia/*/include; do cp -R "$inc"/. "$CUDA/include/"; done
    rm -rf "$tmp"
  done
  rm -rf "$CUDA/wheels"
  printf '{"cuda": {"name": "CUDA SDK", "version": "%s.0"}}\n' "$CUDA_VER" > "$CUDA/version.json"
fi
[[ -f $CUDA/include/cuda_runtime.h && -f $CUDA/include/device_launch_parameters.h && -f $CUDA/include/curand_mtgp32_kernel.h ]] || { echo "CUDA headers missing" >&2; exit 1; }
echo "  .refs/cuda/include ($(ls "$CUDA/include" | wc -l | tr -d ' ') entries)"

# Stubs for the CUDA math-library headers torch's CUDAContext.h includes (cuBLAS / cuSPARSE /
# cuSOLVER wheels are hundreds of MB; for reading code only the handle types are needed).
mkdir -p "$CUDA/stubs"
stub() { [[ -f $CUDA/include/$1 ]] || printf '// IDE-only stub written by setup_ide.sh: the real header is not installed.\n#pragma once\n%s\n' "$2" > "$CUDA/stubs/$1"; }
stub cusparse.h   'typedef struct cusparseContext* cusparseHandle_t;
typedef enum { CUSPARSE_STATUS_SUCCESS = 0 } cusparseStatus_t;
const char* cusparseGetErrorString(cusparseStatus_t status);'
stub cublas_v2.h  'typedef struct cublasContext* cublasHandle_t;
typedef enum { CUBLAS_STATUS_SUCCESS = 0 } cublasStatus_t;'
stub cublasLt.h   'typedef struct cublasLtContext* cublasLtHandle_t;'
stub cusolver_common.h 'typedef enum { CUSOLVER_STATUS_SUCCESS = 0, CUSOLVER_STATUS_INVALID_VALUE = 3 } cusolverStatus_t;'
stub cusolverDn.h '#include <cusolver_common.h>
typedef struct cusolverDnContext* cusolverDnHandle_t;'
# torch generates this header only in CUDA builds; the CPU wheel on a Mac does not ship it.
mkdir -p "$CUDA/stubs/c10/cuda/impl"
[[ -f $TORCH_DIR/include/c10/cuda/impl/cuda_cmake_macros.h ]] || \
  printf '// IDE-only stub written by setup_ide.sh (torch CPU wheel has no CUDA build config).\n#pragma once\n' > "$CUDA/stubs/c10/cuda/impl/cuda_cmake_macros.h"
# Apple's clang driver overwrites the CUDA SDK version with the macOS one, so clang falls back to the
# pre-9.2 kernel-launch API and looks for cudaConfigureCall when it sees <<<...>>>. Declare it.
cat > "$CUDA/ide_shim.h" <<'EOF'
// IDE-only shim written by setup_ide.sh; never used by a real build.
#pragma once
extern "C" cudaError_t cudaConfigureCall(dim3 grid, dim3 block, size_t shared_mem = 0, cudaStream_t stream = 0);
EOF

# ---------------------------------------------------------------- 4. clangd config for .cu / .cuh
cat > .clangd <<EOF
# Generated by setup_ide.sh (absolute paths, machine-specific; gitignored).
If:
  PathMatch: [.*\.cu, .*\.cuh]
CompileFlags:
  Add: [-xcuda, --cuda-host-only, --cuda-path=$ROOT/$CUDA, -nocudalib, --no-cuda-version-check, -std=c++20,
        -isystem$TORCH_DIR/include, -isystem$TORCH_DIR/include/torch/csrc/api/include, -isystem$PY_INC,
        -isystem$ROOT/$CUDA/stubs, -include$ROOT/$CUDA/ide_shim.h,
        -I$ROOT/.refs/flashinfer/include, -Wno-unknown-cuda-version]
Diagnostics:
  # Apple libc++ vs clang's CUDA wrapper headers (host/device strchr overloads): harmless for reading code.
  Suppress: [cuda_ovl_target]
---
If:
  PathMatch: [.*\.h, .*\.hpp, .*\.cpp]
CompileFlags:
  Add: [-std=c++20, -isystem$TORCH_DIR/include, -isystem$TORCH_DIR/include/torch/csrc/api/include, -isystem$PY_INC,
        -I$ROOT/.refs/flashinfer/include, -I$ROOT/$CUDA/include]
Diagnostics:
  Suppress: ["*"]
EOF
echo "wrote .clangd"

# ---------------------------------------------------------------- 5. VS Code workspace settings
mkdir -p .vscode
.venv-ide/bin/python - <<'EOF'
import json, pathlib
p = pathlib.Path(".vscode/settings.json")
s = json.loads(p.read_text()) if p.exists() else {}
s.update({
    "python.defaultInterpreterPath": "${workspaceFolder}/.venv-ide/bin/python",
    "python.analysis.extraPaths": [
        "${workspaceFolder}/.refs/triton/python",
        "${workspaceFolder}/.refs/flashinfer",
        "${workspaceFolder}/.refs/vllm",
    ],
    "python.analysis.exclude": ["**/.refs/**", "**/.venv*/**", "build/sources/**", "**/__pycache__"],
    "C_Cpp.intelliSenseEngine": "disabled",          # clangd does C++/CUDA here; cpptools would fight it
    "clangd.path": "/usr/bin/clangd",
    "clangd.arguments": ["--header-insertion=never"],
    "files.associations": {"*.cu": "cuda-cpp", "*.cuh": "cuda-cpp"},
})
p.write_text(json.dumps(s, indent=2) + "\n")
print("wrote .vscode/settings.json")
EOF
echo
echo "Done. In VS Code: install the 'clangd' extension (llvm-vs-code-extensions.vscode-clangd) if it is not"
echo "installed, then run 'Developer: Reload Window'."
