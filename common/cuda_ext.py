"""Compile a drill's .cu file with torch.utils.cpp_extension.load_inline.

The .cu file must `#include <torch/extension.h>` and define the host functions named in
`functions`; `decls` holds their C++ declarations (load_inline generates the Python bindings).
The module name includes a hash of the source, so editing the .cu file triggers a rebuild.
"""
import functools
import hashlib
from pathlib import Path


@functools.lru_cache(maxsize=None)
def load(cu_path: str, decls: str, functions: tuple[str, ...]):
    from torch.utils.cpp_extension import load_inline

    path = Path(cu_path)
    src = path.read_text()
    tag = hashlib.sha1(src.encode()).hexdigest()[:10]
    return load_inline(
        name=f"{path.parent.name}_{path.stem}_{tag}",
        cpp_sources=decls,
        cuda_sources=src,
        functions=list(functions),
        extra_cuda_cflags=["-O3", "-lineinfo"],
        verbose=False,
    )
