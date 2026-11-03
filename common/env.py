"""Where tests run: a CUDA GPU if present, otherwise the CPU (Triton interpreter mode)."""
import os

import torch

HAS_CUDA = torch.cuda.is_available()
INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
DEVICE = "cuda" if HAS_CUDA else "cpu"


def seed(s: int = 0) -> None:
    torch.manual_seed(s)

# bf16 Tensor Core math needs Ampere (sm_80) or newer; a Colab T4 is sm_75.
BF16_OK = HAS_CUDA and torch.cuda.get_device_capability()[0] >= 8

try:  # FlashInfer is optional: F-track tests compare against it when it is installed (GPU only)
    import flashinfer  # noqa: F401
    HAS_FLASHINFER = HAS_CUDA
except Exception:
    HAS_FLASHINFER = False
