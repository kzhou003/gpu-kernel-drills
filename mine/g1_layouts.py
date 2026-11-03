"""G1 · Layout calculator for Triton / Gluon BlockedLayout and SliceLayout — YOUR IMPLEMENTATION.

Pure Python, runs anywhere. A BlockedLayout says how the elements of a block-level tensor are
spread over the registers, lanes (threads of a warp) and warps of one program:

    size_per_thread  [d] : contiguous elements a thread owns along dim d (one "chunk")
    threads_per_warp [d] : lanes laid out along dim d          (product must be 32)
    warps_per_cta    [d] : warps laid out along dim d          (product = num_warps)
    order                : dims from fastest- to slowest-varying when numbering lanes,
                           warps and registers (order[0] is the fastest)

One "CTA tile" covers tile[d] = spt[d] * tpw[d] * wpc[d] elements along each dim:
    coord[d] = rep[d] * tile[d] + warp_c[d] * (spt[d] * tpw[d]) + lane_c[d] * spt[d] + s[d]
If the tensor is larger than the tile, the pattern repeats (each thread gets more registers);
if it is smaller, coordinates wrap around (several threads hold the same element: broadcast).

Register numbering used in this drill (it matches how Triton numbers registers for blocked
layouts): the elements of one size_per_thread chunk come first (linearised by `order`),
then the repetitions (also linearised by `order`).
"""
from dataclasses import dataclass
from itertools import product
from math import log2, prod


@dataclass(frozen=True)
class BlockedLayout:
    size_per_thread: tuple[int, ...]
    threads_per_warp: tuple[int, ...]
    warps_per_cta: tuple[int, ...]
    order: tuple[int, ...]

    def __post_init__(self):
        r = len(self.size_per_thread)
        assert len(self.threads_per_warp) == len(self.warps_per_cta) == len(self.order) == r
        assert sorted(self.order) == list(range(r)), "order must be a permutation of the dims"
        assert prod(self.threads_per_warp) == 32, "a warp has 32 lanes"

    @property
    def rank(self) -> int:
        return len(self.size_per_thread)

    @property
    def num_warps(self) -> int:
        return prod(self.warps_per_cta)

    def tile(self) -> tuple[int, ...]:
        """Elements covered by one CTA tile along each dim."""
        return tuple(s * t * w for s, t, w in zip(self.size_per_thread, self.threads_per_warp, self.warps_per_cta))


def _delinearize(index: int, sizes: tuple[int, ...], order: tuple[int, ...]) -> list[int]:
    """Inverse of numbering a multi-index with order[0] as the fastest-varying dim."""
    # TODO: split `index` into per-dim coords, order[0] varying fastest
    raise NotImplementedError


def _linearize(coords, sizes, order) -> int:
    # TODO: inverse of _delinearize (optional helper)
    raise NotImplementedError


def reps(layout: BlockedLayout, shape: tuple[int, ...]) -> tuple[int, ...]:
    """How many times the CTA tile repeats along each dim (1 if the tensor is smaller)."""
    # TODO: max(1, shape[d] // tile[d]) per dim
    raise NotImplementedError


def thread_elements(layout: BlockedLayout, shape: tuple[int, ...], warp: int, lane: int) -> list[tuple[int, ...]]:
    """Coordinates held by (warp, lane), indexed by register number."""
    # TODO: loop repetitions (outer) x chunk elements (inner); coord formula in the module docstring; wrap with % shape
    raise NotImplementedError


def owners(layout: BlockedLayout, shape: tuple[int, ...], coord: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Every (warp, lane, register) holding element `coord` (more than one when broadcast)."""
    # TODO: search every (warp, lane, reg)
    raise NotImplementedError


def slice_elements(parent: BlockedLayout, parent_shape: tuple[int, ...], dim: int,
                   warp: int, lane: int) -> list[tuple[int, ...]]:
    """SliceLayout(dim, parent): the layout of a tensor obtained by removing `dim`, e.g. the
    result of tl.sum(x, axis=dim). A thread holds the remaining coordinates of its parent
    elements; duplicates collapse, first occurrence keeps its register order."""
    # TODO: drop `dim` from each parent coordinate, de-duplicate keeping first occurrence
    raise NotImplementedError


def reduce_plan(layout: BlockedLayout, shape: tuple[int, ...], axis: int) -> dict:
    """What tl.sum / tl.max along `axis` costs with this layout.

    in_thread      values each thread combines in its own registers for one output
    shuffle_rounds butterfly __shfl_xor rounds across the lanes laid out along `axis`
    cross_warp     True if warps are laid out along `axis`: partials go through shared
                   memory plus a barrier, the expensive part
    """
    # TODO: in_thread = spt[axis] * reps[axis]; shuffle_rounds = log2(tpw[axis]); cross_warp = wpc[axis] > 1
    raise NotImplementedError


def warp_load_sectors(layout: BlockedLayout, shape: tuple[int, ...], dtype_bytes: int, warp: int = 0) -> tuple[int, int]:
    """Memory efficiency of the first vectorised load a warp issues on a row-major tensor.

    Each lane loads the first `vec` registers of its chunk in one instruction, where
    vec = min(spt[order[0]], 16 // dtype_bytes) (at most 16 bytes per lane per instruction).
    Returns (32-byte sectors touched, bytes actually requested). Ideal: sectors * 32 == bytes.
    """
    # TODO: vec = min(spt[order[0]], 16 // dtype_bytes); collect 32-byte sector ids of every lane's first vec elements
    raise NotImplementedError


def all_coords(shape):
    return list(product(*(range(n) for n in shape)))
