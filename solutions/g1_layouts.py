"""G1 · Layout calculator for Triton / Gluon BlockedLayout and SliceLayout — reference solution.

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
    coords = [0] * len(sizes)
    for d in order:
        coords[d] = index % sizes[d]
        index //= sizes[d]
    return coords


def _linearize(coords, sizes, order) -> int:
    index, mult = 0, 1
    for d in order:
        index += coords[d] * mult
        mult *= sizes[d]
    return index


def reps(layout: BlockedLayout, shape: tuple[int, ...]) -> tuple[int, ...]:
    """How many times the CTA tile repeats along each dim (1 if the tensor is smaller)."""
    out = []
    for n, t in zip(shape, layout.tile()):
        assert n % t == 0 or t % n == 0, "shape and tile must divide each other (powers of two)"
        out.append(max(1, n // t))
    return tuple(out)


def thread_elements(layout: BlockedLayout, shape: tuple[int, ...], warp: int, lane: int) -> list[tuple[int, ...]]:
    """Coordinates held by (warp, lane), indexed by register number."""
    spt, tpw, wpc, order = layout.size_per_thread, layout.threads_per_warp, layout.warps_per_cta, layout.order
    lane_c = _delinearize(lane, tpw, order)    # this lane's position inside the warp
    warp_c = _delinearize(warp, wpc, order)    # this warp's position inside the CTA
    tile = layout.tile()
    r = reps(layout, shape)
    out = []
    for rep_i in range(prod(r)):                       # repetitions: outer
        rep_c = _delinearize(rep_i, r, order)
        for s_i in range(prod(spt)):                   # elements of one chunk: inner
            s_c = _delinearize(s_i, spt, order)
            coord = tuple(
                (rep_c[d] * tile[d] + warp_c[d] * spt[d] * tpw[d] + lane_c[d] * spt[d] + s_c[d]) % shape[d]
                for d in range(layout.rank)
            )
            out.append(coord)
    return out


def owners(layout: BlockedLayout, shape: tuple[int, ...], coord: tuple[int, ...]) -> list[tuple[int, int, int]]:
    """Every (warp, lane, register) holding element `coord` (more than one when broadcast)."""
    found = []
    for warp in range(layout.num_warps):
        for lane in range(32):
            for reg, c in enumerate(thread_elements(layout, shape, warp, lane)):
                if c == tuple(coord):
                    found.append((warp, lane, reg))
    return found


def slice_elements(parent: BlockedLayout, parent_shape: tuple[int, ...], dim: int,
                   warp: int, lane: int) -> list[tuple[int, ...]]:
    """SliceLayout(dim, parent): the layout of a tensor obtained by removing `dim`, e.g. the
    result of tl.sum(x, axis=dim). A thread holds the remaining coordinates of its parent
    elements; duplicates collapse, first occurrence keeps its register order."""
    seen, out = set(), []
    for c in thread_elements(parent, parent_shape, warp, lane):
        rest = c[:dim] + c[dim + 1:]
        if rest not in seen:
            seen.add(rest)
            out.append(rest)
    return out


def reduce_plan(layout: BlockedLayout, shape: tuple[int, ...], axis: int) -> dict:
    """What tl.sum / tl.max along `axis` costs with this layout.

    in_thread      values each thread combines in its own registers for one output
    shuffle_rounds butterfly __shfl_xor rounds across the lanes laid out along `axis`
    cross_warp     True if warps are laid out along `axis`: partials go through shared
                   memory plus a barrier, the expensive part
    """
    r = reps(layout, shape)
    return {
        "in_thread": layout.size_per_thread[axis] * r[axis],
        "shuffle_rounds": int(log2(layout.threads_per_warp[axis])),
        "cross_warp": layout.warps_per_cta[axis] > 1,
    }


def warp_load_sectors(layout: BlockedLayout, shape: tuple[int, ...], dtype_bytes: int, warp: int = 0) -> tuple[int, int]:
    """Memory efficiency of the first vectorised load a warp issues on a row-major tensor.

    Each lane loads the first `vec` registers of its chunk in one instruction, where
    vec = min(spt[order[0]], 16 // dtype_bytes) (at most 16 bytes per lane per instruction).
    Returns (32-byte sectors touched, bytes actually requested). Ideal: sectors * 32 == bytes.
    """
    vec = max(1, min(layout.size_per_thread[layout.order[0]], 16 // dtype_bytes))
    strides = [prod(shape[d + 1:]) for d in range(len(shape))]     # row-major element strides
    sectors = set()
    for lane in range(32):
        for c in thread_elements(layout, shape, warp, lane)[:vec]:
            addr = sum(ci * si for ci, si in zip(c, strides)) * dtype_bytes
            sectors.add(addr // 32)
    return len(sectors), 32 * vec * dtype_bytes


def all_coords(shape):
    return list(product(*(range(n) for n in shape)))
