"""G2 · CuTe layout algebra — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
from dataclasses import dataclass
from math import prod


def flatten(t):
    if isinstance(t, tuple):
        return tuple(x for e in t for x in flatten(e))
    return (t,)


def _product(shape):
    return prod(flatten(shape))


def _idx2crd(idx, shape):
    """Colexicographic: the first mode varies fastest. Follows the nesting of `shape`."""
    # TODO: colexicographic: first mode fastest, recurse into nested modes
    raise NotImplementedError


def _inner(crd, shape, stride):
    # TODO: sum of coord * stride over (nested) modes; unflatten an int coord for a nested mode first
    raise NotImplementedError


@dataclass(frozen=True)
class Layout:
    shape: object
    stride: object

    def __call__(self, crd):
        return _inner(crd, self.shape, self.stride)

    def size(self):                     # number of coordinates (the domain)
        return _product(self.shape)

    def cosize(self):                   # 1 + largest offset (the codomain the layout touches)
        return self(self.size() - 1) + 1 if self.size() else 0

    def __repr__(self):
        return f"{self.shape}:{self.stride}"


def make_layout_right(shape):
    """Compact row-major (LayoutRight): the LAST mode has stride 1."""
    # TODO: last mode stride 1, earlier strides = suffix products
    raise NotImplementedError


def _unflatten(shape, it):
    if isinstance(shape, tuple):
        return tuple(_unflatten(s, it) for s in shape)
    return next(it)


def make_layout_left(shape):
    """Compact column-major (LayoutLeft): stride = exclusive prefix product of the flattened shape."""
    # TODO: stride = exclusive prefix product of the flattened shape
    raise NotImplementedError


def coalesce(layout: Layout) -> Layout:
    """Same function, fewest modes: drop size-1 modes, merge (s0:d0, s1:d1) when d1 == s0 * d0."""
    # TODO: drop size-1 modes; merge consecutive modes when stride_next == shape_prev * stride_prev
    raise NotImplementedError


def composition(a: Layout, b: Layout) -> Layout:
    """R = A o B, i.e. R(i) = A(B(i)) for i < size(B). B is applied mode by mode.

    For one mode s:d of B, walk A's (coalesced) modes: skip the first d elements ("divide"),
    then keep the next s ("mod"). Requires the usual CuTe divisibility conditions."""
    # TODO: by-mode over B; for one mode s:d walk coalesced A: skip d elements, keep s
    raise NotImplementedError


def complement(a: Layout, cotarget: int) -> Layout:
    """The layout of the "holes" of A in [0, cotarget): (A, complement(A)) together cover it.

    Sort A's modes by stride; each gap between where one mode ends and the next begins becomes
    a mode of the complement, plus a final mode up to cotarget."""
    # TODO: sort A's modes by stride; each gap becomes a mode; last mode reaches cotarget
    raise NotImplementedError


def logical_divide(a: Layout, tiler: Layout) -> Layout:
    """Split A into (tile, rest): mode 0 = the elements the tiler picks, mode 1 = repetitions.
    logical_divide(A, B) = A o (B, complement(B, size(A)))."""
    # TODO: A o (tiler, complement(tiler, size(A)))
    raise NotImplementedError


