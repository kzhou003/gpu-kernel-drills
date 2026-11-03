"""G2 · CuTe layout algebra in pure Python — reference solution.

A CuTe Layout is a pair (Shape, Stride) of (nested) integer tuples. It is a FUNCTION from
coordinates to offsets:   offset = sum_i coord_i * stride_i.
A plain integer index is first turned into a coordinate colexicographically (first mode fastest).
Everything in CUTLASS 3 / CuTe DSL — thread-value layouts, tiled MMA/copy, smem swizzles,
FlashAttention-4 and FlashInfer's CuTe-DSL kernels — is built from these few operations.

    Layout((4, 2), (1, 4))   "4x2 column-major"   written (4,2):(1,4)
    Layout((4, 2), (2, 1))   "4x2 row-major"      written (4,2):(2,1)
"""
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
    if isinstance(shape, tuple):
        crd = []
        for s in shape:
            n = _product(s)
            crd.append(_idx2crd(idx % n, s))
            idx //= n
        return tuple(crd)
    return idx


def _inner(crd, shape, stride):
    if isinstance(shape, tuple):
        if not isinstance(crd, tuple):            # an integer for a nested mode: unflatten it first
            crd = _idx2crd(crd, shape)
        return sum(_inner(c, s, d) for c, s, d in zip(crd, shape, stride))
    return crd * stride


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
    fl = flatten(shape)
    strides, acc = [0] * len(fl), 1
    for i in reversed(range(len(fl))):
        strides[i] = acc
        acc *= fl[i]
    return Layout(shape, _unflatten(shape, iter(strides)))


def _unflatten(shape, it):
    if isinstance(shape, tuple):
        return tuple(_unflatten(s, it) for s in shape)
    return next(it)


def make_layout_left(shape):
    """Compact column-major (LayoutLeft): stride = exclusive prefix product of the flattened shape."""
    strides, acc = [], 1
    for s in flatten(shape):
        strides.append(acc)
        acc *= s
    return Layout(shape, _unflatten(shape, iter(strides)))


def coalesce(layout: Layout) -> Layout:
    """Same function, fewest modes: drop size-1 modes, merge (s0:d0, s1:d1) when d1 == s0 * d0."""
    shapes, strides = [], []
    for s, d in zip(flatten(layout.shape), flatten(layout.stride)):
        if s == 1:
            continue
        if shapes and strides[-1] * shapes[-1] == d:
            shapes[-1] *= s                       # the new mode continues the previous one
        else:
            shapes.append(s)
            strides.append(d)
    if not shapes:
        return Layout(1, 0)
    if len(shapes) == 1:
        return Layout(shapes[0], strides[0])
    return Layout(tuple(shapes), tuple(strides))


def composition(a: Layout, b: Layout) -> Layout:
    """R = A o B, i.e. R(i) = A(B(i)) for i < size(B). B is applied mode by mode.

    For one mode s:d of B, walk A's (coalesced) modes: skip the first d elements ("divide"),
    then keep the next s ("mod"). Requires the usual CuTe divisibility conditions."""
    if isinstance(b.shape, tuple):
        modes = [composition(a, Layout(s, d)) for s, d in zip(b.shape, b.stride)]
        return Layout(tuple(m.shape for m in modes), tuple(m.stride for m in modes))
    if b.stride == 0:
        return Layout(b.shape, 0)
    fa = coalesce(a)
    a_shapes, a_strides = flatten(fa.shape), flatten(fa.stride)
    rest_shape, rest_stride = b.shape, b.stride
    shapes, strides = [], []
    for s, d in zip(a_shapes[:-1], a_strides[:-1]):
        assert s % rest_stride == 0 or rest_stride % s == 0, "stride divisibility"
        new_shape = min(max(1, s // rest_stride), rest_shape)
        if new_shape != 1:
            shapes.append(new_shape)
            strides.append(rest_stride * d)
        assert rest_shape % new_shape == 0, "shape divisibility"
        rest_shape //= new_shape
        rest_stride = -(-rest_stride // s)        # ceil division
    if rest_shape != 1 or not shapes:             # whatever is left runs along A's last mode
        shapes.append(rest_shape)
        strides.append(rest_stride * a_strides[-1])
    if len(shapes) == 1:
        return Layout(shapes[0], strides[0])
    return Layout(tuple(shapes), tuple(strides))


def complement(a: Layout, cotarget: int) -> Layout:
    """The layout of the "holes" of A in [0, cotarget): (A, complement(A)) together cover it.

    Sort A's modes by stride; each gap between where one mode ends and the next begins becomes
    a mode of the complement, plus a final mode up to cotarget."""
    modes = sorted((d, s) for s, d in zip(flatten(a.shape), flatten(a.stride)) if s != 1 and d != 0)
    shapes, strides, current = [], [], 1
    for d, s in modes:
        assert d % current == 0, "A must be injective with divisible strides"
        shapes.append(d // current)
        strides.append(current)
        current = s * d
    shapes.append(-(-cotarget // current))
    strides.append(current)
    return coalesce(Layout(tuple(shapes), tuple(strides)))


def logical_divide(a: Layout, tiler: Layout) -> Layout:
    """Split A into (tile, rest): mode 0 = the elements the tiler picks, mode 1 = repetitions.
    logical_divide(A, B) = A o (B, complement(B, size(A)))."""
    c = complement(tiler, a.size())
    return composition(a, Layout((tiler.shape, c.shape), (tiler.stride, c.stride)))
