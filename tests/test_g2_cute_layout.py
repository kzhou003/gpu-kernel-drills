"""Examples are the ones used in the CuTe documentation (layout algebra chapter), checked as functions."""
from drill import load


def L(m, shape, stride):
    return m.Layout(shape, stride)


def same_function(a, b, n):
    return all(a(i) == b(i) for i in range(n))


def test_call_and_size():
    m = load("g2_cute_layout")
    col = L(m, (4, 2), (1, 4))
    row = L(m, (4, 2), (2, 1))
    assert col.size() == 8 and col.cosize() == 8
    assert [col(i) for i in range(8)] == list(range(8))
    assert [row(i) for i in range(8)] == [0, 2, 4, 6, 1, 3, 5, 7]   # index -> (i % 4, i // 4)
    assert row((1, 1)) == 3                                          # coordinate form
    nested = L(m, ((2, 2), 2), ((1, 4), 2))
    assert nested((3, 1)) == 1 * 1 + 1 * 4 + 1 * 2                   # 3 -> (1, 1) inside mode 0
    assert L(m, 8, 2).cosize() == 15


def test_left_right():
    m = load("g2_cute_layout")
    assert m.make_layout_left((4, 3, 2)) == m.Layout((4, 3, 2), (1, 4, 12))
    assert m.make_layout_right((4, 3, 2)) == m.Layout((4, 3, 2), (6, 2, 1))


def test_coalesce():
    m = load("g2_cute_layout")
    assert m.coalesce(L(m, (2, (1, 6)), (1, (6, 2)))) == m.Layout(12, 1)
    assert m.coalesce(L(m, (4, 2), (2, 1))) == m.Layout((4, 2), (2, 1))     # row-major: nothing merges
    a = L(m, ((2, 4), 3), ((1, 2), 16))
    c = m.coalesce(a)
    assert c == m.Layout((8, 3), (1, 16)) and same_function(a, c, a.size())


def test_composition():
    m = load("g2_cute_layout")
    a = L(m, (6, 2), (8, 2))
    b = L(m, (4, 3), (3, 1))
    r = m.composition(a, b)
    assert same_function(r, lambda i: a(b(i)), b.size())
    assert m.coalesce(r) == m.coalesce(m.Layout(((2, 2), 3), ((24, 2), 8)))
    # picking every other element of a 1-D layout
    assert m.composition(L(m, 20, 2), L(m, (5, 4), (4, 1))) == m.Layout((5, 4), (8, 2))


def test_complement():
    m = load("g2_cute_layout")
    assert m.complement(L(m, 4, 1), 24) == m.Layout(6, 4)
    assert m.complement(L(m, 6, 4), 24) == m.Layout(4, 1)
    c = m.complement(L(m, (2, 2), (1, 6)), 24)
    assert c == m.Layout((3, 2), (2, 12))
    a = L(m, (2, 2), (1, 6))
    offsets = sorted(a(i) + c(j) for i in range(a.size()) for j in range(c.size()))
    assert offsets == list(range(24))                     # A and its complement tile [0, 24) exactly


def test_logical_divide():
    m = load("g2_cute_layout")
    a = L(m, 16, 1)
    d = m.logical_divide(a, L(m, 4, 1))                   # 16 elements -> 4 tiles of 4 contiguous
    assert [d((i, j)) for j in range(4) for i in range(4)] == list(range(16))
    a = L(m, 24, 1)
    d = m.logical_divide(a, L(m, 4, 2))                   # tile = 4 elements with stride 2
    assert [d((i, 0)) for i in range(4)] == [0, 2, 4, 6]
    assert sorted(d((i, j)) for i in range(4) for j in range(6)) == list(range(24))
