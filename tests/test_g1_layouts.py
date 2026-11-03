"""Expected values below are worked out by hand from the layout definition (see the drill page)."""
from drill import load

# 4 warps stacked along rows; inside a warp 4 x 8 lanes; each lane owns 1 x 4 contiguous elements.
A = dict(size_per_thread=(1, 4), threads_per_warp=(4, 8), warps_per_cta=(4, 1), order=(1, 0))
# The "wrong" orientation for a row-major tensor: lanes run down the columns.
BAD = dict(size_per_thread=(4, 1), threads_per_warp=(8, 4), warps_per_cta=(4, 1), order=(0, 1))


def L(m, d):
    return m.BlockedLayout(**d)


def test_tile_and_reps():
    m = load("g1_layouts")
    lay = L(m, A)
    assert lay.tile() == (16, 32)
    assert m.reps(lay, (16, 32)) == (1, 1)
    assert m.reps(lay, (64, 64)) == (4, 2)
    assert m.reps(lay, (8, 32)) == (1, 1)      # smaller than the tile -> broadcast, no repetition


def test_thread_elements_basic():
    m = load("g1_layouts")
    lay = L(m, A)
    assert m.thread_elements(lay, (16, 32), warp=0, lane=0) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert m.thread_elements(lay, (16, 32), warp=0, lane=1) == [(0, 4), (0, 5), (0, 6), (0, 7)]
    assert m.thread_elements(lay, (16, 32), warp=0, lane=8) == [(1, 0), (1, 1), (1, 2), (1, 3)]
    assert m.thread_elements(lay, (16, 32), warp=1, lane=0) == [(4, 0), (4, 1), (4, 2), (4, 3)]
    assert m.thread_elements(lay, (16, 32), warp=3, lane=31) == [(15, 28), (15, 29), (15, 30), (15, 31)]


def test_thread_elements_with_repetition():
    m = load("g1_layouts")
    lay = L(m, A)
    # (32, 32): the tile repeats twice along rows -> 8 registers, second chunk 16 rows lower.
    assert m.thread_elements(lay, (32, 32), warp=0, lane=0) == [
        (0, 0), (0, 1), (0, 2), (0, 3), (16, 0), (16, 1), (16, 2), (16, 3)]


def test_every_element_has_exactly_one_owner_without_broadcast():
    m = load("g1_layouts")
    lay = L(m, A)
    shape = (32, 64)
    seen = {}
    for w in range(lay.num_warps):
        for ln in range(32):
            for c in m.thread_elements(lay, shape, w, ln):
                seen[c] = seen.get(c, 0) + 1
    assert len(seen) == 32 * 64 and set(seen.values()) == {1}


def test_owners():
    m = load("g1_layouts")
    lay = L(m, A)
    # row 5 -> warp 1 (rows 4..7), lane row 1; col 13 -> lane col 3, element 1 of the chunk
    assert m.owners(lay, (16, 32), (5, 13)) == [(1, 11, 1)]
    # (8, 32) is half the tile tall: rows wrap, so warps 2 and 3 duplicate warps 0 and 1.
    assert m.owners(lay, (8, 32), (0, 0)) == [(0, 0, 0), (2, 0, 0)]


def test_slice_layout():
    m = load("g1_layouts")
    lay = L(m, A)
    # tl.sum(x, axis=1) keeps rows: lanes 0..7 of warp 0 all end up holding row 0.
    assert m.slice_elements(lay, (16, 32), dim=1, warp=0, lane=0) == [(0,)]
    assert m.slice_elements(lay, (16, 32), dim=1, warp=0, lane=7) == [(0,)]
    assert m.slice_elements(lay, (16, 32), dim=1, warp=0, lane=8) == [(1,)]
    # tl.sum(x, axis=0) keeps columns: lane 0 holds columns 0..3.
    assert m.slice_elements(lay, (16, 32), dim=0, warp=0, lane=0) == [(0,), (1,), (2,), (3,)]


def test_reduce_plan():
    m = load("g1_layouts")
    lay = L(m, A)
    assert m.reduce_plan(lay, (16, 32), axis=1) == {"in_thread": 4, "shuffle_rounds": 3, "cross_warp": False}
    assert m.reduce_plan(lay, (16, 32), axis=0) == {"in_thread": 1, "shuffle_rounds": 2, "cross_warp": True}
    assert m.reduce_plan(lay, (16, 128), axis=1)["in_thread"] == 16    # 4 per chunk x 4 repetitions


def test_warp_load_sectors():
    m = load("g1_layouts")
    # Good layout, fp32: 4 rows x 128 B = 16 sectors for 512 B -> 100% efficient.
    assert m.warp_load_sectors(L(m, A), (16, 32), 4) == (16, 512)
    # fp16: 4 halves per lane (8 B): 4 rows x 64 B = 8 sectors for 256 B.
    assert m.warp_load_sectors(L(m, A), (16, 32), 2) == (8, 256)
    # Bad orientation: each lane's 4 "contiguous" registers sit in 4 different rows.
    # 32 rows x 4 columns (16 B each) -> 32 sectors (1024 B moved) for 512 B used: 50%.
    assert m.warp_load_sectors(L(m, BAD), (128, 32), 4) == (32, 512)
