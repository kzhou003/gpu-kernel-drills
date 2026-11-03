import torch

from common.env import DEVICE
from drill import load


def test_slot_mapping():
    m = load("l1_paged_kv_write")
    block_table = torch.tensor([[7, 2, 5], [0, 3, 0]], dtype=torch.int32)
    seq_idx = torch.tensor([0, 0, 0, 1, 1])
    positions = torch.tensor([0, 3, 4, 0, 5])
    # block_size 4: seq0 pos0 -> block 7 off 0; pos3 -> 7*4+3; pos4 -> block 2 off 0;
    #               seq1 pos0 -> block 0 off 0; pos5 -> block 3 off 1
    out = m.slot_mapping(block_table, seq_idx, positions, 4)
    assert out.dtype == torch.int64
    assert out.tolist() == [28, 31, 8, 0, 13]


def _reference_write(key, value, k_cache, v_cache, slots):
    kc, vc = k_cache.clone(), v_cache.clone()
    NB, BS, H, D = kc.shape
    valid = slots >= 0
    kc.view(NB * BS, H, D)[slots[valid]] = key[valid]
    vc.view(NB * BS, H, D)[slots[valid]] = value[valid]
    return kc, vc


def test_reshape_and_cache():
    m = load("l1_paged_kv_write")
    NB, BS, H, D, T = 6, 16, 2, 64, 37
    k_cache = torch.zeros(NB, BS, H, D, device=DEVICE, dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    key = torch.randn(T, H, D, device=DEVICE).half()
    value = torch.randn(T, H, D, device=DEVICE).half()
    slots = torch.randperm(NB * BS)[:T].to(DEVICE)
    slots[[3, 10]] = -1                                   # padding tokens must be skipped
    ref_k, ref_v = _reference_write(key, value, k_cache, v_cache, slots)
    m.reshape_and_cache(key, value, k_cache, v_cache, slots)
    assert torch.equal(k_cache, ref_k) and torch.equal(v_cache, ref_v)


def test_reshape_and_cache_strided_rows():
    m = load("l1_paged_kv_write")
    T, H, D = 9, 4, 32
    qkv = torch.randn(T, 3 * H, D, device=DEVICE)          # fused projection output
    key, value = qkv[:, H:2 * H], qkv[:, 2 * H:]            # views with row stride 3*H*D
    k_cache = torch.zeros(2, 16, H, D, device=DEVICE)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.arange(T, device=DEVICE) * 3
    ref_k, ref_v = _reference_write(key, value, k_cache, v_cache, slots)
    m.reshape_and_cache(key, value, k_cache, v_cache, slots)
    assert torch.equal(k_cache, ref_k) and torch.equal(v_cache, ref_v)
