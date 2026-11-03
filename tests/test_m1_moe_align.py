import torch

from drill import load


def test_route():
    m = load("m1_moe_align")
    logits = torch.tensor([[0.0, 3.0, 1.0, 2.0], [5.0, 0.0, 0.0, 4.0]])
    w, ids = m.route(logits, 2)
    assert ids.dtype == torch.int32 and ids.tolist() == [[1, 3], [0, 3]]
    torch.testing.assert_close(w.sum(-1), torch.ones(2))
    p = torch.softmax(logits[0], 0)
    torch.testing.assert_close(w[0], torch.stack([p[1], p[3]]) / (p[1] + p[3]))


def test_align_by_hand():
    m = load("m1_moe_align")
    # T=3 tokens, K=2 -> flat ids 0..5; experts: f0->2, f1->0, f2->2, f3->1, f4->0, f5->2
    topk_ids = torch.tensor([[2, 0], [2, 1], [0, 2]], dtype=torch.int32)
    sorted_ids, expert_ids, total = m.moe_align_block_size(topk_ids, num_experts=4, block_m=2)
    # expert 0: [1, 4] | expert 1: [3, pad] | expert 2: [0, 2, 5, pad] | expert 3: none
    assert total == 8
    assert sorted_ids.tolist() == [1, 4, 3, 6, 0, 2, 5, 6]          # pad value = T*K = 6
    assert expert_ids.tolist() == [0, 1, 2, 2]


def test_align_invariants_random():
    m = load("m1_moe_align")
    g = torch.Generator().manual_seed(0)
    T, K, E, BM = 257, 4, 16, 16
    topk_ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)]).int()
    sorted_ids, expert_ids, total = m.moe_align_block_size(topk_ids, E, BM)
    assert total % BM == 0 and sorted_ids.numel() == total and expert_ids.numel() == total // BM
    flat = topk_ids.reshape(-1)
    real = sorted_ids[sorted_ids < T * K].long()
    assert sorted(real.tolist()) == list(range(T * K))               # every pair exactly once
    blocks = sorted_ids.view(-1, BM)
    for b in range(blocks.shape[0]):                                 # each block: one expert, in order
        ids = blocks[b][blocks[b] < T * K].long()
        assert (flat[ids] == expert_ids[b]).all()
        assert (ids[1:] > ids[:-1]).all()
