"""Helpers shared by the paged-attention tests: build a random paged KV cache + a reference."""
import torch


def make_paged(context_lens, num_kv_heads, head_dim, page, dtype, device, extra_blocks=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    pages_per_seq = [(c + page - 1) // page for c in context_lens]
    max_blocks = max(pages_per_seq)
    num_blocks = sum(pages_per_seq) + extra_blocks
    perm = torch.randperm(num_blocks, generator=g)          # physical blocks in random order
    block_tables = torch.zeros(len(context_lens), max_blocks, dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages_per_seq):
        block_tables[b, :n] = perm[used:used + n].to(torch.int32)
        used += n
    k_cache = torch.randn(num_blocks, page, num_kv_heads, head_dim, generator=g).to(dtype)
    v_cache = torch.randn(num_blocks, page, num_kv_heads, head_dim, generator=g).to(dtype)
    return (k_cache.to(device), v_cache.to(device), block_tables.to(device),
            torch.tensor(context_lens, dtype=torch.int32, device=device))


def reference_decode(q, k_cache, v_cache, block_tables, context_lens, scale):
    B, Hq, D = q.shape
    _, page, Hkv, _ = k_cache.shape
    out = torch.empty(B, Hq, D, dtype=torch.float32, device=q.device)
    for b in range(B):
        ctx = int(context_lens[b])
        blocks = block_tables[b, : (ctx + page - 1) // page].long()
        k = k_cache[blocks].reshape(-1, Hkv, D)[:ctx].float()          # [ctx, Hkv, D]
        v = v_cache[blocks].reshape(-1, Hkv, D)[:ctx].float()
        k = k.repeat_interleave(Hq // Hkv, dim=1)                       # GQA: share kv heads
        v = v.repeat_interleave(Hq // Hkv, dim=1)
        s = torch.einsum("hd,thd->ht", q[b].float(), k) * scale         # [Hq, ctx]
        out[b] = torch.einsum("ht,thd->hd", s.softmax(-1), v)
    return out
