"""M1 · MoE routing + moe_align_block_size — reference solution.

route(logits [T, E], top_k)  -> (topk_weights [T, K] fp32, topk_ids [T, K] int32)
    softmax over experts, keep the top_k, renormalise the kept weights to sum to 1
    (Mixtral / Qwen-MoE style; DeepSeek-V3 uses sigmoid scores + group-limited routing).

moe_align_block_size(topk_ids [T, K], num_experts, block_m)
    -> sorted_token_ids [P] int32, expert_ids [P // block_m] int32, num_tokens_post_padded int
The (token, slot) pairs are numbered flat: f = t * K + k. Grouping them by expert and padding each
expert's group to a multiple of block_m lets ONE grouped GEMM launch (M2) process every expert:
each block of block_m rows belongs to exactly one expert. Padding rows hold the value T * K
(an out-of-range id the GEMM masks off). Same contract as vLLM's moe_align_block_size, except
that P here is the exact padded length instead of a preallocated upper bound.
"""
import torch


def route(router_logits: torch.Tensor, top_k: int):
    probs = torch.softmax(router_logits.float(), dim=-1)
    w, ids = torch.topk(probs, top_k, dim=-1)
    w = w / w.sum(-1, keepdim=True)
    return w, ids.to(torch.int32)


def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    flat = topk_ids.reshape(-1).long()                       # [T*K] expert of each (token, slot)
    n = flat.numel()
    dev = flat.device
    counts = torch.bincount(flat, minlength=num_experts)     # tokens per expert
    padded = (counts + block_m - 1) // block_m * block_m     # each group rounded up to block_m
    pad_start = torch.cumsum(padded, 0) - padded             # where each expert's group begins
    total = int(padded.sum().item())

    # Stable sort by expert keeps the original token order inside each expert's group.
    order = torch.argsort(flat, stable=True)                 # flat ids, grouped by expert
    sorted_experts = flat[order]
    group_start = torch.cumsum(counts, 0) - counts           # start of each group in `order`
    rank = torch.arange(n, device=dev) - group_start[sorted_experts]   # position within the group
    dest = pad_start[sorted_experts] + rank                  # position in the padded layout

    sorted_token_ids = torch.full((total,), n, dtype=torch.int32, device=dev)   # padding = T*K
    sorted_token_ids[dest] = order.to(torch.int32)
    expert_ids = torch.repeat_interleave(torch.arange(num_experts, device=dev), padded // block_m).to(torch.int32)
    return sorted_token_ids, expert_ids, total
