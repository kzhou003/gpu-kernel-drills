"""M1 · MoE routing + moe_align_block_size — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import torch


def route(router_logits: torch.Tensor, top_k: int):
    # TODO: softmax (fp32) -> topk -> renormalise the kept weights; ids as int32
    raise NotImplementedError


def moe_align_block_size(topk_ids: torch.Tensor, num_experts: int, block_m: int):
    # TODO: counts = bincount; padded = round each count up to block_m; pad_start = exclusive cumsum
    # TODO: order = stable argsort of the flat expert ids; rank within group; dest = pad_start[e] + rank
    # TODO: sorted_token_ids = full(total, T*K); sorted_token_ids[dest] = order
    # TODO: expert_ids = repeat_interleave(arange(E), padded // block_m)
    raise NotImplementedError


