"""L1 · Paged KV cache: slot mapping + cache write (vLLM's reshape_and_cache) — YOUR IMPLEMENTATION.

Cache layout (one tensor each for K and V):
    k_cache, v_cache : [num_blocks, block_size, num_kv_heads, head_dim]
A sequence's token at logical position p lives in physical block block_table[seq][p // block_size]
at offset p % block_size. Flattening (block, offset) gives a "slot":
    slot = physical_block * block_size + offset
and because the cache is contiguous, slot s starts at element s * num_kv_heads * head_dim.
"""
import torch
import triton
import triton.language as tl


def slot_mapping(block_table: torch.Tensor, seq_idx: torch.Tensor, positions: torch.Tensor,
                 block_size: int) -> torch.Tensor:
    """block_table [B, max_blocks] int32, seq_idx [T], positions [T] -> slots [T] int64.

    Pure PyTorch, vectorised: no Python loop over tokens.
    """
    # TODO: logical block = pos // block_size; physical = block_table[seq, logical];
    #       slot = physical * block_size + pos % block_size  (int64, no Python loop)
    raise NotImplementedError


@triton.jit
def reshape_and_cache_kernel(key_ptr, value_ptr, k_cache_ptr, v_cache_ptr, slot_ptr,
                             key_stride, value_stride, HD, BLOCK: tl.constexpr):
    # TODO: one program per token: load its slot (-1 = skip), copy the token's H*D key and value
    #       values to k_cache / v_cache at element offset slot * HD
    pass


def reshape_and_cache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor,
                      v_cache: torch.Tensor, slots: torch.Tensor) -> None:
    """key/value [T, H, D] -> written in place into k_cache/v_cache [NB, BS, H, D] at `slots`."""
    T, H, D = key.shape
    assert value.shape == key.shape
    assert k_cache.is_contiguous() and v_cache.is_contiguous()
    assert k_cache.shape[2:] == (H, D) and k_cache.dtype == key.dtype
    for t in (key, value):
        assert t.stride(2) == 1 and t.stride(1) == D, "each token's [H, D] must be contiguous"
    if T == 0:
        return
    HD = H * D
    reshape_and_cache_kernel[(T,)](key, value, k_cache, v_cache, slots.long(),
                                   key.stride(0), value.stride(0), HD, BLOCK=triton.next_power_of_2(HD))
