"""L1 · Paged KV cache: slot mapping + cache write (vLLM's reshape_and_cache) — reference solution.

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
    seq_idx = seq_idx.long()
    positions = positions.long()
    logical_block = positions // block_size                                  # which page of the sequence
    physical_block = block_table[seq_idx, logical_block].long()              # gather: [T]
    return physical_block * block_size + positions % block_size


@triton.jit
def reshape_and_cache_kernel(key_ptr, value_ptr, k_cache_ptr, v_cache_ptr, slot_ptr,
                             key_stride, value_stride, HD, BLOCK: tl.constexpr):
    t = tl.program_id(0).to(tl.int64)          # one program per new token
    slot = tl.load(slot_ptr + t)               # int64; -1 marks padding tokens to skip
    offs = tl.arange(0, BLOCK)                 # the token's H*D values, flattened
    # Padding token -> 0 valid elements, every lane masked off. (Written as one comparison
    # rather than `(offs < HD) & (slot >= 0)`: same meaning, and the CPU interpreter rejects
    # a mask built by and-ing a scalar with a block.)
    n_valid = tl.where(slot >= 0, HD, 0)
    mask = offs < n_valid

    # key/value rows may be strided (e.g. a view into a fused qkv tensor), but each row's
    # [H, D] part is contiguous.
    k = tl.load(key_ptr + t * key_stride + offs, mask=mask)
    v = tl.load(value_ptr + t * value_stride + offs, mask=mask)
    # The cache is contiguous: slot s occupies elements [s * HD, (s + 1) * HD).
    tl.store(k_cache_ptr + slot * HD + offs, k, mask=mask)
    tl.store(v_cache_ptr + slot * HD + offs, v, mask=mask)


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
