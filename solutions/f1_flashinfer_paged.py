"""F1 · FlashInfer's paged-KV format (CSR) + plan/run decode — reference solution.

vLLM-style block table (what L1-L3 used):
    block_tables [B, max_blocks] int32 (padded), context_lens [B] int32
FlashInfer's CSR format (no padding; each sequence's pages are contiguous in `indices`):
    indptr        [B + 1] int32   pages of sequence b are indices[indptr[b] : indptr[b+1]]
    indices       [nnz]   int32   physical page ids, concatenated
    last_page_len [B]     int32   tokens used in each sequence's last page, in [1, page_size]
so  context_len[b] = (indptr[b+1] - indptr[b] - 1) * page_size + last_page_len[b].
"""
import torch


def block_table_to_csr(block_tables: torch.Tensor, context_lens: torch.Tensor, page_size: int):
    """Vectorised (no Python loop over sequences). Requires context_lens >= 1."""
    B, max_blocks = block_tables.shape
    ctx = context_lens.long()
    num_pages = (ctx + page_size - 1) // page_size                       # [B]
    indptr = torch.zeros(B + 1, dtype=torch.int32, device=block_tables.device)
    indptr[1:] = torch.cumsum(num_pages, 0).to(torch.int32)
    # Keep the first num_pages[b] entries of row b. Boolean indexing walks the tensor in
    # row-major order, so each sequence's pages stay together and in order.
    keep = torch.arange(max_blocks, device=block_tables.device)[None, :] < num_pages[:, None]
    indices = block_tables[keep].to(torch.int32)
    last_page_len = (ctx - (num_pages - 1) * page_size).to(torch.int32)   # == (ctx - 1) % P + 1
    return indptr, indices, last_page_len


def csr_to_block_table(indptr: torch.Tensor, indices: torch.Tensor, last_page_len: torch.Tensor,
                       page_size: int):
    """Inverse of block_table_to_csr -> (block_tables padded with 0, context_lens)."""
    num_pages = (indptr[1:] - indptr[:-1]).long()                         # [B]
    B = num_pages.numel()
    max_blocks = int(num_pages.max().item()) if B else 0
    col = torch.arange(max_blocks, device=indices.device)[None, :]
    keep = col < num_pages[:, None]
    block_tables = torch.zeros(B, max_blocks, dtype=torch.int32, device=indices.device)
    block_tables[keep] = indices                                          # same row-major order
    context_lens = ((num_pages - 1) * page_size + last_page_len.long()).to(torch.int32)
    return block_tables, context_lens


def flashinfer_decode(q, k_cache, v_cache, block_tables, context_lens, workspace=None, return_lse=False):
    """Run FlashInfer's batch decode on the L2-style inputs.

    q [B, Hq, D]; k_cache/v_cache [num_pages, page_size, Hkv, D] ("NHD" layout, passed as a tuple).
    plan() does the CPU-side scheduling once per batch shape; run() launches the kernel and can
    be called again (e.g. once per layer) with the same plan.
    """
    import flashinfer

    B, Hq, D = q.shape
    _, page_size, Hkv, _ = k_cache.shape
    indptr, indices, last = block_table_to_csr(block_tables, context_lens, page_size)
    if workspace is None:
        workspace = torch.empty(128 << 20, dtype=torch.uint8, device=q.device)   # scratch for split-KV partials
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(indptr, indices, last, Hq, Hkv, D, page_size,
                 q_data_type=q.dtype, kv_data_type=k_cache.dtype)
    return wrapper.run(q, (k_cache, v_cache), return_lse=return_lse)
