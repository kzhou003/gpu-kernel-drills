"""F1 · FlashInfer paged-KV CSR + plan/run decode — YOUR IMPLEMENTATION.

See the drill page for the spec; the reference lives in solutions/."""
import torch


def block_table_to_csr(block_tables: torch.Tensor, context_lens: torch.Tensor, page_size: int):
    """Vectorised (no Python loop over sequences). Requires context_lens >= 1."""
    # TODO: num_pages = ceil(ctx / page_size); indptr = [0, cumsum(num_pages)] (int32)
    # TODO: indices = block_tables entries with column < num_pages[b], row-major (boolean mask)
    # TODO: last_page_len = ctx - (num_pages - 1) * page_size
    # TODO: return indptr, indices, last_page_len
    raise NotImplementedError


def csr_to_block_table(indptr: torch.Tensor, indices: torch.Tensor, last_page_len: torch.Tensor,
                       page_size: int):
    """Inverse of block_table_to_csr -> (block_tables padded with 0, context_lens)."""
    # TODO: num_pages = indptr[1:] - indptr[:-1]; scatter indices back into a zero-padded [B, max] table
    # TODO: context_lens = (num_pages - 1) * page_size + last_page_len
    raise NotImplementedError


def flashinfer_decode(q, k_cache, v_cache, block_tables, context_lens, workspace=None, return_lse=False):
    """Run FlashInfer's batch decode on the L2-style inputs.

    q [B, Hq, D]; k_cache/v_cache [num_pages, page_size, Hkv, D] ("NHD" layout, passed as a tuple).
    plan() does the CPU-side scheduling once per batch shape; run() launches the kernel and can
    be called again (e.g. once per layer) with the same plan.
    """
    # TODO: workspace (128 MB uint8) -> flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout='NHD')
    # TODO: wrapper.plan(indptr, indices, last, Hq, Hkv, D, page_size, q_data_type=..., kv_data_type=...)
    # TODO: return wrapper.run(q, (k_cache, v_cache), return_lse=return_lse)
    raise NotImplementedError


