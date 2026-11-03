"""Benchmarks: your kernel vs. the PyTorch baseline (needs a CUDA GPU).

    python bench.py t2              # benchmark drill T2 from mine/
    DRILL_IMPL=solutions python bench.py t2
    python bench.py all

Reports the median time (triton.testing.do_bench: warmup, then many timed runs with GPU
synchronisation) and the achieved memory bandwidth (GB/s) or compute throughput (TFLOP/s).
Compare these with your GPU's peak numbers to see how close to the roofline you are.
"""
import sys

import torch
import torch.nn.functional as F

from drill import load


def bench(fn):
    import triton.testing
    return triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")   # ms


def row(name, ms, gbytes=None, gflop=None):
    extra = ""
    if gbytes is not None:
        extra = f"{gbytes / (ms * 1e-3):8.1f} GB/s"
    if gflop is not None:
        extra = f"{gflop / (ms * 1e-3) / 1e3:8.1f} TFLOP/s"
    print(f"  {name:<28} {ms * 1e3:9.1f} us  {extra}")


def b_c1():
    m = load("c1_elementwise")
    for n in (1 << 20, 1 << 24, 1 << 27):
        a, b = torch.randn(n, device="cuda"), torch.randn(n, device="cuda")
        print(f"vector_add n={n}")
        gb = 3 * n * 4 / 1e9
        row("torch a + b", bench(lambda: a + b), gb)
        row("yours", bench(lambda: m.vector_add(a, b)), gb)
    x, bias = torch.randn(8192, 8192, device="cuda"), torch.randn(8192, device="cuda")
    print("bias_gelu 8192x8192  (fused: 1 read + 1 write; torch: 2 kernels, 4 passes)")
    gb = 2 * x.numel() * 4 / 1e9
    row("torch gelu(x + bias)", bench(lambda: F.gelu(x + bias, approximate="tanh")), gb)
    row("yours (fused)", bench(lambda: m.bias_gelu(x, bias)), gb)


def b_c2():
    m = load("c2_reduce_sum")
    for n in (1 << 20, 1 << 26):
        x = torch.rand(n, device="cuda")
        print(f"reduce_sum n={n}")
        row("torch x.sum()", bench(lambda: x.sum()), n * 4 / 1e9)
        row("yours", bench(lambda: m.reduce_sum(x)), n * 4 / 1e9)


def b_c3():
    m = load("c3_transpose")
    x = torch.randn(8192, 8192, device="cuda")
    gb = 2 * x.numel() * 4 / 1e9
    print("transpose 8192x8192")
    row("torch x.t().contiguous()", bench(lambda: x.t().contiguous()), gb)
    row("naive", bench(lambda: m.transpose_naive(x)), gb)
    row("tiled", bench(lambda: m.transpose_tiled(x)), gb)


def b_c4():
    m = load("c4_gemm")
    torch.backends.cuda.matmul.allow_tf32 = False
    for n in (512, 2048, 4096):
        A, B = torch.randn(n, n, device="cuda"), torch.randn(n, n, device="cuda")
        gf = 2 * n ** 3 / 1e9
        print(f"fp32 matmul {n}x{n}x{n}")
        row("cuBLAS (torch)", bench(lambda: A @ B), gflop=gf)
        for fn in ("matmul_naive", "matmul_smem", "matmul_regtile"):
            row(fn, bench(lambda: getattr(m, fn)(A, B)), gflop=gf)


def b_t1():
    m = load("t1_add_gelu")
    for n in (1 << 20, 1 << 26):
        x, y = (torch.randn(n, device="cuda", dtype=torch.float16) for _ in range(2))
        print(f"add_gelu fp16 n={n}")
        gb = 3 * n * 2 / 1e9
        row("torch gelu(x + y)", bench(lambda: F.gelu(x + y, approximate="tanh")), gb)
        row("yours", bench(lambda: m.add_gelu(x, y)), gb)


def b_t2():
    m = load("t2_softmax")
    for N in (256, 1024, 4096, 16384):
        x = torch.randn(4096, N, device="cuda", dtype=torch.float16)
        gb = 2 * x.numel() * 2 / 1e9
        print(f"softmax fp16 4096x{N}")
        row("torch.softmax", bench(lambda: torch.softmax(x, -1)), gb)
        row("yours", bench(lambda: m.softmax(x)), gb)


def b_t3():
    m = load("t3_layernorm")
    for N in (1024, 4096, 8192):
        x = torch.randn(8192, N, device="cuda", dtype=torch.float16)
        w, b = torch.randn(N, device="cuda", dtype=torch.float16), torch.randn(N, device="cuda", dtype=torch.float16)
        gb = 2 * x.numel() * 2 / 1e9
        print(f"layer_norm fp16 8192x{N}")
        row("F.layer_norm", bench(lambda: F.layer_norm(x, (N,), w, b)), gb)
        row("yours", bench(lambda: m.layer_norm(x, w, b)), gb)
        row("yours rms_norm", bench(lambda: m.rms_norm(x, w)), gb)


def b_t4():
    m = load("t4_matmul")
    for n in (512, 1024, 2048, 4096, 8192):
        a, b = (torch.randn(n, n, device="cuda", dtype=torch.float16) for _ in range(2))
        gf = 2 * n ** 3 / 1e9
        print(f"fp16 matmul {n}^3")
        row("cuBLAS (torch)", bench(lambda: a @ b), gflop=gf)
        row("yours (autotuned)", bench(lambda: m.matmul(a, b)), gflop=gf)


def b_t5():
    m = load("t5_flash_attn")
    for S in (512, 1024, 2048, 4096):
        for causal in (False, True):
            q, k, v = (torch.randn(4, 16, S, 64, device="cuda", dtype=torch.float16) for _ in range(3))
            gf = 4 * 4 * 16 * S * S * 64 / 1e9 * (0.5 if causal else 1.0)
            print(f"attention B=4 H=16 S={S} D=64 causal={causal}")
            row("torch SDPA", bench(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=causal)), gflop=gf)
            row("yours", bench(lambda: m.flash_attention(q, k, v, causal=causal)), gflop=gf)


def _paged(B, ctx, Hq, Hkv, D, page):
    from tests._paged import make_paged
    k_cache, v_cache, bt, cl = make_paged([ctx] * B, Hkv, D, page, torch.float16, "cuda")
    q = torch.randn(B, Hq, D, device="cuda", dtype=torch.float16)
    kv_gb = 2 * B * ctx * Hkv * D * 2 / 1e9            # bytes of K and V that must be read
    return q, k_cache, v_cache, bt, cl, kv_gb


def b_l1():
    m = load("l1_paged_kv_write")
    T, H, D = 8192, 8, 128
    key, value = (torch.randn(T, H, D, device="cuda", dtype=torch.float16) for _ in range(2))
    k_cache = torch.zeros(T // 16 + 8, 16, H, D, device="cuda", dtype=torch.float16)
    v_cache = torch.zeros_like(k_cache)
    slots = torch.randperm(k_cache.shape[0] * 16, device="cuda")[:T]
    print(f"reshape_and_cache T={T} H={H} D={D}")
    row("yours", bench(lambda: m.reshape_and_cache(key, value, k_cache, v_cache, slots)), 4 * key.numel() * 2 / 1e9)


def b_l2():
    m = load("l2_paged_decode")
    for B, ctx in ((1, 32768), (8, 4096), (64, 1024)):
        q, kc, vc, bt, cl, gb = _paged(B, ctx, 32, 8, 128, 16)
        print(f"paged decode B={B} ctx={ctx} Hq=32 Hkv=8 D=128")
        row("yours", bench(lambda: m.paged_decode_attention(q, kc, vc, bt, cl)), gb)


def b_l3():
    m2, m3 = load("l2_paged_decode"), load("l3_split_kv_decode")
    for B, ctx in ((1, 32768), (8, 4096), (64, 1024)):
        q, kc, vc, bt, cl, gb = _paged(B, ctx, 32, 8, 128, 16)
        print(f"decode B={B} ctx={ctx} Hq=32 Hkv=8 D=128  (KV read = {gb * 1e3:.0f} MB)")
        row("L2 (no split)", bench(lambda: m2.paged_decode_attention(q, kc, vc, bt, cl)), gb)
        for s in (4, 16, 64):
            row(f"L3 num_splits={s}", bench(lambda: m3.split_kv_decode_attention(q, kc, vc, bt, cl, num_splits=s)), gb)


def b_f1():
    m2, f1 = load("l2_paged_decode"), load("f1_flashinfer_paged")
    import flashinfer
    for B, ctx in ((1, 32768), (8, 4096), (64, 1024)):
        q, kc, vc, bt, cl, gb = _paged(B, ctx, 32, 8, 128, 16)
        ws = torch.empty(128 << 20, dtype=torch.uint8, device="cuda")
        indptr, indices, last = f1.block_table_to_csr(bt, cl, 16)
        w = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws, "NHD")
        w.plan(indptr, indices, last, 32, 8, 128, 16, q_data_type=q.dtype, kv_data_type=kc.dtype)
        print(f"decode B={B} ctx={ctx} Hq=32 Hkv=8 D=128")
        row("your L2 (Triton, no split)", bench(lambda: m2.paged_decode_attention(q, kc, vc, bt, cl)), gb)
        row("FlashInfer run() (plan once)", bench(lambda: w.run(q, (kc, vc))), gb)


def b_f2():
    m = load("f2_merge_state")
    import flashinfer
    for S in (128, 8192):
        va, vb = (torch.randn(S, 32, 128, device="cuda", dtype=torch.float16) for _ in range(2))
        sa, sb = (torch.randn(S, 32, device="cuda") for _ in range(2))
        gb = 3 * va.numel() * 2 / 1e9
        print(f"merge_state S={S} H=32 D=128")
        row("yours", bench(lambda: m.merge_state(va, sa, vb, sb)), gb)
        row("flashinfer.merge_state", bench(lambda: flashinfer.merge_state(va, sa, vb, sb)), gb)


def b_f3():
    m = load("f3_fused_norm_act")
    import flashinfer
    for T in (16, 4096):
        x, r = (torch.randn(T, 4096, device="cuda", dtype=torch.bfloat16) for _ in range(2))
        w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
        gb = 4 * x.numel() * 2 / 1e9
        print(f"fused_add_rmsnorm T={T} H=4096 bf16")
        row("yours", bench(lambda: m.fused_add_rmsnorm(x, r, w)), gb)
        row("flashinfer", bench(lambda: flashinfer.fused_add_rmsnorm(x, r, w)), gb)
        y = torch.randn(T, 2 * 14336, device="cuda", dtype=torch.bfloat16)
        gb = 1.5 * y.numel() * 2 / 1e9
        print(f"silu_and_mul T={T} I=14336 bf16")
        row("yours", bench(lambda: m.silu_and_mul(y)), gb)
        row("flashinfer", bench(lambda: flashinfer.silu_and_mul(y)), gb)


def b_f4():
    m = load("f4_topk_topp_renorm")
    import flashinfer
    for B in (1, 64):
        p = torch.softmax(torch.randn(B, 151936, device="cuda"), -1)
        print(f"renorm B={B} V=151936")
        row("yours top_k=20", bench(lambda: m.top_k_renorm_probs(p, 20)))
        row("flashinfer top_k=20", bench(lambda: flashinfer.top_k_renorm_probs(p, 20)))
        row("yours top_p=0.9", bench(lambda: m.top_p_renorm_probs(p, 0.9)))
        row("flashinfer top_p=0.9", bench(lambda: flashinfer.top_p_renorm_probs(p, 0.9)))
        row("torch sort-based top_p", bench(lambda: torch.sort(p, -1, descending=True)))


def b_m2():
    m = load("m2_fused_moe")
    T, E, K, H, I = 4096, 64, 8, 2048, 768            # Qwen3-30B-A3B-like expert shapes
    x = torch.randn(T, H, device="cuda", dtype=torch.bfloat16) * 0.5
    logits = torch.randn(T, E, device="cuda")
    w13 = torch.randn(E, 2 * I, H, device="cuda", dtype=torch.bfloat16) / H ** 0.5
    w2 = torch.randn(E, H, I, device="cuda", dtype=torch.bfloat16) / I ** 0.5
    gf = 2 * T * K * (H * 2 * I + I * H) / 1e9
    print(f"MoE layer T={T} E={E} top{K} H={H} I={I}")
    row("yours (2 grouped GEMMs)", bench(lambda: m.moe_layer(x, logits, w13, w2, K)), gflop=gf)
    # same FLOPs as two dense GEMMs over all T*K rows (an upper bound the grouped kernel chases)
    xd = torch.randn(T * K, H, device="cuda", dtype=torch.bfloat16)
    wd1 = torch.randn(H, 2 * I, device="cuda", dtype=torch.bfloat16)
    wd2 = torch.randn(I, H, device="cuda", dtype=torch.bfloat16)
    row("dense cuBLAS, same FLOPs", bench(lambda: (xd @ wd1)[:, :I] @ wd2), gflop=gf)


def b_m3():
    m = load("m3_fp8_block_gemm")
    for M, N, K in ((128, 7168, 2048), (4096, 7168, 2048), (4096, 4096, 7168)):
        a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        aq, as_ = m.per_token_group_quant_fp8(a)
        bq, bs = m.block_quant_fp8(b)
        gf = 2 * M * N * K / 1e9
        print(f"GEMM M={M} N={N} K={K}")
        row("bf16 cuBLAS", bench(lambda: a @ b.t()), gflop=gf)
        row("yours fp8 block-scaled", bench(lambda: m.fp8_block_gemm(aq, as_, bq, bs)), gflop=gf)
        row("yours act quant only", bench(lambda: m.per_token_group_quant_fp8(a)))


BENCHES = {k[2:]: v for k, v in globals().items() if k.startswith("b_")}


def main():
    if not torch.cuda.is_available():
        sys.exit("benchmarks need a CUDA GPU")
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(torch.cuda.get_device_name(), "| torch", torch.__version__)
    for k in BENCHES if which == "all" else [which]:
        if k not in BENCHES:
            sys.exit(f"no benchmark for {k!r}; choose from {sorted(BENCHES)}")
        print(f"== {k}")
        BENCHES[k]()


if __name__ == "__main__":
    main()
