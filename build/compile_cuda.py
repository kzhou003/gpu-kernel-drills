"""Compile every CUDA drill (solutions + skeletons) through load_inline without a GPU."""
import importlib, os, sys, traceback
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.5;8.0")
sys.path.insert(0, "/w")
ok = True
for impl in ("solutions", "mine"):
    for stem in ("c1_elementwise", "c2_reduce_sum", "c3_transpose", "c4_gemm"):
        try:
            mod = importlib.import_module(f"{impl}.{stem}")
            ext = mod._ext()
            print(f"OK   {impl}/{stem}.cu ->", [n for n in dir(ext) if not n.startswith("_")])
        except Exception as e:
            ok = False
            print(f"FAIL {impl}/{stem}.cu"); traceback.print_exc(limit=1)
            print(str(e)[-3000:])
sys.exit(0 if ok else 1)
