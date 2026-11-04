# gpu-kernel-drills · Kernel Dev 训练营

以 FlashInfer 为主线的 kernel 开发训练，25 道题（约 23 小时）：

- **核心**：CUDA C++（C1–C4）→ Triton（T1–T5）→ 推理 attention 从零写（L1–L3）→ **FlashInfer**（F1–F4 对着官方 API 实现并对拍，F5 贡献实战）
- **进阶**：Layout / CuTe（G1–G2）、MoE 与 FP8（M1–M3）、MLX / Metal（X1–X3）
- 另有：GPU 基础与语法速查、**模型 × 硬件 × 当前最优 kernel 全景地图**、面试题库

每题有背景、要用到的知识、题干、验收测试、三级提示和逐行注释的参考解。

**在线阅读：https://kzhou003.github.io/gpu-kernel-drills/** · [PDF 版（全部提示与参考解展开）](gpu-kernel-drills.pdf) · 本地打开 `index.html` 亦可。

```bash
./check.sh f2                # 测 mine/f2_merge_state.py
./check.sh f2 --solutions    # 测参考解
./check.sh f2 --bench        # 性能（NVIDIA GPU）；X 题用 .venv-mlx/bin/python bench_mlx.py
python3 build/sources.py fetch   # 第一次构建前：按固定 commit 下载被引用的源码文件（需要 gh），并校验所有源码链接
python3 build/check_links.py     # 检查文档 / 论文等外部链接
python build/build.py            # 重新生成 index.html（需要 markdown-it-py；链接解析失败会直接报错）
```

| 环境 | 怎么准备 |
|---|---|
| NVIDIA GPU，没有系统 CUDA toolkit | `./setup_env.sh /path/to/python-with-torch && source env.sh`（pip 安装与 torch 同版本的 nvcc）；F 题再 `pip install flashinfer-python` |
| 只有 CPU | Triton / L / F1–F4 / G / M1–M2 自动走 `TRITON_INTERPRET=1`；macOS 没有 triton wheel，用 `docker run -it -v $PWD:/w -w /w python:3.12-slim bash` |
| Apple silicon | `python3 -m venv .venv-mlx && .venv-mlx/bin/pip install mlx numpy pytest`（arm64 Python） |
| 只想读代码（编辑器里跳转定义） | `./setup_ide.sh`：torch、Triton、FlashInfer、vLLM 的源码和 CUDA 头文件装到 `.venv-ide/`、`.refs/`，并配置 VS Code（Pylance + clangd）。不需要 GPU |

页面中约 240 个源码链接都固定在 commit 上（`build/sources/pins.txt`，共 16 个仓库：FlashInfer、vLLM、SGLang、Triton、CUTLASS、flash-attention、DeepGEMM、MLX 等），`file@symbol` 形式的链接直接跳到定义所在的行。

目录：`mine/` 你的代码 · `solutions/` 参考解 · `tests/` 验收 · `content/` 题面 · `build/` 页面生成。
参考解已在 RTX 5090（torch 2.13 / CUDA 13.0 / Triton 3.7 / FlashInfer 0.7.2）和 Apple M2（MLX 0.32）上全部通过。
