# TinyEngram Environment

This repository keeps a clean, direct-dependency environment in
`requirements.txt`. It is intentionally smaller than `pip freeze`: transitive
packages such as `aiohttp`, `nvidia-*`, `requests`, and `urllib3` are left for
pip to resolve.

## Quick start

```bash
conda create -n tinyengram python=3.10 -y
conda activate tinyengram

pip install --upgrade pip
pip install -r requirements.txt
```

The default `requirements.txt` uses the PyTorch CUDA 12.6 wheel index. If your
machine uses another CUDA version, update the three PyTorch pins and wheel
index in `requirements.txt` to match your local driver/runtime. For example,
CUDA 12.1 users can replace the index line with:

```bash
--extra-index-url https://download.pytorch.org/whl/cu121
```

## Tested anchors

- Python: `3.10`
- PyTorch: `2.8.0`
- Transformers: `4.57.3`
- DeepSpeed: `0.18.5`
- Diffusers: `0.36.0`

`transformers==4.57.3` is pinned because current TinyEngram imports Qwen3
classes and `transformers.masking_utils`. Older versions such as `4.51.3` do
not provide the masking utilities used by `engram_qwen.py`.

## Optional evaluation stack

The training and vision demos do not need the evaluation stack. Install it only
when running scripts under `eval_scripts/`:

```bash
pip install -r requirements-eval.txt
```

`vllm` is included for the optional vLLM backend. Hugging Face evaluation paths
can use `lm-eval` without switching `BACKEND` to `vllm`.
