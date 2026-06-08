# Installation

Visual-OPSD requires Python 3.10, CUDA 12.x, and PyTorch 2.5. The
instructions below mirror the environment used for the paper's
experiments (8 × H800 80 GB, Ubuntu 20.04, CUDA 12.6, PyTorch 2.5.1).

## 1. Python environment

We recommend [uv](https://github.com/astral-sh/uv) for a fast,
reproducible setup.

```bash
git clone <repo-url> Visual-OPSD
cd Visual-OPSD

python -m pip install -U uv
uv venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

Alternatively, with `pip`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 2. flash-attention

`flash-attn` is required by the SigLIP NaViT vision encoder and the
Qwen2-MoT decoder. **Do not** install it with `pip install flash-attn`
(it will try to compile from source against your local CUDA, which is
slow and brittle). Use a prebuilt wheel that matches your CUDA and
PyTorch versions, e.g.:

```bash
# CUDA 12.6 + PyTorch 2.5 (the default used for the paper)
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.8/flash_attn-2.5.9+cu126torch2.5-cp310-cp310-linux_x86_64.whl
```

Other (cu118 / cu121 / torch 2.4) wheels are available in the same
release page. Pick the one matching `torch.version.cuda` and
`torch.__version__`.

## 3. Verify the install

```bash
python -c "
import torch, flash_attn
print('torch:', torch.__version__, 'cuda:', torch.version.cuda)
print('flash-attn:', flash_attn.__version__)
print('cuda available:', torch.cuda.is_available(),
      'devices:', torch.cuda.device_count())
"
```

You should see PyTorch 2.5.x, CUDA 12.x, and flash-attn 2.5.x.

## 4. Base model checkpoint

```bash
python download_model.py            # → models/ThinkMorph-7B/
```

This pulls the BAGEL-7B-MoT model fine-tuned on interleaved CoT traces.
The same checkpoint serves as both the **teacher** (with privileged VT
context) and the **starting point of the student** in all Visual-OPSD
experiments. Visual-OPSD does not modify the model architecture, so the
upstream
[ThinkMorph-7B weights](https://huggingface.co/ThinkMorph/ThinkMorph-7B)
load directly.

## 5. Training data

The four reasoning datasets (`Visual_Search`, `Spatial_Navigation`,
`Jigsaw_Assembly`, `Chart_Refocus`; 24,990 samples total) are released
on the [ThinkMorph Hugging Face hub](https://huggingface.co/ThinkMorph).
After downloading, the expected layout is:

```
datasets/
├── Visual_Search/
│   └── data/*.parquet
├── Spatial_Navigation/
│   └── data/*.parquet
├── Jigsaw_Assembly/
│   └── data/*.parquet
└── Chart_Refocus/
    └── data/*.parquet
```

If you put them somewhere else, either set `VISUAL_OPSD_DATA_ROOT` or
edit the path helpers in [`data/dataset_info.py`](data/dataset_info.py).

```python
from datasets import load_dataset

# Replace any of these with hf-cli / snapshot_download as preferred.
for ds in ["Visual_Search", "Spatial_Navigation",
           "Jigsaw_Assembly", "Chart_Refocus"]:
    load_dataset(f"ThinkMorph/{ds}", split="train")
```

Each parquet record contains: `problem_image`, `question`,
`reasoning_thought_*`, `reasoning_image_*` (the VT images that become
the teacher's privileged context), `answer`, and
`full_text_only_thought`.

## 6. (Optional) Login to Weights & Biases

```bash
export WANDB_API_KEY=<your-key>
# Or disable online sync entirely with --wandb_offline True
```

## Troubleshooting

- **`ImportError: flash_attn`** — install the prebuilt wheel from
  Section 2; do not let `pip` resolve `flash-attn` from PyPI.
- **NCCL hangs on multi-node** — confirm
  `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` is set (the launcher scripts
  default to 1800 s).
- **CUDA OOM during sampling** — drop `--sample_max_new_tokens` or
  `--max_forward_tokens`; the on-policy sampler scales with the
  *sampled* completion length, not the dataset length.
- **CUDA OOM during the dual forward** — the teacher forward is run
  under `torch.no_grad()`; nevertheless, both contexts must fit on each
  rank. Reduce `--max_forward_tokens` and/or enable
  `--cpu_offload True` (the optimizer state is already CPU-offloaded
  during sampling).
- **"Model path not found"** — make sure
  `models/ThinkMorph-7B/llm_config.json` exists. Visual-OPSD reads the
  Qwen2 / SigLIP config files from the checkpoint directory.
