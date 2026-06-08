# Training

This document is the complete reproduction protocol for the paper's
main result (Visual-OPSD), the two baselines (Text-only SFT,
Visual-OPSD-Noise), and the Phase-0 KL diagnostic.

Hyperparameters below are the ones used to produce the reported numbers
(Table 2 of the paper); they are also the defaults baked into the
launcher scripts.

## Prerequisites

- Installed environment per [INSTALL.md](INSTALL.md).
- Base checkpoint at `models/ThinkMorph-7B/` (override via the first
  positional argument to any launcher).
- Training data at `datasets/<task>/data/*.parquet` (or set
  `VISUAL_OPSD_DATA_ROOT`).
- 8 × NVIDIA H800 80 GB (or equivalent). The full-rank on-policy
  sampler requires each rank to materialise the student under
  `FSDP.summon_full_params`, so a single 80 GB GPU is enough capacity
  per rank — but lower-memory GPUs will OOM during the dual forward.

## 1. KL diagnostic (Phase 0)

Before paying for full training, verify the distillable gap on the
frozen base model:

```bash
bash scripts/visual_opsd/run_kl_diagnostic.sh   # ~1 hour on 4× H800
```

Outputs go to `results/visual_opsd_kl_diagnostic/`. The paper observes
**K<sub>gen</sub> ≈ 4.64 nats/token** on 1,000 random samples
(Section 2.2 + Table 1) — a clearly non-trivial divergence,
confirming that the VT context shifts the model's completion
distribution and that there is something to distil.

The diagnostic also serves as a post-training sanity check: re-run it
on the trained student to measure how much of K<sub>gen</sub> has been
closed (Appendix I, Table 5). The paper observes a **58.4 %** gap-
closing for Visual-OPSD versus **3.5 %** for the noise control.

## 2. Text-only SFT baseline

```bash
bash scripts/visual_opsd/run_sft_baseline.sh \
    models/ThinkMorph-7B \
    2000 \
    sft-baseline
```

Reproduces the Text-only SFT row of Table 2. Note that the SFT baseline
is initialised from **BAGEL-7B**, not from ThinkMorph-7B, to avoid
re-fitting on data the base model has already seen — see Appendix D of
the paper for the rationale. If you want a strict reproduction, swap
the `--model_path` to your BAGEL-7B checkpoint.

## 3. Visual-OPSD (main)

```bash
# positional args: MODEL_PATH TEACHER_MODE JSD_BETA JSD_TEMP JSD_WEIGHT TOTAL_STEPS OUTPUT_NAME
bash scripts/visual_opsd/run_visual_opsd.sh \
    models/ThinkMorph-7B \
    ema 0.5 1.0 1.0 \
    1000 \
    visual-opsd-ema-beta0.5
```

The paper's reported checkpoint is at **step 1000**. The launcher
defaults to `TOTAL_STEPS=2000`, which produces a slightly stronger
final checkpoint at the cost of one extra day of compute; use `1000`
if you want to match Table 2 exactly.

## 4. Visual-OPSD-Noise ablation

```bash
bash scripts/visual_opsd/run_visual_opsd_noise.sh \
    models/ThinkMorph-7B \
    ema 0.5 1.0 1.0 \
    1000 \
    visual-opsd-noise-ema-beta0.5
```

Identical to Visual-OPSD except `--noise_vt True` is passed, which
replaces every privileged VT image tensor with
`torch.randn_like(vt)` before the teacher's packed sequence is built.
The teacher context still contains the system prompt, the problem
image, the question, the reference intro, and the transition prompt —
only the VT pixels become Gaussian noise. This isolates whether the
Visual-OPSD gains come from VT *semantic content* (real run) or from
generic regularisation (this run).

The paper observes that the noise control yields only **+0.40 pp** over
Text-only SFT, while real Visual-OPSD yields **+10.28 pp** — a decisive
gap that, together with the KL gap-closing analysis above, rules out
regularisation as the primary mechanism.

## 5. Hyperparameters (paper defaults)

These match Appendix C of the paper and are the values baked into the
launchers.

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Learning rate | 1e-5 | JSD β | 0.5 |
| Min learning rate | 1e-7 | JSD temperature | 1.0 |
| LR scheduler | cosine | JSD top-K | 256 |
| Warmup steps | 200 | JSD token clip | 0.05 |
| Total steps | 1000 | CE weight | 0.0 |
| AdamW (β₁, β₂) | (0.9, 0.95) | JSD weight | 1.0 |
| AdamW ε | 1e-15 | Loss kind | Pure JSD |
| Max grad norm | 1.0 | On-policy sampling | yes |
| Gradient accum. steps | 2 | EMA decay | 0.995 |
| Max forward tokens | 10240 | Teacher mode | EMA |
| Max sampled completion | 1024 | Image skips | 1 |

Image preprocessing follows NaViT: stride 14, max image size 980, min
378, max pixels 2,007,040.

## 6. Output layout

After training, the launcher writes:

```
results/visual-opsd-ema-beta0.5/
├── checkpoints/
│   ├── 0000200/                # warmup boundary
│   ├── 0000500/
│   ├── 0001000/                # paper checkpoint
│   └── ...
├── visual-opsd-ema-beta0.5.log
└── wandb/                      # if not --wandb_offline
```

Each checkpoint folder contains `model.safetensors`, `optimizer.pt`,
and the tokenizer / vit / llm config files needed to reload the
student under `examples/inference_demo.py` or by VLMEvalKit-ThinkMorph.

## 7. Compute budget

The numbers in Table 2 of the paper were produced with the following
budget per run on 8 × H800:

| Run | Steps | Approx. wall time |
|---|---|---|
| KL diagnostic | n/a (1,000 samples) | ~1 hour |
| Text-only SFT | 2,000 | ~6 hours |
| Visual-OPSD-Noise | 1,000 | ~10 hours |
| Visual-OPSD (main) | 1,000 | ~10 hours |

The wall-clock cost of Visual-OPSD is dominated by the on-policy
student sampling step (`FSDP.summon_full_params` + BAGEL generation
primitives), not by the dual forward / backward.
