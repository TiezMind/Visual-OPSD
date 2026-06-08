# Visual-OPSD experiment scripts

This directory contains the training launchers and supporting utilities
for **Visual-OPSD**: a cross-modal on-policy self-distillation framework
that transfers visual-generation knowledge into the text-only
understanding pathway of a unified multimodal model.

## File overview

| File | Purpose |
|------|---------|
| `kl_diagnostic.py`             | Phase 0: measure K<sub>gen</sub> = KL(teacher\_with\_VT ‖ student\_no\_VT) on the frozen base model. Provides a GO/NO-GO gate before training (paper: K<sub>gen</sub> ≈ 4.64 nats/token across 1k samples). |
| `train_visual_opsd.py`         | **Main on-policy Visual-OPSD trainer** — per-step paired dual-forward + JSD loss. No trace cache. (Paper default.) |
| `train_visual_opsd_offline.py` | Visual-OPSD offline (cached-teacher) trainer + SFT baseline trainer. Reads pre-computed teacher logprobs. |
| `on_policy_sampler.py`         | Student on-policy completion sampler (FSDP-aware, handles BAGEL's `<image_start>` skip). |
| `opsd_loss.py`                 | Generalized JSD (forward/reverse interpolation, top-k, per-token clip) + Tinker reverse-KL. |
| `collect_traces.py`            | Pre-compute teacher top-K logprobs for the offline trainer. |
| `test_opsd_dataset.py`         | Smoke test for the paired raw-envelope dataset. |
| `test_opsd_loss.py`            | Unit tests for `opsd_loss.py`. |
| `test_opsd_sampler.py`         | Smoke test for the on-policy sampler. |
| `run_kl_diagnostic.sh`         | Launch the KL diagnostic (recommended GO/NO-GO before training). |
| `run_visual_opsd.sh`           | **Launch on-policy Visual-OPSD training.** |
| `run_visual_opsd_noise.sh`     | Noise-control ablation: replace VT images with Gaussian noise. |
| `run_sft_baseline.sh`          | Launch text-only SFT baseline. |
| `run_collect_traces.sh`        | Cache teacher logprobs (offline trainer only). |
| `run_visual_opsd_offline.sh`   | Launch the offline (cached-teacher) Visual-OPSD trainer. |
| `run_sanity.sh`                | Quick overfit sanity check. |

## Recommended order

```
M0  Sanity         bash scripts/visual_opsd/run_sanity.sh
M1  KL diagnostic  bash scripts/visual_opsd/run_kl_diagnostic.sh    # GO/NO-GO gate
M2  SFT baseline   bash scripts/visual_opsd/run_sft_baseline.sh     # Table 2 row "Text-only SFT"
M3  Visual-OPSD    bash scripts/visual_opsd/run_visual_opsd.sh      # Table 2 row "Visual-OPSD (Ours)"
M4  Ablations      bash scripts/visual_opsd/run_visual_opsd_noise.sh  # Table 2 row "Visual-OPSD-Noise"
```

See [TRAIN.md](../../TRAIN.md) for the full reproduction protocol and
hyperparameter table.

## On-policy Visual-OPSD (paper default)

Per training step, for each raw sample:

1. **On-policy sampling** — the student generates a full completion
   (reasoning + answer) from its current policy under a text-only
   prompt: `[system?, problem_image, question]` → `generate_text` →
   `c_1..c_k`. Runs inside `FSDP.summon_full_params(recurse=True)` so
   every rank materialises the full model for BAGEL's inference
   primitives.
2. **Dual packing** — the same `completion_ids` are wrapped into two
   packed batches sharing a loss=1 tail:
   - student: `[system?, problem_image, question, completion]`
   - teacher: `[system?, problem_image, question, <reference_intro>, (VT_image_i)+, <transition>, completion]`

   The teacher's privileged channel is **strictly visual-only**: only
   the intermediate VT images appear. The text-form `thought_i` traces
   and the ground-truth `answer` are deliberately omitted, so the
   teacher–student information gap isolates the visual generation
   pathway.
3. **Dual forward** — student forward with `return_logits=True` (grad),
   teacher forward with `return_logits=True` (no_grad). Both tensors
   have shape `[len(completion)+1, V]` and align 1-to-1.
4. **Loss** — `ce_weight * CE(student, completion) +
   jsd_weight * generalizedJSD(student_logits, teacher_logits)` over
   the entire completion span. `opsd_loss.py` supports β-interpolation,
   top-K restriction (paper: K=256), per-token JSD clip (paper: 0.05),
   and an optional Tinker reverse-KL variant. **The paper uses pure
   JSD (`ce_weight=0`, `jsd_weight=1`).**
5. **EMA update** — EMA weights are updated for use in the next step's
   teacher (`teacher_mode=ema`, `ema_decay=0.995`).

## Quick launch

```bash
# Paper default: EMA teacher, β=0.5, τ=1.0, jsd_weight=1.0, ce_weight=0.0, 1000 steps
bash scripts/visual_opsd/run_visual_opsd.sh \
    models/ThinkMorph-7B ema 0.5 1.0 1.0 1000 visual-opsd-ema-beta0.5
```

### Teacher modes

- `self`  — teacher = student (trivial no-grad duplicate forward).
- `ema`   — teacher = EMA(student), paper default (`ema_decay≈0.995`).
- `fixed` — teacher = frozen copy of the initial checkpoint.

### Ablation sweep examples

```bash
# β (teacher-weight) sweep
for B in 0.0 0.3 0.5 0.7 1.0; do
    bash scripts/visual_opsd/run_visual_opsd.sh models/ThinkMorph-7B \
        ema $B 1.0 1.0 1000 visual-opsd-beta${B}
done

# Teacher-mode sweep
for T in self ema fixed; do
    bash scripts/visual_opsd/run_visual_opsd.sh models/ThinkMorph-7B \
        $T 0.5 1.0 1.0 1000 visual-opsd-${T}
done

# Noise control (VT replaced with Gaussian noise)
bash scripts/visual_opsd/run_visual_opsd_noise.sh models/ThinkMorph-7B \
    ema 0.5 1.0 1.0 1000
```

The paper reports the K=256, clip=0.05 JSD configuration; see
Appendix J of the paper (Table 6) for sensitivity to these knobs.

## Data

- 24,990 training samples across 4 reasoning datasets:
  `Visual_Search` (6,990), `Spatial_Navigation` (6,000),
  `Jigsaw_Assembly` (6,000), `Chart_Refocus` (6,000).
- Each parquet record contains `problem_image`, `question`,
  `reasoning_thought_*`, `reasoning_image_*` (VT images), `answer`,
  and `full_text_only_thought`. The on-policy paired dataset uses only
  the problem image, question, and VT images (plus `answer` as a
  reference target for accuracy logging).
- Training and evaluation data are disjoint: training uses the
  designated training splits, evaluation uses the held-out test sets of
  the 9 benchmarks listed in [EVAL.md](../../EVAL.md).

See [INSTALL.md](../../INSTALL.md) §5 for the dataset directory layout
and download instructions.

## Model

- **Teacher**: ThinkMorph-7B base checkpoint (frozen / EMA / self,
  depending on `--teacher_mode`). Place it under
  `models/ThinkMorph-7B/` or override via `--model_path`.
- **Student**: initialised from the same ThinkMorph-7B checkpoint.
- Both run in **understanding-only** mode (`--visual_gen False`); the
  VAE / diffusion pathway is not used during Visual-OPSD training.
