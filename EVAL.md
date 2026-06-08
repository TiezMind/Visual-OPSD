# Evaluation

The paper evaluates all systems with the open-source
[VLMEvalKit-ThinkMorph](https://github.com/hychaochao/VLMEvalKit_Thinkmorph)
fork, which already wires up the ThinkMorph / Visual-OPSD inferencer
class and supports all nine benchmarks reported in the paper.

This document records the **exact** protocol used for the paper so that
others can reproduce Table 2.

## 1. Set up the evaluation harness

```bash
git clone https://github.com/hychaochao/VLMEvalKit_Thinkmorph.git
cd VLMEvalKit_Thinkmorph
pip install -r requirements.txt
```

The harness depends on the Visual-OPSD/ThinkMorph training stack (this
repo), so install Visual-OPSD first (per [INSTALL.md](INSTALL.md)) and
activate the same `.venv` for evaluation.

## 2. Point the harness at your checkpoint

Edit the `thinkmorph_series` block in `vlmeval/config.py`:

```python
from functools import partial
from vlmeval.vlm import ThinkMorph

thinkmorph_series = {
    # Visual-OPSD student (text-only).  understanding_output=True
    # tells the harness to skip the VT diffusion step entirely.
    "visual_opsd": partial(
        ThinkMorph,
        model_path="path/to/results/visual-opsd-ema-beta0.5/checkpoints/0001000",
        think=True,
        understanding_output=True,         # text-only inference
        temperature=0.3,
        max_think_token_n=4096,
        save_dir="results_imgs/visual_opsd",
    ),
    # ThinkMorph teacher with full interleaved VT generation.
    "thinkmorph": partial(
        ThinkMorph,
        model_path="path/to/ThinkMorph-7B",
        think=True,
        understanding_output=False,        # diffusion VT pathway active
        temperature=0.3,
        max_think_token_n=4096,
        save_dir="results_imgs/thinkmorph",
    ),
    # Visual-OPSD-Noise ablation.
    "visual_opsd_noise": partial(
        ThinkMorph,
        model_path="path/to/results/visual-opsd-noise-ema-beta0.5/checkpoints/0001000",
        think=True,
        understanding_output=True,
        temperature=0.3,
        max_think_token_n=4096,
        save_dir="results_imgs/visual_opsd_noise",
    ),
    # Text-only SFT baseline.
    "thinkmorph_text_only": partial(
        ThinkMorph,
        model_path="path/to/sft-baseline/checkpoints/0002000",
        think=True,
        understanding_output=True,
        temperature=0.3,
        max_think_token_n=4096,
        save_dir="results_imgs/text_only",
    ),
}
```

The `ThinkMorph` wrapper lives in
`VLMEvalKit_Thinkmorph/vlmeval/vlm/ThinkMorph.py` and ships a
self-contained copy of the Visual-OPSD inferencer (`InterleaveInferencer`
+ packed BAGEL modeling code under `vlmeval/vlm/thinkmorph/`), so it
will work even if this Visual-OPSD repo is not on the Python path.

## 3. Run

The paper reports **9 benchmarks** (3-run average). The launcher below
matches the paper's protocol.

```bash
export ARK_API_KEY=<your-key>             # or OPENAI_API_KEY=<your-key>
export OPENAI_API_BASE=<gateway>          # if not using openai.com directly

MASTER_ADDR=${ARNOLD_WORKER_0_HOST:-127.0.0.1}
MASTER_PORT=${ARNOLD_WORKER_0_PORT:-29500}
NPROC_PER_NODE=${ARNOLD_WORKER_GPU:-8}
NNODES=${ARNOLD_WORKER_NUM:-1}
NODE_RANK=${ARNOLD_ID:-0}

torchrun \
    --master_port=${MASTER_PORT} \
    --master_addr=${MASTER_ADDR} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --nnodes=${NNODES} \
    --node_rank=${NODE_RANK} \
    run.py \
    --data VSP_maze_task_main_original \
           VisPuzzle \
           ChartQA_h_bar ChartQA_v_bar \
           VStarBench \
           BLINK_Jigsaw \
           MMVP \
           BLINK \
           SAT_circular \
           CV-Bench-2D CV-Bench-3D \
    --model visual_opsd \
    --judge gpt-5 \
    --work-dir ./results/visual_opsd
```

## 4. Protocol details (matches paper Appendix M)

- **Decoding.** Greedy (temperature=0), max 1024 output tokens.
- **Hardware.** Single H800 GPU, batch size 1.
- **Image preprocessing.** NaViT (stride 14, max 980 px, min 378 px,
  max 2,007,040 pixels). Identical for all systems.
- **ThinkMorph VT generation.** 50 DDPM denoising steps with classifier
  -free guidance scale 3.5.
- **Latency.** Includes the full pipeline (image preprocessing, ViT
  encoding, LLM decoding, and VT generation where applicable). Excludes
  data loading I/O.
- **V*.** All external visual-search tools (e.g. the SEAL/V* pipeline)
  are disabled across all evaluated VLMs, so every model is scored on
  its native single-pass multimodal reasoning capability.
- **Judges.** GPT-5 is used as the answer judge for free-form
  benchmarks; GPT-4o was used for the original ThinkMorph table.
  Multiple-choice benchmarks (VSP, BLINK, BLINK-J, MMVP, SAT, CV-Bench)
  use exact-match against the option letter; ChartQA / VisPuzzle / V*
  use a judge LM.

The reported numbers in Table 2 are the **average across 3 independent
runs** with different seeds. For a single-seed reproduction the numbers
are within ±0.5 pp of the table values on most benchmarks.

## 5. Output token statistics (Appendix G)

The harness also records mean output tokens per sample, used to compute
the latency table (Table 4 of the paper). Visual-OPSD averages **201
tokens** per sample, roughly 2× shorter than the SFT baseline (411) and
ThinkMorph text tokens (452).
