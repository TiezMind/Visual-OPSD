# Paper figures

This directory contains the figures referenced from the project's
`README.md`. They are the (PNG-rendered) originals used in the paper
**Visual-OPSD: Cross-Modal On-Policy Self-Distillation for Efficient
Unified Multimodal Reasoning**.

| File                       | Paper figure | Caption (short) |
|----------------------------|--------------|-----------------|
| `main_results.png`         | Figure 1     | Headline results: per-benchmark accuracy, Δ-vs-teacher, and accuracy–latency Pareto curve. Visual-OPSD (green) matches or beats the VT teacher on 6/9 benchmarks and dominates the Pareto front at 74.03 % / 10.0 s. |
| `pilot_intervention.png`   | Figure 2     | Pilot intervention on the frozen ThinkMorph teacher: removing or corrupting the intermediate VT images barely changes accuracy across all 9 benchmarks, motivating the move from generative VT to distillation. |
| `latency_comparison.png`   | Figure 3     | Per-sample inference latency on a single H800 — Visual-OPSD is 14.3× faster than the VT teacher because it never invokes the diffusion pathway. |
| `attention_analysis.png`   | Figure 5     | V\* attention-pattern analysis: once a VT is rendered, downstream reasoning attends almost exclusively to the generated image and largely ignores the original input — evidence that VTs concentrate task-relevant visual signal. |

The original LaTeX / PDF sources for these figures live in the paper's
supplementary material. They are bundled here as PNGs so the project
`README.md` can render inline on GitHub.
