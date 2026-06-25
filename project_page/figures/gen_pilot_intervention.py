"""Generate the pilot-intervention figure for the introduction.

Data: per-benchmark accuracy of ThinkMorph under three inference-time VT
interventions (no retraining):
  - Interleaved VT (default)
  - Text-only inference (VT generation suppressed)
  - Noise-VT inference (each VT replaced with Gaussian noise of matching shape)

The figure visually conveys that removing or corrupting intermediate VTs at
inference leaves accuracy largely unchanged across nine benchmarks --
supporting the introduction's claim that the interleaved VT trajectory is, in
large part, not load-bearing for the final answer.

Style mirrors gen_attention_layer.py / gen_win_loss.py so the figure pairs
naturally with the attention-by-layer figure on the same intro figure row.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT_STEM = Path(
    "/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/"
    "figures/pilot_intervention"
)

# ───────────────────────────────────────────────────────────────────────────
# Data
# ───────────────────────────────────────────────────────────────────────────
BENCHMARKS = [
    "VSP", "VisPuz.", "ChartQA", "VStar",
    "BLINK-J", "MMVP", "SAT", "BLINK", "CV-B",
]
INTERLEAVED = np.array(
    [75.83, 77.50, 78.00, 67.01, 66.00, 78.33, 54.66, 59.49, 80.86]
)
TEXT_ONLY = np.array(
    [73.27, 76.25, 76.80, 64.92, 68.00, 79.00, 51.33, 59.50, 80.07]
)
NOISE_VT = np.array(
    [73.79, 77.75, 77.34, 64.40, 61.33, 76.00, 53.33, 59.60, 80.24]
)

# ───────────────────────────────────────────────────────────────────────────
# Style (kept identical to gen_attention_layer.py)
# ───────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset":  "stix",
    "font.size":          9,
    "axes.labelsize":    10,
    "axes.titlesize":    10,
    "xtick.labelsize":   8.5,
    "ytick.labelsize":   8.5,
    "legend.fontsize":   8.5,
    "axes.linewidth":    0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":        300,
    "savefig.dpi":       300,
})

C_INTLV = "#6E59A5"   # muted purple   – interleaved VT (default)
C_TEXT  = "#1F8A56"   # forest green   – text-only inference
C_NOISE = "#E89A45"   # warm amber     – noise-VT inference
GRID    = "#D8DCE0"

fig, ax = plt.subplots(figsize=(5.6, 3.1))

x = np.arange(len(BENCHMARKS))
width = 0.27

bars_i = ax.bar(
    x - width, INTERLEAVED, width,
    color=C_INTLV, edgecolor="white", linewidth=0.6,
    label="Interleaved VT (default)", zorder=3,
)
bars_t = ax.bar(
    x,         TEXT_ONLY,   width,
    color=C_TEXT,  edgecolor="white", linewidth=0.6,
    label="Text-only inference",       zorder=3,
)
bars_n = ax.bar(
    x + width, NOISE_VT,    width,
    color=C_NOISE, edgecolor="white", linewidth=0.6,
    label="Noise-VT inference",        zorder=3,
)

# horizontal grid only
ax.yaxis.grid(True, color=GRID, linestyle="-", linewidth=0.6, zorder=1)
ax.set_axisbelow(True)

ax.set_xticks(x)
ax.set_xticklabels(BENCHMARKS, fontsize=8.4)
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(40, 88)
ax.set_yticks(np.arange(40, 89, 10))
ax.tick_params(axis="both", which="major", length=3.2, color="#777777",
               direction="out")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#777777")

# annotation: convey the headline observation in italics, not bold
ax.text(
    0.99, 0.96,
    "Removing or corrupting intermediate VTs\nbarely changes accuracy.",
    ha="right", va="top", transform=ax.transAxes,
    fontsize=8.0, fontstyle="italic", color="#5A6A86", zorder=5,
)

leg = ax.legend(
    loc="upper left", bbox_to_anchor=(0.005, 0.97),
    frameon=True, fancybox=False, framealpha=0.92,
    edgecolor="#CFCFCF", borderpad=0.45,
    handlelength=1.6, handletextpad=0.5, labelspacing=0.30,
    fontsize=7.8,
)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout()

for fmt in ("pdf", "png"):
    out = OUT_STEM.with_suffix("." + fmt)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")

plt.close()
