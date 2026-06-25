"""Generate per-layer cross-modal attention figure.

Source data: eval_results/attention_vstar_layers/attention_summary.jsonl
- phase=txt0: text before VT generation (vt_frac is always 0 since no VT yet)
- phase=txt1: text after  VT generation (img1 = generated VT, img0 = input)

We plot three series for the ALL category, with a light variance band
between the two task sub-categories (direct_attributes, relative_position)
to convey that the trend is robust across VStar sub-tasks.
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUMMARY = Path(
    "/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/"
    "eval_results/attention_vstar_layers/attention_summary.jsonl"
)
OUT_STEM = Path(
    "/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/"
    "figures/attention_layer_analysis"
)

# ───────────────────────────────────────────────────────────────────────────
# Load + reshape
# ───────────────────────────────────────────────────────────────────────────
records = [json.loads(l) for l in SUMMARY.read_text().splitlines() if l.strip()]

def series(phase: str, field: str, category: str):
    rows = [r for r in records if r["phase"] == phase and r["category"] == category]
    rows.sort(key=lambda r: r["layer"])
    layers = np.array([r["layer"] for r in rows])
    vals = np.array([r[field] for r in rows]) * 100.0  # → percent
    return layers, vals

# main lines: category=ALL
L, txt0_img0 = series("txt0", "input_image_frac", "ALL")
_, txt1_img1 = series("txt1", "vt_frac",          "ALL")
_, txt1_img0 = series("txt1", "input_image_frac", "ALL")

# variance bands: min/max across the two sub-categories
SUBCATS = ("direct_attributes", "relative_position")

def envelope(phase: str, field: str):
    arrs = [series(phase, field, c)[1] for c in SUBCATS]
    arrs = np.stack(arrs, axis=0)
    return arrs.min(0), arrs.max(0)

t00_lo, t00_hi = envelope("txt0", "input_image_frac")
t11_lo, t11_hi = envelope("txt1", "vt_frac")
t10_lo, t10_hi = envelope("txt1", "input_image_frac")

# ───────────────────────────────────────────────────────────────────────────
# Style — matches gen_win_loss.py / gen_kl_heatmap.py conventions
# ───────────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times New Roman", "DejaVu Serif", "Times"],
    "mathtext.fontset": "stix",
    "font.size":        9,
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "xtick.labelsize":  8.5,
    "ytick.labelsize":  8.5,
    "legend.fontsize":  8.5,
    "axes.linewidth":   0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi":       300,
    "savefig.dpi":      300,
})

C_T00 = "#2A6FB5"   # txt0→img0   muted royal blue
C_T11 = "#C53A2E"   # txt1→img1   warm vermillion
C_T10 = "#1F8A56"   # txt1→img0   forest green
GRID  = "#D8DCE0"
INK   = "#2B2B2B"

# Background tints for the two regions called out in the paper text
EARLY = "#F2F4FA"   # very pale blue
DEEP  = "#FAF3F1"   # very pale red

fig, ax = plt.subplots(figsize=(5.6, 3.1))

# region tints (drawn first so everything else sits on top)
ax.axvspan(-0.4, 3.4,  facecolor=EARLY, edgecolor="none", zorder=0)
ax.axvspan(18.6, 27.4, facecolor=DEEP,  edgecolor="none", zorder=0)

# horizontal grid only
ax.yaxis.grid(True, color=GRID, linestyle="-", linewidth=0.6, zorder=1)
ax.set_axisbelow(True)

# variance bands across sub-categories
ax.fill_between(L, t00_lo, t00_hi, color=C_T00, alpha=0.13, linewidth=0, zorder=2)
ax.fill_between(L, t11_lo, t11_hi, color=C_T11, alpha=0.13, linewidth=0, zorder=2)
ax.fill_between(L, t10_lo, t10_hi, color=C_T10, alpha=0.13, linewidth=0, zorder=2)

common = dict(linewidth=1.6, markersize=4.2, markeredgewidth=0.6,
              markeredgecolor="white", zorder=4, clip_on=True)

ax.plot(L, txt0_img0, color=C_T00, marker="o",
        label=r"$\mathrm{txt}_0 \rightarrow \mathrm{img}_0$  (pre-gen $\rightarrow$ input)",
        **common)
ax.plot(L, txt1_img1, color=C_T11, marker="s",
        label=r"$\mathrm{txt}_1 \rightarrow \mathrm{img}_1$  (post-gen $\rightarrow$ generated VT)",
        **common)
ax.plot(L, txt1_img0, color=C_T10, marker="^",
        label=r"$\mathrm{txt}_1 \rightarrow \mathrm{img}_0$  (post-gen $\rightarrow$ input)",
        **common)

# axes formatting
ax.set_xlabel("Transformer layer index")
ax.set_ylabel("Attention fraction (%)")
ax.set_xlim(-0.6, 27.6)
ax.set_ylim(-0.6, 12.5)
ax.set_xticks(np.arange(0, 28, 4))
ax.set_xticks(np.arange(0, 28, 1), minor=True)
ax.set_yticks(np.arange(0, 13, 2))
ax.tick_params(axis="both", which="major", length=3.2, color="#777777",
               direction="out")
ax.tick_params(axis="both", which="minor", length=1.8, color="#AAAAAA",
               direction="out")

for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#777777")

# Region labels (placed just above max value, in tinted bands)
ax.text(1.5, 11.95, "early layers", ha="center", va="top",
        fontsize=8.0, color="#5A6A86", fontstyle="italic", zorder=3)
ax.text(23.0, 11.95, "deep layers", ha="center", va="top",
        fontsize=8.0, color="#8A4F47", fontstyle="italic", zorder=3)

# Annotation: original input is displaced after generation
ax.annotate(
    "input image is\nnearly ignored",
    xy=(15.0, 0.10),
    xytext=(15.5, 3.6),
    fontsize=7.6, color=C_T10, ha="center", va="center",
    arrowprops=dict(arrowstyle="-", color=C_T10, lw=0.7,
                    connectionstyle="arc3,rad=0.18"),
    zorder=5,
)

leg = ax.legend(
    loc="upper center", bbox_to_anchor=(0.46, 0.97),
    frameon=True, fancybox=False, framealpha=0.92,
    edgecolor="#CFCFCF", borderpad=0.45,
    handlelength=1.8, handletextpad=0.5, labelspacing=0.32,
    fontsize=7.8,
)
leg.get_frame().set_linewidth(0.5)

fig.tight_layout()

for fmt in ("pdf", "png"):
    out = OUT_STEM.with_suffix("." + fmt)
    fig.savefig(out, bbox_inches="tight", dpi=300)
    print(f"saved {out}")

# also overwrite the path that the paper currently includes
paper_target = OUT_STEM.parent / "attention.png"
fig.savefig(paper_target, bbox_inches="tight", dpi=300)
print(f"saved {paper_target}")
paper_target_pdf = OUT_STEM.parent / "attention.pdf"
fig.savefig(paper_target_pdf, bbox_inches="tight", dpi=300)
print(f"saved {paper_target_pdf}")

plt.close()
