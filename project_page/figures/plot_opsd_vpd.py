import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
benchmarks = ['VisPuzzle', 'ChartQA', 'VStar', 'BLINK-J', 'MMVP', 'SAT', 'BLINK', 'CV-Bench']

baseline = [77.50, 78.00, 67.01, 66.00, 78.33, 52.67, 59.49, 80.86]
s500     = [80.50, 78.21, 64.40, 75.33, 80.00, 52.00, 60.65, 83.15]
s1000    = [86.00, 78.79, 64.92, 77.33, 77.33, 54.00, 61.44, 80.64]
s1500    = [85.75, 77.24, 60.73, 73.33, 78.67, 48.67, 58.55, 82.50]
s2000    = [86.75, 77.26, 61.25, 72.00, 78.33, 44.00, 57.02, 79.41]

steps = [0, 500, 1000, 1500, 2000]
all_data = [baseline, s500, s1000, s1500, s2000]
N = len(benchmarks)

# ── Style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10.5,
    'ytick.labelsize': 10.5,
    'legend.fontsize': 10,
    'figure.dpi': 300,
})

COL_BASE = '#6C63FF'
COL_BEST = '#00B4A0'

# ── Figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 7.0))
gs = gridspec.GridSpec(1, 2, width_ratios=[1.05, 1.35], wspace=0.25,
                       left=0.06, right=0.87, top=0.84, bottom=0.12)

fig.suptitle('OPSD-VPD Training Dynamics on Official Benchmarks',
             fontsize=16, fontweight='bold', y=0.96, color='#111')

# ═══════════════════════════════════════════════════════════════════════
# (a) Radar chart – Baseline vs. OPSD-VPD-1000step
# ═══════════════════════════════════════════════════════════════════════
ax_r = fig.add_subplot(gs[0], polar=True)
ax_r.set_theta_offset(np.pi / 2)
ax_r.set_theta_direction(-1)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles_closed = angles + [angles[0]]

base_closed = baseline + [baseline[0]]
best_closed = s1000 + [s1000[0]]

ax_r.fill(angles_closed, base_closed, alpha=0.10, color=COL_BASE)
ax_r.plot(angles_closed, base_closed, 'o-', color=COL_BASE,
          linewidth=2.2, markersize=7, label='ThinkMorph (Baseline)',
          markeredgecolor='white', markeredgewidth=1.0, zorder=4)

ax_r.fill(angles_closed, best_closed, alpha=0.10, color=COL_BEST)
ax_r.plot(angles_closed, best_closed, 's-', color=COL_BEST,
          linewidth=2.2, markersize=7, label='OPSD-VPD-1000step',
          markeredgecolor='white', markeredgewidth=1.0, zorder=4)

ax_r.set_xticks(angles)
ax_r.set_xticklabels(benchmarks, fontsize=10.5, fontweight='medium', color='#333')

ax_r.set_ylim(35, 100)
ax_r.set_yticks([50, 60, 70, 80, 90])
ax_r.set_yticklabels(['50', '60', '70', '80', '90'], fontsize=8.5, color='#888')
ax_r.spines['polar'].set_color('#ddd')
ax_r.grid(color='#ddd', linewidth=0.6)

for i in range(N):
    diff = s1000[i] - baseline[i]
    if abs(diff) >= 1.5:
        r_pos = max(base_closed[i], best_closed[i]) + 5.0
        fc = '#E8F5E9' if diff > 0 else '#FFEBEE'
        ec = '#43A047' if diff > 0 else '#E53935'
        tc = '#2E7D32' if diff > 0 else '#C62828'
        ax_r.annotate(f'{diff:+.1f}', xy=(angles[i], r_pos),
                      fontsize=8.5, fontweight='bold', color=tc,
                      ha='center', va='center',
                      bbox=dict(boxstyle='round,pad=0.22', fc=fc, ec=ec,
                                alpha=0.9, linewidth=0.7))

ax_r.legend(loc='upper center', bbox_to_anchor=(0.5, -0.10),
            frameon=True, fancybox=True, edgecolor='#ccc',
            framealpha=0.95, ncol=1, columnspacing=1.5)
ax_r.set_title('(a)  Baseline vs. Best Checkpoint', fontsize=13.5,
               fontweight='bold', pad=28, color='#222')

# ═══════════════════════════════════════════════════════════════════════
# (b) Grouped bar chart – Per-benchmark comparison across checkpoints
# ═══════════════════════════════════════════════════════════════════════
ax_l = fig.add_subplot(gs[1])

method_labels = ['Baseline', '500', '1000', '1500', '2000']
bar_colors = ['#6C63FF', '#E07A5F', '#00B4A0', '#E9C46A', '#264653']
hatches    = ['',  '',  '',  '',  '']

n_methods = len(method_labels)
bar_w = 0.15
x = np.arange(N)

for m_idx in range(n_methods):
    offsets = x + (m_idx - n_methods / 2 + 0.5) * bar_w
    vals = all_data[m_idx]
    bars = ax_l.bar(offsets, vals, bar_w * 0.92, color=bar_colors[m_idx],
                    edgecolor='white', linewidth=0.6,
                    label=method_labels[m_idx], zorder=3,
                    hatch=hatches[m_idx])

    for j, (o, v) in enumerate(zip(offsets, vals)):
        if m_idx == 0:
            continue
        diff = v - baseline[j]
        best_at_bench = max(all_data[k][j] for k in range(n_methods))
        if abs(v - best_at_bench) < 0.01 and diff > 0:
            ax_l.annotate(f'{diff:+.1f}', xy=(o, v + 0.6),
                          fontsize=6.5, fontweight='bold',
                          color='#2E7D32', ha='center', va='bottom')

ax_l.set_ylabel('Accuracy (%)', fontsize=13, fontweight='medium', labelpad=6)
ax_l.set_xticks(x)
ax_l.set_xticklabels(benchmarks, fontsize=10, fontweight='medium', rotation=25, ha='right')
ax_l.set_ylim(38, 96)

ax_l.spines['top'].set_visible(False)
ax_l.spines['right'].set_visible(False)
ax_l.spines['left'].set_color('#bbb')
ax_l.spines['bottom'].set_color('#bbb')
ax_l.tick_params(colors='#555')
ax_l.yaxis.grid(True, color='#eee', linewidth=0.7, zorder=0)

ax_l.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
            frameon=True, fancybox=True, edgecolor='#ccc',
            framealpha=0.95, ncol=5, columnspacing=1.0,
            handlelength=1.5, fontsize=9.5,
            title='Training Steps', title_fontsize=10)

ax_l.set_title('(b)  Per-Benchmark Comparison across Checkpoints', fontsize=13.5,
               fontweight='bold', pad=14, color='#222')

# ── Save ──────────────────────────────────────────────────────────────
for fmt in ('pdf', 'png'):
    fig.savefig(f'/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/figures/opsd_vpd_benchmarks.{fmt}',
                bbox_inches='tight', pad_inches=0.18, dpi=300)
print('Saved figures/opsd_vpd_benchmarks.{pdf,png}')
