"""Generate win/loss analysis figure: VPD vs ThinkMorph per-sample comparison."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Per-sample win/loss data (% of test samples)
# For each benchmark: [VPD_win%, TM_win%, Agree_correct%, Agree_wrong%]
# VPD_win = VPD correct & TM wrong; TM_win = TM correct & VPD wrong
# ══════════════════════════════════════════════════════════════════════

benchmarks = ['BLINK-J', 'VSP', 'VisPuzzle', 'BLINK', 'SAT', 'ChartQA', 'CV-Bench', 'MMVP', 'V*']
vpd_acc    = [77.33, 85.83, 86.00, 61.44, 54.00, 78.79, 80.64, 77.33, 64.92]
tm_acc     = [66.00, 75.83, 77.50, 59.49, 52.67, 78.00, 80.86, 78.33, 67.01]

agree_correct = [60.7, 72.5, 74.5, 52.5, 44.0, 72.5, 76.5, 72.3, 58.6]
vpd_win       = [16.6, 13.3, 11.5,  8.9, 10.0,  6.3,  4.1,  5.0,  6.3]
tm_win        = [ 5.3,  3.3,  3.0,  7.0,  8.7,  5.5,  4.4,  6.0,  8.4]
agree_wrong   = [17.4, 10.9, 11.0, 31.6, 37.3, 15.7, 15.0, 16.7, 26.7]

delta = [v - t for v, t in zip(vpd_acc, tm_acc)]

# Sort by net gain (descending)
order = np.argsort(delta)[::-1]
benchmarks    = [benchmarks[i] for i in order]
vpd_win       = [vpd_win[i] for i in order]
tm_win        = [tm_win[i] for i in order]
agree_correct = [agree_correct[i] for i in order]
agree_wrong   = [agree_wrong[i] for i in order]
delta         = [delta[i] for i in order]

N = len(benchmarks)

# ══════════════════════════════════════════════════════════════════════
# Style (matching existing figures)
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

C_VPD = '#2E8B57'   # VPD wins - sea green
C_TM  = '#7B68EE'   # ThinkMorph wins - slate blue
C_NET = '#333333'    # net label

fig, ax = plt.subplots(figsize=(5.5, 2.8))

y_pos = np.arange(N)
bar_h = 0.55

# VPD wins: bars to the right (positive)
bars_vpd = ax.barh(y_pos, vpd_win, height=bar_h, color=C_VPD, alpha=0.85,
                   edgecolor='white', linewidth=0.5, label='VPD correct, ThinkMorph wrong')
# TM wins: bars to the left (negative)
bars_tm = ax.barh(y_pos, [-v for v in tm_win], height=bar_h, color=C_TM, alpha=0.85,
                  edgecolor='white', linewidth=0.5, label='ThinkMorph correct, VPD wrong')

# Labels on bars
for i in range(N):
    if vpd_win[i] > 2:
        ax.text(vpd_win[i] - 0.5, y_pos[i], f'{vpd_win[i]:.1f}%',
                ha='right', va='center', fontsize=7, color='white', fontweight='bold')
    if tm_win[i] > 2:
        ax.text(-tm_win[i] + 0.5, y_pos[i], f'{tm_win[i]:.1f}%',
                ha='left', va='center', fontsize=7, color='white', fontweight='bold')

# Net gain annotation on the right
for i in range(N):
    net = delta[i]
    color = C_VPD if net > 0 else C_TM
    sign = '+' if net > 0 else ''
    ax.text(20.5, y_pos[i], f'net {sign}{net:.1f}pp',
            ha='left', va='center', fontsize=7.5, color=color, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(benchmarks)
ax.set_xlabel('Samples with different predictions (%)')
ax.axvline(0, color='#666666', linewidth=0.8, zorder=0)

ax.set_xlim(-12, 30)
ax.invert_yaxis()

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0)

ax.legend(loc='upper center', frameon=True, fancybox=True,
          framealpha=0.9, edgecolor='#cccccc', fontsize=7.5,
          ncol=2, bbox_to_anchor=(0.5, 1.12))

plt.tight_layout()

for fmt in ['pdf', 'png']:
    out = f'figures/vpd_win_loss.{fmt}'
    fig.savefig(out, bbox_inches='tight', dpi=300)
    print(f'Saved {out}')

plt.close()
