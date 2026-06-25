import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

# ══════════════════════════════════════════════════════════════════════
# Data from experiments
# ══════════════════════════════════════════════════════════════════════
benchmarks = ['VSP', 'VisPuzzle', 'ChartQA', 'V*', 'BLINK-J', 'MMVP', 'SAT', 'BLINK', 'CV-Bench']
short_names = ['VSP', 'VisPuzzle', 'ChartQA', 'V*', 'BLINK-J', 'MMVP', 'SAT', 'BLINK', 'CV-Bench']

# Accuracy
acc_thinkmorph = [75.83, 77.50, 78.00, 67.01, 66.00, 78.33, 52.67, 59.49, 80.86]
acc_sft        = [49.17, 63.50, 81.66, 56.02, 68.67, 76.33, 46.63, 54.39, 77.37]
acc_vpd_noise  = [50.83, 64.50, 73.77, 61.70, 68.66, 75.33, 48.00, 55.49, 79.09]
acc_vpd        = [85.83, 86.00, 78.79, 64.92, 77.33, 77.33, 54.00, 61.44, 80.64]

# Latency (seconds per sample)
lat_thinkmorph = [142.50, 121.13, 141.45, 153.41, 203.77, 70.02, 130.61, 191.74, 130.45]
lat_sft        = [28.50, 48.01, 28.55, 25.97, 20.05, 17.08, 28.38, 37.07, 23.03]
lat_vpd_noise  = [15.00, 24.69, 11.36, 10.18, 20.17, 9.95, 16.03, 14.13, 12.67]
lat_vpd        = [10.00, 12.01, 7.06, 7.22, 12.27, 7.61, 9.26, 9.71, 15.00]

delta_vpd = [v - b for v, b in zip(acc_vpd, acc_thinkmorph)]

N = len(benchmarks)

# ══════════════════════════════════════════════════════════════════════
# Style
# ══════════════════════════════════════════════════════════════════════
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# NeurIPS-friendly color palette
C_TM   = '#7B68EE'  # VT teacher - medium slate blue
C_SFT  = '#A0A0A0'  # Text-only SFT - neutral gray
C_NOISE = '#F4A460'  # Visual-OPSD-Noise - sandy brown
C_VPD  = '#2EAD6B'  # Visual-OPSD - emerald green (hero color)
C_POS  = '#2E7D32'  # positive delta
C_NEG  = '#C62828'  # negative delta

# ══════════════════════════════════════════════════════════════════════
# Figure 1 — Page-1 hero figure (full-width, ~\textwidth)
# Three panels + unified bottom legend, with pastel-tinted title boxes
# and matching panel-card backgrounds.
# ══════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13.4, 5.5))
# Equal width_ratios + symmetric figure margins → three evenly-distributed panels
gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1],
                       wspace=0.42, left=0.055, right=0.945,
                       top=0.85, bottom=0.26)

# Pastel title-box backgrounds (one tint per panel)
title_box_a = dict(boxstyle='round,pad=0.55', facecolor='#E3F0FB',
                   edgecolor='#9EC5E8', linewidth=1.0)
title_box_b = dict(boxstyle='round,pad=0.55', facecolor='#E5F4E7',
                   edgecolor='#A6D3AE', linewidth=1.0)
title_box_c = dict(boxstyle='round,pad=0.55', facecolor='#EFE3F4',
                   edgecolor='#C9A9D9', linewidth=1.0)

# Light "card" tint for each panel border, matching the title color
panel_face_a = '#F3F8FC'
panel_face_b = '#F4F9F5'
panel_face_c = '#F8F3FA'
panel_edge   = '#c8d0d6'

# Method styling table — single source of truth for legend / panels
METHOD_STYLE = {
    'VT teacher':    dict(color=C_TM,    marker='o', open=False, ls='-',  lw=2.0),
    'Visual-OPSD-Noise':     dict(color=C_NOISE, marker='D', open=False, ls='--', lw=1.6),
    'Text-only SFT': dict(color=C_SFT,   marker='s', open=True,  ls='--', lw=1.4),
    'Visual-OPSD (Ours)':    dict(color=C_VPD,   marker='s', open=False, ls='-',  lw=2.2),
}

# ─────────────────────────────────────────────────────────────────────
# (a) Radar chart: Multi-method benchmark profile
# ─────────────────────────────────────────────────────────────────────
ax_radar = fig.add_subplot(gs[0], polar=True)
ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)

angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
angles_c = angles + [angles[0]]

methods_radar = [
    (acc_thinkmorph, 'VT teacher'),
    (acc_sft,        'Text-only SFT'),
    (acc_vpd_noise,  'Visual-OPSD-Noise'),
    (acc_vpd,        'Visual-OPSD (Ours)'),
]

for vals, label in methods_radar:
    style = METHOD_STYLE[label]
    col = style['color']
    closed = vals + [vals[0]]
    alpha_fill = 0.14 if label == 'Visual-OPSD (Ours)' else 0.05
    ax_radar.fill(angles_c, closed, alpha=alpha_fill, color=col)
    ms = 7.0 if label == 'Visual-OPSD (Ours)' else 5.5
    mfc = 'white' if style['open'] else col
    mec = col if style['open'] else 'white'
    mew = 1.3 if style['open'] else 0.8
    ax_radar.plot(angles_c, closed, color=col, marker=style['marker'],
                  linestyle=style['ls'], linewidth=style['lw'], markersize=ms,
                  markerfacecolor=mfc, markeredgecolor=mec,
                  markeredgewidth=mew, zorder=4)

ax_radar.set_xticks(angles)
ax_radar.set_xticklabels(short_names, fontsize=10.5, fontweight='medium', color='#333')
# Bring angular labels (VSP at top, BLINK on left, etc.) closer to the circle
# so the top label does not collide with the (a) title pill.
ax_radar.tick_params(axis='x', pad=2)
ax_radar.set_ylim(0, 100)
ax_radar.set_yticks([20, 40, 60, 80, 100])
ax_radar.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8.5, color='#999')
ax_radar.spines['polar'].set_color('#e0e0e0')
ax_radar.grid(color='#e0e0e0', linewidth=0.5)

ax_radar.set_title('')  # title placed via fig.text below for alignment

# ─────────────────────────────────────────────────────────────────────
# (b) Delta bar chart: VPD − ThinkMorph improvement
# ─────────────────────────────────────────────────────────────────────
ax_delta = fig.add_subplot(gs[1])

x_pos = np.arange(N)
colors_delta = [C_POS if d > 0 else C_NEG for d in delta_vpd]
alpha_delta = [0.88 if abs(d) > 2 else 0.55 for d in delta_vpd]

bars = ax_delta.bar(x_pos, delta_vpd, width=0.62,
                    color=colors_delta, edgecolor='white', linewidth=0.5,
                    zorder=3)

for i, (bar, d) in enumerate(zip(bars, delta_vpd)):
    bar.set_alpha(alpha_delta[i])
    y_offset = 0.4 if d >= 0 else -0.4
    va = 'bottom' if d >= 0 else 'top'
    ax_delta.text(bar.get_x() + bar.get_width() / 2, d + y_offset,
                  f'{d:+.1f}', ha='center', va=va,
                  fontsize=10, fontweight='bold',
                  color=colors_delta[i])

ax_delta.axhline(y=0, color='#666', linewidth=0.8, zorder=2)
ax_delta.set_xticks(x_pos)
ax_delta.set_xticklabels(short_names, fontsize=10.5, rotation=35, ha='right',
                         fontweight='medium')
ax_delta.set_ylabel(r'$\Delta$ Accuracy (%)', fontsize=13, labelpad=6)
ax_delta.set_ylim(-5, 15)
ax_delta.set_yticks([-4, -2, 0, 2, 4, 6, 8, 10, 12, 14])

ax_delta.spines['top'].set_visible(False)
ax_delta.spines['right'].set_visible(False)
ax_delta.spines['left'].set_color('#ccc')
ax_delta.spines['bottom'].set_color('#ccc')
ax_delta.tick_params(colors='#555')
ax_delta.yaxis.grid(True, color='#f0f0f0', linewidth=0.6, zorder=0)

vt_helpful = [0, 1, 4, 6, 7]  # VSP, VisPuzzle, BLINK-J, SAT, BLINK
for i in vt_helpful:
    ax_delta.get_xticklabels()[i].set_color(C_POS)
    ax_delta.get_xticklabels()[i].set_fontweight('bold')

ax_delta.annotate(r'$\Delta$ vs. Text-only SFT (in bold)',
                  xy=(0.02, 0.96), xycoords='axes fraction',
                  fontsize=9, fontstyle='italic', color='#888')

ax_delta.set_title('')  # title placed via fig.text below for alignment

# ─────────────────────────────────────────────────────────────────────
# (c) Accuracy vs Latency scatter (mean across benchmarks)
# ─────────────────────────────────────────────────────────────────────
ax_scatter = fig.add_subplot(gs[2])

method_data = [
    ('VT teacher',     acc_thinkmorph, lat_thinkmorph, 10),
    ('Text-only SFT',  acc_sft,        lat_sft,        9),
    ('Visual-OPSD-Noise',      acc_vpd_noise,  lat_vpd_noise,  9),
    ('Visual-OPSD (Ours)',     acc_vpd,        lat_vpd,        11),
]

for name, accs, lats, ms in method_data:
    style = METHOD_STYLE[name]
    col = style['color']
    mean_acc = np.mean(accs)
    mean_lat = np.mean(lats)
    speedup = np.mean(lat_thinkmorph) / mean_lat

    fc = 'white' if style['open'] else col
    ec = col if style['open'] else 'white'
    ew = 1.3 if style['open'] else 0.9
    ax_scatter.scatter(mean_lat, mean_acc, s=ms ** 2,
                       facecolor=fc, edgecolor=ec, linewidths=ew,
                       marker=style['marker'], zorder=5, label=name)

    if name == 'VT teacher':
        dx, dy, ha, va = -5, 5, 'right', 'bottom'
    elif name == 'Text-only SFT':
        # place below marker to avoid overlap with VPD-Noise label
        dx, dy, ha, va = 0, -8, 'center', 'top'
    else:
        dx, dy, ha, va = 5, 5, 'left', 'bottom'
    txt = (f'{mean_acc:.1f}%\n({speedup:.0f}\u00d7)' if name != 'VT teacher'
           else f'{mean_acc:.1f}%\n(1\u00d7)')
    ax_scatter.annotate(txt, (mean_lat, mean_acc),
                        xytext=(dx, dy), textcoords='offset points',
                        fontsize=10, fontweight='bold', color=col,
                        ha=ha, va=va)

ax_scatter.annotate('', xy=(20, 74.4), xytext=(125, 71.5),
                    arrowprops=dict(arrowstyle='->', color=C_VPD,
                                    lw=2.0, connectionstyle='arc3,rad=-0.18',
                                    alpha=0.65))
ax_scatter.text(70, 69.5, 'Better accuracy\n& lower latency',
                fontsize=10.5, fontstyle='italic', color=C_VPD,
                ha='center', fontweight='bold', alpha=0.9)

ax_scatter.fill_between([0, 20], [72, 72], [78, 78], alpha=0.07,
                        color=C_VPD, zorder=0)

ax_scatter.set_xlabel('Mean Latency (s / sample)', fontsize=13, labelpad=6)
ax_scatter.set_ylabel('Mean Accuracy (%)', fontsize=13, labelpad=6)
ax_scatter.set_xlim(-5, 160)
ax_scatter.set_ylim(61, 78)

ax_scatter.spines['top'].set_visible(False)
ax_scatter.spines['right'].set_visible(False)
ax_scatter.spines['left'].set_color('#ccc')
ax_scatter.spines['bottom'].set_color('#ccc')
ax_scatter.tick_params(colors='#555')
ax_scatter.xaxis.grid(True, color='#f0f0f0', linewidth=0.6, zorder=0)
ax_scatter.yaxis.grid(True, color='#f0f0f0', linewidth=0.6, zorder=0)

ax_scatter.legend(loc='lower right', frameon=True, fancybox=True,
                  edgecolor='#ddd', framealpha=0.95,
                  fontsize=9.5, handlelength=1.4, labelspacing=0.35,
                  prop={'style': 'italic', 'size': 9.5})

ax_scatter.set_title('')  # title placed via fig.text below for alignment

# ─────────────────────────────────────────────────────────────────────
# Horizontally-aligned panel titles + rounded "card" panels (figure-level)
# ─────────────────────────────────────────────────────────────────────
panel_a_pos = ax_radar.get_position()
panel_b_pos = ax_delta.get_position()
panel_c_pos = ax_scatter.get_position()

# Same y (figure coords) for all three titles → guaranteed horizontal alignment
TITLE_Y = 0.925
PANEL_TOP = 0.975
PANEL_BOTTOM = 0.13  # high enough to leave clear space above the bottom legend

def _panel_cx(pos):
    return pos.x0 + pos.width / 2.0

# Uniform expansion for all panels → cards have identical width and the
# A↔B gap equals the B↔C gap. The single value must (1) cover BLINK/ChartQA
# overhang on panel A, and (2) cover the y-axis label on panels B and C.
EXPAND = 0.040
expand_a = dict(l=EXPAND, r=EXPAND)
expand_b = dict(l=EXPAND+0.005, r=EXPAND)
expand_c = dict(l=EXPAND+0.005, r=EXPAND)

def _add_panel_card(pos, exp, face, edge=panel_edge, lw=1.0):
    rect = mpatches.FancyBboxPatch(
        (pos.x0 - exp['l'], PANEL_BOTTOM),
        pos.width + exp['l'] + exp['r'],
        PANEL_TOP - PANEL_BOTTOM,
        boxstyle='round,pad=0.0,rounding_size=0.014',
        transform=fig.transFigure,
        facecolor=face, edgecolor=edge, linewidth=lw,
        clip_on=False, zorder=-2,
    )
    fig.add_artist(rect)

# Cards (filled tinted background) — drawn first, behind axes content
_add_panel_card(panel_a_pos, expand_a, panel_face_a)
_add_panel_card(panel_b_pos, expand_b, panel_face_b)
_add_panel_card(panel_c_pos, expand_c, panel_face_c)

# Titles — drawn on top (zorder=10) so the colored title pill sits over the card edge
fig.text(_panel_cx(panel_a_pos), TITLE_Y, '(a) Benchmark Profile',
         ha='center', va='center', fontsize=14, fontweight='bold',
         color='#222', bbox=title_box_a, zorder=10)
fig.text(_panel_cx(panel_b_pos), TITLE_Y, r'(b) Visual-OPSD $-$ VT teacher',
         ha='center', va='center', fontsize=14, fontweight='bold',
         color='#222', bbox=title_box_b, zorder=10)
fig.text(_panel_cx(panel_c_pos), TITLE_Y, '(c) Accuracy vs. Latency',
         ha='center', va='center', fontsize=14, fontweight='bold',
         color='#222', bbox=title_box_c, zorder=10)

# ─────────────────────────────────────────────────────────────────────
# Unified bottom legend (figure-level), italic method names
# ─────────────────────────────────────────────────────────────────────
def _legend_handle(label):
    s = METHOD_STYLE[label]
    return Line2D([0], [0], color=s['color'], linestyle=s['ls'], linewidth=s['lw'],
                  marker=s['marker'], markersize=8,
                  markerfacecolor=('white' if s['open'] else s['color']),
                  markeredgecolor=(s['color'] if s['open'] else 'white'),
                  markeredgewidth=(1.3 if s['open'] else 0.8),
                  label=label)

legend_order = ['VT teacher', 'Visual-OPSD-Noise', 'Text-only SFT', 'Visual-OPSD (Ours)']
fig.legend([_legend_handle(n) for n in legend_order], legend_order,
           loc='lower center', bbox_to_anchor=(0.5, 0.025),
           ncol=4, frameon=True, fancybox=True, edgecolor='#ddd',
           framealpha=0.95, columnspacing=3.2, handlelength=2.8,
           handletextpad=0.7,
           prop={'style': 'italic', 'size': 11.5, 'weight': 'medium'})

# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════
out_base = '/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/figures/vpd_main_results'
for fmt in ('pdf', 'png'):
    fig.savefig(f'{out_base}.{fmt}', bbox_inches='tight', pad_inches=0.12, dpi=300)
print(f'Saved {out_base}.pdf and .png')
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Figure 2: Grouped bar chart (detailed per-benchmark comparison)
# ══════════════════════════════════════════════════════════════════════
fig2, ax2 = plt.subplots(figsize=(10, 4.2))

methods_bar = [
    ('VT teacher', acc_thinkmorph, C_TM),
    ('Text-only SFT', acc_sft, C_SFT),
    ('Visual-OPSD-Noise', acc_vpd_noise, C_NOISE),
    ('Visual-OPSD (Ours)', acc_vpd, C_VPD),
]

n_methods = len(methods_bar)
bar_w = 0.19
x2 = np.arange(N)

for m_idx, (name, vals, col) in enumerate(methods_bar):
    offset = x2 + (m_idx - n_methods / 2 + 0.5) * bar_w
    alpha = 1.0 if name == 'Visual-OPSD (Ours)' else 0.75
    edge = '#333' if name == 'Visual-OPSD (Ours)' else 'white'
    edge_w = 1.0 if name == 'Visual-OPSD (Ours)' else 0.5
    bars2 = ax2.bar(offset, vals, bar_w * 0.90, color=col, alpha=alpha,
                    edgecolor=edge, linewidth=edge_w,
                    label=name, zorder=3)

    if name == 'Visual-OPSD (Ours)':
        for j, (o, v) in enumerate(zip(offset, vals)):
            best_val = max(m[1][j] for m in methods_bar)
            if abs(v - best_val) < 0.01:
                ax2.text(o, v + 0.5, f'{v:.1f}',
                         ha='center', va='bottom',
                         fontsize=6.5, fontweight='bold', color=C_VPD)

ax2.set_xticks(x2)
ax2.set_xticklabels(benchmarks, fontsize=9, fontweight='medium',
                    rotation=20, ha='right')
ax2.set_ylabel('Accuracy (%)', fontsize=11, labelpad=6)
ax2.set_ylim(38, 95)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_color('#ccc')
ax2.spines['bottom'].set_color('#ccc')
ax2.tick_params(colors='#555')
ax2.yaxis.grid(True, color='#f0f0f0', linewidth=0.6, zorder=0)

ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
           frameon=True, fancybox=True, edgecolor='#ddd',
           framealpha=0.95, ncol=4, columnspacing=1.2,
           handlelength=1.5, fontsize=9)

bg_spans = [(0, 4, '#E8F5E9', 'VT-helpful'), (5, 8, '#FFF8E1', 'VT-neutral')]
for start, end, bg_col, region_label in bg_spans:
    ax2.axvspan(start - 0.45, end + 0.45, alpha=0.25, color=bg_col, zorder=0)
    mid = (start + end) / 2
    ax2.text(mid, 93, region_label, ha='center', fontsize=7.5,
             fontstyle='italic', color='#666', fontweight='bold')

out_base2 = '/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/figures/vpd_grouped_bar'
for fmt in ('pdf', 'png'):
    fig2.savefig(f'{out_base2}.{fmt}', bbox_inches='tight', pad_inches=0.12, dpi=300)
print(f'Saved {out_base2}.pdf and .png')
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Figure 3: Latency speedup comparison (horizontal bar)
# ══════════════════════════════════════════════════════════════════════
fig3, ax3 = plt.subplots(figsize=(5.5, 3.2))

method_names_lat = ['VT teacher\n(VT Generation)', 'Text-only SFT', 'Visual-OPSD-Noise', 'Visual-OPSD (Ours)']
mean_lats = [np.mean(lat_thinkmorph), np.mean(lat_sft),
             np.mean(lat_vpd_noise), np.mean(lat_vpd)]
colors_lat = [C_TM, C_SFT, C_NOISE, C_VPD]

y_pos = np.arange(len(method_names_lat))[::-1]

for i, (name, lat, col) in enumerate(zip(method_names_lat, mean_lats, colors_lat)):
    alpha = 1.0 if 'Visual-OPSD (Ours)' in name else 0.7
    ax3.barh(y_pos[i], lat, height=0.55, color=col, alpha=alpha,
             edgecolor='white', linewidth=0.5, zorder=3)
    speedup = mean_lats[0] / lat
    label = f'{lat:.1f}s' if i == 0 else f'{lat:.1f}s ({speedup:.1f}x faster)'
    ax3.text(lat + 2, y_pos[i], label,
             ha='left', va='center', fontsize=8, fontweight='bold',
             color=col)

ax3.set_yticks(y_pos)
ax3.set_yticklabels(method_names_lat, fontsize=9, fontweight='medium')
ax3.set_xlabel('Mean Latency (s / sample)', fontsize=10, labelpad=6)
ax3.set_xlim(0, 175)

ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_color('#ccc')
ax3.spines['bottom'].set_color('#ccc')
ax3.tick_params(colors='#555')
ax3.xaxis.grid(True, color='#f0f0f0', linewidth=0.6, zorder=0)

out_base3 = '/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/figures/vpd_latency_comparison'
for fmt in ('pdf', 'png'):
    fig3.savefig(f'{out_base3}.{fmt}', bbox_inches='tight', pad_inches=0.12, dpi=300)
print(f'Saved {out_base3}.pdf and .png')
plt.close()

# ══════════════════════════════════════════════════════════════════════
# Figure 4: Δ(Visual-OPSD − Text-only SFT) bar chart
# ══════════════════════════════════════════════════════════════════════
delta_sft = [v - s for v, s in zip(acc_vpd, acc_sft)]
mean_delta_sft = np.mean(delta_sft)

sorted_idx = np.argsort(delta_sft)[::-1]
sorted_delta = [delta_sft[i] for i in sorted_idx]
sorted_names = [short_names[i] for i in sorted_idx]

fig4, ax4 = plt.subplots(figsize=(7.0, 4.0))

x4 = np.arange(N)
colors_d = [C_POS if d > 0 else C_NEG for d in sorted_delta]
alphas_d = [0.92 if abs(d) > 5 else (0.72 if abs(d) > 1 else 0.55) for d in sorted_delta]

bars4 = ax4.bar(x4, sorted_delta, width=0.58, color=colors_d,
                edgecolor='white', linewidth=0.6, zorder=3)
for i, (bar, d) in enumerate(zip(bars4, sorted_delta)):
    bar.set_alpha(alphas_d[i])
    y_off = 0.55 if d >= 0 else -0.55
    va = 'bottom' if d >= 0 else 'top'
    ax4.text(bar.get_x() + bar.get_width() / 2, d + y_off,
             f'{d:+.2f}', ha='center', va=va,
             fontsize=8.5, fontweight='bold', color=colors_d[i])

ax4.axhline(y=0, color='#666', linewidth=0.8, zorder=2)

ax4.axhline(y=mean_delta_sft, color=C_VPD, linewidth=1.6,
            linestyle='--', alpha=0.7, zorder=4)
ax4.text(N - 0.3, mean_delta_sft + 0.6, f'Mean Δ = +{mean_delta_sft:.2f}',
         ha='right', va='bottom', fontsize=8.5, fontweight='bold',
         color=C_VPD, fontstyle='italic')

ax4.set_xticks(x4)
ax4.set_xticklabels(sorted_names, fontsize=9, fontweight='medium',
                    rotation=25, ha='right')
ax4.tick_params(axis='y', pad=6)

y_lo = float(min(sorted_delta))
y_hi = float(max(sorted_delta))
margin_b, margin_t = 5.0, 5.0  # headroom for bar labels (+36.66) and mean-line annotation
ylim_lo = y_lo - margin_b
ylim_hi = y_hi + margin_t
ax4.set_ylim(ylim_lo, ylim_hi)

tick_step = 4
tick_lo = tick_step * np.floor(ylim_lo / tick_step)
tick_hi = tick_step * np.ceil(ylim_hi / tick_step)
ax4.set_yticks(np.arange(tick_lo, tick_hi + tick_step, tick_step))

ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)
ax4.spines['left'].set_color('#ccc')
ax4.spines['bottom'].set_color('#ccc')
ax4.tick_params(colors='#555')
ax4.yaxis.grid(True, color='#f0f0f0', linewidth=0.6, zorder=0)

for i, idx in enumerate(sorted_idx):
    if idx in vt_helpful:
        ax4.get_xticklabels()[i].set_color(C_POS)
        ax4.get_xticklabels()[i].set_fontweight('bold')

ax4.annotate('VT-helpful benchmarks in bold',
             xy=(0.02, 0.94), xycoords='axes fraction',
             fontsize=7, fontstyle='italic', color='#888',
             va='top')

pos_count = sum(1 for d in delta_sft if d > 0)
ax4.annotate(f'↑ Improved on {pos_count}/{N} benchmarks',
             xy=(0.98, 0.94), xycoords='axes fraction',
             fontsize=7.5, fontweight='bold', color=C_POS,
             ha='right', va='top')

fig4.tight_layout(pad=2.2)
fig4.subplots_adjust(left=0.11, right=0.97, top=0.92, bottom=0.22)
fig4.text(
    0.038, 0.5,
    r'$\Delta$ Accuracy (\%)' '\n' r'Visual-OPSD $-$ Text-only SFT',
    va='center', ha='center', rotation='vertical', fontsize=9.5,
)

out_base4 = '/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/figures/vpd_delta_vs_sft'
for fmt in ('pdf', 'png'):
    fig4.savefig(f'{out_base4}.{fmt}', bbox_inches='tight', pad_inches=0.28, dpi=300)
print(f'Saved {out_base4}.pdf and .png')
plt.close()

print('\nAll figures generated successfully.')
