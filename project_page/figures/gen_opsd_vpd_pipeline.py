"""
Generate OPSD-VPD training pipeline figure (Figure 2 in VPD paper).

Clean, minimal-arrow version:
  - Cross-region arrows removed; color coding indicates data sources
  - Only short, local arrows within each section
  - Clear left-to-right flow: Data → Sample → Dual Forward → Loss/EMA
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'mathtext.fontset': 'stix',
    'font.size': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

C = dict(
    teacher_bg='#E8EAF6',   teacher_ec='#7986CB',  teacher_dk='#283593',
    student_bg='#E8F5E9',   student_ec='#66BB6A',  student_dk='#1B5E20',
    priv='#FFF3E0',         priv_ec='#FFB74D',     priv_dk='#E65100',
    comp='#A5D6A7',         comp_ec='#43A047',     comp_dk='#1B5E20',
    common='#BBDEFB',       common_ec='#64B5F6',   common_dk='#1565C0',
    loss_bg='#FFEBEE',      loss_ec='#EF5350',     loss_dk='#B71C1C',
    ema_bg='#EDE7F6',       ema_ec='#7E57C2',      ema_dk='#4527A0',
    sample_bg='#E0F2F1',    sample_ec='#26A69A',   sample_dk='#00695C',
    data_bg='#ECEFF1',      data_ec='#90A4AE',
    arrow='#455A64',        text='#263238',        section='#37474F',
    vpd='#2EAD6B',          grad='#C62828',
)

fig, ax = plt.subplots(figsize=(16, 7.8))
ax.set_xlim(-0.3, 16.0)
ax.set_ylim(-0.6, 7.8)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('white')


# ═══ Helpers ═══

def rbox(x, y, w, h, fc, ec='#BDBDBD', lw=1.0, zorder=2,
         bs='round,pad=0.1', shadow=False):
    b = FancyBboxPatch(
        (x, y), w, h, boxstyle=bs, facecolor=fc, edgecolor=ec,
        linewidth=lw, zorder=zorder, mutation_scale=0.15)
    if shadow:
        b.set_path_effects([pe.withSimplePatchShadow(
            offset=(1.2, -1.2), shadow_rgbFace='#ddd', alpha=0.25)])
    ax.add_patch(b)

def txt(x, y, s, fs=8, fw='normal', c=C['text'], ha='center', va='center',
        zorder=5, **kw):
    return ax.text(x, y, s, fontsize=fs, fontweight=fw, color=c,
                   ha=ha, va=va, zorder=zorder, **kw)

def arr(x1, y1, x2, y2, c=C['arrow'], lw=1.5, style='->',
        cs=None, zorder=3):
    props = dict(arrowstyle=style, color=c, lw=lw,
                 shrinkA=2, shrinkB=2)
    if cs:
        props['connectionstyle'] = cs
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=props, zorder=zorder)

def badge(x, y, num):
    rbox(x - 0.22, y - 0.22, 0.44, 0.44, C['vpd'], ec=C['vpd'],
         lw=0, zorder=6, bs='round,pad=0.06')
    txt(x, y, str(num), fs=10.5, fw='bold', c='white', zorder=7)


# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — Training Data
# ══════════════════════════════════════════════════════════════════════

rbox(0.0, 0.3, 2.6, 6.8, C['data_bg'], ec=C['data_ec'], lw=0.8,
     zorder=0, bs='round,pad=0.15')
txt(1.3, 7.35, 'Training Sample', fs=11, fw='bold', c=C['section'])

rbox(0.2, 5.8, 2.2, 1.0, '#E3F2FD', ec='#90CAF9', lw=1.0)
txt(1.3, 6.45, 'Problem Image', fs=8, fw='bold', c=C['common_dk'])
txt(1.3, 6.1, '+  Question  Q', fs=8, fw='bold', c=C['common_dk'])

rbox(0.1, 0.5, 2.4, 5.1, C['priv'], ec=C['priv_ec'], lw=1.3,
     zorder=0, bs='round,pad=0.12')
txt(1.3, 5.35, 'Privileged Reference', fs=8.5, fw='bold',
    c=C['priv_dk'], fontstyle='italic')

for iy, lab in [
    (4.5, 'Reference Intro'),
    (3.8, 'Thought_1  +  Thought_2'),
    (3.1, 'VT Image_1  +  VT Image_2'),
    (2.4, 'Reference Answer'),
    (1.7, 'Transition Prompt'),
]:
    rbox(0.25, iy, 2.1, 0.55, '#FFE0B2', ec='#FFCC80', lw=0.7)
    txt(1.3, iy + 0.275, lab, fs=7, c=C['priv_dk'], fw='medium')

txt(1.3, 1.0, '(teacher-only context)', fs=7, c=C['priv_dk'],
    fontstyle='italic')


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — On-Policy Sampling
# ══════════════════════════════════════════════════════════════════════

rbox(3.0, 4.1, 2.5, 2.55, C['sample_bg'], ec=C['sample_ec'],
     lw=1.5, shadow=True, bs='round,pad=0.12')
badge(3.45, 6.95, 1)
txt(3.75, 6.95, 'On-Policy Sampling', fs=10.5, fw='bold',
    c=C['section'], ha='left')

txt(4.25, 6.15, r'Student Model $\theta$', fs=10, fw='bold', c=C['sample_dk'])
txt(4.25, 5.55, 'generate_text( )', fs=8, c='#37474F',
    fontfamily='monospace')
txt(4.25, 5.0, r'$[\mathrm{Sys,\ Img,\ Q}] \rightarrow$  autoregressive',
    fs=7.5, c='#546E7A')
txt(4.25, 4.45, r'image-skip: $\langle$img_start$\rangle'
    r' \to \langle$im_end$\rangle$',
    fs=6.5, c='#90A4AE', fontstyle='italic')

# Completion output
rbox(3.2, 3.1, 2.1, 0.75, C['comp'], ec=C['comp_ec'], lw=1.8)
txt(4.25, 3.6, 'Completion', fs=8, fw='bold', c=C['comp_dk'])
txt(4.25, 3.3, r'$c_1\; c_2\; \ldots\; c_k$', fs=9, fw='bold', c=C['comp_dk'])

# Only 2 local arrows in this section
arr(2.6, 6.3, 3.0, 5.8, lw=1.8, c=C['common_ec'])   # Data → Sampler
arr(4.25, 4.1, 4.25, 3.85, lw=1.5, c=C['sample_ec'])  # Sampler → Completion


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — Dual Forward Pass
# ══════════════════════════════════════════════════════════════════════

rbox(5.7, 0.0, 7.3, 7.4, '#FAFAFA', ec='#CFD8DC', lw=0.8,
     zorder=-1, bs='round,pad=0.15')
badge(6.1, 7.5, 2)
txt(6.45, 7.5, 'Dual Forward Pass  --  Same Model, Different Context',
    fs=10.5, fw='bold', c=C['section'], ha='left')

# ── TEACHER PATH ──
teacher_y0, teacher_h = 4.3, 2.85
rbox(5.9, teacher_y0, 6.9, teacher_h, C['teacher_bg'], ec=C['teacher_ec'],
     lw=1.3, zorder=1, bs='round,pad=0.12')
txt(6.4, teacher_y0 + teacher_h - 0.35,
    r'TEACHER  Forward  $(\theta_{\mathrm{ema}})$  --  no gradient',
    fs=10, fw='bold', c=C['teacher_dk'], ha='left')

ty = teacher_y0 + 0.55
th = 0.85

rbox(6.05, ty, 1.5, th, C['common'], ec=C['common_ec'], lw=1.0, zorder=3)
txt(6.8, ty + th / 2 + 0.12, 'Sys + Img + Q', fs=7.5, fw='bold',
    c=C['common_dk'])
txt(6.8, ty + th / 2 - 0.15, '(shared input)', fs=5.5, c='#90CAF9',
    fontstyle='italic')

rbox(7.75, ty, 2.9, th, C['priv'], ec=C['priv_ec'], lw=1.2, zorder=3)
txt(9.2, ty + th / 2 + 0.15,
    'Ref Intro + Think_1 + VT_1', fs=7, fw='bold', c=C['priv_dk'])
txt(9.2, ty + th / 2 - 0.15,
    '+ Think_2 + VT_2 + Ans + Trans', fs=7, fw='bold', c=C['priv_dk'])

rbox(10.85, ty, 1.55, th, C['comp'], ec=C['comp_ec'], lw=1.8, zorder=3)
txt(11.625, ty + th / 2 + 0.12, 'Completion', fs=7.5, fw='bold',
    c=C['comp_dk'])
txt(11.625, ty + th / 2 - 0.15, r'$c_1\; c_2\; \ldots\; c_k$',
    fs=8, fw='bold', c=C['comp_dk'])

# Short internal arrows (within teacher)
arr(7.55, ty + th / 2, 7.75, ty + th / 2, lw=1.0, c='#9E9E9E', style='->')
arr(10.65, ty + th / 2, 10.85, ty + th / 2, lw=1.0, c='#9E9E9E', style='->')

# Loss mask labels
txt(6.8, ty - 0.18, 'loss = 0', fs=5.5, c='#BDBDBD', fontstyle='italic')
txt(9.2, ty - 0.18, 'loss = 0  (Privileged)', fs=5.5, c=C['priv_dk'],
    fw='bold', fontstyle='italic')
txt(11.625, ty - 0.18, 'loss = 1', fs=5.5, c=C['comp_dk'], fw='bold')

# Teacher → p_T
arr(12.4, ty + th / 2, 12.8, ty + th / 2, lw=1.8, c=C['teacher_ec'])
rbox(12.8, ty + 0.1, 0.6, 0.65, '#C5CAE9', ec=C['teacher_ec'],
     lw=1.3, zorder=3)
txt(13.1, ty + th / 2, r'$p_T$', fs=11, fw='bold', c=C['teacher_dk'])


# ── STUDENT PATH ──
student_y0, student_h = 0.25, 2.85
rbox(5.9, student_y0, 6.9, student_h, C['student_bg'], ec=C['student_ec'],
     lw=1.3, zorder=1, bs='round,pad=0.12')
txt(6.4, student_y0 + student_h - 0.35,
    r'STUDENT  Forward  $(\theta)$  --  with gradient  $\nabla$',
    fs=10, fw='bold', c=C['student_dk'], ha='left')

sy = student_y0 + 0.55
sh = 0.85

rbox(6.05, sy, 1.5, sh, C['common'], ec=C['common_ec'], lw=1.0, zorder=3)
txt(6.8, sy + sh / 2 + 0.12, 'Sys + Img + Q', fs=7.5, fw='bold',
    c=C['common_dk'])
txt(6.8, sy + sh / 2 - 0.15, '(shared input)', fs=5.5, c='#90CAF9',
    fontstyle='italic')

rbox(7.75, sy + 0.08, 2.9, sh - 0.16, '#FAFAFA', ec='#E0E0E0',
     lw=0, zorder=2, bs='round,pad=0.08')
ax.add_patch(FancyBboxPatch(
    (7.75, sy + 0.08), 2.9, sh - 0.16,
    boxstyle='round,pad=0.08', facecolor='none', edgecolor='#BDBDBD',
    linewidth=1.0, linestyle='--', zorder=2, mutation_scale=0.15))
txt(9.2, sy + sh / 2 + 0.05, 'No Privileged Context',
    fs=8, c='#BDBDBD', fontstyle='italic', fw='bold')
txt(9.2, sy + sh / 2 - 0.2, '(information gap)',
    fs=6, c='#BDBDBD', fontstyle='italic')

rbox(10.85, sy, 1.55, sh, C['comp'], ec=C['comp_ec'], lw=1.8, zorder=3)
txt(11.625, sy + sh / 2 + 0.12, 'Completion', fs=7.5, fw='bold',
    c=C['comp_dk'])
txt(11.625, sy + sh / 2 - 0.15, r'$c_1\; c_2\; \ldots\; c_k$',
    fs=8, fw='bold', c=C['comp_dk'])

# Short internal arrow (within student)
arr(7.55, sy + sh / 2, 10.85, sy + sh / 2, lw=1.0, c='#9E9E9E', style='->')

txt(6.8, sy - 0.18, 'loss = 0', fs=5.5, c='#BDBDBD', fontstyle='italic')
txt(11.625, sy - 0.18, 'loss = 1', fs=5.5, c=C['comp_dk'], fw='bold')

# Student → p_S
arr(12.4, sy + sh / 2, 12.8, sy + sh / 2, lw=1.8, c=C['student_ec'])
rbox(12.8, sy + 0.1, 0.6, 0.65, '#C8E6C9', ec=C['student_ec'],
     lw=1.3, zorder=3)
txt(13.1, sy + sh / 2, r'$p_S$', fs=11, fw='bold', c=C['student_dk'])


# ── Token alignment + Core insight (between paths) ──

for dx in [0.2, 0.5, 0.8, 1.1, 1.35]:
    x = 10.85 + dx
    ax.plot([x, x], [sy + sh + 0.02, ty - 0.02],
            color=C['comp_ec'], linewidth=1.0, linestyle=':',
            alpha=0.5, zorder=2)

mid_y = (ty + sy + sh) / 2

rbox(7.0, mid_y - 0.28, 5.3, 0.56, 'white', ec=C['vpd'],
     lw=1.2, zorder=6, bs='round,pad=0.1')
txt(8.3, mid_y + 0.08,
    r'Same completion $\mathbf{c}$,  different context',
    fs=7, fw='bold', c=C['vpd'], zorder=7)
txt(8.3, mid_y - 0.14,
    r'$\Rightarrow$  generation knowledge in $\Delta$logits'
    r'    (token-level alignment)',
    fs=6.5, fw='bold', c=C['vpd'], zorder=7)


# ══════════════════════════════════════════════════════════════════════
# Single clean arrow: Sampling → Dual Forward section
# (replaces all the messy cross-region arrows)
# ══════════════════════════════════════════════════════════════════════

arr(5.3, 3.5, 5.7, 3.5, lw=2.5, c=C['arrow'])

# Inside Dual Forward: completion distributes to both paths (short arrows)
arr(5.85, 3.5, 10.85, ty + th / 2, lw=1.2, c=C['comp_ec'],
    cs='arc3,rad=-0.05')
arr(5.85, 3.5, 10.85, sy + sh / 2, lw=1.2, c=C['comp_ec'],
    cs='arc3,rad=0.05')

pass


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — Loss Computation
# ══════════════════════════════════════════════════════════════════════

lx, ly, lw_, lh = 13.6, 1.6, 2.1, 2.4
rbox(lx, ly, lw_, lh, C['loss_bg'], ec=C['loss_ec'], lw=1.8,
     shadow=True, bs='round,pad=0.12')
badge(13.95, 4.3, 3)
txt(14.25, 4.3, 'Loss', fs=10.5, fw='bold', c=C['section'], ha='left')

txt(lx + lw_ / 2, ly + lh - 0.3,
    'Loss Computation', fs=8.5, fw='bold', c=C['loss_dk'])
txt(lx + lw_ / 2, ly + lh / 2 + 0.25,
    r'$\mathcal{L}_{\mathrm{JSD}} = \mathrm{JSD}_\beta(p_T \| p_S)$',
    fs=9, c=C['text'])
txt(lx + lw_ / 2, ly + lh / 2 - 0.2,
    r'$\mathcal{L}_{\mathrm{CE}} = \mathrm{CE}(p_S,\, \mathbf{c})$',
    fs=9, c=C['text'])
txt(lx + lw_ / 2, ly + 0.3,
    r'$\mathcal{L} = \alpha\, \mathcal{L}_{\mathrm{JSD}}'
    r' + \gamma\, \mathcal{L}_{\mathrm{CE}}$',
    fs=9.5, fw='bold', c=C['loss_dk'])

# p_T → Loss  &  p_S → Loss  (short, straight)
arr(13.4, ty + th / 2, lx, ly + lh - 0.45,
    lw=1.8, c=C['teacher_ec'])
arr(13.4, sy + sh / 2, lx, ly + 0.45,
    lw=1.8, c=C['student_ec'])


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — EMA Update
# ══════════════════════════════════════════════════════════════════════

ema_x, ema_y = 13.6, 5.25
ema_w, ema_h = 2.1, 1.5
rbox(ema_x, ema_y, ema_w, ema_h, C['ema_bg'], ec=C['ema_ec'],
     lw=1.5, shadow=True, bs='round,pad=0.12')
badge(13.95, 7.05, 4)
txt(14.25, 7.05, 'EMA Update', fs=10.5, fw='bold',
    c=C['section'], ha='left')

txt(ema_x + ema_w / 2, ema_y + ema_h - 0.3,
    'Exponential Moving Avg', fs=7.5, fw='bold', c=C['ema_dk'])
txt(ema_x + ema_w / 2, ema_y + ema_h / 2 - 0.1,
    r'$\theta_{\mathrm{ema}} \leftarrow'
    r' 0.995 \cdot \theta_{\mathrm{ema}}$'
    '\n'
    r'$\quad + 0.005 \cdot \theta$',
    fs=8.5, c=C['ema_dk'])

# Short vertical: Loss → EMA
arr(lx + lw_ / 2, ly + lh + 0.05, lx + lw_ / 2, ema_y,
    lw=1.5, c=C['ema_ec'])

# EMA feeds back to Teacher (short horizontal)
arr(ema_x, ema_y + 0.5, 12.8, teacher_y0 + teacher_h - 0.4,
    lw=1.8, c=C['ema_ec'], cs='arc3,rad=0.08')
txt(12.7, 6.6, r'feeds $\theta_{\mathrm{ema}}$',
    fs=6.5, c=C['ema_ec'], fw='bold', fontstyle='italic')

# Gradient flows back to student (short vertical)
arr(lx + lw_ / 2, ly - 0.05, lx + lw_ / 2, 0.55,
    lw=2.0, c=C['grad'])
txt(lx + lw_ / 2 + 0.3, 0.85,
    r'$\nabla_\theta \mathcal{L}$  update $\theta$',
    fs=7.5, fw='bold', c=C['grad'], ha='left')


# ══════════════════════════════════════════════════════════════════════
# Legend
# ══════════════════════════════════════════════════════════════════════

legend_items = [
    (C['common'],  C['common_ec'],  'Shared Input (Img + Q)'),
    (C['priv'],    C['priv_ec'],    'Privileged Context (VT, Thoughts, Answer)'),
    (C['comp'],    C['comp_ec'],    'Shared Completion (on-policy sampled)'),
]
for i, (fc, ec, lab) in enumerate(legend_items):
    lx_ = 0.1 + i * 5.3
    rbox(lx_, -0.4, 0.35, 0.25, fc, ec=ec, lw=0.8, zorder=5)
    txt(lx_ + 0.55, -0.28, lab, fs=7.5, c=C['text'], ha='left', fw='medium')


# ══════════════════════════════════════════════════════════════════════
# Save
# ══════════════════════════════════════════════════════════════════════

out = '/mlx_devbox/users/lipengyu.seed/playground/ThinkMorph/figures/opsd_vpd_pipeline'
for fmt in ('pdf', 'png'):
    fig.savefig(f'{out}.{fmt}', bbox_inches='tight', pad_inches=0.12, dpi=300)
print(f'Saved to {out}.pdf  and  {out}.png')
plt.close()
