"""
Visual-OPSD loss functions.

Ported from OPSD (https://arxiv.org/ to-be-added) trainer — generalized JSD with:
  - beta interpolation between forward/reverse KL
  - top-k restriction over teacher's most-probable tokens
  - per-token JSD clipping (style-token dampening)
  - temperature scaling

Also provides the Thinking-Machines / Tinker reverse-KL policy-gradient loss
(O(1) per token memory, no full-vocab materialization).

All losses expect unscaled logits at the token positions to distill over.
Shapes:
  student_logits / teacher_logits : [N_tokens, V]
  labels (optional)                : [N_tokens]   ; -100 tokens are masked out
Return: scalar loss tensor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
    reduction: str = "mean",
    top_k: int | None = None,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Generalized Jensen-Shannon divergence, as used by OPSD / GKD.

    beta = 0   : KL(student || mixture)         (forward KL in the original paper)
    beta = 1   : KL(teacher || mixture)         (reverse KL)
    beta = 0.5 : symmetric JSD

    Memory optimisation: when ``top_k`` is set the gather is performed in
    the *input* dtype (typically bf16) **before** the float32 upcast, so
    only the top-k slice lives in float32 instead of the full vocabulary.
    For top_k=256 and V=152K this cuts peak logits memory by ~600x.
    """
    assert student_logits.dim() == 2 and teacher_logits.dim() == 2, (
        "logits must be [N_tokens, V]"
    )

    # Apply top_k in the input dtype (bf16) to avoid full-vocab float32.
    if top_k is not None and top_k > 0 and top_k < student_logits.shape[-1]:
        with torch.no_grad():
            _, topk_idx = torch.topk(teacher_logits, k=top_k, dim=-1)
        student_logits = torch.gather(student_logits, dim=-1, index=topk_idx)
        teacher_logits = torch.gather(teacher_logits, dim=-1, index=topk_idx)

    s = (student_logits / temperature).float()
    t = (teacher_logits / temperature).float()
    del student_logits, teacher_logits

    s_logp = F.log_softmax(s, dim=-1)
    t_logp = F.log_softmax(t, dim=-1)
    del s, t

    if beta == 0.0:
        per_tok = F.kl_div(s_logp, t_logp, reduction="none", log_target=True)
    elif beta == 1.0:
        per_tok = F.kl_div(t_logp, s_logp, reduction="none", log_target=True)
    else:
        beta_t = torch.tensor(beta, dtype=s_logp.dtype, device=s_logp.device)
        # log M = log(beta * teacher + (1 - beta) * student)
        mix_logp = torch.logsumexp(
            torch.stack(
                [s_logp + torch.log1p(-beta_t), t_logp + torch.log(beta_t)]
            ),
            dim=0,
        )
        kl_t = F.kl_div(mix_logp, t_logp, reduction="none", log_target=True)
        kl_s = F.kl_div(mix_logp, s_logp, reduction="none", log_target=True)
        per_tok = beta * kl_t + (1.0 - beta) * kl_s  # [N, V_used]

    if token_clip is not None and token_clip > 0:
        per_tok = per_tok.clamp(max=token_clip)

    # reduce over vocab dim first, keep a per-token divergence scalar
    per_tok_sum = per_tok.sum(dim=-1)  # [N]

    if labels is not None:
        mask = labels != -100
        per_tok_sum = per_tok_sum[mask]
        if per_tok_sum.numel() == 0:
            return per_tok_sum.new_zeros(())

    if reduction == "sum":
        loss = per_tok_sum.sum()
    elif reduction == "mean":
        loss = per_tok_sum.mean()
    elif reduction == "none":
        loss = per_tok_sum
    else:
        raise ValueError(f"Unknown reduction: {reduction}")

    # Standard KD temperature-squared scaling (Hinton).
    return loss * (temperature ** 2)


def tinker_reverse_kl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor,
    labels: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Thinking-Machines / Tinker memory-efficient reverse-KL policy-gradient loss.

    L = - E[ stop_grad(log pi_teacher(x) - log pi_student(x)) * log pi_student(x) ]

    Avoids materializing full-vocab probability tensors by gathering only the
    log-prob of the actually-sampled token.

    sampled_token_ids : [N_tokens] int64
    """
    assert student_logits.shape == teacher_logits.shape
    assert student_logits.shape[0] == sampled_token_ids.shape[0]

    s_logp = F.log_softmax(student_logits / temperature, dim=-1)
    with torch.no_grad():
        t_logp = F.log_softmax(teacher_logits / temperature, dim=-1)

    idx = sampled_token_ids.unsqueeze(-1)
    s_lp_sampled = torch.gather(s_logp, -1, idx).squeeze(-1)  # [N]
    t_lp_sampled = torch.gather(t_logp, -1, idx).squeeze(-1)  # [N]

    advantage = (t_lp_sampled - s_lp_sampled).detach()

    if labels is not None:
        mask = labels != -100
        advantage = advantage[mask]
        s_lp_sampled = s_lp_sampled[mask]
        if s_lp_sampled.numel() == 0:
            return student_logits.new_zeros(())

    return -(advantage * s_lp_sampled).mean()


def compute_student_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    sampled_token_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    *,
    loss_kind: str = "jsd",
    beta: float = 0.5,
    temperature: float = 1.0,
    top_k: int | None = None,
    token_clip: float | None = None,
) -> torch.Tensor:
    """Dispatch to the requested KD loss flavour."""
    if loss_kind == "jsd":
        return generalized_jsd_loss(
            student_logits,
            teacher_logits,
            labels=labels,
            beta=beta,
            temperature=temperature,
            top_k=top_k,
            token_clip=token_clip,
        )
    if loss_kind == "tinker":
        if sampled_token_ids is None:
            raise ValueError("tinker loss requires `sampled_token_ids`")
        return tinker_reverse_kl_loss(
            student_logits,
            teacher_logits,
            sampled_token_ids=sampled_token_ids,
            labels=labels,
            temperature=temperature,
        )
    raise ValueError(f"Unknown loss_kind: {loss_kind!r}")
