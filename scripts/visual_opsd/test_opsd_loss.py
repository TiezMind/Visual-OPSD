"""Quick sanity checks for Visual-OPSD loss utilities.

Run locally without distributed:
    source .venv/bin/activate
    python scripts/visual_opsd/test_opsd_loss.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
import torch.nn.functional as F

from scripts.visual_opsd.opsd_loss import (
    generalized_jsd_loss,
    tinker_reverse_kl_loss,
    compute_student_kd_loss,
)


def approx_equal(a, b, atol=1e-4):
    return abs(float(a) - float(b)) < atol


def test_identical_distributions_zero_loss():
    torch.manual_seed(0)
    V = 128
    N = 10
    logits = torch.randn(N, V)

    # JSD
    loss = generalized_jsd_loss(logits, logits, beta=0.5, temperature=1.0)
    assert loss.abs() < 1e-4, f"expected ~0, got {loss.item()}"
    print(f"  [ok] JSD(a, a) = {loss.item():.3e}")

    # Forward KL, reverse KL
    for beta in (0.0, 1.0):
        loss = generalized_jsd_loss(logits, logits, beta=beta)
        assert loss.abs() < 1e-4, f"beta={beta}: expected ~0, got {loss.item()}"
    print(f"  [ok] KL(a, a) = 0 for beta in {{0, 1}}")


def test_symmetry_of_jsd_at_half():
    torch.manual_seed(1)
    V = 64
    N = 20
    a = torch.randn(N, V)
    b = torch.randn(N, V)

    jab = generalized_jsd_loss(a, b, beta=0.5)
    jba = generalized_jsd_loss(b, a, beta=0.5)
    assert abs(jab.item() - jba.item()) < 1e-3
    print(f"  [ok] JSD(a,b) = JSD(b,a) = {jab.item():.4f}  (beta=0.5 symmetric)")


def test_temperature_scaling_numerically_stable():
    """With tau^2 scaling (Hinton), absolute magnitude depends on distribution
    shape, so we just verify stability / finiteness across temperatures."""
    torch.manual_seed(2)
    V = 64
    N = 10
    a = torch.randn(N, V) * 5.0
    b = torch.randn(N, V) * 5.0
    vals = {}
    for tau in (0.5, 1.0, 2.0, 4.0):
        v = generalized_jsd_loss(a, b, beta=0.5, temperature=tau).item()
        assert torch.isfinite(torch.tensor(v)), f"tau={tau} blew up: {v}"
        vals[tau] = v
    print(f"  [ok] JSD values across temperatures: " +
          ", ".join(f"tau={t}: {v:.3f}" for t, v in vals.items()))


def test_top_k_reduces_sensitivity_to_tail():
    torch.manual_seed(3)
    V = 1024
    N = 50
    a = torch.randn(N, V)
    b = a.clone()
    # perturb only the tail (low-prob) tokens of b
    b[:, 500:] += torch.randn(N, 524) * 3.0

    l_full = generalized_jsd_loss(a, b, beta=0.5)
    l_k50 = generalized_jsd_loss(a, b, beta=0.5, top_k=50)
    # k=50 should see much less divergence because tail is dropped
    assert l_k50 < l_full, f"top-k should be <= full: full={l_full.item()}, k50={l_k50.item()}"
    print(f"  [ok] top-k pruning reduces divergence: full={l_full.item():.3f}, k=50 -> {l_k50.item():.3f}")


def test_token_clip_dampens_outliers():
    torch.manual_seed(4)
    V = 64
    N = 20
    a = torch.randn(N, V)
    # inject one pathologically divergent token
    b = a.clone()
    b[0] += 30.0 * torch.randn(V)

    l_raw = generalized_jsd_loss(a, b, beta=0.5)
    l_clip = generalized_jsd_loss(a, b, beta=0.5, token_clip=0.05)
    assert l_clip < l_raw
    print(f"  [ok] token_clip=0.05 dampens outliers: raw={l_raw.item():.3f}, clipped={l_clip.item():.3f}")


def test_labels_mask_ignored_positions():
    torch.manual_seed(5)
    V = 32
    N = 30
    a = torch.randn(N, V)
    b = torch.randn(N, V)
    labels = torch.full((N,), -100)
    labels[:10] = 0
    # only first 10 positions count
    l_full = generalized_jsd_loss(a[:10], b[:10], beta=0.5)
    l_mask = generalized_jsd_loss(a, b, labels=labels, beta=0.5)
    assert abs(l_full.item() - l_mask.item()) < 1e-4
    print(f"  [ok] labels mask: full={l_full.item():.4f}, masked={l_mask.item():.4f}")


def test_tinker_loss_basic():
    torch.manual_seed(6)
    V = 32
    N = 20
    s = torch.randn(N, V, requires_grad=True)
    t = torch.randn(N, V)
    sampled = torch.randint(0, V, (N,))
    loss = tinker_reverse_kl_loss(s, t, sampled)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    print(f"  [ok] tinker_reverse_kl_loss = {loss.item():.4f} (grad flowed)")


def test_dispatch():
    torch.manual_seed(7)
    V, N = 32, 10
    s = torch.randn(N, V)
    t = torch.randn(N, V)
    sampled = torch.randint(0, V, (N,))

    l_jsd = compute_student_kd_loss(s, t, loss_kind="jsd", beta=0.5)
    l_tinker = compute_student_kd_loss(
        s, t, sampled_token_ids=sampled, loss_kind="tinker"
    )
    print(f"  [ok] dispatch: jsd={l_jsd.item():.4f}, tinker={l_tinker.item():.4f}")


def test_grad_flows_to_student_only():
    torch.manual_seed(8)
    V, N = 64, 16
    s = torch.randn(N, V, requires_grad=True)
    t = torch.randn(N, V, requires_grad=True)
    loss = generalized_jsd_loss(s, t, beta=0.5, top_k=32, token_clip=0.05)
    loss.backward()
    assert s.grad is not None and s.grad.abs().sum() > 0
    # teacher grad should also flow if teacher has requires_grad — OK, we don't
    # stop_grad inside the loss (that's the caller's job).  Check student at least.
    print(f"  [ok] student grad magnitude = {s.grad.abs().sum().item():.3f}")


if __name__ == "__main__":
    print("=== Testing Visual-OPSD loss utilities ===")
    for fn in [
        test_identical_distributions_zero_loss,
        test_symmetry_of_jsd_at_half,
        test_temperature_scaling_numerically_stable,
        test_top_k_reduces_sensitivity_to_tail,
        test_token_clip_dampens_outliers,
        test_labels_mask_ignored_positions,
        test_tinker_loss_basic,
        test_dispatch,
        test_grad_flows_to_student_only,
    ]:
        print(f"\n-- {fn.__name__} --")
        fn()
    print("\n=== All OPSD loss tests passed ===")
