"""
Is the regulariser-driven oscillation PATCH-SIZE dependent, as the 2026-10-23
log entry claims via D3? dR/d(delta_j) = sens_j * sign(delta_j - base_j) has no
dependence on the number of terms in the sum, so it should NOT be.
"""
import sys
from pathlib import Path

import torch

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patchreach.patch import iap


def run(P, lr, w3, reduce="sum", steps=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    base = (torch.rand(3, P, P, generator=g) * 0.8 + 0.1)
    sens = torch.rand(P, P, generator=g) * 50 + 2.0
    delta = (base + 0.05).clamp(0, 1).clone().requires_grad_(True)
    hist = []
    for _ in range(steps):
        if delta.grad is not None:
            delta.grad = None
        (w3 * iap.regulariser(delta, base, sens, reduce)).backward()
        iap.colour_constant_step_(delta, sens, lr)
        delta.data.clamp_(0.0, 1.0)
        hist.append(float((delta.detach() - base).abs().mean()))
    tail = hist[-50:]
    return hist, max(tail) - min(tail)


print("=" * 74)
print(" reg-only oscillation vs PATCH SIZE, at the same lr and w3")
print("=" * 74)
print(f"{'P':>5} {'elements':>10} {'lr*w3':>7}  {'|d| @1/2/3/10':>30}  swing(last 50)")
print("-" * 74)
for P in (32, 64, 128, 256):
    hist, swing = run(P, 0.1, 3.0)
    at = " ".join(f"{hist[i]:.4f}" for i in (0, 1, 2, 9))
    print(f"{P:>5} {3 * P * P:>10} {0.3:>7.3f}  {at:>30}  {swing:.4f}"
          + ("  LIMIT CYCLE" if swing > 1e-3 else "  settled"))

print()
print("  If these trajectories agree, the instability is SIZE-INDEPENDENT and")
print("  the log entry's D3 link is wrong: the gradient of a sum w.r.t. one")
print("  element does not depend on how many elements the sum has.")

print()
print("=" * 74)
print(" and does sum84 still help? (it scales the whole term by (84/P)**2)")
print("=" * 74)
for P in (84, 256):
    _, s_sum = run(P, 0.1, 3.0, "sum")
    _, s_84 = run(P, 0.1, 3.0, "sum84")
    print(f"  P={P:>4}  sum   swing {s_sum:.4f}"
          + ("  LIMIT CYCLE" if s_sum > 1e-3 else "  settled"))
    print(f"         sum84 swing {s_84:.4f}"
          + ("  LIMIT CYCLE" if s_84 > 1e-3 else "  settled")
          + f"   (effective w3 = {3.0 * (84 / P) ** 2:.3f})")
