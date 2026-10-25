"""
Two checks against the observed runs.

A. Is the regulariser alone an OSCILLATOR at lr*w3 = 0.3? Iterate the IAP step
   with ONLY the regulariser's gradient — no model, no adversarial term — and
   watch |delta - base|.

B. Under slide at 1024x2048, how much of the frame can a patch at
   (623, 1792) reach at all?
"""
import sys
from pathlib import Path

import torch

ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from patchreach.models.wrapper import slide_windows
from patchreach.patch import iap

print("=" * 74)
print(" A. the regulariser alone, iterated — is lr*w3 an oscillator?")
print("=" * 74)
torch.manual_seed(0)
P = 64
base = torch.rand(3, P, P).clamp(0.1, 0.9)
# the observed sens profile: median 25, max at the 1/eps ceiling
sens = (torch.rand(P, P) * 50 + 2.0)
sens[torch.rand(P, P) < 0.05] = 1.0 / iap.SENS_EPS

print("  delta starts 0.05 ABOVE base — at delta == base, sign(delta-base) is")
print("  0 and the reg-only dynamic is a fixed point, so it has to be kicked")
print("  first. The attack term does that kicking in a real run.")
print()
print(f"{'lr':>6} {'w3':>5} {'lr*w3':>7}   mean|d-b| at step 1/2/3/5/10/50/200")
print("-" * 74)
for lr, w3 in ((0.1, 3.0), (0.1, 1.0), (0.1, 0.3), (0.01, 3.0), (0.003, 3.0)):
    delta = (base + 0.05).clamp(0, 1).clone().requires_grad_(True)
    traj, hist = [], []
    for t in range(1, 201):
        if delta.grad is not None:
            delta.grad = None
        # ONLY the regulariser. No model, no adversarial term.
        (w3 * iap.regulariser(delta, base, sens, "sum")).backward()
        iap.colour_constant_step_(delta, sens, lr)
        delta.data.clamp_(0.0, 1.0)
        d = float((delta.detach() - base).abs().mean())
        hist.append(d)
        if t in (1, 2, 3, 5, 10, 50, 200):
            traj.append(d)
    # does it settle, or keep swinging? compare the last 50 steps' spread
    tail = hist[-50:]
    swing = max(tail) - min(tail)
    print(f"{lr:>6} {w3:>5} {lr * w3:>7.3f}   "
          + " ".join(f"{v:.4f}" for v in traj)
          + f"   | last-50 swing {swing:.4f}"
          + ("  <- LIMIT CYCLE" if swing > 1e-3 else "  settled"))

print()
print("  The reg's gradient is proportional to sens and the step divides by")
print("  sens, so the two cancel: the reg contributes a SENS-INDEPENDENT")
print("  relative change of lr*w3*sign(delta-base) per step. At lr*w3 = 0.3")
print("  that is a 30% multiplicative kick, and `* delta` shrinks toward ZERO")
print("  rather than toward base, so it overshoots and reverses sign.")

print()
print("=" * 74)
print(" B. what a patch at (623, 1792) can reach under slide")
print("=" * 74)
H, W, p = 1024, 2048, 256
top, left = 623, 1792
CROP, STRIDE = (1024, 1024), (768, 768)
wins = list(slide_windows(H, W, CROP, STRIDE))
print(f"  windows: {[(y1, y2, x1, x2) for y1, y2, x1, x2 in wins]}")
print(f"  patch occupies x [{left}, {left + p}), y [{top}, {top + p})")

reach = torch.zeros(H, W, dtype=torch.bool)
hit = []
for y1, y2, x1, x2 in wins:
    overlaps = not (left + p <= x1 or left >= x2 or
                    top + p <= y1 or top >= y2)
    if overlaps:
        hit.append((x1, x2))
        reach[y1:y2, x1:x2] = True
print(f"  windows CONTAINING any patch pixel: {hit}")
print(f"  -> reachable pixels: {int(reach.sum()):,} of {H * W:,} "
      f"({100 * float(reach.float().mean()):.1f}% of the frame)")
print(f"  -> an upper bound on any_flip_rate and target_hit_rate, before the")
print(f"     attack even starts.")
print()
print(f"  for contrast, a CENTRED patch at ({(H - p) // 2}, {(W - p) // 2}):")
reach2 = torch.zeros(H, W, dtype=torch.bool)
t2, l2 = (H - p) // 2, (W - p) // 2
for y1, y2, x1, x2 in wins:
    if not (l2 + p <= x1 or l2 >= x2 or t2 + p <= y1 or t2 >= y2):
        reach2[y1:y2, x1:x2] = True
print(f"  -> reachable pixels: {int(reach2.sum()):,} "
      f"({100 * float(reach2.float().mean()):.1f}% of the frame)")
