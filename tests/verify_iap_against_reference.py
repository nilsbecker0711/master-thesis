r"""
Step-by-step verification of the IAP port against the reference's STRUCTURE.

    python tests/verify_iap_against_reference.py

NOT A PYTEST MODULE, and the name is deliberate — pytest's default collection
pattern is test_*.py, so this is skipped by `pytest tests/`. It is a DIFFERENTIAL
check: it transcribes the reference's inner loop structure for structure and
holds patchreach/patch/iap.py equal to it, which is a different kind of evidence
from tests/test_iap.py's per-function pins and is slow enough (it runs the real
attack loop several times over) not to belong in the suite.

WHAT ONLY THIS CAN CATCH. test_iap.py pins each of IAP's four parts in
isolation. The loop can still be wrong in ways every one of those tests passes:
the order of grad-zero / stop-check / backward / update, whether each rung
restarts from the base, whether the clamp lands before or after the step, and
whether the number in the record describes the patch written to disk. Deviation
D9 — the released regulariser's two gradient paths cancelling exactly — was
found here, not there.

Exits non-zero on the first failing check, so it can gate a commit.
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# argv[1] stays supported so the check can be pointed at another checkout.
ROOT = (Path(sys.argv[1]) if len(sys.argv) > 1
        else Path(__file__).resolve().parents[1])
sys.path.insert(0, str(ROOT))

from patchreach.data.cityscapes import norm_tensors, upsample_to
from patchreach.metrics.miou import single_image_miou
from patchreach.patch import iap
from patchreach.patch.spec import Patch, PatchConfig

K, H, W, P = 19, 64, 128, 16
dev = torch.device("cpu")
MEAN, STD = norm_tensors(dev)
torch.manual_seed(0)

ok, bad = [], []


def step(n, name, passed, detail=""):
    (ok if passed else bad).append(n)
    print(f"  [{'PASS' if passed else 'FAIL'}] {n}. {name}")
    if detail:
        for line in detail.strip().splitlines():
            print(f"            {line}")


class Stub(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = nn.Conv2d(3, K, 3, padding=1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.c(x)


class StubCam:
    n_degenerate = 0

    def __init__(self, m):
        self.m = m

    def __call__(self, imgs, labels):
        c = torch.zeros(1, 1, H, W)
        c[:, :, 8:40, 70:110] = 1.0
        with torch.no_grad():
            return c, self.m(imgs)

    def close(self):
        pass


model = Stub().eval()
cam = StubCam(model)
x01 = torch.rand(1, 3, H, W).clamp(0.02, 0.98)
img = (x01 - MEAN) / STD
label = torch.randint(0, K, (1, H, W))
label[:, :4, :] = 255


def mk_patch():
    cfg = PatchConfig(mode="raw", pixel_param="direct", size=P,
                      scale=P / H, placement="center")
    return Patch(cfg, dev, MEAN, STD)


print("=" * 72)
print(" PART 3 — THE REGULARISER'S GRADIENT")
print("=" * 72)

# ─── 1. the reference's own expression, transcribed exactly ──────────────────
base_vals = x01[0, :, 8:8 + P, 70:70 + P].clone()
sens_t = torch.rand(P, P) * 20 + 1.0

delta_ref = torch.nn.Parameter(base_vals.clone()[None])   # [1,3,p,p]
temp_ref = delta_ref.clone()                              # <-- the reference
r_ref = 3 * (sens_t.detach() * torch.abs(delta_ref - temp_ref)).sum()
r_ref.backward()
g_from_clone = delta_ref.grad.abs().sum().item()
step(1, "reference `temp = delta.clone()` keeps the graph",
     temp_ref.requires_grad and temp_ref.grad_fn is not None,
     f"temp.requires_grad={temp_ref.requires_grad}  "
     f"grad_fn={type(temp_ref.grad_fn).__name__ if temp_ref.grad_fn else None}")

# Move delta away from temp so sign() is non-zero everywhere, then re-check.
delta_ref2 = torch.nn.Parameter(base_vals.clone()[None])
temp_ref2 = delta_ref2.clone()
delta_ref2.data.add_(0.1)                        # |delta - temp| = 0.1 > 0
r_ref2 = 3 * (sens_t.detach() * torch.abs(delta_ref2 - temp_ref2)).sum()
r_ref2.backward()
g2 = delta_ref2.grad.abs().sum().item()
step(2, "reference regulariser contributes ZERO gradient (paths cancel)",
     g_from_clone == 0.0 and g2 == 0.0,
     f"grad sum at delta==temp : {g_from_clone:.3e}\n"
     f"grad sum at |d-t|=0.1   : {g2:.3e}   (value of the term: "
     f"{float(r_ref2):.1f}, so the LOSS moves but the GRADIENT does not)")

# ─── 3. the port's regulariser, with a detached base ─────────────────────────
delta_mine = base_vals.clone().requires_grad_(True)
r_mine = 3 * iap.regulariser(delta_mine, base_vals + 0.0, sens_t, "sum")
# shift so sign() != 0
delta_mine2 = (base_vals + 0.1).clone().requires_grad_(True)
r_mine2 = 3 * iap.regulariser(delta_mine2, base_vals, sens_t, "sum")
r_mine2.backward()
g_mine = delta_mine2.grad
want = 3 * sens_t * torch.ones(3, P, P)          # sign(+0.1) = +1
step(3, "port's regulariser gradient is live and equals 3*w*sens*sign(d-b)",
     torch.allclose(g_mine, want, atol=1e-5),
     f"max |grad - 3*sens*sign| = {float((g_mine - want).abs().max()):.3e}")

print()
print("=" * 72)
print(" PARTS 1, 2, 4 AND THE LOOP — TRAJECTORY AGAINST THE REFERENCE")
print("=" * 72)


def reference_style_rung(objective, top, left, base, sens, lr, steps):
    """
    The reference's inner loop, structure for structure, with the SAME
    objective the port uses (the loss substitution is the declared deviation;
    everything else here is claimed unchanged, and that is what is tested).

    base is passed in DETACHED, i.e. the paper's regulariser, so the two sides
    differ only in the mechanics under test.
    """
    delta = torch.nn.Parameter(base.clone()[None])
    temp = base.clone()                                  # detached: no clone()
    traj = []
    fp = torch.zeros(1, H, W, dtype=torch.bool)
    fp[:, top:top + P, left:left + P] = True
    sup = ~fp
    for _ in range(steps):
        x_p = x01.clone()
        x_p[:, :, top:top + P, left:left + P] = delta
        logits = upsample_to(model((x_p - MEAN) / STD), (H, W))
        parts = objective(logits, label, fp, sup, delta[0], temp, sens)
        parts["total"].backward()
        delta.data = (delta.data
                      - lr * delta.grad.mean(dim=1) * delta.data * (1 / sens))
        delta.grad.zero_()
        traj.append(delta.detach()[0].clone())
    return traj


patch = mk_patch()
prep = iap.prepare(model, img, patch, mean_t=MEAN, std_t=STD, label=label,
                   cam=cam, score="cam", log=lambda *_: None)
top, left, p_side = prep["box"]

# ─── 4. localisation ─────────────────────────────────────────────────────────
step(4, "localisation lands on the CAM hotspot, not the centre",
     (top, left) == (8, 70) and (top, left) != ((H - P) // 2, (W - P) // 2),
     f"(top,left)=({top},{left})  centre=({(H-P)//2},{(W-P)//2})")

# ─── 5. base is the image region, in [0,1] ───────────────────────────────────
step(5, "delta initialises to the denormalised image region it replaces",
     torch.allclose(patch.param.detach(), x01[0, :, top:top + P,
                                               left:left + P], atol=1e-6),
     f"max |param - x01[region]| = "
     f"{float((patch.param.detach() - x01[0,:,top:top+P,left:left+P]).abs().max()):.2e}")

# ─── 6. the sensitivity crop is aligned with the window ──────────────────────
step(6, "sens_patch is the sensitivity map cropped at the same window",
     torch.equal(prep["sens_patch"],
                 prep["sens"][0, top:top + P, left:left + P]))

# ─── 7. sens is computed in [0,1], not in normalised space ───────────────────
s01 = iap.sensitivity(x01)
snorm = iap.sensitivity(img)
#     allclose, not equal: prepare() recovers [0,1] by un-normalising, which
#     round-trips through float32 and lands ~6e-8 from the original frame.
step(7, "sens is computed on the [0,1] frame (eps=1e-3 is calibrated there)",
     torch.allclose(prep["sens"], s01, rtol=1e-4) and
     not torch.allclose(s01, snorm),
     f"median sens on [0,1] = {float(s01.median()):.2f}   "
     f"on normalised = {float(snorm.median()):.2f}  "
     f"(ratio {float(s01.median()/snorm.median()):.2f}x)")

# ─── 8. the footprint apply() produces matches the resolved window ───────────
_, fp_applied = patch.apply(img)
fp_want = torch.zeros(1, H, W, dtype=torch.bool)
fp_want[:, top:top + P, left:left + P] = True
step(8, "Patch.apply footprint == the window localisation resolved",
     torch.equal(fp_applied, fp_want))

# ─── 9. the composite is the reference's paste, exactly ──────────────────────
patched, _ = patch.apply(img)
manual = x01.clone()
manual[:, :, top:top + P, left:left + P] = patch.param.detach()
step(9, "normalised composite == norm(reference's in-place paste)",
     torch.allclose(patched, (manual - MEAN) / STD, atol=1e-6),
     f"max abs diff = {float((patched - (manual-MEAN)/STD).abs().max()):.2e}")

# ─── 10. the trajectory, step for step ───────────────────────────────────────
obj = iap.Objective(tar="ipatch_cospgd", untar="ce", target_class=16,
                    w1=1.5, w2=1.0, w3=3.0, reg_reduce="sum")
LR, NSTEP = 0.09, 5
ref_traj = reference_style_rung(obj, top, left, prep["base"],
                                prep["sens_patch"], LR, NSTEP)

diffs = []
for k in range(1, NSTEP + 1):
    pk = mk_patch()
    pk.cfg.placement, pk.cfg.placement_xy = "fixed", ((top + P / 2) / H,
                                                      (left + P / 2) / W)
    pk.placement = (top, left)
    r = iap._rung(model, img, label, pk, prep, obj, LR, k,
                  stop="none", stop_thresh=1.0, target_class=16,
                  num_classes=K, exclude_footprint=True, classes="gt",
                  clamp=False, clean_remote=0.0, log_every=10 ** 6,
                  verbose=False, log=lambda *_: None)
    diffs.append(float((r["final_param"] - ref_traj[k - 1]).abs().max()))
step(10, f"_rung trajectory == reference update, each of {NSTEP} steps",
     max(diffs) < 1e-6,
     "max |param_mine - param_ref| after k steps:\n" +
     "\n".join(f"  k={i+1}: {d:.3e}" for i, d in enumerate(diffs)))

# ─── 11. each rung restarts from the base ────────────────────────────────────
pk = mk_patch()
pk.placement = (top, left)
rec = iap.attack_image(model, img, label, pk, mean_t=MEAN, std_t=STD, cam=cam,
                       target_class=16, lrs=(0.05, 0.2), steps=3, stop="none",
                       num_classes=K, clamp=False, log_every=10 ** 6,
                       verbose=False, log=lambda *_: None)
r0, r1 = rec["iap_rungs"][0], rec["iap_rungs"][1]
step(11, "the ladder re-initialises delta per rung (rungs are independent)",
     r0["lr"] == 0.05 and r1["lr"] == 0.2 and
     r0["steps_run"] == 3 and r1["steps_run"] == 3 and
     rec["iap_steps_run"] == 6)

print()
print("=" * 72)
print(" THE STOP, THE CLAMP, AND RECORD/ARTEFACT AGREEMENT")
print("=" * 72)

# ─── 12. the stop is evaluated BEFORE the update ─────────────────────────────
pk = mk_patch()
pk.placement = (top, left)
rec_s = iap.attack_image(model, img, label, pk, mean_t=MEAN, std_t=STD,
                         cam=cam, target_class=16, lrs=(0.05,), steps=50,
                         stop="flip", stop_thresh=0.0,   # fires at step 1
                         num_classes=K, clamp=False, log_every=10 ** 6,
                         verbose=False, log=lambda *_: None)
step(12, "stop at step 1 leaves the UNMODIFIED base as the result",
     rec_s["iap_stopped_at"] == 1 and
     torch.allclose(pk.param.detach(), prep["base"], atol=1e-6),
     f"stopped_at={rec_s['iap_stopped_at']}  "
     f"max |param - base| = "
     f"{float((pk.param.detach() - prep['base']).abs().max()):.2e}  "
     f"(the break precedes the update, so no step was taken)")

# ─── 13. the clamp ───────────────────────────────────────────────────────────
pk = mk_patch()
pk.placement = (top, left)
iap.attack_image(model, img, label, pk, mean_t=MEAN, std_t=STD, cam=cam,
                 target_class=16, lrs=(8.0,), steps=4, stop="none",
                 num_classes=K, clamp=True, log_every=10 ** 6,
                 verbose=False, log=lambda *_: None)
in_range = float(pk.param.min()) >= 0.0 and float(pk.param.max()) <= 1.0
pk2 = mk_patch()
pk2.placement = (top, left)
iap.attack_image(model, img, label, pk2, mean_t=MEAN, std_t=STD, cam=cam,
                 target_class=16, lrs=(8.0,), steps=4, stop="none",
                 num_classes=K, clamp=False, log_every=10 ** 6,
                 verbose=False, log=lambda *_: None)
escaped = float(pk2.param.min()) < 0.0 or float(pk2.param.max()) > 1.0
step(13, "clamp=True holds [0,1]; clamp=False reproduces the released escape",
     in_range and escaped,
     f"clamp=True  range [{float(pk.param.min()):.3f}, "
     f"{float(pk.param.max()):.3f}]\n"
     f"clamp=False range [{float(pk2.param.min()):.3f}, "
     f"{float(pk2.param.max()):.3f}]")

# ─── 14. the record describes the patch left in the object ───────────────────
pk = mk_patch()
pk.placement = (top, left)
rec_f = iap.attack_image(model, img, label, pk, mean_t=MEAN, std_t=STD,
                         cam=cam, target_class=16, lrs=(0.05, 0.2), steps=6,
                         stop="none", num_classes=K, clamp=True,
                         log_every=2, verbose=False, log=lambda *_: None)
with torch.no_grad():
    pz, fpz = pk.apply(img)
    lz = upsample_to(model(pz), (H, W))
    remote_now = single_image_miou(lz, label, K, exclude=fpz, classes="gt")
step(14, "record's final_remote is measured on the patch the object holds",
     abs(remote_now - rec_f["final_remote"]) < 1e-4,
     f"recomputed {remote_now:.6f}  vs record {rec_f['final_remote']:.6f}")

# ─── 15. drop_remote is internally consistent ────────────────────────────────
step(15, "drop_remote == clean_remote - final_remote",
     abs((rec_f["clean_remote"] - rec_f["final_remote"])
         - rec_f["drop_remote"]) < 1e-9)

# ─── 16. invisibility is measured on that same patch ─────────────────────────
adv01 = (pz * STD + MEAN).clamp(0, 1)
# AGAINST prepared["x01"], NOT the original frame — and that is the right
# reference, not a convenience. Clean and adversarial must sit in the SAME
# round-tripped space, or the ~6e-8 un-normalisation error is attributed to the
# patch. It is invisible at any real drop and dominates PSNR at 138 dB, where
# the stub patch has barely moved and the metric is measuring float noise.
v = iap.invisibility(prep["x01"], adv01, (top, left, P))
step(16, "SSIM/PSNR in the record describe the same composite",
     abs(v["ssim_patch"] - rec_f["ssim_patch"]) < 1e-5 and
     abs(v["psnr_image"] - rec_f["psnr_image"]) < 1e-3,
     f"ssim_patch {v['ssim_patch']:.6f} vs {rec_f['ssim_patch']:.6f}\n"
     f"psnr_image {v['psnr_image']:.4f} vs {rec_f['psnr_image']:.4f}")

print()
print("=" * 72)
print(f" {len(ok)} passed, {len(bad)} failed" +
      (f"   FAILED: {bad}" if bad else ""))
print("=" * 72)
sys.exit(1 if bad else 0)
