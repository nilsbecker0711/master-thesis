"""
train.py --resume: slicing a universal run must be transparent.

A population run is N independent per-image attacks and slices for free. This
is ONE continuous optimisation, so the cut has to carry Adam's moments, the
cosine schedule's position, the epoch counter and the RNG. If any of those is
dropped the run still completes and still writes plausible numbers — it just
silently becomes a different optimisation, restarted cold at every slice
boundary. That is the failure this file exists to catch.

Slices are simulated by killing the process immediately after a checkpoint
write, which is the realistic case: the driver hands out 30-minute dev-GPU
slices and SLURM terminates whatever is running when the wall clock expires.

The model and the dataset are stubs. What is under test is the state machine,
not the segmentor.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train as train_mod

H, W, K = 32, 64, 4


class _StubModel(torch.nn.Module):
    """A tiny conv, genuinely differentiable w.r.t. its input, so the patch
    receives a real gradient and Adam accumulates real moments."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.c = torch.nn.Conv2d(3, K, 3, padding=1)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.c(x)


class _StubDS(torch.utils.data.Dataset):
    def __init__(self, n):
        g = torch.Generator().manual_seed(1)
        self.imgs = torch.rand(n, 3, H, W, generator=g)
        self.lbls = torch.randint(0, K, (n, H, W), generator=g)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.lbls[i]


class _Killed(Exception):
    """Stands in for SLURM terminating the slice at its walltime."""


@pytest.fixture
def stub(monkeypatch):
    monkeypatch.setattr(train_mod, "setup_model",
                        lambda a: (_StubModel(), K, K, None))
    monkeypatch.setattr(train_mod, "make_dataset",
                        lambda a, split: _StubDS(8 if split == "train" else 4))

    # The val loader hardcodes num_workers=2 (train.py). That is fine on the
    # cluster, where DataLoader forks; under pytest on Windows it spawns
    # subprocesses that re-import the test module and the run hangs. Force
    # single-process loading for both loaders — it changes nothing about the
    # state machine under test.
    real_dl = train_mod.DataLoader
    monkeypatch.setattr(train_mod, "DataLoader",
                        lambda *a, **kw: real_dl(*a, **{**kw,
                                                        "num_workers": 0}))
    return monkeypatch


def _argv(out_dir, epochs, *extra):
    """Note --epochs is ALWAYS the full run, exactly as the driver passes it;
    a slice stops early because it is killed, never because it was told to do
    less work. Telling it to do less would change the cosine T_max."""
    return ["train.py", "--arch", "segformer", "--cityscapes_root", "/none",
            "--img_h", str(H), "--img_w", str(W), "--num_classes", str(K),
            "--patch_mode", "raw", "--patch_size", "8", "--patch_scale", "0.25",
            "--epochs", str(epochs), "--lr", "0.1", "--lr_schedule", "cosine",
            "--batch_size", "2", "--num_workers", "0",
            "--val_images", "4", "--val_every", "1000",
            "--no_diagnostics", "--panel_images", "",
            "--out_dir", str(out_dir), *extra]


def _slice(mp, argv, kill_after=None):
    """Run train.main(), optionally killed right after the Nth checkpoint."""
    real = train_mod.atomic_save
    seen = {"n": 0}

    def spy(obj, path):
        real(obj, path)
        seen["n"] += 1
        if kill_after is not None and seen["n"] >= kill_after:
            raise _Killed
    mp.setattr(train_mod, "atomic_save", spy)
    mp.setattr(sys, "argv", argv)
    try:
        train_mod.main()
    except _Killed:
        pass
    finally:
        mp.setattr(train_mod, "atomic_save", real)


def _final(out_dir):
    return torch.load(out_dir / "final.pt", map_location="cpu")["param"]


# ── the one that matters ─────────────────────────────────────────────────────

def test_sliced_run_reproduces_the_uninterrupted_one(stub, tmp_path):
    """Six epochs in one go versus 2 + 1 + 3, with the process state thrown
    away in between, must land on the same patch."""
    whole = tmp_path / "whole"
    _slice(stub, _argv(whole, 6))
    assert (whole / "results.json").exists()

    sliced = tmp_path / "sliced"
    _slice(stub, _argv(sliced, 6, "--resume"), kill_after=2)
    _slice(stub, _argv(sliced, 6, "--resume"), kill_after=1)
    _slice(stub, _argv(sliced, 6, "--resume"))
    assert (sliced / "results.json").exists()

    a, b = _final(whole), _final(sliced)
    assert torch.equal(a, b), f"max diff {(a - b).abs().max().item():.3e}"


def test_a_cold_resume_would_fail_this_suite(stub, tmp_path):
    """Guard on the guard: if the state carried only the parameter and Adam
    restarted cold, the patches would differ. Confirms the test above is
    actually sensitive to what it claims to test."""
    whole = tmp_path / "whole"
    _slice(stub, _argv(whole, 6))

    cold = tmp_path / "cold"
    _slice(stub, _argv(cold, 6, "--resume"), kill_after=3)
    st = torch.load(cold / "train_state.pt", map_location="cpu")
    st["opt"]["state"] = {}                      # drop the moments
    torch.save(st, cold / "train_state.pt")
    _slice(stub, _argv(cold, 6, "--resume"))

    assert not torch.equal(_final(whole), _final(cold))


# ── the driver's contract ────────────────────────────────────────────────────

def test_resume_is_idempotent_once_results_exist(stub, tmp_path):
    """The driver relaunches until the sentinel appears; a relaunch after it
    appears must not start a second run into the same directory."""
    d = tmp_path / "r"
    _slice(stub, _argv(d, 2))
    before = _final(d).clone()
    mtime = (d / "results.json").stat().st_mtime_ns

    _slice(stub, _argv(d, 2, "--resume"))
    assert (d / "results.json").stat().st_mtime_ns == mtime
    assert torch.equal(_final(d), before)


def test_resume_with_no_state_yet_starts_fresh(stub, tmp_path):
    """The driver's FIRST slice passes --resume too; it must not need a
    special case."""
    d = tmp_path / "first"
    _slice(stub, _argv(d, 2, "--resume"))
    assert (d / "results.json").exists()


def test_resume_without_out_dir_is_refused(stub, tmp_path):
    argv = [a for a in _argv(tmp_path / "x", 2, "--resume")
            if a not in ("--out_dir", str(tmp_path / "x"))]
    stub.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="out_dir"):
        train_mod.main()


def test_resume_refuses_a_config_that_drifted(stub, tmp_path):
    """A slice relaunched with a different --lr or --epochs continues a
    DIFFERENT optimisation: cosine T_max and Adam's moments both depend on
    them, and nothing downstream would show the seam."""
    d = tmp_path / "drift"
    _slice(stub, _argv(d, 6), kill_after=2)

    stub.setattr(sys, "argv", _argv(d, 6, "--resume", "--lr", "0.9"))
    with pytest.raises(SystemExit, match="lr: recorded"):
        train_mod.main()

    stub.setattr(sys, "argv", _argv(d, 12, "--resume"))
    with pytest.raises(SystemExit, match="epochs: recorded"):
        train_mod.main()


# ── what the state has to contain ────────────────────────────────────────────

def test_state_carries_everything_the_next_step_needs(stub, tmp_path):
    d = tmp_path / "s"
    _slice(stub, _argv(d, 3), kill_after=3)
    st = torch.load(d / "train_state.pt", map_location="cpu")
    for k in ("epoch", "gstep", "best", "history", "elapsed_s", "param",
              "opt", "sched", "py_rng", "torch_rng"):
        assert k in st, f"train_state.pt is missing {k}"
    assert st["epoch"] == 3
    assert st["gstep"] == 3 * 4                  # 8 train imgs / batch 2
    assert st["sched"] is not None and st["sched"]["T_max"] == 3 * 4
    assert st["opt"]["state"], "Adam's moments are missing"


def test_wall_clock_accumulates_across_slices(stub, tmp_path):
    """wall_clock_s is a cost column in the index; per-slice it understates
    the run by however many slices it took."""
    d = tmp_path / "w"
    _slice(stub, _argv(d, 4, "--resume"), kill_after=2)
    part = torch.load(d / "train_state.pt", map_location="cpu")["elapsed_s"]
    assert part > 0
    _slice(stub, _argv(d, 4, "--resume"))
    total = json.loads((d / "results.json").read_text())["wall_clock_s"]
    assert total >= part


def test_the_sampled_subset_is_pinned_across_a_resume(stub, tmp_path):
    d = tmp_path / "n"
    _slice(stub, _argv(d, 3, "--n_train", "4", "--train_seed", "68",
                       "--resume"), kill_after=1)
    rec = json.loads((d / "train_images.json").read_text())
    assert len(rec["indices"]) == 4 and rec["train_seed"] == 68

    # Tamper with the record: a resume that quietly trained on other images
    # would be invisible in every output, so it must refuse.
    (d / "train_images.json").write_text(
        json.dumps(dict(rec, indices=[1, 2, 3, 5])))
    stub.setattr(sys, "argv", _argv(d, 3, "--n_train", "4",
                                    "--train_seed", "68", "--resume"))
    with pytest.raises(SystemExit, match="training subset"):
        train_mod.main()
