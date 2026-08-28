r"""
The --images resolver. Run with: pytest -q tests/test_image_selection.py

Cross-image validation is the claim these back: a patch overfit on image x
reaching 50 points THERE says nothing about anywhere else, and the difference
between the two numbers is entirely in which images got selected. So the
selection is tested rather than trusted — in particular that the exclusion
actually removes x, that it does not quietly shrink the sample, and that
`fixed` does not move when a seed does.
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

N_VAL = 500


def _sel(*args, **kw):
    pytest.importorskip("torchvision")
    from _common import resolve_images
    return resolve_images(*args, **kw)


# ── a) one random image, b) n random images ──────────────────────────────────

def test_random_one_image():
    assert len(_sel("random", N_VAL, 1, sample_seed=7)) == 1


def test_random_n_and_seed_reproducible():
    a = _sel("random", N_VAL, 25, sample_seed=3)
    assert len(a) == 25 and len(set(a)) == 25
    assert a == _sel("random", N_VAL, 25, sample_seed=3)
    assert a != _sel("random", N_VAL, 25, sample_seed=4)


def test_random_sample_is_nested_in_n():
    """
    n=50 must be n=25 plus 25 new images under the same seed, or --resume on a
    grown population re-attacks images it already has.
    """
    small = set(_sel("random", N_VAL, 25, sample_seed=0))
    assert small <= set(_sel("random", N_VAL, 50, sample_seed=0))


def test_random_is_not_the_first_n_indices():
    """The trap the old truncate-after-sort call site fell into."""
    assert _sel("random", N_VAL, 10, sample_seed=0) != list(range(10))


# ── c) n fixed images ────────────────────────────────────────────────────────

def test_fixed_ten_is_fixed10():
    from _common import FIXED10
    assert _sel("fixed", N_VAL, 10) == FIXED10
    assert _sel("fixed10", N_VAL) == FIXED10


def test_fixed_ignores_sample_seed():
    """A 'fixed' set that moves with a seed is not fixed."""
    assert _sel("fixed", N_VAL, 40, sample_seed=0) == \
           _sel("fixed", N_VAL, 40, sample_seed=99)


def test_fixed_is_nested_and_starts_with_fixed10():
    from _common import FIXED10
    big = _sel("fixed", N_VAL, 40)
    assert len(big) == 40 and len(set(big)) == 40
    assert big[:10] == FIXED10
    assert big[:25] == _sel("fixed", N_VAL, 25)


# ── d) everything except image x ─────────────────────────────────────────────

def test_all_is_the_whole_split():
    assert _sel("all", N_VAL) == list(range(N_VAL))


def test_exclude_holds_out_the_training_image():
    out = _sel("all", N_VAL, exclude=[420])
    assert len(out) == N_VAL - 1 and 420 not in out


def test_exclude_minus_one_excludes_nothing():
    """-1 is the default, and it must mean 'all 500'."""
    assert _sel("all", N_VAL, exclude=[-1]) == list(range(N_VAL))


def test_exclude_applies_before_truncation():
    """--n_images 100 --exclude_image x must still return 100 images."""
    out = _sel("random", N_VAL, 100, sample_seed=0, exclude=[420])
    assert len(out) == 100 and 420 not in out


def test_exclude_works_for_every_mode():
    from _common import FIXED10
    x = FIXED10[0]
    assert x not in _sel("fixed10", N_VAL, exclude=[x])
    assert x not in _sel("fixed", N_VAL, 30, exclude=[x])
    assert x not in _sel(f"{x} 1 2", N_VAL, exclude=[x])


def test_exclude_accepts_several():
    out = _sel("all", N_VAL, exclude=[1, 2, 3])
    assert len(out) == N_VAL - 3


# ── the failure modes ────────────────────────────────────────────────────────

def test_empty_selection_raises():
    """Excluding the only requested image is a mistake, not an empty run."""
    with pytest.raises(SystemExit):
        _sel("42", N_VAL, exclude=[42])


def test_out_of_range_explicit_index_raises():
    with pytest.raises(SystemExit):
        _sel("2 5 9999", N_VAL)


def test_explicit_list_keeps_its_order_and_both_separators():
    assert _sel("5,2,45", N_VAL) == [5, 2, 45]
    assert _sel("5 2 45", N_VAL) == [5, 2, 45]


# ── the tag, so two selections do not share a directory name ─────────────────

@pytest.mark.parametrize("kw,expect", [
    (dict(images="random", n_images=25, sample_seed=3), "rand25s3"),
    (dict(images="fixed", n_images=25), "fixed25"),
    (dict(images="fixed10"), "fixed10"),
    (dict(images="all"), "all"),
    (dict(images="all", exclude_image=[420]), "all_ex420"),
])
def test_sample_tag(kw, expect):
    pytest.importorskip("torchvision")
    import argparse
    from _common import sample_tag
    base = dict(images="fixed10", n_images=None, sample_seed=0,
                exclude_image=[-1])
    assert sample_tag(argparse.Namespace(**{**base, **kw})) == expect


# ── the parsers that have to accept all of it ────────────────────────────────

@pytest.mark.parametrize("module", ["evaluate", "overfit_population",
                                    "measure_frequency_sensitivity",
                                    "export_conditional_patches"])
def test_scripts_expose_the_image_flags(module):
    pytest.importorskip("torchvision")
    import argparse
    import importlib
    mod = importlib.import_module(module)
    p = (mod.build_parser() if hasattr(mod, "build_parser")
         else _parser_of(mod))
    opts = {s for act in p._actions for s in act.option_strings}
    for f in ("--images", "--n_images", "--sample_seed", "--exclude_image"):
        assert f in opts, f"{module} is missing {f}"


def _parser_of(mod):
    """evaluate.py builds its parser inline in main(); rebuild the same one."""
    import argparse
    from _common import add_model_args, add_image_args
    return add_image_args(add_model_args(argparse.ArgumentParser()))


def test_from_image_is_recorded_in_the_patch_config():
    """
    A csf base is NOT in the checkpoint, so nothing but this flag tells a later
    evaluation to re-derive it from the image being evaluated. Without it,
    render() falls back to 0.5 grey and the cross-image number describes a
    patch that was never optimised.
    """
    from patchreach.patch.spec import PatchConfig
    assert PatchConfig(mode="csf").from_image is False
    assert PatchConfig(mode="csf", from_image=True).validate().from_image


def test_from_image_survives_a_save_load_round_trip(tmp_path):
    from patchreach.patch.spec import Patch, PatchConfig
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    p = Patch(PatchConfig(mode="csf", size=32, scale=0.25, from_image=True),
              torch.device("cpu"), mean, std)
    p.save(tmp_path / "best.pt")
    assert Patch.load(tmp_path / "best.pt", torch.device("cpu"),
                      mean, std).cfg.from_image is True


def test_csf_patch_rebases_onto_the_image_it_is_evaluated_on(tmp_path):
    """
    THE cross-image operation. A csf patch is base + bounded residual, and the
    base is not in the checkpoint. Loaded onto a DIFFERENT image it must take
    its base there — otherwise render() returns 0.5 grey plus a residual, i.e.
    a visible square that no optimisation ever produced, and the transfer
    number measures that instead of the attack.
    """
    from patchreach.patch.spec import Patch, PatchConfig
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    dark = ((torch.zeros(1, 3, 128, 256) + 0.05) - mean) / std   # "image A"
    light = ((torch.zeros(1, 3, 128, 256) + 0.95) - mean) / std  # "image B"

    p = Patch(PatchConfig(mode="csf", size=32, scale=0.25, from_image=True),
              torch.device("cpu"), mean, std)
    p.resolve_placement(128, 256)
    p.set_reference_from_image(dark, mean, std)
    p.save(tmp_path / "best.pt")

    q = Patch.load(tmp_path / "best.pt", torch.device("cpu"), mean, std)
    q.resolve_placement(128, 256)
    grey = float(q.render().mean())            # no re-derivation: 0.5 grey
    q.set_reference_from_image(light, mean, std)
    rebased = float(q.render().mean())         # re-derived on image B

    assert grey == pytest.approx(0.5, abs=0.1)
    assert rebased > 0.9, "the base did not follow the evaluated image"
