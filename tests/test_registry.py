"""
Registry invariants.

The factorial will grow — every cell added to answer "encoder or decoder?"
lands here — and the failure modes of this file are all IMPORT-time and
therefore total: registry.py is imported by scripts/_common.py, so a bad entry
takes down every script in the repo before it prints anything. An alias that
outlived its entry once did exactly that, with a bare KeyError and no message.
"""
import pytest

from patchreach.models.registry import (ArchSpec, BRACKETS, DECODERS, ENCODERS,
                                        REGISTRY, _ALIASES, grid)


# ── the declared axes ────────────────────────────────────────────────────────
def test_every_entry_declares_both_axes():
    """
    `bracket` alone cannot express the factorial, which is the whole reason
    encoder/decoder exist. An entry missing one is filed under an axis it
    cannot be grouped on, and silently drops out of the T04/T16 grid.
    """
    for name, spec in REGISTRY.items():
        assert spec.encoder, f"{name} declares no encoder"
        assert spec.decoder, f"{name} declares no decoder"


def test_declared_axes_are_known_keys():
    for name, spec in REGISTRY.items():
        assert spec.encoder in ENCODERS, f"{name}: encoder {spec.encoder!r}"
        assert spec.decoder in DECODERS, f"{name}: decoder {spec.decoder!r}"


def test_bracket_agrees_with_the_encoder_everywhere():
    """The bracket is a property of the encoder, so the two cannot disagree."""
    for name, spec in REGISTRY.items():
        assert spec.bracket == ENCODERS[spec.encoder], (
            f"{name}: bracket {spec.bracket!r} vs encoder {spec.encoder!r}")
        assert spec.bracket in BRACKETS


# ── the guards fire at construction, not at load ─────────────────────────────
def test_unknown_encoder_is_refused():
    with pytest.raises(ValueError, match="unknown encoder"):
        ArchSpec("probe", "none", encoder="resnet99", decoder="aspp")


def test_unknown_decoder_is_refused():
    with pytest.raises(ValueError, match="unknown decoder"):
        ArchSpec("probe", "none", encoder="resnet50", decoder="magic")


def test_a_bracket_that_contradicts_its_encoder_is_refused():
    """ViT-S is global attention; filing it as 'none' is a typo, not a variant."""
    with pytest.raises(ValueError, match="disagrees with encoder"):
        ArchSpec("probe", "none", encoder="vit_s", decoder="aspp")


def test_axes_are_optional_so_a_bare_spec_still_builds():
    """Nothing forces a throwaway spec to declare an axis it is not filed under."""
    spec = ArchSpec("probe", "none")
    assert spec.encoder == "" and spec.decoder == ""


# ── aliases ──────────────────────────────────────────────────────────────────
def test_every_alias_points_at_a_real_entry():
    for short, full in _ALIASES.items():
        assert full in REGISTRY, f"alias {short!r} -> missing {full!r}"
        assert REGISTRY[short] is REGISTRY[full]


# ── the grid ─────────────────────────────────────────────────────────────────
def test_grid_counts_each_cell_once():
    """An alias is the same object under a second key and must not double-count."""
    rows = grid()
    assert len(rows) == len(REGISTRY) - len(_ALIASES)
    assert len({r["arch"] for r in rows}) == len(rows)
    assert not ({r["arch"] for r in rows} & set(_ALIASES))


def test_grid_carries_both_axes_and_availability():
    for r in grid():
        assert r["encoder"] in ENCODERS and r["decoder"] in DECODERS
        assert isinstance(r["available"], bool)


def test_the_factorial_actually_crosses():
    """
    The point of the lineup is that at least one encoder appears under more
    than one decoder, and at least one decoder under more than one encoder.
    Without both, no cell isolates either axis and the grid is still a list.
    """
    rows = grid()
    by_encoder, by_decoder = {}, {}
    for r in rows:
        by_encoder.setdefault(r["encoder"], set()).add(r["decoder"])
        by_decoder.setdefault(r["decoder"], set()).add(r["encoder"])

    assert any(len(d) > 1 for d in by_encoder.values()), (
        "no encoder appears under two decoders — nothing isolates the decoder")
    assert any(len(e) > 1 for e in by_decoder.values()), (
        "no decoder appears over two encoders — nothing isolates the encoder")


def test_the_decoder_contrast_at_fixed_encoder_exists():
    """
    resnet50 under both aspp and unet is the pair that answers the question the
    factorial was built for: reach never built, or reach discarded.
    """
    cells = {(r["encoder"], r["decoder"]) for r in grid()}
    assert ("resnet50", "aspp") in cells
    assert ("resnet50", "unet") in cells


def test_the_hybrid_probe_exists():
    """
    dcnv3_t under both upernet and mask_tf: same encoder, one convolutional
    decoder and one transformer decoder. Asks whether a global decoder can
    supply reach the encoder never builds.
    """
    cells = {(r["encoder"], r["decoder"]) for r in grid()}
    assert ("dcnv3_t", "upernet") in cells
    assert ("dcnv3_t", "mask_tf") in cells
