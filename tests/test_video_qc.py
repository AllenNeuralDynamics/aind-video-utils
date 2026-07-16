"""Tests for the transcode QC numeric helpers and the shade_regions plot option."""

import numpy as np
import pytest

# The QC module imports matplotlib at import time; skip this whole module when the
# plotting extra is absent (e.g. the no-extras wheel smoke-test in CI).
matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from aind_video_utils.plotting import intensity_histogram  # noqa: E402
from aind_video_utils.video_qc import (  # noqa: E402
    _box_mean,
    _oetf_derivative,
    _robust_variance,
    bt709_noise_shape,
    classify_gamma,
    noise_transfer_curve,
)

# ---------------------------------------------------------------------------
# _oetf_derivative / bt709_noise_shape
# ---------------------------------------------------------------------------


def test_oetf_derivative_expansive_toe():
    """The BT.709 OETF is expansive (slope > 1) in the dark toe."""
    d = _oetf_derivative(np.array([0.001, 0.005, 0.01]))
    assert np.all(d > 1.0)


def test_oetf_derivative_compressive_highlights():
    """Above the crossover the OETF is gently compressive (slope < 1)."""
    d = _oetf_derivative(np.array([0.5, 0.8, 0.99]))
    assert np.all(d < 1.0)


def test_bt709_noise_shape_positive_finite():
    means = np.linspace(5, 250, 20)
    shape = bt709_noise_shape(means)
    assert shape.shape == means.shape
    assert np.all(np.isfinite(shape))
    assert np.all(shape > 0)


# ---------------------------------------------------------------------------
# _box_mean
# ---------------------------------------------------------------------------


def test_box_mean_constant_field():
    """Box mean of a constant field is that constant (edge-replicated)."""
    img = np.full((10, 12), 7.0)
    out = _box_mean(img, radius=2)
    assert out.shape == img.shape
    assert np.allclose(out, 7.0)


def test_box_mean_smooths_noise():
    rng = np.random.default_rng(0)
    img = 100.0 + rng.normal(0, 10, size=(64, 64))
    out = _box_mean(img, radius=2)
    # Averaging a 5x5 box shrinks variance well below the pixel variance.
    assert out.var() < img.var() / 5


# ---------------------------------------------------------------------------
# _robust_variance
# ---------------------------------------------------------------------------


def test_robust_variance_matches_gaussian():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 3.0, size=20000)
    v = _robust_variance(x)
    assert np.isclose(v, 9.0, rtol=0.1)


def test_robust_variance_too_few():
    assert np.isnan(_robust_variance(np.zeros(31)))


# ---------------------------------------------------------------------------
# noise_transfer_curve
# ---------------------------------------------------------------------------


def _flat_noise_frames(mean_levels, sigma, rng, shape=(120, 120)):
    """Frames of vertically-stacked flat patches, each with additive Gaussian noise."""
    frames = []
    n = len(mean_levels)
    band = shape[0] // n
    for _ in range(6):
        img = np.empty(shape, dtype=np.float64)
        for i, m in enumerate(mean_levels):
            lo = i * band
            hi = shape[0] if i == n - 1 else (i + 1) * band
            img[lo:hi, :] = m + rng.normal(0, sigma, size=(hi - lo, shape[1]))
        frames.append(img)
    return frames


def test_noise_transfer_curve_recovers_flat_variance():
    rng = np.random.default_rng(2)
    levels = [40.0, 100.0, 180.0]
    frames = _flat_noise_frames(levels, sigma=2.0, rng=rng)
    mean, variance = noise_transfer_curve(frames, min_count=200)
    assert mean.size >= 3
    # High-pass residual of white noise keeps its variance to within a modest factor.
    assert np.all(variance > 0)
    assert np.median(variance) < 25.0  # sigma^2 == 4, well under this ceiling


# ---------------------------------------------------------------------------
# classify_gamma
# ---------------------------------------------------------------------------


def test_classify_gamma_linear():
    """A Var ∝ mean photon-transfer curve is called LINEAR."""
    mean = np.linspace(20, 240, 12)
    variance = 0.5 * mean  # perfect linear shot-noise
    result = classify_gamma(mean, variance)
    assert result["call"] == "LINEAR"
    assert result["resid_linear"] < result["resid_bt709"]


def test_classify_gamma_bt709():
    """A BT.709-shaped curve is called GAMMA."""
    mean = np.linspace(20, 240, 12)
    variance = 300.0 * bt709_noise_shape(mean)
    result = classify_gamma(mean, variance)
    assert result["call"] == "GAMMA"
    assert result["resid_bt709"] < result["resid_linear"]


def test_classify_gamma_insufficient_bins():
    mean = np.array([5.0, 8.0])  # both below the toe cutoff
    variance = np.array([1.0, 2.0])
    result = classify_gamma(mean, variance)
    assert result["call"] == "AMBIGUOUS"
    assert np.isnan(result["resid_linear"])


def test_classify_gamma_excludes_toe():
    """Bins below the toe cutoff do not enter the verdict mask."""
    mean = np.array([3.0, 6.0, 20.0, 60.0, 120.0, 200.0])
    variance = 0.5 * mean
    result = classify_gamma(mean, variance, toe_cutoff=12.0)
    assert not result["ok_mask"][0]
    assert not result["ok_mask"][1]
    assert result["ok_mask"][2:].all()


# ---------------------------------------------------------------------------
# intensity_histogram shade_regions
# ---------------------------------------------------------------------------


def test_shade_regions_adds_spans():
    img = np.random.default_rng(3).integers(0, 256, size=(32, 32)).astype(np.uint8)
    fig, ax = plt.subplots()
    n_before = len(ax.patches)
    intensity_histogram(
        img,
        intensity_range=(0, 255),
        ax=ax,
        shade_regions=[(-0.5, 15.5, "#2166ac"), (235.5, 255.5, "#b2182b")],
    )
    assert len(ax.patches) >= n_before + 2
    plt.close(fig)


def test_shade_regions_none_is_noop():
    img = np.random.default_rng(4).integers(0, 256, size=(32, 32)).astype(np.uint8)
    fig, ax = plt.subplots()
    intensity_histogram(img, intensity_range=(0, 255), ax=ax, shade_regions=None)
    plt.close(fig)
