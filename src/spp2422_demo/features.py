"""Handcrafted shape descriptors for one normalized-force curve.

Ported from the research code (`predict_roughness.py`) that this demo draws its data
from. Curves live on the shared 0..1 event-time axis with 500 samples; the descriptors
cover essentially every generic way two such curves can differ in shape, not just the
textbook few:

- Amplitude: peak height, area, start/end value, end-of-window undershoot.
- Timing/steepness, position-invariant: rise/fall durations between the 10th and 90th
  percentile of peak height. Unlike `peak_x` or a fixed-window slope this describes
  *how fast* the curve moves without depending on *where* it peaks, which matters
  because real ironing's peak position is weak evidence.
- Local shape at the peak: curvature from a local quadratic fit.
- Global shape: skewness and kurtosis of the curve read as a distribution over time,
  plus total variation as a generic roughness measure.
- Whole-segment trends: the curve is split into rise / plateau / fall around the peak,
  and each segment gets a linear-fit slope and R^2. Added because the real deep-drawing
  plateau is not flat -- it declines, close to linearly.
- Region variance: the standard deviation within each tenth of the stroke. Shape
  descriptors say where the curve goes; this says how steadily it gets there.
- `burst=True` (ironing only -- deep drawing never rings) adds the contact-transition
  burst's position and amplitude (residual energy against a median-filtered copy,
  searched only before the curve's own peak), the oscillation energy per tenth, and the
  first draw-down: ironing rises to a shoulder, dips through the burst, then climbs to
  its real peak, and the depth and steepness of that dip are wear-sensitive.
"""

from __future__ import annotations

import hashlib

import numpy as np
from scipy.signal import medfilt

PERCENTILES = (0.1, 0.25, 0.5, 0.75, 0.9)
BANDS = 10  # slices of event time that the variance features are measured over

# Extracting features for all 4500 curves takes a few seconds, and both the validation
# sweeps and the dashboard ask for the same matrices over and over. Keyed on the content
# of the input arrays, so it can never return features for the wrong curves.
_CACHE: dict[tuple, tuple[np.ndarray, list[str]]] = {}
_CACHE_LIMIT = 64


def _digest(array: np.ndarray | None) -> bytes | None:
    if array is None:
        return None
    contiguous = np.ascontiguousarray(array)
    return hashlib.blake2b(contiguous.view(np.uint8), digest_size=16).digest()


def _percentile_crossings(curve: np.ndarray, peak_i: int, peak_v: float) -> dict[str, float]:
    """When the curve first reaches / drops back below each fraction of its peak."""
    n = len(curve)
    out: dict[str, float] = {}
    rise = curve[: peak_i + 1]
    for p in PERCENTILES:
        idx = np.flatnonzero(rise >= p * peak_v)
        out[f"rise_t{int(p * 100)}"] = (idx[0] / n) if len(idx) else 0.0
    fall = curve[peak_i:]
    for p in PERCENTILES:
        idx = np.flatnonzero(fall <= p * peak_v)
        out[f"fall_t{int(p * 100)}"] = ((peak_i + idx[0]) / n) if len(idx) else 1.0
    out["rise_duration_10_90"] = out["rise_t90"] - out["rise_t10"]
    out["fall_duration_90_10"] = out["fall_t10"] - out["fall_t90"]
    return out


def _shape_moments(curve: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Skewness and kurtosis of the curve as a distribution over event time.

    Weight = force, mass = when it happens. A long tail after the peak gives positive
    skew; a peakier-than-Gaussian curve gives positive excess kurtosis.
    """
    weights = np.clip(curve - curve.min(), 0.0, None)
    total = weights.sum()
    if total <= 0:
        return {"skewness": 0.0, "kurtosis": 0.0}
    weights = weights / total
    mean_x = float(np.sum(x * weights))
    var_x = float(np.sum(weights * (x - mean_x) ** 2))
    if var_x <= 0:
        return {"skewness": 0.0, "kurtosis": 0.0}
    skew = float(np.sum(weights * (x - mean_x) ** 3) / var_x**1.5)
    kurt = float(np.sum(weights * (x - mean_x) ** 4) / var_x**2 - 3.0)
    return {"skewness": skew, "kurtosis": kurt}


def _plateau_bounds(curve: np.ndarray, peak_i: int, threshold: float = 0.85) -> tuple[int, int]:
    """[lo, hi) of the contiguous region around the peak staying within `threshold` of it.

    Finds the genuine flat top of a broad curve (deep drawing) and collapses to a narrow
    band on a sharply peaked one (ironing) -- correct in both cases, because it adapts to
    the curve's own shape instead of assuming one fixed x-range fits both channels.
    """
    peak_v = curve[peak_i]
    above = curve >= threshold * peak_v
    lo, hi, n = peak_i, peak_i, len(curve)
    while lo > 0 and above[lo - 1]:
        lo -= 1
    while hi < n - 1 and above[hi + 1]:
        hi += 1
    return lo, hi + 1


def _segment_stats(
    curve: np.ndarray, x: np.ndarray, lo: int, hi: int, prefix: str
) -> dict[str, float]:
    """Linear-fit slope and R^2 over `curve[lo:hi]`.

    A whole-segment trend rather than a point-sampled derivative: the slope says how the
    segment moves on average, R^2 how much of its shape a straight line explains.
    """
    if hi - lo < 3:
        return {f"{prefix}_slope": 0.0, f"{prefix}_r2": 0.0}
    xs, ys = x[lo:hi], curve[lo:hi]
    slope, intercept = np.polyfit(xs, ys, 1)
    fit = slope * xs + intercept
    ss_res = float(np.sum((ys - fit) ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {f"{prefix}_slope": float(slope), f"{prefix}_r2": r2}


def _first_drawdown(
    smooth: np.ndarray, x: np.ndarray, burst_i: int, peak_i: int
) -> dict[str, float]:
    """The dip an ironing stroke makes before its real peak.

    The force climbs to a shoulder, the tool takes contact -- the burst -- and the force
    drops away before climbing again to the true peak. Measured on the burst-suppressed
    curve so the oscillation itself cannot stand in for the shoulder or the dip.
    """
    shoulder_i = int(smooth[: max(burst_i, 1)].argmax())
    stop = max(peak_i, shoulder_i + 2)
    dip_i = shoulder_i + int(smooth[shoulder_i:stop].argmin())
    shoulder_v, dip_v, peak_v = (
        float(smooth[shoulder_i]),
        float(smooth[dip_i]),
        float(smooth[peak_i]),
    )
    depth = shoulder_v - dip_v
    width = float(x[dip_i] - x[shoulder_i])
    return {
        "shoulder_v": shoulder_v,
        "shoulder_x": float(x[shoulder_i]),
        "dip_v": dip_v,
        "dip_x": float(x[dip_i]),
        "drawdown_depth": depth,
        "drawdown_width": width,
        "drawdown_rate": depth / max(width, 1e-6),
        "drawdown_relative": depth / max(peak_v, 1e-6),
        "peak_minus_shoulder": peak_v - shoulder_v,
    }


def curve_features(
    curve: np.ndarray, *, burst: bool = False, peak_ref: np.ndarray | None = None
) -> dict[str, float]:
    """Shape descriptors for one normalized-force curve. See the module docstring.

    `peak_ref` (real ironing only): the curve's own argmax is not a reliable peak
    location there -- the contact-transition burst is taller than the bell-shaped arc's
    genuine peak for ~85% of real ironing strokes. When given, the peak *index* is
    located on `peak_ref` instead, while the amplitude is still read off `curve` so it
    stays the genuine unsmoothed value.
    """
    curve = np.asarray(curve, dtype=np.float64)
    n = len(curve)
    x = np.linspace(0.0, 1.0, n)
    peak_i = int(np.asarray(peak_ref).argmax()) if peak_ref is not None else int(curve.argmax())
    peak_v = float(curve[peak_i])

    features = {
        "peak_height": peak_v,
        "peak_x": peak_i / n,
        "area": float(curve.mean()),
        "start_val": float(curve[0]),
        "end_val": float(curve[-1]),
        "undershoot": float(curve[-int(0.1 * n) :].min()),
        "onset_slope": float((curve[int(0.05 * n)] - curve[0]) / 0.05),
        "total_variation": float(np.abs(np.diff(curve)).sum()),
    }

    # Slope profile at several fixed points, not just one rise/fall window.
    for center in (0.2, 0.35, 0.5, 0.65, 0.8):
        lo, hi = max(0, int((center - 0.05) * n)), min(n - 1, int((center + 0.05) * n))
        features[f"slope_{int(center * 100)}"] = float((curve[hi] - curve[lo]) / (x[hi] - x[lo]))

    # Local curvature at the peak: leading coefficient of a quadratic fit around it.
    window = max(5, n // 25)
    lo, hi = max(0, peak_i - window), min(n, peak_i + window)
    if hi - lo >= 3:
        features["peak_curvature"] = float(np.polyfit(x[lo:hi] - x[peak_i], curve[lo:hi], 2)[0])
    else:
        features["peak_curvature"] = 0.0

    features.update(_percentile_crossings(curve, peak_i, peak_v))
    features.update(_shape_moments(curve, x))

    plateau_lo, plateau_hi = _plateau_bounds(curve, peak_i)
    features.update(_segment_stats(curve, x, 0, plateau_lo, "rise_seg"))
    features.update(_segment_stats(curve, x, plateau_lo, plateau_hi, "plateau_seg"))
    features.update(_segment_stats(curve, x, plateau_hi, n, "fall_seg"))

    half = peak_v / 2
    above = np.flatnonzero(curve > half)
    features["width_half"] = float((above[-1] - above[0]) / n) if len(above) else 0.0

    edges = np.linspace(0, n, BANDS + 1).astype(int)
    bands = list(zip(edges[:-1], edges[1:], strict=True))
    for i, (lo, hi) in enumerate(bands):
        features[f"band_std_{i}"] = float(curve[lo:hi].std())

    if burst:
        smooth = np.asarray(peak_ref, dtype=np.float64) if peak_ref is not None else None
        residual = curve - (smooth if smooth is not None else medfilt(curve, kernel_size=9))
        energy = np.convolve(residual**2, np.ones(9), mode="same")
        lo = max(1, round(0.05 * n))
        hi = max(lo + 1, peak_i)
        burst_i = lo + int(np.argmax(energy[lo:hi]))
        features["burst_x"] = burst_i / n
        features["burst_amplitude"] = float(np.sqrt(energy[lo:hi].max()))
        features["residual_energy"] = float(np.sqrt(np.mean(residual**2)))
        for i, (band_lo, band_hi) in enumerate(bands):
            features[f"band_osc_{i}"] = float(np.abs(residual[band_lo:band_hi]).mean())
        features.update(
            _first_drawdown(smooth if smooth is not None else curve, x, burst_i, peak_i)
        )

    return features


def feature_matrix(
    curves: np.ndarray, *, burst: bool = False, peak_ref: np.ndarray | None = None
) -> tuple[np.ndarray, list[str]]:
    """Stack `curve_features` over many curves. Returns (matrix, feature names)."""
    key = (_digest(curves), _digest(peak_ref), burst)
    if key in _CACHE:
        return _CACHE[key]

    refs = peak_ref if peak_ref is not None else [None] * len(curves)
    rows = [curve_features(c, burst=burst, peak_ref=r) for c, r in zip(curves, refs, strict=True)]
    names = list(rows[0].keys())
    result = (np.array([[row[name] for name in names] for row in rows]), names)

    if len(_CACHE) >= _CACHE_LIMIT:
        del _CACHE[next(iter(_CACHE))]
    _CACHE[key] = result
    return result
