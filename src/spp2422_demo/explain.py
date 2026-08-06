"""Where in the stroke did the model find its evidence?

Both methods return one attribution value per time sample, positive where the region
supports the predicted level and negative where it argues against it. That makes them
directly overlayable on the force curve, which is the point: an operator should be able
to look at the highlighted stretch of the stroke and recognise the physical event.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d

from .data import EVENT_TIME, LEVELS
from .models import CnnModel, WearModel

OCCLUSION_WINDOW = 25  # samples, 5% of the stroke
OCCLUSION_STRIDE = 5
IG_STEPS = 64
# Gradient attributions are noisy sample to sample, and the question being asked is
# "which part of the stroke", not "which single sample". Smoothing over 4% of the stroke
# keeps the regions and drops the speckle -- the same reason attribution maps are
# normally shown blurred.
SMOOTHING = 21


@dataclass(frozen=True)
class Attribution:
    """Per-sample evidence for one prediction, plus how to describe it in words."""

    values: np.ndarray  # (500,) signed, scaled so max |value| is 1
    method: str
    level: int

    @property
    def focus(self) -> tuple[float, float]:
        """The contiguous stretch of event time carrying most of the positive evidence.

        Found by smoothing the positive part and taking the widest run above half its
        maximum, so a single spike does not stand in for a broad region.
        """
        positive = uniform_filter1d(np.clip(self.values, 0, None), size=OCCLUSION_WINDOW)
        if not positive.any():
            return 0.0, 1.0
        above = np.flatnonzero(positive >= 0.5 * positive.max())
        # Widest contiguous run of samples above the threshold.
        splits = np.split(above, np.flatnonzero(np.diff(above) > 1) + 1)
        widest = max(splits, key=len)
        return float(EVENT_TIME[widest[0]]), float(EVENT_TIME[widest[-1]])

    def summary(self, station_name: str) -> str:
        start, end = self.focus
        return (
            f"The model reads the {station_name.lower()} stroke between "
            f"x = {start:.2f} and x = {end:.2f} of event time as the decisive evidence "
            f"for this level."
        )


def _normalise(values: np.ndarray, level: int, method: str) -> Attribution:
    values = uniform_filter1d(values, size=SMOOTHING, mode="nearest")
    peak = np.abs(values).max()
    return Attribution(values / peak if peak > 0 else values, method, level)


def occlusion(
    model: WearModel, curve: np.ndarray, peak_ref: np.ndarray | None, level: int
) -> Attribution:
    """Slide a window across the stroke, flatten it, and watch the confidence move.

    Flattening means replacing the window with a straight line between its endpoints:
    the curve stays continuous, so the model is not handed an artificial step edge, but
    whatever shape lived inside the window is gone. Works for any model.
    """
    curve = np.asarray(curve, dtype=np.float32)
    n = len(curve)
    class_index = LEVELS.index(level)
    single_ref = peak_ref[None, :] if peak_ref is not None else None
    baseline = float(model.predict_proba(curve[None, :], single_ref)[0, class_index])

    starts = np.arange(0, n - OCCLUSION_WINDOW + 1, OCCLUSION_STRIDE)
    variants = np.repeat(curve[None, :], len(starts), axis=0)
    for row, start in enumerate(starts):
        stop = start + OCCLUSION_WINDOW
        variants[row, start:stop] = np.linspace(curve[start], curve[stop - 1], OCCLUSION_WINDOW)

    refs = np.repeat(peak_ref[None, :], len(starts), axis=0) if peak_ref is not None else None
    dropped = baseline - model.predict_proba(variants, refs)[:, class_index]

    # Each window's effect is spread back over the samples it covered, so overlapping
    # windows accumulate and the result lines up sample-for-sample with the curve.
    values = np.zeros(n)
    counts = np.zeros(n)
    for row, start in enumerate(starts):
        values[start : start + OCCLUSION_WINDOW] += dropped[row]
        counts[start : start + OCCLUSION_WINDOW] += 1
    return _normalise(values / np.maximum(counts, 1), level, "Occlusion sensitivity")


def integrated_gradients(model: CnnModel, curve: np.ndarray, level: int) -> Attribution:
    """Accumulate the gradient along a straight path from the mean curve to this one.

    The baseline is the training-set mean, so the attribution answers "what makes *this*
    stroke different from a typical one", rather than re-explaining the shape every
    stroke shares.
    """
    class_index = LEVELS.index(level)
    x = model._standardise(np.asarray(curve, dtype=np.float32)[None, :])
    baseline = torch.zeros_like(x)  # standardised space: zero is the training mean

    alphas = torch.linspace(1.0 / IG_STEPS, 1.0, IG_STEPS).view(-1, 1, 1)
    path = (baseline + alphas * (x - baseline)).requires_grad_(True)
    model.net.zero_grad()
    torch.log_softmax(model.net(path), dim=1)[:, class_index].sum().backward()

    average_gradient = path.grad.mean(dim=0)
    values = (average_gradient * (x - baseline)[0]).detach().numpy().ravel()
    return _normalise(values, level, "Integrated gradients")


def explain(
    model: WearModel, curve: np.ndarray, peak_ref: np.ndarray | None, level: int
) -> Attribution:
    """Best available attribution for this model: gradients where we have them."""
    if isinstance(model, CnnModel):
        return integrated_gradients(model, curve, level)
    return occlusion(model, curve, peak_ref, level)
