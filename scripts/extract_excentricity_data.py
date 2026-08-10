"""Extract the strip-misalignment experiments into one committed `.npz`.

Source is the raw NUMISHEET capture in `_excentricity_data/real_numisheet/` (926 MB of
per-channel arrays, not committed). Only the axial punch force of the deep-drawing module
is modelled, so this pulls that one channel out and reproduces the preprocessing of the
research code (`numisheet/dataset.py`, `TimeSeriesDataset`) exactly:

1. Seven folders, one per infeed level, each `(50, ~8300)` -- 50 strokes of one
   uninterrupted run. Raw lengths differ per series (8295..8299), which is more than four
   distinct values, so the loader resamples every stroke to `TARGET_LENGTH` rather than to
   the mean length.
2. The first stroke of every series is dropped as a warm-up transient: 7 x 49 = 343.
3. Normalization divides the whole set by the *median of the per-stroke maxima* -- one
   scalar for the entire dataset, not a per-stroke or per-series scale.

Notch filtering is deliberately not applied: the paper's headline numbers come from the
unfiltered path.

Run from the repo root: `uv run python scripts/extract_excentricity_data.py`
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import interpolate

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "_excentricity_data" / "real_numisheet"
OUT = ROOT / "data" / "excentricity.npz"

SENSOR = "K2_Ch2_Mod2AI4"  # axial punch force of the deep-drawing module
TARGET_LENGTH = int(8296 * 0.1099) + 1  # 912, the sim-matched time base

# The capture runs at ~24 kHz, so a segmented stroke spans 8296 / 24000 = 0.3457 s. The
# paper reports the plateau at 0.255..0.294 s, which fixes where the segment starts on the
# press cycle; carrying that offset here keeps the dashboard's time axis comparable with
# the paper's figures rather than starting an arbitrary count at zero.
STROKE_SECONDS = 8296 / 24_000
PLATEAU_START, PLATEAU_END = 352, 455  # indices into the 912-point trace
PLATEAU_START_SECONDS = 0.255


def _resample(series: np.ndarray, target_length: int) -> np.ndarray:
    """Linear interpolation onto a common length -- `TimeSeriesDataset._downsample`."""
    if len(series) == target_length:
        return series
    interpolate_at = interpolate.interp1d(np.arange(len(series)), series, kind="linear")
    return interpolate_at(np.linspace(0, len(series) - 1, target_length))


def main() -> None:
    folders = sorted(f for f in SOURCE.iterdir() if f.is_dir())
    if not folders:
        raise SystemExit(f"no experiment folders under {SOURCE}")

    curves, labels, stroke_index = [], [], []
    for folder in folders:
        # "Messergebnisse_Ex_60_15mm_50h_100hm" -> 15, i.e. hundredths of a millimetre.
        label = int(folder.name.split("_")[3][:2])
        raw = np.load(folder / "npy" / f"{SENSOR}_segmented.npy")
        resampled = np.array([_resample(row, TARGET_LENGTH) for row in raw])[1:]

        curves.append(resampled)
        labels.append(np.full(len(resampled), label))
        stroke_index.append(np.arange(1, len(resampled) + 1))
        print(f"{folder.name}: {raw.shape} -> {resampled.shape}, label {label}")

    data = np.concatenate(curves)
    # One scalar for the whole set. Kept alongside the curves: the raw capture is already
    # in kN, so multiplying back by it is what lets the dashboard plot physical force and
    # quote a slope in kN/s, while the models see exactly what the research code fed them.
    force_scale = float(np.median(np.max(data, axis=1)))
    data /= force_scale

    dt = STROKE_SECONDS / TARGET_LENGTH
    time = PLATEAU_START_SECONDS - PLATEAU_START * dt + np.arange(TARGET_LENGTH) * dt

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT,
        curves=data.astype(np.float32),
        labels=np.concatenate(labels).astype(np.int16),
        stroke_index=np.concatenate(stroke_index).astype(np.int16),
        time=time.astype(np.float32),
        plateau=np.array([PLATEAU_START, PLATEAU_END], dtype=np.int16),
        force_scale=np.float32(force_scale),
    )
    print(f"\n{OUT.relative_to(ROOT)}: {data.shape}, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
