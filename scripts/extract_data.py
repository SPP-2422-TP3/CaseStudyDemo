"""One-time extraction of the demo's dataset from the research pipeline's output.

The research repo (see README) downloads ~14 GB of raw measurements and simulation
exports and reduces them to `data/prepared.hdf5`. This demo only needs the normalized
curves and their labels, which compress to a few MB -- small enough to commit, so the
app runs anywhere without the big download.

Usage:

    uv run python scripts/extract_data.py [path/to/prepared.hdf5]
"""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "context-material" / "sheet-metal-synthetic" / "data" / "prepared.hdf5"
DESTINATION = ROOT / "data" / "curves.npz"

# Curves are normalized force -- O(1) values used for display and shape features, where
# float16's ~3 decimal digits are ample and halve the committed file.
CURVE_DTYPE = np.float16


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SOURCE
    if not source.exists():
        raise SystemExit(
            f"{source} not found. Run scripts/download.sh and scripts/prepare.py in the "
            "research repo first (see README)."
        )

    out: dict[str, np.ndarray] = {}
    with h5py.File(source, "r") as f:
        real = f["real"]
        out["real_deep_drawing"] = real["deep_drawing/normalized"][:].astype(CURVE_DTYPE)
        out["real_ironing"] = real["ironing/normalized"][:].astype(CURVE_DTYPE)
        burst_suppressed = real["ironing/normalized_burst_suppressed"]
        out["real_ironing_burst_suppressed"] = burst_suppressed[:].astype(CURVE_DTYPE)

        meta = real["meta_data"]
        out["T"] = meta["Deep Drawing"][:].astype(np.int8)
        out["A"] = meta["Ironing"][:].astype(np.int8)
        out["V"] = meta["V"][:].astype(np.int8)
        # Position of the stroke within its 500-stroke production run: the axis along
        # which tool bedding-in drift shows up, and what the demo's live stream walks.
        out["stroke_index"] = meta["index"][:].astype(np.int16)

        sim = f["simulated"]
        out["sim_deep_drawing"] = sim["deep_drawing/normalized"][:].astype(CURVE_DTYPE)
        out["sim_dd_mu"] = sim["deep_drawing/meta_data/mu"][:].astype(np.float32)
        out["sim_ironing"] = sim["ironing/normalized"][:].astype(CURVE_DTYPE)
        out["sim_ir_mu"] = sim["ironing/meta_data/mu"][:].astype(np.float32)
        out["sim_ir_mu_dd"] = sim["ironing/meta_data/mu_deep_drawing"][:].astype(np.float32)

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DESTINATION, **out)

    print(f"==> wrote {DESTINATION.relative_to(ROOT)} ({DESTINATION.stat().st_size / 1e6:.1f} MB)")
    for key, value in out.items():
        print(f"    {key:32s} {str(value.shape):14s} {value.dtype}")


if __name__ == "__main__":
    main()
