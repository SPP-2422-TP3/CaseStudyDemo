"""Why ironing fails the unseen-run test, and what it does manage instead.

The dashboard reports leave-one-run-out, where ironing sits at chance. That is a real
result but not the whole story: the held-out run's two neighbours at the same upstream
deep-drawing state stay in training carrying *different* ironing labels, so the model
matches a test stroke to them and answers with their label. Grouping the folds by
upstream state instead removes that confound.

    uv run python scripts/ironing_protocols.py [--permutations 300]

Prints, for each feature set and both protocols, the accuracy and -- with
--permutations -- a p-value against relabelled runs.
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from spp2422_demo.data import LEVELS, load_station
from spp2422_demo.features import feature_matrix
from spp2422_demo.models import SEED

# The descriptors that read how noisy the stroke is and how deep its first dip goes,
# as opposed to where the curve goes.
VARIANCE_AND_DRAWDOWN = ("band_std_", "band_osc_", "shoulder_", "dip_", "drawdown_", "peak_minus_")


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def accuracy(X: np.ndarray, labels: np.ndarray, folds: list[np.ndarray]) -> float:
    scores = []
    for held in folds:
        fitted = model().fit(X[~held], labels[~held])
        scores.append(np.mean(fitted.predict(X[held]) == labels[held]))
    return float(np.mean(scores))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=0)
    args = parser.parse_args()

    data = load_station("ironing")
    a, t = data.labels, data.other_labels
    X, names = feature_matrix(data.curves, burst=True, peak_ref=data.peak_ref)
    selected = [i for i, name in enumerate(names) if name.startswith(VARIANCE_AND_DRAWDOWN)]

    sets = {
        "all descriptors": X,
        "variance + draw-down only": X[:, selected],
    }
    protocols = {
        "leave one run out": [(a == own) & (t == other) for own, other in data.runs()],
        "leave one upstream T out": [t == level for level in np.unique(t)],
    }

    # Which run carries which ironing level is what the permutation shuffles.
    run_of_row = np.full(len(a), -1)
    for i, (own, other) in enumerate(data.runs()):
        run_of_row[(a == own) & (t == other)] = i
    run_levels = np.array([own for own, _ in data.runs()])
    rng = np.random.default_rng(SEED)

    print(f"ironing, {len(a)} strokes, chance = {1 / len(LEVELS):.2%}")
    for protocol, folds in protocols.items():
        print(f"\n{protocol}")
        for label, matrix in sets.items():
            score = accuracy(matrix, a, folds)
            line = f"  {label:28s} {score:7.2%}"
            if args.permutations:
                null = np.array(
                    [
                        accuracy(matrix, rng.permutation(run_levels)[run_of_row], folds)
                        for _ in range(args.permutations)
                    ]
                )
                p = (np.sum(null >= score) + 1) / (args.permutations + 1)
                line += f"   null {null.mean():.2%} +- {null.std():.2%}   p = {p:.4f}"
            print(line)


if __name__ == "__main__":
    main()
