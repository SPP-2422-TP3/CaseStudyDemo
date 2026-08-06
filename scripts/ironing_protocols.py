"""Why ironing fails the unseen-run test, and whether the upstream state rescues it.

The dashboard reports leave-one-run-out, where ironing sits at chance. The failure is
structured rather than noisy: withholding run A1·T3 leaves A2·T3 and A3·T3 in training,
so a test stroke's nearest neighbours share the upstream deep-drawing state but carry
*different* ironing labels, and the model answers with theirs.

Deep drawing predicts its own state at 99.6-100% on an unseen run, so that upstream state
is available in production -- from the same press stroke, since `real_deep_drawing` and
`real_ironing` are extracted row for row. This script asks whether handing it to the
ironing model fixes the split, two ways: appended as a feature, or subtracted from the
descriptors as a known effect.

    uv run python scripts/ironing_protocols.py [--permutations 200]

Prints accuracy per variant and, with --permutations, a p-value against runs relabelled
at random -- nine runs make for a wide null, so a point estimate alone means little.

The upstream model is refitted per fold, so it never sees the held-out run either. Its
predictions for *training* rows are in-sample; T is ~100% accurate either way, and the
per-fold upstream accuracy is printed so that assumption stays visible.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable

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

# A variant builds the design matrix for one fold, given the (possibly permuted) labels.
Variant = Callable[[int, np.ndarray], np.ndarray]


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED))


def accuracy(build: Variant, labels: np.ndarray, folds: list[np.ndarray]) -> float:
    scores = []
    for i, held in enumerate(folds):
        matrix = build(i, labels)
        fitted = model().fit(matrix[~held], labels[~held])
        scores.append(np.mean(fitted.predict(matrix[held]) == labels[held]))
    return float(np.mean(scores))


def upstream_probabilities(
    folds: list[np.ndarray], t_labels: np.ndarray
) -> tuple[np.ndarray, list[float]]:
    """Deep-drawing state per stroke, from a model that never saw the held-out run.

    Returns (n_folds, n_strokes, n_levels) probabilities and the per-fold accuracy on the
    held-out run. Computed once: a permutation relabels ironing, never deep drawing, so
    these models would otherwise be refitted thousands of times for nothing.
    """
    upstream = load_station("deep_drawing")
    features, _ = feature_matrix(upstream.curves, burst=False, peak_ref=None)

    probabilities, scores = [], []
    for held in folds:
        fitted = model().fit(features[~held], upstream.labels[~held])
        probabilities.append(fitted.predict_proba(features))
        scores.append(float(np.mean(fitted.predict(features[held]) == t_labels[held])))
    return np.array(probabilities), scores


def residualise(
    features: np.ndarray,
    a_labels: np.ndarray,
    t_labels: np.ndarray,
    run_of_row: np.ndarray,
    train_mask: np.ndarray,
) -> np.ndarray:
    """Remove the upstream state's contribution to each descriptor.

    Reads every feature as `x = mu + alpha(A) + beta(T)` and fits both effects by least
    squares over the run-level means of the training runs, then subtracts `beta(T)`. A
    two-way fit rather than centring within each T group: the held-out run's group has
    only two runs left and they carry two of the three ironing levels, so a plain group
    mean would be biased exactly where it matters. Eight of nine cells keep the additive
    model identifiable, and only training labels enter the fit.
    """
    runs = np.unique(run_of_row[train_mask])
    means = np.array([features[run_of_row == run].mean(axis=0) for run in runs])
    own = np.array([a_labels[run_of_row == run][0] for run in runs])
    other = np.array([t_labels[run_of_row == run][0] for run in runs])

    # Treatment coding against level 1: intercept, A2..An, T2..Tn.
    design = np.column_stack(
        [np.ones(len(runs))]
        + [(own == level).astype(float) for level in LEVELS[1:]]
        + [(other == level).astype(float) for level in LEVELS[1:]]
    )
    coefficients, *_ = np.linalg.lstsq(design, means, rcond=None)

    beta = np.zeros((max(LEVELS) + 1, features.shape[1]))
    for offset, level in enumerate(LEVELS[1:], start=len(LEVELS)):
        beta[level] = coefficients[offset]
    return features - beta[t_labels]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutations", type=int, default=0)
    args = parser.parse_args()

    data = load_station("ironing")
    a, t = data.labels, data.other_labels
    features, names = feature_matrix(data.curves, burst=True, peak_ref=data.peak_ref)
    subset = features[
        :, [i for i, name in enumerate(names) if name.startswith(VARIANCE_AND_DRAWDOWN)]
    ]

    # Both stations are extracted row for row from one press stroke; the chain depends on it.
    upstream = load_station("deep_drawing")
    assert np.array_equal(upstream.labels, t), "deep drawing rows are not the ironing rows"
    assert np.array_equal(upstream.other_labels, a), "station labels disagree"

    folds = [(a == own) & (t == other) for own, other in data.runs()]
    run_of_row = np.full(len(a), -1)
    for i, held in enumerate(folds):
        run_of_row[held] = i

    predicted_t, upstream_scores = upstream_probabilities(folds, t)
    print(f"ironing, {len(a)} strokes, chance = {1 / len(LEVELS):.2%}")
    print(
        f"upstream deep-drawing model on its held-out run: "
        f"{np.mean(upstream_scores):.2%} mean, {min(upstream_scores):.2%} worst fold"
    )

    variants: dict[str, Variant] = {
        "ironing descriptors alone": lambda fold, labels: features,
        "variance and draw-down only": lambda fold, labels: subset,
        "+ predicted upstream T": lambda fold, labels: np.hstack([features, predicted_t[fold]]),
        "upstream T-effect removed": lambda fold, labels: residualise(
            features, labels, t, run_of_row, ~folds[fold]
        ),
    }

    # A permutation reassigns which run carries which ironing level, leaving the runs and
    # the folds intact -- the null of "these descriptors say nothing about wear".
    run_levels = np.array([own for own, _ in data.runs()])
    rng = np.random.default_rng(SEED)

    print("\nleave one run out")
    for label, build in variants.items():
        score = accuracy(build, a, folds)
        line = f"  {label:28s} {score:7.2%}"
        if args.permutations:
            null = np.array(
                [
                    accuracy(build, rng.permutation(run_levels)[run_of_row], folds)
                    for _ in range(args.permutations)
                ]
            )
            p = (np.sum(null >= score) + 1) / (args.permutations + 1)
            line += f"   null {null.mean():.2%} +- {null.std():.2%}   p = {p:.4f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
