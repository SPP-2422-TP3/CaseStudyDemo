"""The four selectable wear-state classifiers.

All of them map one normalized-force curve to a probability over wear levels 1..3, so
the dashboard can swap between them without knowing what is inside. Two read handcrafted
shape features, one reads the raw curve, and one reads both.

Every model offers the same pair of entry points twice over: `fit`/`predict_proba` take
curves and derive whatever else they need, which is what the dashboard calls for a single
stroke; `fit_matrix`/`predict_proba_matrix` additionally take a feature matrix computed
once outside. The validation sweeps refit the same models over many splits of one curve
set, and re-extracting features per split is what makes that unaffordable.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data import LEVELS
from .features import feature_matrix

SEED = 42


class WearModel(Protocol):
    """What the dashboard needs from a classifier."""

    key: str
    name: str
    description: str

    def fit(self, curves: np.ndarray, labels: np.ndarray, peak_ref: np.ndarray | None) -> None: ...

    def predict_proba(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        """(n, 3) probabilities over wear levels 1..3."""
        ...

    def fit_matrix(self, curves: np.ndarray, features: np.ndarray, labels: np.ndarray) -> None:
        """Fit from features extracted once by the caller. Ignore whichever input the
        model does not read."""
        ...

    def predict_proba_matrix(self, curves: np.ndarray, features: np.ndarray) -> np.ndarray: ...


class FeatureModel:
    """A scikit-learn estimator on top of the handcrafted shape descriptors."""

    def __init__(self, key: str, name: str, description: str, estimator, *, burst: bool) -> None:
        self.key = key
        self.name = name
        self.description = description
        self.estimator = estimator
        self.burst = burst
        self.feature_names: list[str] = []

    def _features(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        matrix, names = feature_matrix(curves, burst=self.burst, peak_ref=peak_ref)
        self.feature_names = names
        return matrix

    def fit(self, curves: np.ndarray, labels: np.ndarray, peak_ref: np.ndarray | None) -> None:
        self.estimator.fit(self._features(curves, peak_ref), labels)

    def predict_proba(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        return self.estimator.predict_proba(self._features(curves, peak_ref))

    def fit_matrix(self, curves: np.ndarray, features: np.ndarray, labels: np.ndarray) -> None:
        del curves  # the features are the whole input
        self.estimator.fit(features, labels)

    def predict_proba_matrix(self, curves: np.ndarray, features: np.ndarray) -> np.ndarray:
        del curves
        return self.estimator.predict_proba(features)


EMBEDDING = 64  # width of the curve encoder's output


def _conv_block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(),
        nn.MaxPool1d(4),
    )


def _curve_encoder() -> nn.Sequential:
    """Three conv blocks and global pooling: one curve in, one EMBEDDING-wide vector out."""
    return nn.Sequential(
        _conv_block(1, 16, 9),
        _conv_block(16, 32, 7),
        _conv_block(32, 64, 5),
        nn.AdaptiveAvgPool1d(1),
        nn.Flatten(),
    )


class _Net(nn.Module):
    """Small 1-D CNN: the curve encoder with a linear head on top."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            _curve_encoder(), nn.Dropout(0.2), nn.Linear(EMBEDDING, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class _HybridNet(nn.Module):
    """Two encoders joined below one head.

    The curve goes through the same convolutional stack as the CNN; the handcrafted
    features go through a small dense layer. Concatenating the two before the head lets
    the network use whichever is informative for a given stroke -- and lets the features
    supply what a short receptive field never sees, such as how the whole stroke is
    positioned in event time.
    """

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.curve = _curve_encoder()
        self.tabular = nn.Sequential(nn.Linear(n_features, 32), nn.BatchNorm1d(32), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(EMBEDDING + 32, n_classes))

    def forward(self, curve: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.curve(curve), self.tabular(features)], dim=1))


def _train_net(net: nn.Module, inputs: tuple[torch.Tensor, ...], y: torch.Tensor, epochs: int):
    """AdamW under a cosine schedule. Shared so the CNN and the hybrid train identically
    and their scores stay comparable."""
    optimiser = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(SEED)

    net.train()
    for _ in range(epochs):
        for batch in torch.randperm(len(y), generator=generator).split(128):
            if len(batch) < 2:  # batch norm has no variance to work with
                continue
            optimiser.zero_grad()
            loss_fn(net(*(tensor[batch] for tensor in inputs)), y[batch]).backward()
            optimiser.step()
        schedule.step()
    net.eval()


def _standardise(values: np.ndarray, mean, std) -> torch.Tensor:
    return torch.from_numpy(
        ((np.asarray(values, dtype=np.float32) - mean) / std).astype(np.float32)
    )


class CnnModel:
    """1-D CNN on the raw 500-sample curve -- no handcrafted features at all."""

    key = "cnn"
    name = "1-D CNN"
    description = "Convolutional network reading the raw curve, no handcrafted features."

    def __init__(self, epochs: int = 60) -> None:
        self.epochs = epochs
        # Seeded here rather than in fit(): the initial weights are as much a part of
        # the trained result as the batch order is.
        torch.manual_seed(SEED)
        self.net = _Net(len(LEVELS))
        self.mean = 0.0
        self.std = 1.0

    def standardise(self, curves: np.ndarray) -> torch.Tensor:
        """Curves as the network sees them. Public: `explain` integrates gradients here."""
        return _standardise(curves, self.mean, self.std).unsqueeze(1)

    def fit(self, curves: np.ndarray, labels: np.ndarray, peak_ref: np.ndarray | None) -> None:
        del peak_ref  # the raw curve is the whole input
        self.fit_matrix(curves, np.empty((len(curves), 0)), labels)

    def fit_matrix(self, curves: np.ndarray, features: np.ndarray, labels: np.ndarray) -> None:
        del features
        torch.manual_seed(SEED)
        self.mean = float(curves.mean())
        self.std = float(curves.std()) or 1.0
        y = torch.from_numpy(np.asarray(labels) - min(LEVELS)).long()
        _train_net(self.net, (self.standardise(curves),), y, self.epochs)

    def predict_proba(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        del peak_ref
        return self.predict_proba_matrix(curves, np.empty((len(curves), 0)))

    def predict_proba_matrix(self, curves: np.ndarray, features: np.ndarray) -> np.ndarray:
        del features
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(self.standardise(curves)), dim=1).numpy()


class HybridModel:
    """The curve and the shape features, read together by one network."""

    key = "hybrid"
    name = "Hybrid CNN"
    description = "Convolutional network reading the raw curve and the shape features together."

    def __init__(self, *, burst: bool, epochs: int = 60) -> None:
        self.burst = burst
        self.epochs = epochs
        # Built in fit_matrix, where the number of features is known: it differs by
        # station, since only ironing carries the burst and draw-down descriptors.
        self.net: _HybridNet | None = None
        self.curve_mean = 0.0
        self.curve_std = 1.0
        self.feature_mean = np.zeros(1)
        self.feature_std = np.ones(1)

    def _extract(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        return feature_matrix(curves, burst=self.burst, peak_ref=peak_ref)[0]

    def _inputs(
        self, curves: np.ndarray, features: np.ndarray
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            _standardise(curves, self.curve_mean, self.curve_std).unsqueeze(1),
            _standardise(features, self.feature_mean, self.feature_std),
        )

    def fit(self, curves: np.ndarray, labels: np.ndarray, peak_ref: np.ndarray | None) -> None:
        self.fit_matrix(curves, self._extract(curves, peak_ref), labels)

    def fit_matrix(self, curves: np.ndarray, features: np.ndarray, labels: np.ndarray) -> None:
        torch.manual_seed(SEED)
        self.net = _HybridNet(features.shape[1], len(LEVELS))
        self.curve_mean = float(curves.mean())
        self.curve_std = float(curves.std()) or 1.0
        self.feature_mean = features.mean(axis=0)
        self.feature_std = np.where(features.std(axis=0) > 0, features.std(axis=0), 1.0)

        y = torch.from_numpy(np.asarray(labels) - min(LEVELS)).long()
        _train_net(self.net, self._inputs(curves, features), y, self.epochs)

    def predict_proba(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        return self.predict_proba_matrix(curves, self._extract(curves, peak_ref))

    def predict_proba_matrix(self, curves: np.ndarray, features: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(*self._inputs(curves, features)), dim=1).numpy()


def build_models(*, burst: bool) -> list[WearModel]:
    """The four models, in the order the dashboard offers them.

    `burst` turns on the ironing-only contact-transition and draw-down descriptors.
    """
    return [
        FeatureModel(
            key="logistic",
            name="Logistic regression",
            description="Linear model on shape features -- the transparent baseline.",
            estimator=make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, random_state=SEED),
            ),
            burst=burst,
        ),
        FeatureModel(
            key="forest",
            name="Random forest",
            description="Tree ensemble on the same shape features.",
            estimator=RandomForestClassifier(
                n_estimators=300, min_samples_leaf=2, random_state=SEED, n_jobs=-1
            ),
            burst=burst,
        ),
        CnnModel(),
        HybridModel(burst=burst),
    ]
