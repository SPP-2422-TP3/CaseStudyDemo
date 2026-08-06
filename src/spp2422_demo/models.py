"""The three selectable wear-state classifiers.

All of them map one normalized-force curve to a probability over wear levels 1..3, so
the dashboard can swap between them without knowing what is inside. Two read handcrafted
shape features, one reads the raw curve.
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

    # Validation sweeps refit the same model over many splits of one curve set. Going
    # through a feature matrix computed once, rather than re-extracting per split, is
    # what makes those sweeps affordable.
    def fit_matrix(self, features: np.ndarray, labels: np.ndarray) -> None:
        self.estimator.fit(features, labels)

    def predict_proba_matrix(self, features: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(features)


class _Net(nn.Module):
    """Small 1-D CNN: three strided conv blocks, global pooling, linear head."""

    def __init__(self, n_classes: int) -> None:
        super().__init__()

        def block(in_ch: int, out_ch: int, kernel: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel, padding=kernel // 2),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(4),
            )

        self.body = nn.Sequential(
            block(1, 16, 9),
            block(16, 32, 7),
            block(32, 64, 5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)


class CnnModel:
    """1-D CNN on the raw 500-sample curve -- no handcrafted features at all."""

    key = "cnn"
    name = "1-D CNN"
    description = "Convolutional network reading the raw curve, no handcrafted features."

    def __init__(self, epochs: int = 60) -> None:
        self.epochs = epochs
        self.net = _Net(len(LEVELS))
        self.mean = 0.0
        self.std = 1.0

    def _standardise(self, curves: np.ndarray) -> torch.Tensor:
        x = (np.asarray(curves, dtype=np.float32) - self.mean) / self.std
        return torch.from_numpy(x).unsqueeze(1)

    def fit(self, curves: np.ndarray, labels: np.ndarray, peak_ref: np.ndarray | None) -> None:
        del peak_ref  # the raw curve is the whole input
        torch.manual_seed(SEED)
        self.mean = float(curves.mean())
        self.std = float(curves.std()) or 1.0

        x = self._standardise(curves)
        y = torch.from_numpy(np.asarray(labels) - min(LEVELS)).long()
        optimiser = torch.optim.AdamW(self.net.parameters(), lr=3e-3, weight_decay=1e-4)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=self.epochs)
        loss_fn = nn.CrossEntropyLoss()
        generator = torch.Generator().manual_seed(SEED)

        self.net.train()
        for _ in range(self.epochs):
            for batch in torch.randperm(len(x), generator=generator).split(128):
                optimiser.zero_grad()
                loss_fn(self.net(x[batch]), y[batch]).backward()
                optimiser.step()
            schedule.step()
        self.net.eval()

    def predict_proba(self, curves: np.ndarray, peak_ref: np.ndarray | None) -> np.ndarray:
        del peak_ref
        self.net.eval()
        with torch.no_grad():
            return torch.softmax(self.net(self._standardise(curves)), dim=1).numpy()


def build_models(*, burst: bool) -> list[WearModel]:
    """The three models, in the order the dashboard offers them.

    `burst` turns on the ironing-only contact-transition descriptors.
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
    ]
