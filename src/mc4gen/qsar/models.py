"""Single-model wrappers used by the ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from lightgbm import LGBMRegressor
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


class Regressor(Protocol):
    def fit(self, X: NDArray[np.float32], y: NDArray[np.float32]) -> None: ...
    def predict(self, X: NDArray[np.float32]) -> NDArray[np.float32]: ...


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    factory: str
    params: dict[str, float | int | str | bool | None]


DEFAULT_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec("rf", "rf", {"n_estimators": 400, "max_depth": None, "n_jobs": -1, "random_state": 0}),
    ModelSpec(
        "lgbm",
        "lgbm",
        {
            "n_estimators": 600,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "n_jobs": -1,
            "random_state": 0,
            "verbose": -1,
        },
    ),
    ModelSpec("svr", "svr", {"C": 4.0, "gamma": "scale", "kernel": "rbf"}),
    ModelSpec("ridge", "ridge", {"alpha": 1.0}),
    ModelSpec(
        "mlp",
        "mlp",
        {
            "hidden_layer_sizes": (512, 256, 128),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-4,
            "max_iter": 300,
            "random_state": 0,
        },
    ),
)


def build(spec: ModelSpec) -> Regressor:
    kind = spec.factory
    params = dict(spec.params)
    if kind == "rf":
        return RandomForestRegressor(**params)
    if kind == "lgbm":
        return LGBMRegressor(**params)
    if kind == "svr":
        return SVR(**params)
    if kind == "ridge":
        return Ridge(**params)
    if kind == "mlp":
        return MLPRegressor(**params)
    raise ValueError(f"Unknown model factory: {kind!r}")
