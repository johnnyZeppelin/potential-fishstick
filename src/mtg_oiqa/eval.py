from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import curve_fit


def _logistic5(x: np.ndarray, b0: float, b1: float, b2: float, b3: float, b4: float) -> np.ndarray:
    return b0 * (0.5 - 1.0 / (1.0 + np.exp(b1 * (x - b2)))) + b3 * x + b4


def nonlinear_map(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x = predictions.astype(np.float64)
    y = targets.astype(np.float64)
    if len(x) < 5:
        return x
    initial = (1.0, 1.0, np.mean(x), 0.0, np.mean(y))
    params, _ = curve_fit(_logistic5, x, y, p0=initial, maxfev=10000)
    return _logistic5(x, *params)


def plcc(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.size == 0:
        return float("nan")
    return float(np.corrcoef(predictions, targets)[0, 1])


def srcc(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.size == 0:
        return float("nan")
    pred_rank = predictions.argsort().argsort()
    target_rank = targets.argsort().argsort()
    return float(np.corrcoef(pred_rank, target_rank)[0, 1])


def rmse(predictions: np.ndarray, targets: np.ndarray) -> float:
    if predictions.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean((predictions - targets) ** 2)))


@dataclass
class EvaluationResult:
    plcc: float
    srcc: float
    rmse: float


def evaluate_predictions(predictions: Iterable[float], targets: Iterable[float]) -> EvaluationResult:
    predictions_np = np.asarray(list(predictions), dtype=np.float64)
    targets_np = np.asarray(list(targets), dtype=np.float64)
    mapped = nonlinear_map(predictions_np, targets_np)
    return EvaluationResult(
        plcc=plcc(mapped, targets_np),
        srcc=srcc(mapped, targets_np),
        rmse=rmse(mapped, targets_np),
    )
