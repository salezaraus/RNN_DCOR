"""
Error metrics for forecasting evaluations.
"""
import numpy as np


def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Symmetric mean absolute percentage error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    denom = (np.abs(actual) + np.abs(predicted)) / 2
    return float(np.mean(np.abs(predicted - actual) / denom) * 100)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Weighted absolute percentage error
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    num = np.sum(np.abs(actual - predicted))
    den = np.sum(np.abs(actual))
    if den == 0:
        raise ValueError("Denominator zero in WAPE calculation")
    return float(num / den)


def relative_mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Relative MSE: mean((y - ŷ)^2 / y^2)
    """
    actual = np.array(actual)
    predicted = np.array(predicted)
    return float(np.mean((actual - predicted) ** 2 / (actual ** 2)))
