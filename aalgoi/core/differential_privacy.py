"""
Differential privacy utilities for federated metrics.

Uses diffprivlib to add calibrated noise to shared metrics,
preventing dataset fingerprinting via aggregate statistics.
"""

from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

_DP_AVAILABLE = False
try:
    from diffprivlib.mechanisms import Laplace, Gaussian
    _DP_AVAILABLE = True
except ImportError:
    Laplace = None
    Gaussian = None


def privatize_float(
    value: float,
    sensitivity: float = 1.0,
    epsilon: float = 1.0,
    mechanism: str = "laplace",
) -> float:
    if not _DP_AVAILABLE:
        return value
    if mechanism == "gaussian":
        mech = Gaussian(epsilon=epsilon, sensitivity=sensitivity)
    else:
        mech = Laplace(epsilon=epsilon, sensitivity=sensitivity)
    return mech.randomise(value)


def privatize_metric(
    metric_name: str,
    value: float,
    epsilon: float = 1.0,
) -> float:
    sensitivities = {
        "execution_time_ms": 0.01,
        "success_rate": 0.01,
        "data_size": 0.001,
        "reward": 0.1,
        "confidence": 0.01,
    }
    sensitivity = sensitivities.get(metric_name, 1.0)
    return privatize_float(value, sensitivity=sensitivity, epsilon=epsilon)


def prepare_federation_payload(
    metrics: Dict[str, float],
    epsilon: float = 1.0,
) -> Dict[str, float]:
    if not _DP_AVAILABLE:
        return metrics
    return {
        k: privatize_metric(k, v, epsilon=epsilon)
        for k, v in metrics.items()
    }
