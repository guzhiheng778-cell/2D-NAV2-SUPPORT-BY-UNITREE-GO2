#!/usr/bin/env python3

import math
from typing import Tuple


def validate_error_rates(false_positive_rate: float, false_negative_rate: float) -> None:
    rates = {
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }
    for name, value in rates.items():
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    if false_positive_rate + false_negative_rate >= 1.0:
        raise ValueError(
            "false_positive_rate + false_negative_rate must be less than 1 "
            "for an informative binary sensor"
        )


def observation_probabilities(
    concentration: float,
    detection_threshold: float,
    likelihood_temperature: float,
    false_positive_rate: float,
    false_negative_rate: float,
) -> Tuple[float, float]:
    """Return complementary hit/void probabilities for a noisy binary sensor."""
    temperature = max(1e-12, likelihood_temperature)
    logit = max(
        -60.0,
        min(60.0, (concentration - detection_threshold) / temperature),
    )
    ideal_hit_probability = 1.0 / (1.0 + math.exp(-logit))

    hit_probability = (
        (1.0 - false_negative_rate) * ideal_hit_probability
        + false_positive_rate * (1.0 - ideal_hit_probability)
    )
    void_probability = 1.0 - hit_probability
    return hit_probability, void_probability
