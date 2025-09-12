from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import MetricSampler


class PriceSampler(MetricSampler):
    """Sampler for Bitcoin price.

    Formulas derived from legacy monte_carlo.py:
    - Lognormal distribution: mu = ln(mean) - 0.5 * sigma^2
    - Sample from lognormal distribution for realistic price modeling
    """

    def __init__(self, config):
        self.type = config.type
        if self.type == "lognormal":
            self.mean = config.mean
            self.sigma = config.sigma
        elif self.type == "fixed":
            self.fixed_value = config.fixed_value
        else:
            raise ValueError(f"Unsupported price sampler type: {self.type}")

    def sample(self) -> float:
        """Sample Bitcoin price.

        For lognormal: Uses the formula from legacy code:
        mu = ln(mean) - 0.5 * sigma^2
        This ensures E[X] = mean for lognormal distribution.
        """
        if self.type == "lognormal":
            # Formula from legacy: sample_price function
            mu = np.log(self.mean) - 0.5 * self.sigma**2
            return float(np.random.lognormal(mu, self.sigma))
        elif self.type == "fixed":
            return self.fixed_value
        else:
            raise ValueError(f"Unsupported price sampler type: {self.type}")

    def get_description(self) -> str:
        if self.type == "lognormal":
            return f"Lognormal(mu={self.mean}, sigma={self.sigma})"
        elif self.type == "fixed":
            return f"Fixed({self.fixed_value})"
        return self.type