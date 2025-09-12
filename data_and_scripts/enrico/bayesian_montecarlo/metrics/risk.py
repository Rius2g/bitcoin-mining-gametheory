from abc import ABC, abstractmethod
from typing import Optional
import numpy as np

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import MetricSampler


class RiskSampler(MetricSampler):
    """Sampler for miner risk aversion.

    Formulas derived from legacy monte_carlo.py:
    - Uniform sampling: Random sample from [0,1] range
    - Beta sampling: Sample from beta distribution for more realistic risk modeling
    - Risk penalty in utility: risk_aversion * revenue (variance approximation)
    """

    def __init__(self, config):
        self.type = config.type
        if self.type == "uniform":
            self.low = config.low
            self.high = config.high
        elif self.type == "beta":
            self.a = config.a
            self.b = config.b
        else:
            raise ValueError(f"Unsupported risk sampler type: {self.type}")

    def sample(self) -> float:
        """Sample miner risk aversion.

        Based on legacy: sample_risk() function
        - Uniform: Simple random sampling from [low, high]
        - Beta: More sophisticated distribution for risk preferences
        """
        if self.type == "uniform":
            return float(np.random.uniform(self.low, self.high))
        elif self.type == "beta":
            return float(np.random.beta(self.a, self.b))
        else:
            raise ValueError(f"Unsupported risk sampler type: {self.type}")

    def get_description(self) -> str:
        if self.type == "uniform":
            return f"Uniform({self.low}, {self.high})"
        elif self.type == "beta":
            return f"Beta(alpha={self.a}, beta={self.b})"
        return self.type