from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import MetricSampler


class CostSampler(MetricSampler):
    """Sampler for electricity cost.

    Formulas derived from legacy monte_carlo.py:
    - Empirical sampling: Load from Excel file and sample randomly
    - Uniform sampling: Sample from uniform distribution
    - Original cost_per_day formula: cost($/kWh) * (efficiency J/TH / 3.6e6 J/kWh) * hash_rate(TH/s) * 86400s/day
    """

    def __init__(self, config, data_path: Optional[str] = None):
        self.type = config.type
        if self.type == "empirical_excel":
            if not config.file:
                raise ValueError("Excel file path required for empirical_excel cost sampler")
            self.file_path = Path(data_path) / config.file if data_path else Path(config.file)
            self._load_data()
        elif self.type == "uniform":
            self.low = config.low
            self.high = config.high
        else:
            raise ValueError(f"Unsupported cost sampler type: {self.type}")

    def _load_data(self):
        """Load cost data from Excel file.

        Based on legacy: sample_cost_excel function
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Cost data file not found: {self.file_path}")
        df = pd.read_excel(self.file_path)
        if 'cost' not in df.columns:
            raise ValueError("Cost data file must contain 'cost' column")
        self.costs = df['cost'].values

    def sample(self) -> float:
        """Sample electricity cost.

        Based on legacy sampling functions:
        - empirical_excel: Random choice from Excel data
        - uniform: Random sample from uniform distribution
        """
        if self.type == "empirical_excel":
            return float(np.random.choice(self.costs))
        elif self.type == "uniform":
            return float(np.random.uniform(self.low, self.high))
        else:
            raise ValueError(f"Unsupported cost sampler type: {self.type}")

    def get_description(self) -> str:
        if self.type == "empirical_excel":
            return f"Empirical from {self.file_path.name} (n={len(self.costs)})"
        elif self.type == "uniform":
            return f"Uniform({self.low}, {self.high})"
        return self.type