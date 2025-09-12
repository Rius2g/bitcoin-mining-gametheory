from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class MetricSampler(ABC):
    """Abstract base class for all metric samplers.

    This defines the interface that all metric samplers must implement.
    Based on the legacy monte_carlo.py sampling functions.
    """

    @abstractmethod
    def sample(self) -> float:
        """Sample a value for this metric."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this sampler."""
        pass


# Import concrete implementations from individual files
from .price import PriceSampler
from .cost import CostSampler
from .risk import RiskSampler
