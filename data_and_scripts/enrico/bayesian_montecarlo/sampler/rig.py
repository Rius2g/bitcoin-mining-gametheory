from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class RigSampler:
    """Sampler for mining rig specifications."""

    def __init__(self, config: dict, data_path: Optional[str] = None):
        self.file_path = Path(data_path) / config['file'] if data_path else Path(config['file'])
        self._load_data()

    def _load_data(self):
        """Load rig specifications from Excel file."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Rig specs file not found: {self.file_path}")
        df = pd.read_excel(self.file_path)
        required_cols = {'capacity_THs', 'efficiency'}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Rig specs file must contain columns: {required_cols}")
        self.capacities = df['capacity_THs'].values
        self.efficiencies = df['efficiency'].values

    def sample(self) -> Tuple[float, float]:
        """Sample rig capacity and efficiency.

        Returns:
            Tuple of (capacity_THs, efficiency_J_per_TH)
        """
        cap_idx = np.random.randint(0, len(self.capacities))
        eff_idx = np.random.randint(0, len(self.efficiencies))
        return float(self.capacities[cap_idx]), float(self.efficiencies[eff_idx])

    def get_description(self) -> str:
        return f"Empirical from {self.file_path.name} (n={len(self.capacities)} rigs)"


class NetworkSampler:
    """Sampler for network parameters."""

    def __init__(self, config):
        self.base_hash_rate = config.hash_rate
        self.pool_fee = config.pool_fee
        self.block_time = config.block_time
        self.block_reward = config.block_reward
        self.block_subsidy = config.block_subsidy
        self.discount_factor = config.discount_factor

    def sample_total_hash(self) -> float:
        """Sample total network hash rate."""
        # For now, return constant value
        return self.base_hash_rate

    def sample_pool_fee(self) -> float:
        """Sample pool fee."""
        return self.pool_fee

    def sample_block_time(self) -> float:
        """Sample block time."""
        return self.block_time

    def sample_block_reward(self, price: float) -> float:
        """Sample block reward in USD."""
        btc_reward = self.block_reward + self.block_subsidy
        discounted_reward = btc_reward * np.exp(-self.discount_factor / self.block_time)
        return discounted_reward * price

    def get_description(self) -> str:
        return f"Network(hash={self.base_hash_rate/1e6:.0f}EH/s, fee={self.pool_fee:.3f})"
