from abc import ABC, abstractmethod
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


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
