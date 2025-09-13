from dataclasses import dataclass, field
from typing import Optional
from abc import ABC, abstractmethod
import numpy as np

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class MinerType:
    """Represents a miner's characteristics and current state."""
    cost_per_kwh: float  # $/kWh
    efficiency: float    # J/TH
    risk_aversion: float # 0-1 scale
    max_hash_rate: float # TH/s

    # Game state (set during simulation)
    network_hash_rate: float = field(init=False, default=0.0)
    pool_hash_rate: float = field(init=False, default=0.0)
    block_reward: float = field(init=False, default=0.0)
    block_time: float = field(init=False, default=600.0)
    pool_fee: float = field(init=False, default=0.025)
    btc_price: float = field(init=False, default=0.0)
    discount_factor: float = field(init=False, default=0.5)

    # Decision variables
    strategy: str = field(init=False, default="offline")
    actual_hash_rate: float = field(init=False, default=0.0)

    def cost_per_day(self) -> float:
        """Calculate daily electricity cost."""
        # Convert efficiency from J/TH to kWh/TH
        kwh_per_th = self.efficiency / 3.6e6
        # Daily energy consumption in kWh (use max_hash_rate for utility calculation)
        daily_kwh = kwh_per_th * self.max_hash_rate * 86400
        return self.cost_per_kwh * daily_kwh

    def revenue_solo(self) -> float:
        """Expected daily revenue from solo mining."""
        # Blocks per day found by network
        network_blocks_per_day = 86400 / self.block_time
        # Miner's share of network blocks (use max_hash_rate for utility calculation)
        miner_share = self.max_hash_rate / (self.network_hash_rate + self.max_hash_rate)
        # Expected blocks per day
        expected_blocks = network_blocks_per_day * miner_share
        # Revenue per block (block_reward is already in USD)
        return expected_blocks * self.block_reward

    def revenue_pool(self) -> float:
        """Expected daily revenue from pool mining."""
        if self.pool_hash_rate == 0:
            return 0.0

        # Blocks per day found by the pool (proportional to pool's network share)
        network_blocks_per_day = 86400 / self.block_time
        pool_blocks_per_day = self.pool_hash_rate / self.network_hash_rate * network_blocks_per_day
        # Miner's share of pool blocks (use max_hash_rate for utility calculation)
        miner_share = self.max_hash_rate / self.pool_hash_rate
        # Revenue per block (after pool fee, block_reward is already in USD)
        revenue_per_block = self.block_reward * (1 - self.pool_fee)
        return pool_blocks_per_day * miner_share * revenue_per_block

    def utility_solo(self) -> float:
        """Calculate utility from solo mining strategy."""
        revenue = self.revenue_solo()
        cost = self.cost_per_day()
        # Risk penalty (variance approximation)
        variance_penalty = self.risk_aversion * revenue
        return revenue - cost - variance_penalty

    def utility_pool(self) -> float:
        """Calculate utility from pool mining strategy."""
        revenue = self.revenue_pool()
        cost = self.cost_per_day()
        return revenue - cost

    def utility_offline(self) -> float:
        """Calculate utility from offline strategy."""
        return 0.0

    def choose_strategy(self) -> str:
        """Choose the optimal strategy based on utility maximization."""
        u_solo = self.utility_solo()
        u_pool = self.utility_pool()
        u_offline = self.utility_offline()

        # Choose strategy with highest utility
        utilities = {'solo': u_solo, 'pool': u_pool, 'offline': u_offline}
        best_strategy = max(utilities, key=utilities.get)

        self.strategy = best_strategy
        if best_strategy == 'offline':
            self.actual_hash_rate = 0.0
        else:
            self.actual_hash_rate = self.max_hash_rate

        return best_strategy

    def update_game_state(self, network_hash: float, pool_hash: float,
                         block_reward: float, block_time: float,
                         pool_fee: float, btc_price: float,
                         discount_factor: float):
        """Update the game state parameters."""
        self.network_hash_rate = network_hash
        self.pool_hash_rate = pool_hash
        self.block_reward = block_reward
        self.block_time = block_time
        self.pool_fee = pool_fee
        self.btc_price = btc_price
        self.discount_factor = discount_factor
