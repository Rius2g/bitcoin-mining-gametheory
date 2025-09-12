from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from .miner import MinerType
from .strategy import Strategy, get_strategy

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BayesianGame:
    """Bayesian game for Bitcoin mining strategy equilibrium."""

    def __init__(self, network_sampler, strategy: Optional[Strategy] = None):
        self.network_sampler = network_sampler
        self.strategy = strategy or get_strategy('utility_maximizing')
        self.miners: List[MinerType] = []
        self.convergence_tolerance = 1e-6
        self.max_iterations = 10

    def add_miner(self, miner: MinerType):
        """Add a miner to the game."""
        self.miners.append(miner)

    def add_miners(self, miners: List[MinerType]):
        """Add multiple miners to the game."""
        self.miners.extend(miners)

    def update_game_state(self, btc_price: float, pool_hash_override: Optional[float] = None):
        """Update the game state for all miners."""
        # Sample network parameters
        network_hash = self.network_sampler.sample_total_hash()
        pool_fee = self.network_sampler.sample_pool_fee()
        block_time = self.network_sampler.sample_block_time()
        block_reward = self.network_sampler.sample_block_reward(btc_price)

        # Calculate pool hash rate
        if pool_hash_override is not None:
            pool_hash = pool_hash_override
        else:
            # Will be updated in equilibrium finding
            pool_hash = 0.0

        # Update each miner
        for miner in self.miners:
            miner.update_game_state(
                network_hash=network_hash,
                pool_hash=pool_hash,
                block_reward=block_reward,
                block_time=block_time,
                pool_fee=pool_fee,
                btc_price=btc_price,
                discount_factor=self.network_sampler.discount_factor
            )

    def find_equilibrium(self, max_iterations: Optional[int] = None) -> bool:
        """Find Bayesian Nash equilibrium through best-response iteration.

        Returns:
            True if converged, False otherwise
        """
        if max_iterations is None:
            max_iterations = self.max_iterations

        for iteration in range(max_iterations):
            prev_strategies = [miner.strategy for miner in self.miners]

            # Update pool hash based on current strategies
            pool_hash = sum(miner.actual_hash_rate for miner in self.miners
                          if miner.strategy == 'pool')

            # Update pool hash for all miners
            for miner in self.miners:
                miner.pool_hash_rate = pool_hash

            # Each miner chooses best response
            for miner in self.miners:
                self.strategy.choose_action(miner, self)

            # Check convergence
            current_strategies = [miner.strategy for miner in self.miners]
            if all(curr == prev for curr, prev in zip(current_strategies, prev_strategies)):
                return True

        return False

    def get_game_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the current game state."""
        if not self.miners:
            return {}

        strategies = [miner.strategy for miner in self.miners]
        hash_rates = [miner.actual_hash_rate for miner in self.miners]

        total_hash = sum(hash_rates)
        active_miners = sum(1 for hr in hash_rates if hr > 0)
        pool_hash = sum(hr for miner, hr in zip(self.miners, hash_rates)
                       if miner.strategy == 'pool')

        strategy_counts = pd.Series(strategies).value_counts()

        return {
            'total_miners': len(self.miners),
            'active_miners': active_miners,
            'pct_active': active_miners / len(self.miners),
            'total_hash_rate': total_hash,
            'pool_hash_rate': pool_hash,
            'network_hash_rate': self.miners[0].network_hash_rate if self.miners else 0,
            'strategy_distribution': strategy_counts.to_dict(),
            'btc_price': self.miners[0].btc_price if self.miners else 0,
            'block_reward': self.miners[0].block_reward if self.miners else 0,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert current game state to DataFrame."""
        data = []
        for i, miner in enumerate(self.miners):
            data.append({
                'miner_id': i,
                'cost_per_kwh': miner.cost_per_kwh,
                'efficiency': miner.efficiency,
                'risk_aversion': miner.risk_aversion,
                'max_hash_rate': miner.max_hash_rate,
                'actual_hash_rate': miner.actual_hash_rate,
                'strategy': miner.strategy,
                'revenue_solo': miner.revenue_solo(),
                'revenue_pool': miner.revenue_pool(),
                'cost_per_day': miner.cost_per_day(),
                'utility_solo': miner.utility_solo(),
                'utility_pool': miner.utility_pool(),
                'utility_offline': miner.utility_offline(),
            })
        return pd.DataFrame(data)
