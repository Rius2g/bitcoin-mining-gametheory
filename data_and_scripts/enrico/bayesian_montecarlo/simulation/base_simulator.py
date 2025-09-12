from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.loader import Config
from metrics.base import PriceSampler, CostSampler, RiskSampler
from sampler.rig import RigSampler
from sampler.network import NetworkSampler
from game.miner import MinerType
from game.bayes_game import BayesianGame


class SimulationResult:
    """Container for simulation results."""

    def __init__(self, summary_df: pd.DataFrame, miners_df: Optional[pd.DataFrame] = None):
        self.summary_df = summary_df
        self.miners_df = miners_df
        self.metadata = {}

    def add_metadata(self, key: str, value: Any):
        """Add metadata to the result."""
        self.metadata[key] = value

    def save(self, output_dir: str, prefix: str = "simulation"):
        """Save results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_path / f"{prefix}_summary.csv"
        self.summary_df.to_csv(summary_path, index=False)

        # Save miners data if available
        if self.miners_df is not None:
            miners_path = output_path / f"{prefix}_miners.csv"
            self.miners_df.to_csv(miners_path, index=False)

        # Save metadata
        if self.metadata:
            metadata_path = output_path / f"{prefix}_metadata.json"
            import json
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)


class MonteCarloSimulator:
    """Base Monte Carlo simulator for Bayesian mining games."""

    def __init__(self, config: Config):
        self.config = config

        # Initialize samplers
        self.price_sampler = PriceSampler(config.metrics.price)
        self.cost_sampler = CostSampler(config.metrics.cost)
        self.risk_sampler = RiskSampler(config.metrics.risk)
        self.rig_sampler = RigSampler(config.rig_specs)
        self.network_sampler = NetworkSampler(config.network)

        # Initialize game
        self.game = BayesianGame(self.network_sampler)

    def create_miners(self, n_miners: int) -> List[MinerType]:
        """Create a population of miners with sampled characteristics."""
        miners = []
        for _ in range(n_miners):
            # Sample miner characteristics
            cost = self.cost_sampler.sample()
            risk = self.risk_sampler.sample()
            capacity, efficiency = self.rig_sampler.sample()

            # Sample number of rigs (1-10)
            n_rigs = np.random.randint(1, 11)
            total_hash = capacity * n_rigs

            miner = MinerType(
                cost_per_kwh=cost,
                efficiency=efficiency,
                risk_aversion=risk,
                max_hash_rate=total_hash
            )
            miners.append(miner)

        return miners

    def run_single(self, n_miners: Optional[int] = None,
                  pool_hash_override: Optional[float] = None) -> pd.DataFrame:
        """Run a single simulation draw.

        Returns:
            DataFrame with miner-level data
        """
        if n_miners is None:
            n_miners = self.config.simulation.miners

        # Sample price
        btc_price = self.price_sampler.sample()

        # Create miners
        miners = self.create_miners(n_miners)

        # Create a new game instance for this draw to avoid accumulation
        game = BayesianGame(self.network_sampler)
        game.add_miners(miners)

        # Update game state
        game.update_game_state(btc_price, pool_hash_override)

        # Find equilibrium
        converged = game.find_equilibrium(self.config.simulation.max_iterations)

        # Get results
        df = game.to_dataframe()
        df.attrs['btc_price'] = btc_price
        df.attrs['converged'] = converged
        df.attrs['pool_hash_override'] = pool_hash_override

        return df

    def run(self, draws: Optional[int] = None,
           pool_hash_override: Optional[float] = None,
           record_miners: bool = False) -> SimulationResult:
        """Run multiple simulation draws.

        Args:
            draws: Number of draws (default from config)
            pool_hash_override: Fixed pool hash rate
            record_miners: Whether to save individual miner data

        Returns:
            SimulationResult with summary and optional miner data
        """
        if draws is None:
            draws = self.config.simulation.draws

        summaries = []
        miners_dfs = [] if record_miners else None

        for draw_idx in range(draws):
            df = self.run_single(self.config.simulation.miners, pool_hash_override)

            # Create summary row
            summary = {
                'draw_idx': draw_idx,
                'btc_price': df.attrs['btc_price'],
                'converged': df.attrs['converged'],
                'total_miners': len(df),
                'active_miners': (df['actual_hash_rate'] > 0).sum(),
                'pct_active': (df['actual_hash_rate'] > 0).mean(),
                'total_hash_rate': df['actual_hash_rate'].sum(),
                'pool_hash_rate': df[df['strategy'] == 'pool']['actual_hash_rate'].sum(),
                'pct_solo': (df['strategy'] == 'solo').mean(),
                'pct_pool': (df['strategy'] == 'pool').mean(),
                'pct_offline': (df['strategy'] == 'offline').mean(),
            }
            summaries.append(summary)

            if record_miners:
                df_copy = df.copy()
                df_copy['draw_idx'] = draw_idx
                miners_dfs.append(df_copy)

        summary_df = pd.DataFrame(summaries)

        if record_miners:
            miners_df = pd.concat(miners_dfs, ignore_index=True)
        else:
            miners_df = None

        result = SimulationResult(summary_df, miners_df)

        # Add metadata
        result.add_metadata('config', {
            'draws': draws,
            'miners_per_draw': self.config.simulation.miners,
            'pool_hash_override': pool_hash_override,
        })
        result.add_metadata('samplers', {
            'price': self.price_sampler.get_description(),
            'cost': self.cost_sampler.get_description(),
            'risk': self.risk_sampler.get_description(),
            'rig': self.rig_sampler.get_description(),
            'network': self.network_sampler.get_description(),
        })

        return result
