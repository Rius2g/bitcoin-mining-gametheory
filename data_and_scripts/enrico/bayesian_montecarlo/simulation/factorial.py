from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base_simulator import MonteCarloSimulator, SimulationResult
from config.loader import Config


class FactorialMonteCarlo:
    """Factorial Monte Carlo simulator for scenario analysis."""

    def __init__(self, config: Config):
        self.config = config
        self.base_simulator = MonteCarloSimulator(config)

    def run_context_sweep(self, contexts: Optional[Dict[str, Dict[str, Any]]] = None,
                         draws_per_context: Optional[int] = None,
                         pool_sizes: Optional[List[float]] = None) -> Dict[str, SimulationResult]:
        """Run simulations across different market contexts.

        Args:
            contexts: Dictionary of context configurations
            draws_per_context: Number of draws per context
            pool_sizes: List of pool sizes to test

        Returns:
            Dictionary mapping context names to SimulationResults
        """
        if contexts is None:
            contexts = self._get_default_contexts()

        if draws_per_context is None:
            draws_per_context = self.config.simulation.draws

        if pool_sizes is None:
            pool_sizes = self.config.factorial.pool_sizes

        results = {}

        for context_name, context_params in contexts.items():
            print(f"Running context: {context_name}")

            # Configure samplers for this context
            self._configure_context(context_params)

            # Run simulations for different pool sizes
            context_results = []
            for pool_size in pool_sizes:
                result = self.base_simulator.run(
                    draws=draws_per_context,
                    pool_hash_override=pool_size,
                    record_miners=True
                )
                result.add_metadata('pool_size', pool_size)
                context_results.append(result)

            # Combine results across pool sizes
            combined_result = self._combine_results(context_results, context_name)
            results[context_name] = combined_result

        return results

    def _get_default_contexts(self) -> Dict[str, Dict[str, Any]]:
        """Get default market contexts from config."""
        contexts = {}
        for name, ctx_config in self.config.factorial.contexts.items():
            contexts[name] = {
                'btc_price': ctx_config.price,
                'network_hash': ctx_config.network_hash,
                'block_subsidy': ctx_config.block_subsidy,
            }
        return contexts

    def _configure_context(self, context_params: Dict[str, Any]):
        """Configure samplers for a specific market context."""
        # Store original samplers
        if not hasattr(self, '_original_price_sampler'):
            self._original_price_sampler = self.base_simulator.price_sampler
            self._original_network_sampler = self.base_simulator.network_sampler

        # Override price sampler
        if 'btc_price' in context_params:
            # Create a simple object with sample method
            class FixedPriceSampler:
                def __init__(self, price):
                    self.price = price
                def sample(self):
                    return self.price
                def get_description(self):
                    return f"Fixed({self.price})"

            self.base_simulator.price_sampler = FixedPriceSampler(context_params['btc_price'])

        # Override network parameters
        if 'network_hash' in context_params:
            # Create a modified network sampler
            class ModifiedNetworkSampler:
                def __init__(self, original_sampler, network_hash, block_subsidy=None):
                    self.original_sampler = original_sampler
                    self.network_hash = network_hash
                    self.block_subsidy = block_subsidy or original_sampler.block_subsidy
                    # Copy other attributes
                    self.discount_factor = original_sampler.discount_factor
                    self.pool_fee = original_sampler.pool_fee
                    self.block_time = original_sampler.block_time
                    self.block_reward = original_sampler.block_reward

                def sample_total_hash(self):
                    return self.network_hash

                def sample_pool_fee(self, pool_hash_rate: float = 0.0):
                    return self.original_sampler.sample_pool_fee(pool_hash_rate)

                def sample_block_time(self):
                    return self.original_sampler.sample_block_time()

                def sample_block_reward(self, price: float):
                    btc_reward = 3.125 + self.block_subsidy  # R + M
                    discounted_reward = btc_reward * np.exp(-self.original_sampler.discount_factor / self.original_sampler.block_time)
                    return discounted_reward * price

                def get_description(self):
                    return f"Modified({self.network_hash/1e6:.0f}EH/s)"

            self.base_simulator.network_sampler = ModifiedNetworkSampler(
                self._original_network_sampler,
                context_params['network_hash'],
                context_params.get('block_subsidy')
            )

    def _combine_results(self, results: List[SimulationResult], context_name: str) -> SimulationResult:
        """Combine results from multiple pool sizes into a single result."""
        if not results:
            return SimulationResult(pd.DataFrame(), None)

        # Combine summary DataFrames
        summary_dfs = []
        miners_dfs = []

        for i, result in enumerate(results):
            summary_df = result.summary_df.copy()
            summary_df['pool_size'] = result.metadata.get('pool_size', i)
            summary_dfs.append(summary_df)

            if result.miners_df is not None:
                miners_df = result.miners_df.copy()
                miners_df['pool_size'] = result.metadata.get('pool_size', i)
                miners_dfs.append(miners_df)

        combined_summary = pd.concat(summary_dfs, ignore_index=True)
        combined_miners = pd.concat(miners_dfs, ignore_index=True) if miners_dfs else None

        combined_result = SimulationResult(combined_summary, combined_miners)
        combined_result.add_metadata('context', context_name)
        combined_result.add_metadata('pool_sizes', [r.metadata.get('pool_size') for r in results])

        return combined_result

    def run_scenario_analysis(self, scenarios: Dict[str, Dict[str, Any]],
                            draws: Optional[int] = None) -> Dict[str, SimulationResult]:
        """Run analysis for specific scenarios.

        Args:
            scenarios: Dictionary of scenario configurations
            draws: Number of draws per scenario

        Returns:
            Dictionary mapping scenario names to results
        """
        if draws is None:
            draws = self.config.simulation.draws

        results = {}

        for scenario_name, scenario_params in scenarios.items():
            print(f"Running scenario: {scenario_name}")

            # Configure for scenario
            self._configure_scenario(scenario_params)

            # Run simulation
            result = self.base_simulator.run(draws=draws, record_miners=True)
            result.add_metadata('scenario', scenario_name)
            result.add_metadata('scenario_params', scenario_params)

            results[scenario_name] = result

        return results

    def _configure_scenario(self, scenario_params: Dict[str, Any]):
        """Configure simulator for a specific scenario."""
        # This can be extended to handle various scenario parameters
        for param, value in scenario_params.items():
            if param == 'price':
                self.base_simulator.price_sampler = lambda: value
            elif param == 'network_hash':
                self.base_simulator.network_sampler.base_hash_rate = value
            elif param == 'pool_fee':
                self.base_simulator.network_sampler.pool_fee = value
            # Add more parameter overrides as needed
