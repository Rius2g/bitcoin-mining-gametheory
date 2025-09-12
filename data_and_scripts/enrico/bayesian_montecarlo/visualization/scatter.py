from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PlotScatter:
    """Scatter plot visualization for simulation results."""

    def __init__(self, config=None):
        self.config = config
        self.colors = {
            'solo': '#a8dadc',
            'pool': '#ffb4a2',
            'offline': '#e9c46a'
        } if config is None else config.visualization.colors

    def plot(self, df: pd.DataFrame, x_col: str = 'cost_per_kwh',
             y_col: str = 'actual_hash_rate', color_col: str = 'strategy',
             title_suffix: str = "", save_path: Optional[str] = None):
        """Create scatter plot of simulation results.

        Args:
            df: DataFrame with miner data
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for coloring points
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(10, 6))

        # Plot each strategy
        for strategy in df[color_col].unique():
            subset = df[df[color_col] == strategy]
            if subset.empty:
                continue

            # Size points by risk aversion
            sizes = 50 + 150 * subset['risk_aversion'] if 'risk_aversion' in subset.columns else 100
            alphas = 0.3 + 0.7 * subset['risk_aversion'] if 'risk_aversion' in subset.columns else 0.7

            plt.scatter(
                subset[x_col],
                subset[y_col],
                label=strategy.capitalize(),
                color=self.colors.get(strategy, 'gray'),
                edgecolor='white',
                s=sizes,
                alpha=alphas
            )

        plt.xlabel(self._format_label(x_col))
        plt.ylabel(self._format_label(y_col))
        plt.title(f'Miner Strategies{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_with_rug(self, df: pd.DataFrame, x_col: str = 'cost_per_kwh',
                     y_col: str = 'actual_hash_rate', title_suffix: str = "",
                     save_path: Optional[str] = None):
        """Create scatter plot with rug for offline miners.

        Args:
            df: DataFrame with miner data
            x_col: Column for x-axis
            y_col: Column for y-axis
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(10, 6))

        # Split active and offline
        active = df[df[y_col] > 0].copy()
        offline = df[df[y_col] == 0]

        # Plot active miners
        for strategy in active['strategy'].unique():
            subset = active[active['strategy'] == strategy]
            if subset.empty:
                continue

            sizes = 50 + 150 * subset['risk_aversion'] if 'risk_aversion' in subset.columns else 100
            alphas = 0.3 + 0.7 * subset['risk_aversion'] if 'risk_aversion' in subset.columns else 0.7

            plt.scatter(
                subset[x_col],
                subset[y_col],
                label=strategy.capitalize(),
                color=self.colors.get(strategy, 'gray'),
                marker='o',
                edgecolor='white',
                s=sizes,
                alpha=alphas
            )

        # Add rug for offline miners
        if not offline.empty:
            plt.plot(offline[x_col], np.zeros_like(offline[x_col]), '|',
                    markersize=8, alpha=0.5, color='gray', label='offline')

        plt.xlabel(self._format_label(x_col))
        plt.ylabel(self._format_label(y_col))
        plt.title(f'Miner Strategies with Offline Rug{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _format_label(self, col_name: str) -> str:
        """Format column name for display."""
        if col_name == 'cost_per_kwh':
            return 'Electricity Cost ($/kWh)'
        elif col_name == 'actual_hash_rate':
            return 'Hash Rate (TH/s)'
        elif col_name == 'max_hash_rate':
            return 'Max Hash Rate (TH/s)'
        elif col_name == 'risk_aversion':
            return 'Risk Aversion'
        else:
            return col_name.replace('_', ' ').title()
