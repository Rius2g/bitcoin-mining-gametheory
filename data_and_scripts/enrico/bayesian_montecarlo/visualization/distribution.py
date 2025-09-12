from typing import Optional, List
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PlotDistribution:
    """Distribution plot visualization for simulation results."""

    def __init__(self, config=None):
        self.config = config

    def plot_summary_distributions(self, summary_df: pd.DataFrame,
                                 columns: Optional[List[str]] = None,
                                 title_suffix: str = "",
                                 save_path: Optional[str] = None):
        """Plot distributions of summary statistics.

        Args:
            summary_df: DataFrame with summary statistics
            columns: Columns to plot (default: key metrics)
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        if columns is None:
            columns = ['btc_price', 'total_hash_rate', 'pct_active', 'pct_pool']

        n_cols = len(columns)
        n_rows = (n_cols + 1) // 2

        fig, axes = plt.subplots(n_rows, 2, figsize=(12, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(columns):
            row = i // 2
            col_idx = i % 2
            ax = axes[row, col_idx]

            if col in summary_df.columns:
                summary_df[col].hist(ax=ax, bins=30, alpha=0.7, edgecolor='black')
                ax.set_title(f'{self._format_label(col)} Distribution')
                ax.set_xlabel(self._format_label(col))
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for i in range(len(columns), n_rows * 2):
            row = i // 2
            col_idx = i % 2
            axes[row, col_idx].set_visible(False)

        plt.suptitle(f'Simulation Summary Distributions{title_suffix}', fontsize=14)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_price_vs_pool(self, summary_df: pd.DataFrame,
                          title_suffix: str = "",
                          save_path: Optional[str] = None):
        """Plot scatter of BTC price vs pool participation.

        Args:
            summary_df: DataFrame with summary statistics
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(8, 6))

        plt.scatter(summary_df['btc_price'], summary_df['pct_pool'],
                   alpha=0.6, edgecolor='black', s=50)

        plt.xlabel('BTC Price ($)')
        plt.ylabel('Pool Participation Rate')
        plt.title(f'BTC Price vs Pool Participation{title_suffix}')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_strategy_distribution(self, summary_df: pd.DataFrame,
                                 title_suffix: str = "",
                                 save_path: Optional[str] = None):
        """Plot distribution of strategy choices.

        Args:
            summary_df: DataFrame with summary statistics
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        strategy_cols = ['pct_solo', 'pct_pool', 'pct_offline']
        available_cols = [col for col in strategy_cols if col in summary_df.columns]

        if not available_cols:
            print("No strategy columns found in summary DataFrame")
            return

        plt.figure(figsize=(10, 6))

        for col in available_cols:
            plt.hist(summary_df[col], bins=20, alpha=0.7,
                    label=self._format_label(col), edgecolor='black')

        plt.xlabel('Proportion')
        plt.ylabel('Frequency')
        plt.title(f'Strategy Choice Distributions{title_suffix}')
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def _format_label(self, col_name: str) -> str:
        """Format column name for display."""
        if col_name == 'btc_price':
            return 'BTC Price ($)'
        elif col_name == 'total_hash_rate':
            return 'Total Hash Rate (TH/s)'
        elif col_name == 'pct_active':
            return '% Active Miners'
        elif col_name == 'pct_pool':
            return '% Pool Miners'
        elif col_name == 'pct_solo':
            return '% Solo Miners'
        elif col_name == 'pct_offline':
            return '% Offline Miners'
        else:
            return col_name.replace('_', ' ').title()
