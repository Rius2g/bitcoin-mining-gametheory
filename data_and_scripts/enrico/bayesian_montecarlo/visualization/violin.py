from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PlotViolin:
    """Violin plot visualization for simulation results."""

    def __init__(self, config=None):
        self.config = config
        self.colors = {
            'solo': '#a8dadc',
            'pool': '#ffb4a2',
            'offline': '#e9c46a',
            'potential': '#a8dadc',  # For potential hash rate
            'actual': '#ffb4a2'      # For actual hash rate
        } if config is None else {**config.visualization.colors, 
                                  'potential': '#a8dadc', 'actual': '#ffb4a2'}

    def plot_potential_vs_actual(self, df: pd.DataFrame,
                               cost_bins: int = 3,
                               title_suffix: str = "",
                               save_path: Optional[str] = None):
        """Plot violin plots of potential vs actual hash rates by cost level.

        Based on legacy plot_potential_vs_actual function.

        Args:
            df: DataFrame with miner data
            cost_bins: Number of cost bins to create
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        # Pick three cost levels like in legacy code
        cost_vals = sorted(df['cost_per_kwh'].unique())
        low = cost_vals[0]
        med = cost_vals[len(cost_vals)//2] 
        high = cost_vals[-1]
        
        df3 = df[df['cost_per_kwh'].isin([low, med, high])].copy()
        level_map = {low: 'Low', med: 'Medium', high: 'High'}
        df3['energy_level'] = df3['cost_per_kwh'].map(level_map)

        # Prepare data arrays
        levels = ['Low', 'Medium', 'High']
        positions = np.arange(1, len(levels) + 1)
        offset = 0.2
        
        potential_data = []
        actual_data = []
        
        for level in levels:
            level_data = df3[df3['energy_level'] == level]
            # Potential = max_hash_rate (capacity)
            potential_data.append(level_data['max_hash_rate'].values)
            # Actual = actual_hash_rate  
            actual_data.append(level_data['actual_hash_rate'].values)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot potential (capacity)
        vp_pot = ax.violinplot(potential_data, positions=positions - offset,
                              widths=0.3, showmeans=False, showmedians=True)

        # Plot actual
        vp_act = ax.violinplot(actual_data, positions=positions + offset,
                              widths=0.3, showmeans=False, showmedians=True)

        # Color the potential bodies
        for body in vp_pot['bodies']:
            body.set_facecolor(self.colors['potential'])
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        
        # Color the actual bodies  
        for body in vp_act['bodies']:
            body.set_facecolor(self.colors['actual'])
            body.set_edgecolor('black')
            body.set_alpha(0.7)

        # Customize plot
        ax.set_xticks(positions)
        ax.set_xticklabels(levels)
        ax.set_xlabel('Energy Cost Level')
        ax.set_ylabel('Hash Rate (TH/s)')
        ax.set_title(f'Potential vs. Actual Hash Rate by Cost Level{title_suffix}')

        # Create legend
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=self.colors['potential'], edgecolor='black', 
                  label='Potential', alpha=0.7),
            Patch(facecolor=self.colors['actual'], edgecolor='black', 
                  label='Actual', alpha=0.7)
        ]
        ax.legend(handles=legend_handles, loc='upper right')

        ax.grid(linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_strategy_by_cost(self, df: pd.DataFrame,
                            cost_bins: int = 3,
                            title_suffix: str = "",
                            save_path: Optional[str] = None):
        """Plot strategy distribution by cost level using violins.

        Args:
            df: DataFrame with miner data
            cost_bins: Number of cost bins
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        # Create cost bins
        df = df.copy()
        df['cost_bin'] = pd.qcut(df['cost_per_kwh'], cost_bins,
                                labels=[f'Low ({i+1})' for i in range(cost_bins)])

        # Prepare data by strategy
        strategies = ['solo', 'pool', 'offline']
        bin_labels = sorted(df['cost_bin'].unique())

        fig, axes = plt.subplots(1, len(strategies), figsize=(15, 5))

        for i, strategy in enumerate(strategies):
            ax = axes[i]

            strategy_data = []
            for bin_label in bin_labels:
                subset = df[(df['cost_bin'] == bin_label) & (df['strategy'] == strategy)]
                if strategy == 'offline':
                    values = subset['max_hash_rate'].values  # Use capacity for offline
                else:
                    values = subset['actual_hash_rate'].values
                strategy_data.append(values if len(values) > 0 else np.array([0.0]))

            positions = np.arange(len(bin_labels))
            vp = ax.violinplot(strategy_data, positions=positions,
                             widths=0.8, showmeans=True, showmedians=True)

            # Color violins
            color = self.colors.get('potential', 'blue') if strategy == 'offline' else self.colors.get('actual', 'red')
            for body in vp['bodies']:
                body.set_facecolor(color)
                body.set_edgecolor('black')
                body.set_alpha(0.7)

            ax.set_xticks(positions)
            ax.set_xticklabels(bin_labels)
            ax.set_xlabel('Cost Level')
            ax.set_ylabel('Hash Rate (TH/s)')
            ax.set_title(f'{strategy.capitalize()} Strategy{title_suffix}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_risk_by_strategy(self, df: pd.DataFrame,
                            title_suffix: str = "",
                            save_path: Optional[str] = None):
        """Plot risk distribution by strategy.

        Args:
            df: DataFrame with miner data
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        strategies = df['strategy'].unique()
        strategy_data = []

        for strategy in strategies:
            subset = df[df['strategy'] == strategy]
            if 'risk_aversion' in subset.columns:
                strategy_data.append(subset['risk_aversion'].values)
            else:
                strategy_data.append(np.array([]))

        plt.figure(figsize=(10, 6))

        positions = np.arange(len(strategies))
        vp = plt.violinplot(strategy_data, positions=positions,
                           widths=0.8, showmeans=True, showmedians=True)

        # Color violins
        colors = ['#a8dadc', '#ffb4a2', '#e9c46a']  # solo, pool, offline
        for i, body in enumerate(vp['bodies']):
            color = colors[i % len(colors)]
            body.set_facecolor(color)
            body.set_edgecolor('black')
            body.set_alpha(0.7)

        plt.xticks(positions, [s.capitalize() for s in strategies])
        plt.xlabel('Strategy')
        plt.ylabel('Risk Aversion')
        plt.title(f'Risk Distribution by Strategy{title_suffix}')
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()
