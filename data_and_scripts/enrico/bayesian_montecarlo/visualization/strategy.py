from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PlotStrategy:
    """Strategy-focused visualization for simulation results."""

    def __init__(self, config=None):
        self.config = config
        self.colors = {
            'solo': '#a8dadc',
            'pool': '#ffb4a2', 
            'offline': '#e9c46a'
        } if config is None else config.visualization.colors

    def plot_strategy_shares(self, df: pd.DataFrame, 
                           xcol: str = 'cost_per_kwh',
                           kind: str = 'area',
                           title_suffix: str = "",
                           save_path: Optional[str] = None):
        """Plot strategy shares along a continuous variable.

        Based on legacy plot_strategy_shares function.
        
        Args:
            df: DataFrame with miner data or summary data
            xcol: Column for x-axis (e.g., 'cost_per_kwh', 'btc_price')
            kind: 'area' for stacked area, 'bar' for stacked bars
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        # Check if already summarized or raw miner data
        if 'pct_active' in df.columns and 'pct_pool' in df.columns:
            summary = df.copy()
            summary['pct_offline'] = 1.0 - summary['pct_active']
            summary['pct_solo'] = summary['pct_active'] - summary['pct_pool']
        else:
            # Raw per-miner data - need to aggregate
            df_work = df.copy()
            
            # Create strategy indicators
            df_work['offline'] = (df_work['actual_hash_rate'] == 0).astype(float)
            df_work['solo'] = (df_work['strategy'] == 'solo').astype(float)
            df_work['pool'] = (df_work['strategy'] == 'pool').astype(float)
            
            # Group by xcol and calculate means (fractions)
            summary = df_work.groupby(xcol)[['offline', 'solo', 'pool']].mean().reset_index()
            summary.rename(columns={
                'offline': 'pct_offline',
                'solo': 'pct_solo', 
                'pool': 'pct_pool'
            }, inplace=True)

        # Sort by x-axis for smooth plotting
        summary = summary.sort_values(xcol)
        x = summary[xcol].values
        y_off = summary['pct_offline'].values
        y_solo = summary['pct_solo'].values
        y_pool = summary['pct_pool'].values

        plt.figure(figsize=(10, 6))

        if kind == 'area':
            # Stacked area plot
            plt.fill_between(x, 0, y_off, 
                           color=self.colors['offline'], alpha=0.7, label='Offline')
            plt.fill_between(x, y_off, y_off + y_solo,
                           color=self.colors['solo'], alpha=0.7, label='Solo')
            plt.fill_between(x, y_off + y_solo, y_off + y_solo + y_pool,
                           color=self.colors['pool'], alpha=0.7, label='Pool')
        
        elif kind == 'bar':
            # Stacked bar plot
            width = (x.max() - x.min()) / len(x) * 0.8  # Bar width
            plt.bar(x, y_off, width, color=self.colors['offline'], 
                   alpha=0.7, label='Offline')
            plt.bar(x, y_solo, width, bottom=y_off,
                   color=self.colors['solo'], alpha=0.7, label='Solo')
            plt.bar(x, y_pool, width, bottom=y_off + y_solo,
                   color=self.colors['pool'], alpha=0.7, label='Pool')
        
        else:
            raise ValueError("kind must be 'area' or 'bar'")

        plt.xlabel(xcol.replace('_', ' ').title())
        plt.ylabel('Fraction of Miners')
        plt.title(f"Strategy Shares vs. {xcol.replace('_', ' ').title()}{title_suffix}")
        plt.legend(loc='upper right')
        plt.grid(linestyle='--', alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_scatter_with_rug(self, df: pd.DataFrame,
                             xcol: str = 'cost_per_kwh',
                             ycol: str = 'actual_hash_rate', 
                             title_suffix: str = "",
                             save_path: Optional[str] = None):
        """Scatter plot with rug for offline miners at y=0.

        Based on legacy plot_scatter_with_rug function.
        
        Args:
            df: DataFrame with miner data
            xcol: Column for x-axis
            ycol: Column for y-axis  
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        # Pick three cost levels for clarity
        cost_vals = sorted(df[xcol].unique())
        low = cost_vals[0]
        med = cost_vals[len(cost_vals)//2] 
        high = cost_vals[-1]
        
        df3 = df[df[xcol].isin([low, med, high])].copy()
        
        # Split active vs offline
        active = df3[df3[ycol] > 0]
        offline = df3[df3[ycol] == 0]
        
        plt.figure(figsize=(10, 6))
        
        # Plot active miners as scatter
        for strategy in ['solo', 'pool']:
            subset = active[active['strategy'] == strategy]
            if not subset.empty:
                sizes = 50 + 150 * subset['risk_aversion'] if 'risk_aversion' in subset.columns else 100
                alphas = 0.3 + 0.7 * subset['risk_aversion'] if 'risk_aversion' in subset.columns else 0.7
                
                plt.scatter(subset[xcol], subset[ycol],
                           color=self.colors[strategy], 
                           s=sizes, alpha=alphas,
                           label=strategy.capitalize(),
                           edgecolor='white', linewidth=0.5)
        
        # Add rug plot for offline miners at y=0
        if not offline.empty:
            rug_y = np.zeros(len(offline))
            plt.scatter(offline[xcol], rug_y, 
                       color=self.colors['offline'], 
                       s=20, alpha=0.6, marker='|',
                       label='Offline (rug)')
        
        plt.xlabel(xcol.replace('_', ' ').title())
        plt.ylabel(ycol.replace('_', ' ').title())
        plt.title(f'Miner Strategies with Offline Rug{title_suffix}')
        plt.legend()
        plt.grid(linestyle='--', alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()

    def plot_activation_heatmap(self, df: pd.DataFrame,
                               cost_col: str = 'cost_per_kwh',
                               eff_col: str = 'efficiency',
                               x_bins: int = 10,
                               y_bins: int = 10,
                               title_suffix: str = "",
                               save_path: Optional[str] = None):
        """Plot heatmap of activation probability across cost and efficiency.

        Based on legacy plot_activation_heatmap function.
        
        Args:
            df: DataFrame with miner data
            cost_col: Column for electricity cost (x-axis)
            eff_col: Column for efficiency (y-axis)
            x_bins: Number of bins for cost
            y_bins: Number of bins for efficiency
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        df_work = df.copy()
        
        # Mark active miners
        df_work['active'] = (df_work['actual_hash_rate'] > 0).astype(float)
        
        # Create bins
        df_work['cost_bin'] = pd.cut(df_work[cost_col], bins=x_bins)
        df_work['eff_bin'] = pd.cut(df_work[eff_col], bins=y_bins)
        
        # Pivot to get mean activation per (eff_bin, cost_bin)
        pivot = df_work.pivot_table(
            index='eff_bin',
            columns='cost_bin', 
            values='active',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        
        # Create heatmap
        im = plt.imshow(pivot.values, cmap='RdYlBu_r', aspect='auto',
                       vmin=0, vmax=1, origin='lower')
        
        # Customize ticks and labels
        plt.colorbar(im, label='Activation Probability')
        
        # Set x-axis (cost bins)
        x_labels = [f'{interval.left:.3f}-{interval.right:.3f}' 
                   for interval in pivot.columns]
        plt.xticks(range(len(x_labels)), x_labels, rotation=45)
        
        # Set y-axis (efficiency bins) 
        y_labels = [f'{interval.left:.1f}-{interval.right:.1f}'
                   for interval in pivot.index]
        plt.yticks(range(len(y_labels)), y_labels)
        
        plt.xlabel(f'{cost_col.replace("_", " ").title()} Bins')
        plt.ylabel(f'{eff_col.replace("_", " ").title()} Bins')
        plt.title(f'Miner Activation Heatmap{title_suffix}')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        plt.show()