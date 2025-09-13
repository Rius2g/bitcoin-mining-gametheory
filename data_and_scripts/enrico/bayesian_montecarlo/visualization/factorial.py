from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from matplotlib.lines import Line2D

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class PlotFactorial:
    """Factorial analysis visualization for simulation results."""

    def __init__(self, config=None):
        self.config = config
        self.colors = {
            'solo': '#a8dadc',
            'pool': '#ffb4a2', 
            'offline': '#e9c46a'
        } if config is None else config.visualization.colors
        
        # Context colors for factorial analysis
        self.context_colors = {
            'good': '#b7e4c7',    # pastel green
            'average': '#ffd6a5', # pastel orange  
            'bad': '#ffb4a2'      # pastel red
        }

    def plot_context_violin_by_cost(self, 
                                   dfs_by_context: Dict[str, pd.DataFrame],
                                   cost_col: str = 'cost_per_kwh',
                                   hash_col: str = 'actual_hash_rate',
                                   strategies: list = None,
                                   cost_bins: int = 3,
                                   title_suffix: str = "",
                                   save_path: Optional[str] = None):
        """Create grid of violin plots: one row per context, one column per cost-bin.

        Based on legacy plot_context_violin_by_cost function.
        
        Args:
            dfs_by_context: Dict mapping context names to DataFrames
            cost_col: Column for electricity cost
            hash_col: Column for hash rate
            strategies: List of strategies to plot
            cost_bins: Number of cost bins
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        if strategies is None:
            strategies = ['solo', 'pool', 'offline']
            
        contexts = list(dfs_by_context.keys())
        
        # Define bin edges based on all data
        all_costs = pd.concat(dfs_by_context.values())[cost_col]
        if isinstance(cost_bins, int):
            bins = np.linspace(all_costs.min(), all_costs.max(), cost_bins + 1)
        else:
            bins = np.array(cost_bins)
        
        n_ctx = len(contexts)
        n_bins = len(bins) - 1
        
        fig, axes = plt.subplots(n_ctx, n_bins, figsize=(4*n_bins, 3*n_ctx), 
                                sharey='row')
        
        # Handle single context or single bin cases
        if n_ctx == 1 and n_bins == 1:
            axes = np.array([[axes]])
        elif n_ctx == 1:
            axes = axes.reshape(1, -1)
        elif n_bins == 1:
            axes = axes.reshape(-1, 1)
        
        # Create strategy legend
        legend_handles = [
            Line2D([0], [0], marker='s', markersize=15, 
                   markerfacecolor=self.colors[strat], 
                   markeredgecolor='black', linestyle='None', 
                   label=strat.capitalize())
            for strat in strategies
        ]
        
        # Compute global sample counts for width scaling
        global_counts = []
        for ctx in contexts:
            df = dfs_by_context[ctx].copy()
            df['_cost_bin'] = pd.cut(df[cost_col], bins=bins, labels=False, 
                                   include_lowest=True)
            for j in range(n_bins):
                subset = df[df['_cost_bin'] == j]
                for strat in strategies:
                    grp = subset[subset['strategy'] == strat]
                    global_counts.append(len(grp))
        
        max_count = max(global_counts) if global_counts else 1
        
        # Plot each context and cost bin
        for i, ctx in enumerate(contexts):
            df = dfs_by_context[ctx].copy()
            df['_cost_bin'] = pd.cut(df[cost_col], bins=bins, labels=False,
                                   include_lowest=True)
            
            for j in range(n_bins):
                ax = axes[i, j]
                subset = df[df['_cost_bin'] == j]
                
                # Prepare data for violin plot
                data = []
                widths = []
                positions = np.arange(len(strategies))
                
                for strat in strategies:
                    grp = subset[subset['strategy'] == strat]
                    
                    if strat == 'offline':
                        # For offline, show capacity (potential)
                        samples = grp['max_hash_rate'].values if 'max_hash_rate' in grp.columns else grp.get('capacity', [0.0])
                    else:
                        # For active strategies, show actual hash rate
                        samples = grp[hash_col].values if not grp.empty else np.array([0.0])
                    
                    if len(samples) == 0:
                        samples = np.array([0.0])
                        
                    data.append(samples)
                    
                    # Scale width by sample count
                    count = len(grp)
                    width = 0.15 + 0.45 * (count / max_count) if max_count > 0 else 0.6
                    widths.append(width)
                
                # Create violin plot
                parts = ax.violinplot(data, positions=positions, widths=widths,
                                    showmedians=True)
                
                # Color the violins
                for idx, body in enumerate(parts['bodies']):
                    strat = strategies[idx]
                    body.set_facecolor(self.colors[strat])
                    body.set_edgecolor('black')
                    body.set_alpha(0.7)
                
                # Add risk visualization as secondary y-axis
                ax_risk = ax.twinx()
                ax_risk.set_ylim(0, 1)
                ax_risk.spines['right'].set_color('#cccccc')
                ax_risk.tick_params(axis='y', colors='#cccccc', labelsize=10)
                
                # Only show risk label on leftmost plot
                if j == 0:
                    ax_risk.set_ylabel('Risk Aversion', color='#cccccc', fontsize=12)
                else:
                    ax_risk.set_ylabel("")
                ax_risk.yaxis.label.set_color('#cccccc')
                ax_risk.yaxis.set_tick_params(color='#cccccc')
                
                # Plot mean risk aversion with error bars for each strategy
                for idx, strat in enumerate(strategies):
                    grp = subset[subset['strategy'] == strat]
                    if not grp.empty and 'risk_aversion' in grp.columns:
                        mean_risk = grp['risk_aversion'].mean()
                        std_risk = grp['risk_aversion'].std()
                        
                        # Plot mean risk aversion point
                        ax_risk.scatter(positions[idx], mean_risk, color='black', 
                                       marker='o', s=40, zorder=5, 
                                       label='Mean Risk Aversion' if (i==0 and j==0 and idx==0) else None)
                        
                        # Add error bars
                        ax_risk.errorbar(positions[idx], mean_risk, yerr=std_risk, 
                                        fmt='none', ecolor='black', capsize=4, zorder=4)
                
                # Add percentage labels
                total_count = len(subset)
                for idx, strat in enumerate(strategies):
                    strat_count = len(subset[subset['strategy'] == strat])
                    percentage = strat_count / total_count * 100 if total_count > 0 else 0.0
                    ax.text(positions[idx], -0.08, f"{percentage:.1f}%", 
                           ha='center', va='top', fontsize=10, 
                           transform=ax.get_xaxis_transform())
                
                # Customize subplot
                if i == n_ctx - 1:  # Bottom row
                    ax.set_xlabel('Strategy')
                if j == 0:  # Left column
                    ax.set_ylabel(f'{ctx.title()}\n{hash_col.replace("_", " ").title()}')
                
                # Set title for top row
                if i == 0:
                    bin_start = bins[j]
                    bin_end = bins[j+1] 
                    ax.set_title(f'Cost: {bin_start:.3f}-{bin_end:.3f}')
                
                ax.set_xticks(positions)
                ax.set_xticklabels([s.capitalize() for s in strategies])
                ax.grid(True, alpha=0.3)
        
        # Add mean risk aversion to legend
        mean_risk_handle = Line2D([0], [0], color='black', marker='o', linestyle='None', 
                                 markersize=8, label='Mean Risk Aversion')
        all_handles = legend_handles + [mean_risk_handle]
        
        fig.legend(handles=all_handles, loc='upper center', 
                  ncol=len(all_handles), fontsize=12, frameon=False, 
                  bbox_to_anchor=(0.5, 1.02))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_context_comparison(self, dfs_by_context: Dict[str, pd.DataFrame],
                               metric: str = 'pct_pool',
                               title_suffix: str = "",
                               save_path: Optional[str] = None):
        """Compare a metric across different contexts.
        
        Args:
            dfs_by_context: Dict mapping context names to summary DataFrames
            metric: Metric to compare (e.g., 'pct_pool', 'pct_active')
            title_suffix: Suffix to append to title
            save_path: Path to save plot (optional)
        """
        contexts = list(dfs_by_context.keys())
        
        plt.figure(figsize=(10, 6))
        
        for i, ctx in enumerate(contexts):
            df = dfs_by_context[ctx]
            if metric in df.columns:
                values = df[metric].values
                positions = np.random.normal(i, 0.04, size=len(values))  # Add jitter
                
                plt.scatter(positions, values, 
                           color=self.context_colors.get(ctx, '#cccccc'),
                           alpha=0.6, s=30, label=ctx.title())
                
                # Add median line
                median_val = np.median(values)
                plt.plot([i-0.3, i+0.3], [median_val, median_val], 
                        color='black', linewidth=2)
        
        plt.xlabel('Market Context')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Across Contexts{title_suffix}')
        plt.xticks(range(len(contexts)), [c.title() for c in contexts])
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()