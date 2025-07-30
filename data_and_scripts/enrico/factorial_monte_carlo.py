import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from monte_carlo import (
    MonteCarloSimulator,
    sample_price,
)

# Function: Faceted violin plot by cost-bins per context
def plot_context_violin_by_cost(
    dfs_by_context,
    cost_col='cost',
    hash_col='hash_rate',
    strategies=['solo', 'pool', 'offline'],
    cost_bins=3,
    colors=None,
    title_suffix=""
):
    """
    Creates a grid of violin plots: one row per context, one column per cost-bin.
    Each cell shows hash-rate distribution for solo/pool/offline in that cost range.

    Parameters
    ----------
    dfs_by_context : dict of DataFrame
        Keys are context labels ('good','avg','low'), values are DataFrames with columns:
        cost_col, hash_col, 'capacity', 'strategy', 'risk'.
    cost_bins : int or sequence
        Number of equal-width bins (or bin edges) for electricity cost.
    colors : dict
        Mapping strategy to color.
    title_suffix : str
        Text to append to figure title.
    """
    if colors is None:
        colors = {'solo': '#a8dadc', 'pool': '#ffb4a2', 'offline': '#e9c46c'}

    import matplotlib.patheffects as path_effects
    contexts = list(dfs_by_context.keys())
    # define bins edges
    all_costs = pd.concat(dfs_by_context.values())[cost_col]
    if isinstance(cost_bins, int):
        bins = np.linspace(all_costs.min(), all_costs.max(), cost_bins + 1)
    else:
        bins = np.array(cost_bins)

    n_ctx = len(contexts)
    n_bins = len(bins) - 1
    fig, axes = plt.subplots(n_ctx, n_bins, figsize=(4*n_bins, 3*n_ctx), sharey='row')
    # Define context colors
    context_colors = {
        'good': '#b7e4c7',   # pastel green
        'avg':  '#ffd6a5',   # pastel orange
        'bad':  '#ffb4a2'    # pastel red
    }

    # Create a single legend for strategies/colors
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker='s', markersize=15, markerfacecolor=colors[strat], markeredgecolor='black', linestyle='None', label=strat.capitalize())
        for strat in strategies
    ]

    # --- Compute global sample counts for all strategies in all bins ---
    global_counts = []
    for i, ctx in enumerate(contexts):
        df = dfs_by_context[ctx].copy()
        df['_cost_bin'] = pd.cut(df[cost_col], bins=bins, labels=False, include_lowest=True)
        for j in range(n_bins):
            subset = df[df['_cost_bin'] == j]
            for strat in strategies:
                grp = subset[subset['strategy'] == strat]
                count = len(grp)
                global_counts.append(count)
    max_count = max(global_counts) if global_counts else 1

    # --- Plot with consistent width scaling by sample count ---
    for i, ctx in enumerate(contexts):
        df = dfs_by_context[ctx].copy()
        df['_cost_bin'] = pd.cut(df[cost_col], bins=bins, labels=False, include_lowest=True)
        row_color = context_colors.get(ctx, '#cccccc')
        for j in range(n_bins):
            ax = axes[i, j] if n_ctx > 1 else axes[j]
            subset = df[df['_cost_bin'] == j]
            data = []
            widths = []
            positions = np.arange(len(strategies))
            for strat in strategies:
                grp = subset[subset['strategy'] == strat]
                if strat == 'offline':
                    samples = grp['capacity'].values if not grp.empty else np.array([0.0])
                else:
                    samples = grp[hash_col].values if not grp.empty else np.array([0.0])
                data.append(samples)
                count = len(grp)
                width = 0.15 + 0.45 * (count / max_count) if max_count > 0 else 0.6
                widths.append(width)
            parts = ax.violinplot(data, positions=positions, widths=widths, showmedians=True)
            for idx, body in enumerate(parts['bodies']):
                strat = strategies[idx]
                body.set_facecolor(colors[strat])
                body.set_edgecolor('black')
                body.set_alpha(0.7)
            total_count = len(subset)
            ax_risk = ax.twinx()
            ax_risk.set_ylim(0, 1)
            ax_risk.spines['right'].set_color('#cccccc')
            ax_risk.tick_params(axis='y', colors='#cccccc', labelsize=10)
            if i == 0 and j == 0:
                ax_risk.set_ylabel('Mean Risk', color='#cccccc', fontsize=12)
            else:
                ax_risk.set_ylabel("")
            ax_risk.yaxis.label.set_color('#cccccc')
            ax_risk.yaxis.set_tick_params(color='#cccccc')
            for idx, strat in enumerate(strategies):
                strat_count = len(subset[subset['strategy'] == strat])
                percentage = strat_count / total_count * 100 if total_count > 0 else 0.0
                ax.text(positions[idx], -0.08, f"{percentage:.1f}%", ha='center', va='top', fontsize=12, color='black', fontweight='normal', transform=ax.get_xaxis_transform())
                grp = subset[subset['strategy'] == strat]
                if not grp.empty and 'risk' in grp:
                    mean_risk = grp['risk'].mean()
                    std_risk = grp['risk'].std()
                    ax_risk.scatter(positions[idx], mean_risk, color='black', marker='o', s=40, zorder=5, label='Mean Risk' if (i==0 and j==0 and idx==0) else None)
                    ax_risk.errorbar(positions[idx], mean_risk, yerr=std_risk, fmt='none', ecolor='black', capsize=4, zorder=4)
            ax.set_xticks(positions)
            ax.set_xticklabels(["" for _ in strategies])
            # Set subplot background to 90% transparent context color
            ax.set_facecolor(row_color)
            ax.patch.set_alpha(0.1)
            # Remove colored border and add light gray grid
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1)
            ax.grid(True, color='#dddddd', linestyle='-', linewidth=1, alpha=0.7)
            # Only show country/cost at top row
            if i == 0:
                bin_min = bins[j]
                bin_max = bins[j+1]
                try:
                    cost_rates = pd.read_excel('cost_rates.xlsx')
                except Exception:
                    cost_rates = None
                if cost_rates is not None:
                    mask = (cost_rates['cost'] >= bin_min) & (cost_rates['cost'] < bin_max)
                    countries_in_bin = cost_rates['country'][mask].tolist()
                    costs_in_bin = cost_rates['cost'][mask].tolist()
                    if countries_in_bin:
                        country_label = countries_in_bin[0]
                        min_cost = costs_in_bin[0]
                        title_txt = f"{country_label} (~{min_cost:.3f} $/kWh)"
                        cost_val = min_cost
                    else:
                        title_txt = f"~{bin_min:.3f}-{bin_max:.3f} $/kWh"
                        cost_val = (bin_min + bin_max) / 2
                else:
                    title_txt = f"~{bin_min:.3f}-{bin_max:.3f} $/kWh"
                    cost_val = (bin_min + bin_max) / 2
                ax.set_title(title_txt, fontsize=13, color='black')
            else:
                ax.set_title("")
            if i == 0 and j == 0:
                ax.set_ylabel("Hash Rate/Capacity", fontsize=15)
            else:
                ax.set_ylabel("")
        # Add context label box at the start of each row (leftmost subplot)
        left_ax = axes[i, 0] if n_bins > 1 else axes[0]
        left_ax.annotate(
            ctx,
            xy=(-0.5, 1.05),
            xycoords='axes fraction',
            fontsize=18,
            color='black',
            ha='left',
            va='center',
            bbox=dict(boxstyle='round,pad=0.4', fc=row_color, ec=row_color, alpha=0.98)
        )
    # Remove figure title
    # Add legend above all subplots, including mean risk
    from matplotlib.lines import Line2D
    mean_risk_handle = Line2D([0], [0], color='black', marker='o', linestyle='None', markersize=8, label='Mean Risk')
    fig.legend(handles=legend_handles + [mean_risk_handle], loc='upper center', ncol=len(strategies)+1, fontsize=15, frameon=False, bbox_to_anchor=(0.5, 1.02))
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-context Bitcoin mining Monte Carlo")
    parser.add_argument("--rig-specs", type=str, required=True,
                        help="Path to Excel file with rig specifications")
    parser.add_argument("--cost-rates", type=str, required=True,
                        help="Path to Excel file with electricity costs")
    parser.add_argument("--miners", "-n", type=int, default=500,
                        help="Number of miners per context draw")
    parser.add_argument("--bins", type=int, default=3,
                        help="Number of cost bins")
    args = parser.parse_args()

    # Load data
    rig_specs = pd.read_excel(args.rig_specs)
    cost_rates = pd.read_excel(args.cost_rates)

    # Instantiate simulator
    sim = MonteCarloSimulator(
        N=args.miners,
        rig_specs=rig_specs,
        cost_rates=cost_rates,
        price_sampler=sample_price,
        R=3.125,
        M=0.035,
        tau=0.5,
        T=600,
        pool_fee=0.01
    )

    # Define contexts
    contexts = {
        'good': {'price': 200000, 'H': 400e6, 'M': 0.05},
        'avg':  {'price': 120000, 'H': 600e6, 'M': 0.025},
        'bad':  {'price':  60000, 'H': 1000e6, 'M': 0.01}
    }

    # Run single-draw for each context
    dfs_by_context = {}
    for label, params in contexts.items():
        sim.price_sampler = lambda *args, p=params['price']: p
        sim.network_hash_const = params['H']
        sim.M = params['M']
        # Randomly choose pool size: small (50e6), medium (100e6), or big (200e6)
        pool_sizes = [50e6, 100e6, 200e6]
        pool_hash_override = np.random.choice(pool_sizes)
        dfs_by_context[label] = sim.run_single(pool_hash_override=pool_hash_override)

    # Plot faceted violins by cost-bin
    plot_context_violin_by_cost(dfs_by_context, cost_bins=args.bins, title_suffix=" (Faceted)")
