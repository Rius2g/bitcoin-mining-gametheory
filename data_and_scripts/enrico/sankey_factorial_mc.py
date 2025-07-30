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
        Appended to subplot titles.
    """
    contexts = list(dfs_by_context.keys())
    fig, axes = plt.subplots(len(contexts), cost_bins, figsize=(cost_bins * 4, len(contexts) * 3), sharex=True, sharey=True)

    for i, ctx in enumerate(contexts):
        df = dfs_by_context[ctx]
        df['cost_bin'] = pd.cut(df[cost_col], bins=cost_bins)

        for j, bin in enumerate(sorted(df['cost_bin'].dropna().unique())):
            ax = axes[i, j] if cost_bins > 1 else axes[i]
            sub = df[df['cost_bin'] == bin]
            for strat in strategies:
                sub_strat = sub[sub['strategy'] == strat]
                if not sub_strat.empty:
                    ax.violinplot(sub_strat[hash_col], positions=[strategies.index(strat)], showmeans=False, showmedians=True)
            ax.set_xticks(range(len(strategies)))
            ax.set_xticklabels(strategies)
            ax.set_title(f"{ctx.capitalize()} - {bin}{title_suffix}")
    plt.tight_layout()
    plt.show()

# -----------------------
# Sankey Plot Integration
# -----------------------

import plotly.graph_objects as go

def categorize_hash_power(h):
    # Individual miner hash rate, categories: low (100-999), medium (1000-1999), big (2000-3000)
    if h < 1000:
        return "low hash"
    elif h < 2000:
        return "medium hash"
    else:
        return "big hash"

def categorize_risk(r):
    if r < 0.3:
        return "low risk"
    elif r < 0.7:
        return "medium risk"
    else:
        return "high risk"

def plot_miner_sankey(df):
    df['hash_cat'] = df['hash_rate'].apply(categorize_hash_power)
    df['risk_cat'] = df['risk'].apply(categorize_risk)
    df['strategy'] = df['strategy'].fillna('offline')

    # Pool size categorization for pool miners only (50/100/200 EH/s)
    def categorize_pool_size(row):
        if row['strategy'] != 'pool':
            return None
        # Assume pool_hash column exists or set by simulation, else use hash_rate as proxy
        pool_hash = row.get('pool_hash', row['hash_rate'])
        if pool_hash < 100e6:
            return 'small pool'
        elif pool_hash < 200e6:
            return 'medium pool'
        else:
            return 'big pool'
    df['pool_size_cat'] = df.apply(categorize_pool_size, axis=1)

    # Build flows for each stage
    flows = []
    # Stage 1: hash_cat -> risk_cat
    flow1 = df.groupby(['hash_cat', 'risk_cat']).size().reset_index(name='count')
    flows.append(flow1)
    # Stage 2: risk_cat -> strategy
    flow2 = df.groupby(['risk_cat', 'strategy']).size().reset_index(name='count')
    flows.append(flow2)
    # Stage 3: strategy -> pool_size_cat (only for pool miners)
    flow3 = df[df['strategy']=='pool'].groupby(['strategy', 'pool_size_cat']).size().reset_index(name='count')
    flows.append(flow3)

    # Collect all unique labels
    labels = pd.concat([
        flows[0][flows[0].columns[0]],
        flows[0][flows[0].columns[1]],
        flows[1][flows[1].columns[1]],
        flows[2][flows[2].columns[1]]
    ]).dropna().unique().tolist()
    label_to_index = {label: i for i, label in enumerate(labels)}

    sources, targets, values = [], [], []
    for flow in flows:
        for _, row in flow.iterrows():
            src = label_to_index[row.iloc[0]]
            tgt = label_to_index[row.iloc[1]]
            val = row['count']
            sources.append(src)
            targets.append(tgt)
            values.append(val)

    fig = go.Figure(data=[go.Sankey(
        node=dict(label=labels, pad=15, thickness=20),
        link=dict(source=sources, target=targets, value=values)
    )])
    fig.update_layout(title="Miner Decision Flow (with Pool Size)", font_size=10)
    fig.show()

# NOTE:
# To use the Sankey diagram, after running your simulation and getting the resulting DataFrame
# (e.g., df containing columns: 'hash_rate', 'risk', 'strategy'), call:
# plot_miner_sankey(df)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-specs", required=True, help="Path to rig_specs.xlsx")
    parser.add_argument("--cost-rates", required=True, help="Path to cost_rates.xlsx")
    parser.add_argument("--miners", type=int, default=100, help="Number of miners to simulate")
    args = parser.parse_args()

    # Load rig and cost data
    rig_specs = pd.read_excel(args.rig_specs)
    cost_rates = pd.read_excel(args.cost_rates)

    # Instantiate simulator
    sim = MonteCarloSimulator(
        rig_specs=rig_specs,
        cost_rates=cost_rates,
        N=args.miners
    )

    # Run simulation
    results_df, miners_df = sim.run(record_miners=True)
    print(miners_df.head())

    # Plot Sankey if columns exist
    required_cols = {'hash_rate', 'risk', 'strategy'}
    if required_cols.issubset(miners_df.columns):
        plot_miner_sankey(miners_df)
    else:
        print("Missing columns for Sankey diagram:", required_cols - set(miners_df.columns))

    # --- Miner Regime Insights ---
    print("\n--- Miner Regime Insights ---")
    active = miners_df[miners_df.hash_rate > 0]
    # 1. Cost cut-off for participation
    cutoff_cost = active.cost.max()
    print(f"Cost cut-off (max cost among active): {cutoff_cost:.4f} $/kWh")
    # 2. Hash rate cut-off for participation
    cutoff_hash = active.hash_rate.min()
    print(f"Hash rate cut-off (min hash among active): {cutoff_hash:.2f} TH/s")
    # 3. Risk cut-off for participation
    cutoff_risk = active.risk.max()
    print(f"Risk cut-off (max risk among active): {cutoff_risk:.2f}")
    # 4. Strategy switch thresholds by cost
    offline_miners = miners_df[miners_df.strategy == 'offline']
    if not offline_miners.empty:
        first_offline_cost = offline_miners.cost.min()
        print(f"First cost where miners go offline: {first_offline_cost:.4f} $/kWh")
    else:
        print("No miners went offline.")
    # 5. Strategy distribution by cost regime
    cost_bins = pd.qcut(miners_df.cost, 3, labels=['low', 'medium', 'high'])
    strat_by_cost = miners_df.groupby([cost_bins, 'strategy']).size().unstack(fill_value=0)
    print("\nStrategy distribution by cost regime:\n", strat_by_cost)
    # 6. Strategy distribution by hash rate regime
    hash_bins = pd.qcut(miners_df.hash_rate, 3, labels=['low', 'medium', 'high'])
    strat_by_hash = miners_df.groupby([hash_bins, 'strategy']).size().unstack(fill_value=0)
    print("\nStrategy distribution by hash rate regime:\n", strat_by_hash)
    # 7. Strategy distribution by risk regime
    risk_bins = pd.qcut(miners_df.risk, 3, labels=['low', 'medium', 'high'])
    strat_by_risk = miners_df.groupby([risk_bins, 'strategy']).size().unstack(fill_value=0)
    print("\nStrategy distribution by risk regime:\n", strat_by_risk)
    # 8. Average and std of hash rate, cost, and risk for each strategy
    for strat in miners_df['strategy'].unique():
        sub = miners_df[miners_df.strategy == strat]
        print(f"\nStats for {strat}:")
        print(f"  Avg hash rate: {sub.hash_rate.mean():.2f} ± {sub.hash_rate.std():.2f} TH/s")
        print(f"  Avg cost: {sub.cost.mean():.4f} ± {sub.cost.std():.4f} $/kWh")
        print(f"  Avg risk: {sub.risk.mean():.2f} ± {sub.risk.std():.2f}")
    # 9. Pool size distribution for pool miners (if pool_hash column exists)
    if 'pool_hash' in miners_df.columns:
        pool_sizes = miners_df[miners_df.strategy == 'pool']['pool_hash']
        print(f"\nPool size distribution for pool miners:\n{pool_sizes.describe()}")

    # --- Table of all miner variable distributions ---
    print("\n--- Miner Variable Distributions Table ---")
    summary_table = miners_df[['cost', 'hash_rate', 'risk', 'capacity', 'efficiency']].describe().T
    print(summary_table)

