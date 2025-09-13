# monte_carlo.py
# -------------------------------
# Monte Carlo Simulation of Miner Decisions
# with Fixed Network Hash and Pool‐Size Scenarios
# -------------------------------

from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import pandas as pd
from typing import Callable, List

# -------------------------------
# Sampling Functions (unchanged)
# -------------------------------
def sample_price(vol=0.47, last_price=119000):
    mu = np.log(last_price) - 0.5*vol**2
    return float(np.random.lognormal(mu, vol))

def sample_cost_excel(df_cost):
    return float(df_cost['cost'].sample(1).iloc[0])

def sample_eff_excel(df_rigs):
    return float(df_rigs['efficiency'].sample(1).iloc[0])

def sample_capacity_excel(df_rigs):
    return float(df_rigs['capacity_THs'].sample(1).iloc[0])

def sample_risk():
    return float(np.random.rand())

def cost_mixture_sampler():
    if np.random.rand() < 0.5:
        return np.random.normal(0.03, 0.005)   # low‐cost cluster
    else:
        return np.random.normal(0.10, 0.010)   # high‐cost cluster

# -------------------------------
# Miner Class with Profit Formulas
# -------------------------------
@dataclass
class Miner:
    cost: float           # $/kWh
    efficiency: float     # J per TH
    risk: float           # 0–1 scale
    max_hash: float       # TH/s
    # these will be set by decide():
    network_hash: float   = field(init=False, repr=False)
    pool_hash: float      = field(init=False, repr=False)
    R: float              = field(init=False, repr=False)
    M: float              = field(init=False, repr=False)
    tau: float            = field(init=False, repr=False)
    T: float              = field(init=False, repr=False)
    pool_fee: float       = field(init=False, repr=False)
    price: float          = field(init=False, repr=False)
    strategy: str         = field(init=False, default="offline")
    hash_rate: float      = field(init=False, default=0.0)

    def cost_per_day(self) -> float:
        # cost($/kWh) * (efficiency J/TH / 3.6e6 J/kWh) * max_hash(TH/s) * 86400s/day
        return self.cost * (self.efficiency / 3.6e6) * self.max_hash * 86400

    def revenue_solo(self) -> float:
        # Expected blocks per day: (hash rate / network hash rate) * blocks per day
        blocks_per_day = 86400 / self.T  # seconds per day / block time
        expected_blocks = self.max_hash / (self.network_hash + self.max_hash) * blocks_per_day
        # Discounted block reward
        block_reward = (self.R + self.M) * np.exp(-self.tau/self.T)
        # Daily revenue in USD
        return expected_blocks * block_reward * self.price

    def revenue_pool(self) -> float:
        # Expected blocks per day found by pool
        blocks_per_day = 86400 / self.T
        expected_blocks_pool = self.pool_hash / self.network_hash * blocks_per_day
        # Your share of pool reward
        miner_share = self.max_hash / self.pool_hash if self.pool_hash > 0 else 0.0
        # Discounted block reward after pool fee
        block_reward = (self.R + self.M) * np.exp(-self.tau/self.T) * (1 - self.pool_fee)
        # Daily revenue in USD
        return expected_blocks_pool * miner_share * block_reward * self.price


    def decide(self, price, network_hash, pool_hash,
               R, M, tau, T, pool_fee):
        self.price = price
        self.network_hash = network_hash
        self.pool_hash    = pool_hash
        self.R = R; self.M = M; self.tau = tau; self.T = T; self.pool_fee = pool_fee

        rev_s = self.revenue_solo()
        rev_p = self.revenue_pool()
        cost  = self.cost_per_day()
        var_pen = self.risk * rev_s  # variance penalty approx

        u_s = rev_s - cost - var_pen
        u_p = rev_p - cost

        # Choose strategy with highest utility
        if max(u_s, u_p, 0) == u_s and u_s > 0:
            self.strategy = 'solo'
            self.hash_rate = self.max_hash
        elif max(u_s, u_p, 0) == u_p:
            self.strategy = 'pool'
            self.hash_rate = self.max_hash
        else:
            self.strategy = 'offline'
            self.hash_rate = 0

# -------------------------------
# Monte Carlo Simulator
# -------------------------------
class MonteCarloSimulator:
    def __init__(self,
                 N: int,
                 rig_specs: pd.DataFrame,
                 cost_rates: pd.DataFrame,
                 price_sampler: Callable = sample_price,
                 R: float = 3.125,
                 M: float = 0.035,  # 0.035 BTC block subsidy
                 tau: float = 0.5,
                 T: float = 600,
                 pool_fee: float = 0.025):
        self.N = N
        self.rig_specs = rig_specs
        self.cost_rates = cost_rates
        self.price_sampler = price_sampler
        # THESE ARE NOW CONSTANT:
        self.network_hash_const = 990e6  # 990 EH/s in TH/s
        self.R = R; self.M = M; self.tau = tau; self.T = T; self.pool_fee = pool_fee

    def rig_sampler(self):
        return (
            sample_capacity_excel(self.rig_specs),
            sample_eff_excel(self.rig_specs)
        )

    def cost_sampler(self):
        return sample_cost_excel(self.cost_rates)

    def risk_sampler(self):
        return sample_risk()

    def run_single(self, pool_hash_override: float = None) -> pd.DataFrame:
        price = self.price_sampler()
        H_tot = self.network_hash_const

        # draw all miners
        miners: List[Miner] = []
        for _ in range(self.N):
            cap, eff = self.rig_sampler()
            cost = self.cost_sampler()
            risk = self.risk_sampler()
            # multiple rigs per miner
            n_rigs = np.random.randint(1, 11)  # 1 to 10 rigs
            total_hash = cap * n_rigs  # total hash rate for this miner
            miners.append(Miner(cost, eff, risk, total_hash))

        # init everyone at full hash
        for m in miners:
            m.hash_rate = m.max_hash

        # pool‐hash: either override or sum of those who pick ‘pool’
        pool_actual = pool_hash_override if pool_hash_override is not None else 0.0

        # iterate best responses
        converged = False
        it = 0
        max_iter = 10
        while not converged and it < max_iter:
            prev_strats = [m.strategy for m in miners]
            for m in miners:
                m.decide(price, H_tot, pool_actual,
                         self.R, self.M, self.tau, self.T, self.pool_fee)
            converged = all(m.strategy == old
                            for m, old in zip(miners, prev_strats))
            it += 1
            # keep H_tot constant; update pool_actual only if endogenous
            if pool_hash_override is None:
                pool_actual = sum(m.hash_rate for m in miners if m.strategy == 'pool')

        df = pd.DataFrame([{
            'cost': m.cost,
            'efficiency': m.efficiency,
            'risk': m.risk,
            'capacity': m.max_hash,
            'hash_rate': m.hash_rate,
            'strategy': m.strategy
        } for m in miners])
        # store some attrs for summary
        df.attrs['price']  = price
        df.attrs['H_tot']  = H_tot
        df.attrs['pool_hash'] = pool_actual
        return df

    def run(self, draws=500, pool_hash_override: float = None, record_miners: bool = False):
        rec = []
        miner_dfs = []
        for draw_idx in range(draws):
            df = self.run_single(pool_hash_override)
            rec.append({
                'price':      df.attrs['price'],
                'H_tot':      df.attrs['H_tot'],
                'pool_hash':  df.attrs['pool_hash'],
                'total_hash': df['hash_rate'].sum(),
                'pct_active': (df['hash_rate']>0).mean(),
                'pct_pool':   (df['strategy']=='pool').mean()
            })
            if record_miners:
                df = df.copy()
                df['draw_idx'] = draw_idx
                miner_dfs.append(df)
        summary_df = pd.DataFrame(rec)
        if record_miners:
            miners_df = pd.concat(miner_dfs, ignore_index=True)
            return summary_df, miners_df
        else:
            return summary_df

# -------------------------------
# Plotting Helpers (unchanged, except titles below)
# -------------------------------
def plot_scatter(df, title_suffix=""):
    colors = {'solo':'#a8dadc','pool':'#ffb4a2','offline':'#e9c46a'}
    plt.figure(figsize=(8,5))
    for strat, col in colors.items():
        sub = df[df['strategy']==strat]
        if sub.empty:
            continue     # skip any group with zero points

        # marker size grows with risk, alpha grows with risk
        sizes = 50 + 150 * sub['risk']        # from 50 to 200
        alphas = 0.3 + 0.7 * sub['risk']      # from 0.3 to 1.0

        plt.scatter(
            sub['cost'],
            sub['hash_rate'],
            label=strat,
            color=col,
            edgecolor='white',
            s=sizes,
            alpha=alphas
        )
    plt.xlabel('Electricity Cost ($/kWh)')
    plt.ylabel('Hash Rate (TH/s)')
    plt.title(f'One Simulation Draw{title_suffix}')
    plt.legend(); plt.grid(linestyle='--',alpha=0.5); plt.tight_layout()
    plt.show()

def plot_dist(summary, title_suffix=""):
    fig, axes = plt.subplots(2,2, figsize=(10,8))
    summary['price'].hist(ax=axes[0,0])
    axes[0,0].set_title('BTC Price')
    summary['total_hash'].hist(ax=axes[0,1])
    axes[0,1].set_title('Total Active Hash')
    summary['pct_active'].hist(ax=axes[1,0])
    axes[1,0].set_title('% Active')
    summary.plot.scatter(x='price',y='pct_pool',ax=axes[1,1],alpha=0.6)
    axes[1,1].set_title(f'% Pool vs Price{title_suffix}')
    plt.tight_layout(); plt.show()

def plot_scatter_with_rug(df, title_suffix=""):
    """
    Scatter of hash_rate vs. electricity cost for the three energy‐cost levels,
    plus a rug at hash_rate=0 marking every offline miner. Active miners are
    all plotted with the same marker, sized by risk and colored by strategy.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    xcol = 'cost'  # fix x-axis to electricity cost

    # 1) pick the three cost‐levels
    cost_vals = sorted(df['cost'].unique())
    low, med, high = cost_vals[0], cost_vals[len(cost_vals)//2], cost_vals[-1]
    df3 = df[df['cost'].isin([low, med, high])].copy()

    # 2) split active vs. offline
    active  = df3[df3['hash_rate'] >  0].copy()
    offline = df3[df3['hash_rate'] == 0]

    # 3) compute overlap‐based alpha for active points
    counts = active.groupby([xcol, 'hash_rate'])['risk'].transform('size')
    if counts.max() > 1:
        dens_norm = (counts - 1) / (counts.max() - 1)
    else:
        dens_norm = np.zeros_like(counts, dtype=float)
    base_alpha = 0.3 + 0.7 * active['risk']
    active['alpha'] = base_alpha + (1 - base_alpha) * dens_norm

    # 4) plot
    plt.figure(figsize=(8,5))
    colors = {'solo':'#a8dadc','pool':'#ffb4a2'}
    marker = 'o'

    # active miners, one marker by strategy
    for strat, col in colors.items():
        sub = active[active['strategy']==strat]
        if sub.empty:
            continue
        sizes = 50 + 150 * sub['risk']  # from 50 to 200
        plt.scatter(
            sub[xcol], sub['hash_rate'],
            label=strat,
            color=col,
            marker=marker,
            edgecolor='white',
            s=sizes,
            alpha=sub['alpha']
        )

    # rug for offline miners
    if not offline.empty:
        x_off = offline[xcol].values
        y_off = np.zeros_like(x_off)
        plt.plot(x_off, y_off, '|', markersize=8, alpha=0.5, color='gray')
        legend_off = Line2D([0],[0], marker='|', color='gray', linestyle='None',
                            markersize=8, alpha=0.5, label='offline')
        handles, labels = plt.gca().get_legend_handles_labels()
        handles.append(legend_off)
        labels.append('offline')
        plt.legend(handles, labels, ncol=2, fontsize='small')

    plt.xlabel('Electricity Cost ($/kWh)')
    plt.ylabel('Hash Rate (TH/s)')
    plt.title(f'Hash Rate vs. Electricity Cost with Offline Rug{title_suffix}')
    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
# -------------------------------

def plot_strategy_shares(df, xcol='cost', kind='area', title_suffix=""):
    """
    Plots the share of miners choosing Offline, Solo, and Pool along xcol.
    
    - If df contains 'pct_active' & 'pct_pool', uses those directly.
    - Otherwise assumes df is raw (one row per miner) and groups by xcol.
    
    Parameters
    ----------
    df : pandas.DataFrame
      Either a per-miner table with ['cost','hash_rate','strategy'], or a
      summary with [xcol, 'pct_active','pct_pool'].
    xcol : str
      Column to use on horizontal axis (e.g. 'cost', 'price').
    kind : {'area','bar'}
      'area' for stacked-area; 'bar' for stacked bars.
    title_suffix : str
      Suffix appended to the title.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # ---- prepare summary_df with columns: xcol, pct_offline, pct_solo, pct_pool
    if 'pct_active' in df.columns and 'pct_pool' in df.columns:
        summary = df.copy()
        summary['pct_offline'] = 1.0 - summary['pct_active']
        summary['pct_solo']    = summary['pct_active'] - summary['pct_pool']
    else:
        # raw per-miner df
        df2 = df.copy()
        # boolean masks
        df2['offline'] = (df2['hash_rate'] == 0).astype(float)
        df2['solo']    = (df2['strategy'] == 'solo').astype(float)
        df2['pool']    = (df2['strategy'] == 'pool').astype(float)
        # group & mean to get fractions
        summary = df2.groupby(xcol)[['offline','solo','pool']].mean().reset_index()
        summary.rename(columns={
            'offline': 'pct_offline',
            'solo':    'pct_solo',
            'pool':    'pct_pool'
        }, inplace=True)

    # sort by xcol for area plots
    summary = summary.sort_values(xcol)
    x       = summary[xcol].values
    y_off   = summary['pct_offline'].values
    y_solo  = summary['pct_solo'].values
    y_pool  = summary['pct_pool'].values

    colors = {
        'offline': '#e9c46a',
        'solo':    '#a8dadc',
        'pool':    '#ffb4a2'
    }

    plt.figure(figsize=(10,6))

    if kind == 'area':
        plt.stackplot(
            x,
            y_off, y_solo, y_pool,
            labels=['Offline','Solo','Pool'],
            colors=[colors['offline'], colors['solo'], colors['pool']],
            alpha=0.8
        )
    elif kind == 'bar':
        # bar width: small fraction of total span
        if len(x) > 1:
            width = (x.max() - x.min()) / (len(x)*1.5)
        else:
            width = 0.8
        plt.bar(x, y_off,  width=width, label='Offline', color=colors['offline'])
        plt.bar(x, y_solo, width=width, bottom=y_off,            label='Solo',   color=colors['solo'])
        plt.bar(x, y_pool, width=width, bottom=y_off + y_solo,   label='Pool',   color=colors['pool'])
    else:
        raise ValueError("kind must be 'area' or 'bar'")

    plt.xlabel(xcol.replace('_',' ').title())
    plt.ylabel('Fraction of Miners')
    plt.title(f"Strategy Shares vs. {xcol.replace('_',' ').title()}{title_suffix}")
    plt.legend(loc='upper right', ncol=1)
    plt.grid(linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
# -------------------------------

def plot_potential_vs_actual(
    df,
    title_suffix="",
    kind="violin",
    potential_color="#a8dadc",
    actual_color="#ffb4a2"
):
    """
    Plots the distribution of potential (capacity) vs. actual (hash_rate)
    hash rates for miners across three energy cost levels (Low, Medium, High).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ['cost', 'capacity', 'hash_rate'].
    title_suffix : str, default ""
        Suffix to append to the plot title.
    kind : {'violin', 'boxplot'}, default 'violin'
        Type of plot to draw.
    potential_color : color, default '#a8dadc'
        Color for the potential distribution bodies/boxes.
    actual_color : color, default '#ffb4a2'
        Color for the actual distribution bodies/boxes.
    """
    from matplotlib.patches import Patch

    # 1) pick the three cost‐levels
    cost_vals = sorted(df['cost'].unique())
    low, med, high = cost_vals[0], cost_vals[len(cost_vals)//2], cost_vals[-1]
    df3 = df[df['cost'].isin([low, med, high])].copy()
    level_map = {low: 'Low', med: 'Medium', high: 'High'}
    df3['energy_level'] = df3['cost'].map(level_map)

    # 2) prepare data arrays
    levels    = ['Low', 'Medium', 'High']
    positions = np.arange(1, len(levels) + 1)
    offset    = 0.2
    potential_data = [df3[df3['energy_level'] == lvl]['capacity'].values for lvl in levels]
    actual_data    = [df3[df3['energy_level'] == lvl]['hash_rate'].values for lvl in levels]

    # 3) plotting
    plt.figure(figsize=(10, 6))

    if kind == "violin":
        vp_pot = plt.violinplot(
            potential_data,
            positions=positions - offset,
            widths=0.3,
            showmeans=False,
            showmedians=True
        )
        vp_act = plt.violinplot(
            actual_data,
            positions=positions + offset,
            widths=0.3,
            showmeans=False,
            showmedians=True
        )

        # color the potential bodies
        for body in vp_pot['bodies']:
            body.set_facecolor(potential_color)
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        # color the potential median line
        if 'cmedians' in vp_pot and isinstance(vp_pot['cmedians'], plt.Line2D) is False:
            # vp_pot['cmedians'] is a LineCollection
            vp_pot['cmedians'].set_color('black')
            vp_pot['cmedians'].set_linewidth(1)

        # color the actual bodies
        for body in vp_act['bodies']:
            body.set_facecolor(actual_color)
            body.set_edgecolor('black')
            body.set_alpha(0.7)
        # color the actual median line
        if 'cmedians' in vp_act and isinstance(vp_act['cmedians'], plt.Line2D) is False:
            vp_act['cmedians'].set_color('black')
            vp_act['cmedians'].set_linewidth(1)

    elif kind == "boxplot":
        bp_pot = plt.boxplot(
            potential_data,
            positions=positions - offset,
            widths=0.3,
            patch_artist=True,
            boxprops={'facecolor': potential_color, 'edgecolor': 'black', 'alpha': 0.7},
            medianprops={'color': 'black'}
        )
        bp_act = plt.boxplot(
            actual_data,
            positions=positions + offset,
            widths=0.3,
            patch_artist=True,
            boxprops={'facecolor': actual_color, 'edgecolor': 'black', 'alpha': 0.7},
            medianprops={'color': 'black'}
        )
    else:
        raise ValueError("kind must be 'violin' or 'boxplot'")

    # 4) finalize
    plt.xticks(positions, levels)
    plt.xlabel('Energy Cost Level')
    plt.ylabel('Hash Rate (TH/s)')
    plt.title(f'Potential vs. Actual Hash Rate by Energy Cost Level {title_suffix}')

    # legend
    legend_handles = [
        Patch(facecolor=potential_color, edgecolor='black', label='Potential', alpha=0.7),
        Patch(facecolor=actual_color,   edgecolor='black', label='Actual',    alpha=0.7)
    ]
    plt.legend(handles=legend_handles, loc='upper right')

    plt.grid(linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
# -------------------------------

def plot_activation_heatmap(
    df,
    cost_col='cost',
    eff_col='efficiency',
    x_bins=10,
    y_bins=10,
    title_suffix=""
):
    """
    Plots a heatmap of activation probability (hash_rate > 0) 
    across bins of electricity cost vs. rig efficiency.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns [cost_col, eff_col, 'hash_rate'].
    cost_col : str
        Column name for electricity cost (x-axis).
    eff_col : str
        Column name for rig efficiency (y-axis).
    x_bins : int or sequence
        Number of bins (or bin edges) for cost.
    y_bins : int or sequence
        Number of bins (or bin edges) for efficiency.
    title_suffix : str
        Suffix to append to the plot title.
    """

    # 1) mark active miners
    df2 = df.copy()
    df2['active'] = (df2['hash_rate'] > 0).astype(float)

    # 2) create cost/efficiency bins
    df2['cost_bin'] = pd.cut(df2[cost_col], bins=x_bins)
    df2['eff_bin']  = pd.cut(df2[eff_col],  bins=y_bins)

    # 3) pivot to get mean activation per (eff_bin, cost_bin)
    pivot = df2.pivot_table(
        index='eff_bin',
        columns='cost_bin',
        values='active',
        aggfunc='mean'
    )

    # 4) plot heatmap
    plt.figure(figsize=(8,6))
    # imshow with origin='lower' so low-eff/low-cost in bottom-left
    plt.imshow(pivot, 
               origin='lower', 
               aspect='auto', 
               vmin=0, vmax=1,
               cmap='viridis')

    # 5) tick labels at bin centers
    # extract bin mids
    cost_mids = [interval.left + (interval.right-interval.left)/2 for interval in pivot.columns]
    eff_mids  = [interval.left + (interval.right-interval.left)/2 for interval in pivot.index]

    plt.xticks(np.arange(len(cost_mids)), [f"{c:.2f}" for c in cost_mids], rotation=45)
    plt.yticks(np.arange(len(eff_mids)),  [f"{e:.1f}" for e in eff_mids])

    plt.xlabel('Electricity Cost ($/kWh)')
    plt.ylabel('Rig Efficiency (J/TH)')
    plt.title(f'Activation Probability Heatmap{title_suffix}')

    cbar = plt.colorbar()
    cbar.set_label('Fraction Active')

    plt.tight_layout()
    plt.show()
# -------------------------------

# -------------------------------
# Main: loop over the three pool‐size scenarios
# -------------------------------
if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--rig-specs", required=True,
                        help="Excel file with columns ['model','capacity_THs','efficiency']")
    parser.add_argument("--cost-rates", required=True,
                        help="Excel file with columns ['country','cost']")
    parser.add_argument("-d", "--draws", type=int, default=500,
                        help="Number of Monte Carlo draws")
    parser.add_argument("-n", "--miners", type=int, default=500,
                        help="Number of miners per draw")
    args = parser.parse_args()

    # load inputs
    if not os.path.isfile(args.rig_specs):
        raise FileNotFoundError(f"Excel file not found at {args.rig_specs}")
    if not os.path.isfile(args.cost_rates):
        raise FileNotFoundError(f"Excel file not found at {args.cost_rates}")
    df_rigs  = pd.read_excel(args.rig_specs)
    df_costs = pd.read_excel(args.cost_rates)

    needed_rig  = {'model','capacity_THs','efficiency'}
    needed_cost = {'country','cost'}
    if not needed_rig.issubset(df_rigs.columns):
        raise ValueError(f"Missing columns in rig_specs: {needed_rig - set(df_rigs.columns)}")
    if not needed_cost.issubset(df_costs.columns):
        raise ValueError(f"Missing columns in cost_rates: {needed_cost - set(df_costs.columns)}")

    # define your three scenarios (TH/s)
    SCENARIOS = {
        'dominant': 200e6,   # 200 EH/s
        'average': 100e6,    # 100 EH/s
        'small':    50e6     # 50 EH/s
    }

    for name, pool_hash in SCENARIOS.items():
        print(f"\n=== Scenario: {name.capitalize()} Pool (pool_hash = {pool_hash/1e6:.0f} EH/s) ===")
        sim = MonteCarloSimulator(N=args.miners,
                                  rig_specs=df_rigs,
                                  cost_rates=df_costs)

        # One draw
        df1 = sim.run_single(pool_hash_override=pool_hash)
        plot_scatter(df1, title_suffix=f" ({name.capitalize()})")
        # plot_scatter_with_rug(df1, title_suffix=f" ({name.capitalize()})")
        # enhanced_plot(df1)
        plot_strategy_shares(df1, xcol='cost', kind='area',
                             title_suffix=f" ({name.capitalize()})")
        plot_potential_vs_actual(df1,
                                 title_suffix=f" ({name.capitalize()})",
                                 kind="violin")
        plot_activation_heatmap(df1, cost_col='cost', eff_col='efficiency',
                                title_suffix=f" ({name.capitalize()})")

        # Full Monte Carlo
        summary = sim.run(draws=args.draws,
                          pool_hash_override=pool_hash)
        plot_dist(summary, title_suffix=f" ({name.capitalize()})")
        summary.to_csv(f"monte_carlo_summary_{name}.csv", index=False)
    
    ## Extra summary for the last scenario
    price = df1.attrs['price']
    H_tot = df1.attrs['H_tot']
    pool_hash = df1.loc[df1.strategy=='pool','capacity'].sum()

    # 1) Cost cutoff: highest cost among active miners
    active = df1[df1.hash_rate > 0]
    cutoff_cost = active.cost.max()

    # 2) Breakdown by strategy
    n = len(df1)
    pct_solo = np.mean(df1.strategy=='solo')
    pct_pool = np.mean(df1.strategy=='pool')
    pct_off = np.mean(df1.strategy=='offline')

    # 3) Risk‐split among active
    active_solo = df1[(df1.hash_rate>0)&(df1.strategy=='solo')]
    active_pool = df1[(df1.hash_rate>0)&(df1.strategy=='pool')]
    avg_rho_solo = active_solo.risk.mean() if len(active_solo) else np.nan
    avg_rho_pool = active_pool.risk.mean() if len(active_pool) else np.nan

    print(f"Drawn BTC price: ${price:,.0f}")
    print(f"Cost‐participation cutoff: ${cutoff_cost:.3f}/kWh")
    print(f"Active miners: {100*pct_solo:.1f}% solo, {100*pct_pool:.1f}% pool, {100*pct_off:.1f}% offline")
    print(f"Avg risk ρ among solo miners: {avg_rho_solo:.2f}")
    print(f"Avg risk ρ among pool miners: {avg_rho_pool:.2f}")

    print("All scenarios done.")
