#!/usr/bin/env python3
"""
Strategic Transition Heatmaps
Visualize how mining strategies change under different market conditions:
1. Energy Cost vs Strategy Distribution by Miner Size
2. Geographic Strategy Migration Patterns  
3. Risk Preference vs Strategic Approach Matrix
4. Capacity Utilization Strategy Transitions
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os

def load_all_summary_data(plots_dir):
    """Load all summary CSV files and combine them"""
    all_data = []
    miner_sizes = ["Small", "Medium", "Large", "Industrial"]
    
    for size in miner_sizes:
        file_path = f"{plots_dir}/{size}_summary.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            df['Miner Size'] = size
            all_data.append(df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=False)
    else:
        raise FileNotFoundError("No summary CSV files found!")

def classify_energy_region(energy_sector):
    """Classify energy regions"""
    region_mapping = {
        'kazakhstan': 'Kazakhstan',
        'china': 'China', 
        'texas': 'Texas',
        'denmark': 'Denmark'
    }
    return region_mapping.get(energy_sector, 'Texas')

def classify_energy_cost_tier(energy_sector):
    """Classify energy cost tiers"""
    cost_mapping = {
        'kazakhstan': 'Very Low\n($0.04)',
        'china': 'Low\n($0.07)', 
        'texas': 'Medium\n($0.10)',
        'denmark': 'High\n($0.15)'
    }
    return cost_mapping.get(energy_sector, 'Medium\n($0.10)')

def classify_risk_preference(strategy_name):
    """Classify risk preference based on strategy"""
    if 'Solo' in strategy_name:
        return 'Risk Seeking'
    elif 'Ocean' in strategy_name or 'TIDES' in strategy_name:
        return 'Risk Neutral'
    elif any(term in strategy_name for term in ['PPLNS', 'Pool']):
        return 'Risk Averse'
    else:
        return 'Risk Neutral'

def classify_strategic_approach(main_strategy, underlying_strategy):
    """Classify strategic approach"""
    if 'Solo' in underlying_strategy:
        if 'Always-On' in main_strategy:
            return 'Solo Aggressive'
        elif 'Stop-Loss' in main_strategy:
            return 'Solo Conservative'
        else:
            return 'Solo Adaptive'
    elif 'Ocean' in underlying_strategy or 'TIDES' in underlying_strategy:
        return 'Zero-Fee Pool'
    elif 'PPLNS' in underlying_strategy:
        return 'PPLNS Pool'
    elif any(pool in underlying_strategy for pool in ['FPPS', 'Antpool', 'F2Pool']):
        return 'Major Pool'
    else:
        return 'Alternative Pool'

def classify_capacity_tier(hashrate_usage):
    """Classify capacity utilization"""
    if hashrate_usage >= 95:
        return 'Maximum\n(95%+)'
    elif hashrate_usage >= 80:
        return 'High\n(80-95%)'
    elif hashrate_usage >= 60:
        return 'Moderate\n(60-80%)'
    elif hashrate_usage >= 40:
        return 'Conservative\n(40-60%)'
    else:
        return 'Minimal\n(<40%)'

def create_energy_strategy_heatmap(combined_df):
    """Create heatmap showing strategy distribution across energy costs and miner sizes"""
    print("Creating Energy Cost vs Strategy Distribution heatmap...")
    
    # Apply classifications
    combined_df['Energy Tier'] = combined_df['Energy Sector'].apply(classify_energy_cost_tier)
    combined_df['Strategic Approach'] = combined_df.apply(
        lambda row: classify_strategic_approach(row['Main Strategy'], row['Underlying Strategy']), 
        axis=1
    )
    
    # Create pivot table for heatmap
    strategy_counts = combined_df.groupby(['Energy Tier', 'Strategic Approach', 'Miner Size']).size().reset_index(name='Count')
    
    # Calculate percentages within each energy tier
    energy_totals = combined_df.groupby('Energy Tier').size().reset_index(name='Total')
    strategy_counts = strategy_counts.merge(energy_totals, on='Energy Tier')
    strategy_counts['Percentage'] = (strategy_counts['Count'] / strategy_counts['Total'] * 100).round(1)
    
    # Create subplots for each miner size
    miner_sizes = ['Small', 'Medium', 'Large', 'Industrial']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{size} Miners' for size in miner_sizes],
        shared_xaxes=True,
        shared_yaxes=True
    )
    
    energy_tiers = ['Very Low\n($0.04)', 'Low\n($0.07)', 'Medium\n($0.10)', 'High\n($0.15)']
    strategies = sorted(combined_df['Strategic Approach'].unique())
    
    for idx, size in enumerate(miner_sizes):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        # Filter data for this miner size
        size_data = strategy_counts[strategy_counts['Miner Size'] == size]
        
        # Create pivot table for heatmap
        heatmap_data = size_data.pivot_table(
            index='Strategic Approach', 
            columns='Energy Tier', 
            values='Percentage', 
            fill_value=0
        )
        
        # Ensure all energy tiers are present
        for tier in energy_tiers:
            if tier not in heatmap_data.columns:
                heatmap_data[tier] = 0
        
        # Ensure all strategies are present
        for strategy in strategies:
            if strategy not in heatmap_data.index:
                heatmap_data.loc[strategy] = 0
        
        # Reorder columns and rows
        heatmap_data = heatmap_data.reindex(columns=energy_tiers, fill_value=0)
        heatmap_data = heatmap_data.reindex(strategies, fill_value=0)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='RdYlBu_r',
                showscale=(idx == 0),  # Only show scale for first subplot
                zmin=0,
                zmax=100,
                text=[[f"{val:.1f}%" for val in row] for row in heatmap_data.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                hovertemplate='<b>%{y}</b><br>Energy Cost: %{x}<br>Percentage: %{z:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Strategic Approach Distribution Across Energy Cost Tiers by Miner Size',
        font_size=12,
        height=800,
        width=1200
    )
    
    return fig

def create_geographic_migration_heatmap(combined_df):
    """Create heatmap showing strategy preferences by geographic region"""
    print("Creating Geographic Strategy Migration heatmap...")
    
    # Apply classifications
    combined_df['Region'] = combined_df['Energy Sector'].apply(classify_energy_region)
    combined_df['Strategic Approach'] = combined_df.apply(
        lambda row: classify_strategic_approach(row['Main Strategy'], row['Underlying Strategy']), 
        axis=1
    )
    combined_df['Risk Preference'] = combined_df['Underlying Strategy'].apply(classify_risk_preference)
    
    # Create pivot table showing strategy distribution by region and miner size
    migration_data = combined_df.groupby(['Region', 'Strategic Approach', 'Miner Size']).agg({
        'Final USD Profit': ['count', 'mean'],
        'USD Risk-Adjusted Return': 'mean'
    }).round(2)
    
    migration_data.columns = ['Count', 'Avg_Profit', 'Avg_Risk_Return']
    migration_data = migration_data.reset_index()
    
    # Calculate regional strategy preferences (percentage within each region)
    region_totals = combined_df.groupby(['Region', 'Miner Size']).size().reset_index(name='Region_Total')
    migration_data = migration_data.merge(region_totals, on=['Region', 'Miner Size'])
    migration_data['Region_Percentage'] = (migration_data['Count'] / migration_data['Region_Total'] * 100).round(1)
    
    # Create separate heatmaps for each miner size
    miner_sizes = ['Small', 'Medium', 'Large', 'Industrial']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{size} Miners - Regional Strategy Preferences' for size in miner_sizes]
    )
    
    regions = ['Kazakhstan', 'China', 'Texas', 'Denmark']
    strategies = sorted(combined_df['Strategic Approach'].unique())
    
    for idx, size in enumerate(miner_sizes):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        # Filter data for this miner size
        size_data = migration_data[migration_data['Miner Size'] == size]
        
        # Create pivot table
        heatmap_data = size_data.pivot_table(
            index='Strategic Approach',
            columns='Region',
            values='Region_Percentage',
            fill_value=0
        )
        
        # Ensure all regions and strategies are present
        for region in regions:
            if region not in heatmap_data.columns:
                heatmap_data[region] = 0
        
        for strategy in strategies:
            if strategy not in heatmap_data.index:
                heatmap_data.loc[strategy] = 0
        
        # Reorder
        heatmap_data = heatmap_data.reindex(columns=regions, fill_value=0)
        heatmap_data = heatmap_data.reindex(strategies, fill_value=0)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Viridis',
                showscale=(idx == 0),
                text=[[f"{val:.1f}%" for val in row] for row in heatmap_data.values],
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertemplate='<b>%{y}</b><br>Region: %{x}<br>Preference: %{z:.1f}%<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Regional Strategy Preferences by Miner Size (%)',
        font_size=12,
        height=800,
        width=1200
    )
    
    return fig

def create_risk_strategy_matrix(combined_df):
    """Create risk preference vs strategic approach matrix"""
    print("Creating Risk Preference vs Strategic Approach matrix...")
    
    # Apply classifications
    combined_df['Risk Preference'] = combined_df['Underlying Strategy'].apply(classify_risk_preference)
    combined_df['Strategic Approach'] = combined_df.apply(
        lambda row: classify_strategic_approach(row['Main Strategy'], row['Underlying Strategy']), 
        axis=1
    )
    
    # Create matrix showing average profitability for each risk-strategy combination
    risk_strategy_matrix = combined_df.groupby(['Risk Preference', 'Strategic Approach', 'Miner Size']).agg({
        'Final USD Profit': ['count', 'mean'],
        'USD Risk-Adjusted Return': 'mean',
        'Avg Hashrate Usage (%)': 'mean'
    }).round(2)
    
    risk_strategy_matrix.columns = ['Count', 'Avg_Profit', 'Avg_Risk_Return', 'Avg_Hashrate']
    risk_strategy_matrix = risk_strategy_matrix.reset_index()
    
    # Create subplots for each miner size
    miner_sizes = ['Small', 'Medium', 'Large', 'Industrial']
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'{size} Miners - Risk vs Strategy Profitability ($)' for size in miner_sizes]
    )
    
    risk_preferences = ['Risk Averse', 'Risk Neutral', 'Risk Seeking']
    strategies = sorted(combined_df['Strategic Approach'].unique())
    
    for idx, size in enumerate(miner_sizes):
        row = (idx // 2) + 1
        col = (idx % 2) + 1
        
        # Filter data for this miner size
        size_data = risk_strategy_matrix[risk_strategy_matrix['Miner Size'] == size]
        
        # Create pivot table
        profit_matrix = size_data.pivot_table(
            index='Strategic Approach',
            columns='Risk Preference',
            values='Avg_Profit',
            fill_value=0
        )
        
        # Ensure all categories are present
        for risk in risk_preferences:
            if risk not in profit_matrix.columns:
                profit_matrix[risk] = 0
        
        for strategy in strategies:
            if strategy not in profit_matrix.index:
                profit_matrix.loc[strategy] = 0
        
        # Reorder
        profit_matrix = profit_matrix.reindex(columns=risk_preferences, fill_value=0)
        profit_matrix = profit_matrix.reindex(strategies, fill_value=0)
        
        # Add heatmap
        fig.add_trace(
            go.Heatmap(
                z=profit_matrix.values,
                x=profit_matrix.columns,
                y=profit_matrix.index,
                colorscale='RdYlGn',
                showscale=(idx == 0),
                text=[[f"${val:,.0f}" if val > 0 else "N/A" for val in row] for row in profit_matrix.values],
                texttemplate="%{text}",
                textfont={"size": 9},
                hovertemplate='<b>%{y}</b><br>Risk: %{x}<br>Avg Profit: $%{z:,.0f}<extra></extra>'
            ),
            row=row, col=col
        )
    
    fig.update_layout(
        title='Average Profitability by Risk Preference and Strategic Approach',
        font_size=12,
        height=800,
        width=1200
    )
    
    return fig

def create_capacity_transition_heatmap(combined_df):
    """Create capacity utilization vs strategy transition heatmap"""
    print("Creating Capacity Utilization vs Strategy heatmap...")
    
    # Apply classifications
    combined_df['Capacity Tier'] = combined_df['Avg Hashrate Usage (%)'].apply(classify_capacity_tier)
    combined_df['Strategic Approach'] = combined_df.apply(
        lambda row: classify_strategic_approach(row['Main Strategy'], row['Underlying Strategy']), 
        axis=1
    )
    combined_df['Risk Preference'] = combined_df['Underlying Strategy'].apply(classify_risk_preference)
    
    # Create transition matrix
    capacity_data = combined_df.groupby(['Capacity Tier', 'Strategic Approach']).agg({
        'Final USD Profit': ['count', 'mean'],
        'USD Risk-Adjusted Return': 'mean'
    }).round(2)
    
    capacity_data.columns = ['Count', 'Avg_Profit', 'Avg_Risk_Return']
    capacity_data = capacity_data.reset_index()
    
    # Calculate percentages within each capacity tier
    capacity_totals = combined_df.groupby('Capacity Tier').size().reset_index(name='Total')
    capacity_data = capacity_data.merge(capacity_totals, on='Capacity Tier')
    capacity_data['Percentage'] = (capacity_data['Count'] / capacity_data['Total'] * 100).round(1)
    
    # Create pivot table
    capacity_tiers = ['Minimal\n(<40%)', 'Conservative\n(40-60%)', 'Moderate\n(60-80%)', 'High\n(80-95%)', 'Maximum\n(95%+)']
    strategies = sorted(combined_df['Strategic Approach'].unique())
    
    heatmap_data = capacity_data.pivot_table(
        index='Strategic Approach',
        columns='Capacity Tier',
        values='Percentage',
        fill_value=0
    )
    
    # Ensure all categories are present
    for tier in capacity_tiers:
        if tier not in heatmap_data.columns:
            heatmap_data[tier] = 0
    
    for strategy in strategies:
        if strategy not in heatmap_data.index:
            heatmap_data.loc[strategy] = 0
    
    # Reorder
    heatmap_data = heatmap_data.reindex(columns=capacity_tiers, fill_value=0)
    heatmap_data = heatmap_data.reindex(strategies, fill_value=0)
    
    # Create single heatmap
    fig = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale='Blues',
        text=[[f"{val:.1f}%" for val in row] for row in heatmap_data.values],
        texttemplate="%{text}",
        textfont={"size": 11},
        hovertemplate='<b>%{y}</b><br>Capacity: %{x}<br>Usage: %{z:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title='Strategic Approach vs Capacity Utilization Distribution (%)',
        xaxis_title='Capacity Utilization Tier',
        yaxis_title='Strategic Approach',
        font_size=12,
        height=600,
        width=1000
    )
    
    return fig

def main():
    """Main execution function"""
    print("=== Strategic Transition Heatmaps ===")
    
    # Load data
    plots_dir = "/Users/riius/bitcoin-mining-gametheory/data_and_scripts/strategies/equilibrium_plots"
    combined_df = load_all_summary_data(plots_dir)
    
    # Create output directory
    output_dir = "/Users/riius/bitcoin-mining-gametheory/data_and_scripts/marius/bayesian_montecarlo/strategic_transition_heatmaps"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. Energy Cost vs Strategy Distribution
        fig1 = create_energy_strategy_heatmap(combined_df.copy())
        fig1.write_html(f"{output_dir}/energy_strategy_distribution.html")
        print("✓ Energy-Strategy Distribution heatmap saved")
        
        # 2. Geographic Strategy Migration
        fig2 = create_geographic_migration_heatmap(combined_df.copy())
        fig2.write_html(f"{output_dir}/geographic_strategy_migration.html")
        print("✓ Geographic Strategy Migration heatmap saved")
        
        # 3. Risk-Strategy Profitability Matrix
        fig3 = create_risk_strategy_matrix(combined_df.copy())
        fig3.write_html(f"{output_dir}/risk_strategy_profitability.html")
        print("✓ Risk-Strategy Profitability matrix saved")
        
        # 4. Capacity Utilization Transitions
        fig4 = create_capacity_transition_heatmap(combined_df.copy())
        fig4.write_html(f"{output_dir}/capacity_strategy_transitions.html")
        print("✓ Capacity-Strategy Transitions heatmap saved")
        
        print(f"\n=== Success! Created 4 Strategic Transition Heatmaps ===")
        print(f"Location: {output_dir}/")
        print("\nHeatmaps created:")
        print("  1. energy_strategy_distribution.html - Strategy adoption across energy cost tiers")
        print("  2. geographic_strategy_migration.html - Regional strategy preferences")
        print("  3. risk_strategy_profitability.html - Risk-strategy profitability combinations")
        print("  4. capacity_strategy_transitions.html - Capacity utilization patterns")
        
        # Print insights
        print(f"\n=== Strategic Transition Insights ===")
        
        # Energy cost insights
        energy_insights = combined_df.groupby(['Energy Sector', 'Miner Size']).agg({
            'Final USD Profit': 'mean'
        }).round(0)
        print("\nAverage Profit by Energy Region and Size:")
        for (energy, size), profit in energy_insights['Final USD Profit'].items():
            print(f"  {energy.title()} - {size}: ${profit:,.0f}")
        
        # Strategy popularity
        combined_df['Strategic Approach'] = combined_df.apply(
            lambda row: classify_strategic_approach(row['Main Strategy'], row['Underlying Strategy']), 
            axis=1
        )
        strategy_popularity = combined_df['Strategic Approach'].value_counts()
        print(f"\nMost Popular Strategic Approaches:")
        for strategy, count in strategy_popularity.head(5).items():
            print(f"  {strategy}: {count} combinations ({count/len(combined_df)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error creating strategic transition heatmaps: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()