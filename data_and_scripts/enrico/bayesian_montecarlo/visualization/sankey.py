"""
Sankey diagram visualizations for Bitcoin mining game theory simulations.

This module provides comprehensive flow diagram capabilities for visualizing
miner decision flows and transitions in the Bitcoin mining ecosystem.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from abc import ABC, abstractmethod


class SankeyPlotter(ABC):
    """Abstract base class for Sankey diagram plotters."""
    
    @abstractmethod
    def plot(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Generate Sankey diagram from simulation data."""
        pass


class MinerDecisionFlowPlotter(SankeyPlotter):
    """
    Sankey diagram for miner decision flow visualization.
    
    Shows flow from hash power categories -> risk categories -> strategies -> pool sizes.
    """
    
    def __init__(self):
        self.hash_power_bins = [0, 1000, 2000, float('inf')]
        self.hash_power_labels = ['low hash', 'medium hash', 'big hash']
        self.risk_bins = [0, 0.3, 0.7, 1.0]
        self.risk_labels = ['low risk', 'medium risk', 'high risk']
        self.pool_size_bins = [0, 100e6, 200e6, float('inf')]
        self.pool_size_labels = ['small pool', 'medium pool', 'big pool']
    
    def categorize_hash_power(self, hash_rate: float) -> str:
        """Categorize hash power into bins."""
        for i, (low, high) in enumerate(zip(self.hash_power_bins[:-1], self.hash_power_bins[1:])):
            if low <= hash_rate < high:
                return self.hash_power_labels[i]
        return self.hash_power_labels[-1]
    
    def categorize_risk(self, risk: float) -> str:
        """Categorize risk into bins."""
        for i, (low, high) in enumerate(zip(self.risk_bins[:-1], self.risk_bins[1:])):
            if low <= risk < high:
                return self.risk_labels[i]
        return self.risk_labels[-1]
    
    def categorize_pool_size(self, row: pd.Series) -> Optional[str]:
        """Categorize pool size for pool miners only."""
        if row.get('strategy') != 'pool':
            return None
        
        # Use pool_hash if available, otherwise use actual_hash_rate or hash_rate as proxy
        pool_hash = row.get('pool_hash', row.get('actual_hash_rate', row.get('hash_rate', 0)))
        
        for i, (low, high) in enumerate(zip(self.pool_size_bins[:-1], self.pool_size_bins[1:])):
            if low <= pool_hash < high:
                return self.pool_size_labels[i]
        return self.pool_size_labels[-1]
    
    def prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data with categorized columns for Sankey diagram."""
        df = data.copy()
        
        # Map column names to standard names
        hash_rate_col = 'actual_hash_rate' if 'actual_hash_rate' in df.columns else 'hash_rate'
        risk_col = 'risk_aversion' if 'risk_aversion' in df.columns else 'risk'
        
        # Add categorization columns
        df['hash_cat'] = df[hash_rate_col].apply(self.categorize_hash_power)
        df['risk_cat'] = df[risk_col].apply(self.categorize_risk)
        df['strategy'] = df['strategy'].fillna('offline')
        df['pool_size_cat'] = df.apply(self.categorize_pool_size, axis=1)
        
        return df
    
    def build_flows(self, df: pd.DataFrame) -> List[pd.DataFrame]:
        """Build flow data for each stage of the Sankey diagram."""
        flows = []
        
        # Stage 1: hash_cat -> risk_cat
        flow1 = df.groupby(['hash_cat', 'risk_cat']).size().reset_index(name='count')
        flows.append(flow1)
        
        # Stage 2: risk_cat -> strategy
        flow2 = df.groupby(['risk_cat', 'strategy']).size().reset_index(name='count')
        flows.append(flow2)
        
        # Stage 3: strategy -> pool_size_cat (only for pool miners)
        pool_df = df[df['strategy'] == 'pool']
        if not pool_df.empty and 'pool_size_cat' in pool_df.columns:
            flow3 = pool_df.groupby(['strategy', 'pool_size_cat']).size().reset_index(name='count')
            flows.append(flow3)
        
        return flows
    
    def create_node_mapping(self, flows: List[pd.DataFrame]) -> Tuple[List[str], Dict[str, int]]:
        """Create node labels and mapping from flows."""
        labels = []
        
        # Collect all unique labels from flows
        for flow in flows:
            if len(flow.columns) >= 3:  # Ensure flow has source, target, count columns
                labels.extend(flow.iloc[:, 0].unique())  # Source nodes
                labels.extend(flow.iloc[:, 1].unique())  # Target nodes
        
        # Remove duplicates and None values
        labels = [label for label in pd.Series(labels).dropna().unique() if label is not None]
        
        # Create mapping
        label_to_index = {label: i for i, label in enumerate(labels)}
        
        return labels, label_to_index
    
    def extract_links(self, flows: List[pd.DataFrame], label_to_index: Dict[str, int]) -> Tuple[List[int], List[int], List[float]]:
        """Extract source, target, and value lists for Sankey links."""
        sources, targets, values = [], [], []
        
        for flow in flows:
            for _, row in flow.iterrows():
                if len(row) >= 3:  # Ensure row has source, target, count
                    source_label = row.iloc[0]
                    target_label = row.iloc[1]
                    value = row.iloc[2]
                    
                    # Skip if either label is None or not in mapping
                    if (source_label is None or target_label is None or 
                        source_label not in label_to_index or target_label not in label_to_index):
                        continue
                    
                    sources.append(label_to_index[source_label])
                    targets.append(label_to_index[target_label])
                    values.append(float(value))
        
        return sources, targets, values
    
    def plot(self, data: pd.DataFrame, title: str = "Miner Decision Flow", **kwargs) -> go.Figure:
        """
        Generate Sankey diagram showing miner decision flow.
        
        Args:
            data: DataFrame with columns ['actual_hash_rate' or 'hash_rate', 'risk_aversion' or 'risk', 'strategy', 'pool_hash']
            title: Title for the diagram
            **kwargs: Additional arguments (font_size, node_color, etc.)
        
        Returns:
            Plotly Figure object
        """
        # Prepare data
        df = self.prepare_data(data)
        
        # Build flows
        flows = self.build_flows(df)
        
        if not flows:
            raise ValueError("No valid flows found in data")
        
        # Create node mapping
        labels, label_to_index = self.create_node_mapping(flows)
        
        if not labels:
            raise ValueError("No valid labels found in flows")
        
        # Extract links
        sources, targets, values = self.extract_links(flows, label_to_index)
        
        if not sources:
            raise ValueError("No valid links found in flows")
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=labels,
                pad=kwargs.get('node_pad', 15),
                thickness=kwargs.get('node_thickness', 20),
                color=kwargs.get('node_color', 'rgba(0,0,255,0.8)')
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values,
                color=kwargs.get('link_color', 'rgba(0,0,255,0.2)')
            )
        )])
        
        fig.update_layout(
            title=title,
            font_size=kwargs.get('font_size', 10),
            width=kwargs.get('width', 1000),
            height=kwargs.get('height', 600)
        )
        
        return fig


class StrategyTransitionPlotter(SankeyPlotter):
    """
    Sankey diagram for strategy transitions over time.
    
    Shows how miners change strategies between different time periods.
    """
    
    def plot(self, data: pd.DataFrame, time_column: str = 'period', 
             strategy_column: str = 'strategy', **kwargs) -> go.Figure:
        """
        Generate Sankey diagram showing strategy transitions over time.
        
        Args:
            data: DataFrame with time and strategy columns
            time_column: Column name for time periods
            strategy_column: Column name for strategies
            **kwargs: Additional arguments
        
        Returns:
            Plotly Figure object
        """
        # Get unique time periods and strategies
        periods = sorted(data[time_column].unique())
        strategies = data[strategy_column].unique()
        
        if len(periods) < 2:
            raise ValueError("Need at least 2 time periods for transition analysis")
        
        # Build transition flows between consecutive periods
        flows = []
        labels = []
        
        for i in range(len(periods) - 1):
            current_period = periods[i]
            next_period = periods[i + 1]
            
            # Get miners in both periods
            current_data = data[data[time_column] == current_period]
            next_data = data[data[time_column] == next_period]
            
            # Merge on miner ID to track transitions
            if 'miner_id' in data.columns:
                transitions = current_data.merge(
                    next_data, on='miner_id', suffixes=('_from', '_to')
                )
                
                # Count transitions
                transition_counts = transitions.groupby([
                    f'{strategy_column}_from', 
                    f'{strategy_column}_to'
                ]).size().reset_index(name='count')
                
                # Add period labels
                transition_counts['from_label'] = transition_counts[f'{strategy_column}_from'] + f'_t{i}'
                transition_counts['to_label'] = transition_counts[f'{strategy_column}_to'] + f'_t{i+1}'
                
                flows.append(transition_counts[['from_label', 'to_label', 'count']])
                
                # Collect labels
                labels.extend(transition_counts['from_label'].unique())
                labels.extend(transition_counts['to_label'].unique())
        
        # Remove duplicate labels
        labels = list(set(labels))
        label_to_index = {label: i for i, label in enumerate(labels)}
        
        # Extract links
        sources, targets, values = [], [], []
        for flow in flows:
            for _, row in flow.iterrows():
                sources.append(label_to_index[row['from_label']])
                targets.append(label_to_index[row['to_label']])
                values.append(row['count'])
        
        # Create diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                label=labels,
                pad=kwargs.get('node_pad', 15),
                thickness=kwargs.get('node_thickness', 20)
            ),
            link=dict(
                source=sources,
                target=targets,
                value=values
            )
        )])
        
        fig.update_layout(
            title=kwargs.get('title', "Strategy Transitions Over Time"),
            font_size=kwargs.get('font_size', 10)
        )
        
        return fig


class SankeyVisualizer:
    """Main interface for creating Sankey diagrams."""
    
    def __init__(self):
        self.plotters = {
            'decision_flow': MinerDecisionFlowPlotter(),
            'strategy_transition': StrategyTransitionPlotter()
        }
    
    def plot_decision_flow(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create miner decision flow Sankey diagram."""
        return self.plotters['decision_flow'].plot(data, **kwargs)
    
    def plot_strategy_transitions(self, data: pd.DataFrame, **kwargs) -> go.Figure:
        """Create strategy transition Sankey diagram."""
        return self.plotters['strategy_transition'].plot(data, **kwargs)
    
    def show_decision_flow(self, data: pd.DataFrame, **kwargs):
        """Display miner decision flow Sankey diagram."""
        fig = self.plot_decision_flow(data, **kwargs)
        fig.show()
    
    def show_strategy_transitions(self, data: pd.DataFrame, **kwargs):
        """Display strategy transition Sankey diagram."""
        fig = self.plot_strategy_transitions(data, **kwargs)
        fig.show()
    
    def save_decision_flow(self, data: pd.DataFrame, filename: str, **kwargs):
        """Save miner decision flow Sankey diagram."""
        fig = self.plot_decision_flow(data, **kwargs)
        if filename.endswith('.html'):
            fig.write_html(filename)
        elif filename.endswith('.png'):
            fig.write_image(filename)
        elif filename.endswith('.pdf'):
            fig.write_image(filename)
        else:
            raise ValueError("Unsupported file format. Use .html, .png, or .pdf")
    
    def save_strategy_transitions(self, data: pd.DataFrame, filename: str, **kwargs):
        """Save strategy transition Sankey diagram."""
        fig = self.plot_strategy_transitions(data, **kwargs)
        if filename.endswith('.html'):
            fig.write_html(filename)
        elif filename.endswith('.png'):
            fig.write_image(filename)
        elif filename.endswith('.pdf'):
            fig.write_image(filename)
        else:
            raise ValueError("Unsupported file format. Use .html, .png, or .pdf")