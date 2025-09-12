from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import yaml
import os
from pathlib import Path


@dataclass
class PriceConfig:
    type: str
    mean: Optional[float] = None
    sigma: Optional[float] = None
    fixed_value: Optional[float] = None


@dataclass
class CostConfig:
    type: str
    file: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None


@dataclass
class RiskConfig:
    type: str
    low: Optional[float] = None
    high: Optional[float] = None
    a: Optional[float] = None  # for beta distribution
    b: Optional[float] = None


@dataclass
class MetricsConfig:
    price: PriceConfig
    cost: CostConfig
    risk: RiskConfig


@dataclass
class NetworkConfig:
    hash_rate: float
    pool_fee: float
    block_time: float
    block_reward: float
    block_subsidy: float
    discount_factor: float


@dataclass
class GameConfig:
    strategies: List[str]
    convergence_tolerance: float


@dataclass
class VisualizationConfig:
    output_dir: str
    plots_dir: str
    data_dir: str
    colors: Dict[str, str]


@dataclass
class ContextConfig:
    price: float
    network_hash: float
    block_subsidy: float


@dataclass
class FactorialConfig:
    contexts: Dict[str, ContextConfig]
    pool_sizes: List[float]


@dataclass
class SimulationConfig:
    draws: int
    miners: int
    max_iterations: int


@dataclass
class Config:
    simulation: SimulationConfig
    metrics: MetricsConfig
    rig_specs: Dict[str, str]
    network: NetworkConfig
    game: GameConfig
    visualization: VisualizationConfig
    factorial: FactorialConfig


class ConfigLoader:
    """Loads and validates configuration from YAML file."""

    def __init__(self, config_path: str = "config/default.yaml"):
        self.config_path = Path(config_path)

    def load(self) -> Config:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)

        return self._parse_config(data)

    def _parse_config(self, data: Dict[str, Any]) -> Config:
        """Parse raw YAML data into Config dataclass."""
        # Parse simulation config
        sim_data = data['simulation']
        simulation = SimulationConfig(
            draws=sim_data['draws'],
            miners=sim_data['miners'],
            max_iterations=sim_data['max_iterations']
        )

        # Parse metrics config
        metrics_data = data['metrics']
        metrics = MetricsConfig(
            price=PriceConfig(**metrics_data['price']),
            cost=CostConfig(**metrics_data['cost']),
            risk=RiskConfig(**metrics_data['risk'])
        )

        # Parse network config
        network_data = data['network']
        network = NetworkConfig(**network_data)

        # Parse game config
        game_data = data['game']
        game = GameConfig(**game_data)

        # Parse visualization config
        viz_data = data['visualization']
        visualization = VisualizationConfig(**viz_data)

        # Parse factorial config
        fact_data = data['factorial']
        contexts = {}
        for name, ctx_data in fact_data['contexts'].items():
            contexts[name] = ContextConfig(**ctx_data)
        factorial = FactorialConfig(
            contexts=contexts,
            pool_sizes=fact_data['pool_sizes']
        )

        return Config(
            simulation=simulation,
            metrics=metrics,
            rig_specs=data['rig_specs'],
            network=network,
            game=game,
            visualization=visualization,
            factorial=factorial
        )

    def save_default(self, output_path: str = "config/default.yaml"):
        """Save current configuration as default."""
        # This would be used to save modified configs
        pass
