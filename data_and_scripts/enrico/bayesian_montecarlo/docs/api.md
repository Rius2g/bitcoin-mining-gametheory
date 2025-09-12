# API Reference

## Core Classes

### MonteCarloSimulator

Main simulation orchestrator for Bayesian Monte Carlo analysis.

```python
class MonteCarloSimulator:
    def __init__(self, config: Config)
    def run(self, draws: int = None,
           pool_hash_override: float = None,
           record_miners: bool = False) -> SimulationResult
    def run_single(self, n_miners: int = None,
                  pool_hash_override: float = None) -> pd.DataFrame
    def create_miners(self, n_miners: int) -> List[MinerType]
```

**Parameters:**
- `config`: Configuration object with simulation parameters
- `draws`: Number of Monte Carlo draws (default: from config)
- `pool_hash_override`: Fixed pool hash rate in TH/s
- `record_miners`: Whether to save individual miner data

**Returns:**
- `SimulationResult`: Container with summary and optional miner data

### BayesianGame

Implements the Bayesian game framework for miner strategy equilibrium.

```python
class BayesianGame:
    def __init__(self, network_sampler, strategy: Strategy = None)
    def add_miner(self, miner: MinerType)
    def add_miners(self, miners: List[MinerType])
    def update_game_state(self, btc_price: float,
                         pool_hash_override: float = None)
    def find_equilibrium(self, max_iterations: int) -> bool
    def to_dataframe(self) -> pd.DataFrame
```

**Methods:**
- `add_miner()`: Add single miner to game
- `add_miners()`: Add multiple miners to game
- `update_game_state()`: Update network conditions
- `find_equilibrium()`: Find Nash equilibrium through iteration
- `to_dataframe()`: Export results as pandas DataFrame

### MinerType

Represents an individual miner's characteristics and decision-making.

```python
@dataclass
class MinerType:
    cost_per_kwh: float      # Electricity cost ($/kWh)
    efficiency: float        # Mining efficiency (J/TH)
    risk_aversion: float     # Risk preference (0-1)
    max_hash_rate: float     # Maximum hash rate (TH/s)

    # Game state (set during simulation)
    network_hash_rate: float = field(init=False, default=0.0)
    pool_hash_rate: float = field(init=False, default=0.0)
    block_reward: float = field(init=False, default=0.0)
    block_time: float = field(init=False, default=600.0)
    pool_fee: float = field(init=False, default=0.025)
    btc_price: float = field(init=False, default=0.0)
    discount_factor: float = field(init=False, default=0.5)

    # Decision variables
    strategy: str = field(init=False, default="offline")
    actual_hash_rate: float = field(init=False, default=0.0)

    def cost_per_day(self) -> float
    def revenue_solo(self) -> float
    def revenue_pool(self) -> float
    def choose_best_strategy(self) -> str
```

## Samplers

### MetricSampler (Abstract Base)

```python
class MetricSampler(ABC):
    @abstractmethod
    def sample(self) -> float:
        """Sample a value for this metric."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of this sampler."""
        pass
```

### PriceSampler

Samples Bitcoin price from various distributions.

```python
class PriceSampler(MetricSampler):
    def __init__(self, config: dict)
    def sample(self) -> float
    def get_description(self) -> str
```

**Supported distributions:**
- `lognormal`: Log-normal distribution with configurable μ and σ
- Parameters: `mean`, `sigma`

### CostSampler

Samples electricity costs for miners.

```python
class CostSampler(MetricSampler):
    def __init__(self, config: dict)
    def sample(self) -> float
    def get_description(self) -> str
```

**Supported methods:**
- `empirical_excel`: Sample from Excel file data
- `uniform`: Uniform distribution with low/high bounds
- Parameters: `file` (for Excel), `low`, `high` (for uniform)

### RiskSampler

Samples miner risk aversion levels.

```python
class RiskSampler(MetricSampler):
    def __init__(self, config: dict)
    def sample(self) -> float
    def get_description(self) -> str
```

**Supported distributions:**
- `uniform`: Uniform distribution (0-1)
- `beta`: Beta distribution with α and β parameters
- Parameters: `low`, `high` (uniform) or `alpha`, `beta` (beta)

### RigSampler

Samples mining hardware specifications.

```python
class RigSampler:
    def __init__(self, config: dict)
    def sample(self) -> tuple[float, float]
    def get_description(self) -> str
```

**Returns:** `(hash_rate_capacity, efficiency)`
- `hash_rate_capacity`: TH/s per rig
- `efficiency`: J/TH (energy efficiency)

### NetworkSampler

Samples network-level parameters.

```python
class NetworkSampler:
    def __init__(self, config: dict)
    def sample_total_hash(self) -> float
    def sample_pool_fee(self) -> float
    def sample_block_time(self) -> float
    def sample_block_reward(self, btc_price: float) -> float
    def get_description(self) -> str
```

## Strategy Classes

### Strategy (Abstract Base)

```python
class Strategy(ABC):
    @abstractmethod
    def calculate_utility(self, miner: MinerType, game_state: dict) -> float:
        """Calculate expected utility for this strategy."""
        pass
```

### Concrete Strategies

```python
class SoloMining(Strategy):
    def calculate_utility(self, miner: MinerType, game_state: dict) -> float

class PoolMining(Strategy):
    def calculate_utility(self, miner: MinerType, game_state: dict) -> float

class Offline(Strategy):
    def calculate_utility(self, miner: MinerType, game_state: dict) -> float

class UtilityMaximizing(Strategy):
    def calculate_utility(self, miner: MinerType, game_state: dict) -> float
```

## Configuration Classes

### Config (Main Configuration)

```python
@dataclass
class Config:
    simulation: SimulationConfig
    metrics: MetricsConfig
    network: NetworkConfig
    game: GameConfig
    visualization: VisualizationConfig
    factorial: FactorialConfig
```

### Component Configurations

```python
@dataclass
class SimulationConfig:
    draws: int
    miners: int
    max_iterations: int

@dataclass
class MetricsConfig:
    price: dict
    cost: dict
    risk: dict

@dataclass
class NetworkConfig:
    hash_rate: float
    pool_fee: float
    block_time: float
    block_reward: float
    block_subsidy: float
    discount_factor: float
```

## Visualization Classes

### PlotScatter

Creates scatter plots of simulation results.

```python
class PlotScatter:
    def __init__(self, config=None)
    def plot(self, df: pd.DataFrame, x_col: str = 'cost_per_kwh',
             y_col: str = 'actual_hash_rate', color_col: str = 'strategy',
             title_suffix: str = "", save_path: Optional[str] = None)
```

### PlotDistribution

Creates distribution plots and histograms.

```python
class PlotDistribution:
    def __init__(self, config=None)
    def plot(self, df: pd.DataFrame, column: str = 'strategy',
             title_suffix: str = "", save_path: Optional[str] = None)
```

### PlotViolin

Creates violin plots for distribution comparison.

```python
class PlotViolin:
    def __init__(self, config=None)
    def plot(self, df: pd.DataFrame, x_col: str = 'strategy',
             y_col: str = 'risk_aversion', title_suffix: str = "",
             save_path: Optional[str] = None)
```

## Utility Classes

### SimulationResult

Container for simulation results.

```python
class SimulationResult:
    def __init__(self, summary_df: pd.DataFrame, miners_df: Optional[pd.DataFrame] = None)
    def add_metadata(self, key: str, value: Any)
    def save(self, output_dir: str, prefix: str = "simulation")
```

### ConfigLoader

Loads and validates YAML configuration files.

```python
class ConfigLoader:
    def __init__(self, config_path: str = "config/default.yaml")
    def load(self) -> Config
```

## Extension Points

### Creating Custom Samplers

```python
from metrics.base import MetricSampler

class CustomPriceSampler(MetricSampler):
    def __init__(self, config: dict):
        self.parameter = config.get('parameter', 1.0)

    def sample(self) -> float:
        # Implement custom sampling logic
        return custom_distribution(self.parameter)

    def get_description(self) -> str:
        return f"Custom distribution (parameter={self.parameter})"
```

### Creating Custom Strategies

```python
from game.strategy import Strategy

class CustomStrategy(Strategy):
    def calculate_utility(self, miner: MinerType, game_state: dict) -> float:
        # Implement custom utility calculation
        revenue = self.calculate_revenue(miner, game_state)
        cost = self.calculate_cost(miner, game_state)
        risk_penalty = self.calculate_risk_penalty(miner, game_state)

        return revenue - cost - risk_penalty
```

### Adding Custom Visualizations

```python
import matplotlib.pyplot as plt
from typing import Optional
import pandas as pd

class CustomPlot:
    def __init__(self, config=None):
        self.config = config

    def plot(self, df: pd.DataFrame, save_path: Optional[str] = None):
        plt.figure(figsize=(10, 6))

        # Custom plotting logic
        plt.scatter(df['x'], df['y'], c=df['color'])

        if save_path:
            plt.savefig(save_path)
        plt.show()
```

## Error Handling

The system uses structured logging and exception handling:

```python
from utils.logger import logger

try:
    # Simulation code
    result = simulator.run(draws=1000)
except Exception as e:
    logger.log_error(e, "simulation execution")
    raise
```

## Type Hints and Dataclasses

The codebase extensively uses:
- **Type hints**: For function parameters and return values
- **Dataclasses**: For configuration and data structures
- **Abstract base classes**: For extensible interfaces
- **Generic types**: For flexible data handling

## Performance Considerations

- **Vectorization**: NumPy operations for efficient computation
- **Lazy loading**: Excel data loaded only when needed
- **Memory management**: Optional miner data recording
- **Configurable iterations**: Adjustable convergence criteria</content>
<parameter name="filePath">/Users/enricotedeschi/projects/bitcoin-mining-gametheory/data_and_scripts/enrico/bayesian_montecarlo/docs/api.md