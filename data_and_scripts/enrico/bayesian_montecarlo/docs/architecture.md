# Architecture Diagrams

## System Architecture

```mermaid
graph TB
    subgraph "User Interface"
        CLI[Command Line Interface<br/>main.py]
    end

    subgraph "Configuration Layer"
        YAML[YAML Config Files<br/>config/default.yaml]
        Loader[ConfigLoader<br/>config/loader.py]
    end

    subgraph "Sampling Layer"
        Price[PriceSampler<br/>metrics/price.py]
        Cost[CostSampler<br/>metrics/cost.py]
        Risk[RiskSampler<br/>metrics/risk.py]
        Rig[RigSampler<br/>sampler/rig.py]
        Network[NetworkSampler<br/>sampler/network.py]
    end

    subgraph "Simulation Layer"
        MCS[MonteCarloSimulator<br/>simulation/base_simulator.py]
        FMC[FactorialMonteCarlo<br/>simulation/factorial.py]
    end

    subgraph "Game Theory Layer"
        BG[BayesianGame<br/>game/bayes_game.py]
        Miner[MinerType<br/>game/miner.py]
        Strategy[Strategy Classes<br/>game/strategy.py]
    end

    subgraph "Visualization Layer"
        Scatter[PlotScatter<br/>visualization/scatter.py]
        Dist[PlotDistribution<br/>visualization/distribution.py]
        Violin[PlotViolin<br/>visualization/violin.py]
    end

    subgraph "Data Sources"
        Excel[Excel Files<br/>data/*.xlsx]
        Legacy[Legacy Code<br/>legacy/*.py]
    end

    subgraph "Outputs"
        CSV[CSV Files<br/>outputs/data/*.csv]
        Plots[Plot Files<br/>outputs/plots/*.png]
        JSON[Metadata<br/>outputs/data/*.json]
    end

    CLI --> Loader
    YAML --> Loader
    Loader --> MCS
    Loader --> FMC

    MCS --> Price
    MCS --> Cost
    MCS --> Risk
    MCS --> Rig
    MCS --> Network

    MCS --> BG
    FMC --> BG

    BG --> Miner
    BG --> Strategy

    MCS --> Scatter
    MCS --> Dist
    MCS --> Violin

    Excel --> Cost
    Excel --> Rig
    Legacy --> Loader

    Scatter --> Plots
    Dist --> Plots
    Violin --> Plots
    MCS --> CSV
    MCS --> JSON
```

## Data Flow Diagram

```mermaid
sequenceDiagram
    participant U as User
    participant CLI as main.py
    participant CL as ConfigLoader
    participant MCS as MonteCarloSimulator
    participant Samp as Samplers
    participant BG as BayesianGame
    participant Viz as Visualizers
    participant Out as Output Files

    U->>CLI: python main.py basic --draws 1000
    CLI->>CL: Load configuration
    CL-->>CLI: Config object
    CLI->>MCS: Initialize simulator

    loop For each Monte Carlo draw
        MCS->>Samp: Sample parameters<br/>(price, cost, risk, hardware)
        Samp-->>MCS: Parameter values
        MCS->>BG: Create game with miners
        BG->>BG: Find Nash equilibrium
        BG-->>MCS: Strategy decisions
    end

    MCS->>Viz: Generate visualizations
    Viz-->>MCS: Plot files
    MCS->>Out: Save CSV results
    MCS->>Out: Save metadata
    CLI-->>U: Simulation complete
```

## Class Hierarchy

```mermaid
classDiagram
    class MetricSampler {
        <<abstract>>
        +sample() float
        +get_description() str
    }

    class PriceSampler {
        +sample() float
        +get_description() str
    }

    class CostSampler {
        +sample() float
        +get_description() str
    }

    class RiskSampler {
        +sample() float
        +get_description() str
    }

    class RigSampler {
        +sample() tuple
        +get_description() str
    }

    class NetworkSampler {
        +sample_total_hash() float
        +sample_pool_fee() float
        +sample_block_time() float
        +sample_block_reward() float
    }

    class MonteCarloSimulator {
        -config: Config
        -price_sampler: PriceSampler
        -cost_sampler: CostSampler
        -risk_sampler: RiskSampler
        -rig_sampler: RigSampler
        -network_sampler: NetworkSampler
        -game: BayesianGame
        +run() SimulationResult
        +run_single() DataFrame
    }

    class BayesianGame {
        -network_sampler: NetworkSampler
        -miners: List[MinerType]
        +add_miner()
        +add_miners()
        +update_game_state()
        +find_equilibrium() bool
        +to_dataframe() DataFrame
    }

    class MinerType {
        +cost_per_kwh: float
        +efficiency: float
        +risk_aversion: float
        +max_hash_rate: float
        +cost_per_day() float
        +revenue_solo() float
        +revenue_pool() float
        +choose_best_strategy() str
    }

    class Strategy {
        <<abstract>>
        +calculate_utility() float
    }

    class SoloMining {
        +calculate_utility() float
    }

    class PoolMining {
        +calculate_utility() float
    }

    class Offline {
        +calculate_utility() float
    }

    MetricSampler <|-- PriceSampler
    MetricSampler <|-- CostSampler
    MetricSampler <|-- RiskSampler

    MonteCarloSimulator --> PriceSampler
    MonteCarloSimulator --> CostSampler
    MonteCarloSimulator --> RiskSampler
    MonteCarloSimulator --> RigSampler
    MonteCarloSimulator --> NetworkSampler
    MonteCarloSimulator --> BayesianGame

    BayesianGame --> MinerType
    BayesianGame --> NetworkSampler

    MinerType --> Strategy
    Strategy <|-- SoloMining
    Strategy <|-- PoolMining
    Strategy <|-- Offline
```

## Component Interaction Flow

```mermaid
flowchart TD
    A[Start Simulation] --> B{Load Configuration}
    B --> C[Initialize Samplers]
    C --> D[Create Miner Population]

    D --> E[Sample Market Conditions]
    E --> F[Sample Network Parameters]
    F --> G[Update Game State]

    G --> H{Find Equilibrium}
    H --> I{Converged?}
    I --> J[Update Strategies] --> H
    I --> K[Record Results]

    K --> L{More Draws?}
    L --> D
    L --> M[Aggregate Results]

    M --> N[Generate Visualizations]
    N --> O[Save Outputs]
    O --> P[End Simulation]
```</content>
<parameter name="filePath">/Users/enricotedeschi/projects/bitcoin-mining-gametheory/data_and_scripts/enrico/bayesian_montecarlo/docs/architecture.md