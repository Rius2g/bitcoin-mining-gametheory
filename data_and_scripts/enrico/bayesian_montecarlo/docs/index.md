# Documentation Index

## 📚 Bayesian Monte Carlo Simulation Documentation

T├── data/
│   ├── cost_rates.xlsx     # Electricity cost data
│   └── rig_specs.xlsx      # Mining hardware specs
├── legacy/
│   ├── *.py                 # Legacy simulation code (deprecated)
│   └── *.xlsx              # Legacy Excel files (moved to data/) documentation suite provides comprehensive guidance for the Bayesian Monte Carlo simulation system for Bitcoin mining game theory analysis.

## 📖 Documentation Files

### [📋 Main Documentation](./README.md)
Complete technical documentation covering:
- System architecture and design
- Input sources and data flow
- Core components and classes
- Configuration system
- Usage examples and API reference
- Performance considerations and troubleshooting

### [🏗️ Architecture Diagrams](./architecture.md)
Visual representations of the system:
- System architecture overview
- Data flow diagrams
- Class hierarchy
- Component interaction flows
- Mermaid diagrams for visualization

### [🚀 Quick Start Guide](./quickstart.md)
Practical guide for getting started:
- Installation instructions
- Basic usage examples
- Configuration customization
- Output interpretation
- Troubleshooting common issues
- Performance optimization tips

### [🔧 API Reference](./api.md)
Developer-focused documentation:
- Complete class and method references
- Extension points for customization
- Type hints and data structures
- Error handling patterns
- Performance considerations

## 📁 Project Structure

```
docs/
├── README.md           # Main comprehensive documentation
├── architecture.md     # Architecture diagrams and flows
├── quickstart.md       # Quick start guide
└── api.md             # API reference for developers

bayesian_montecarlo/
├── main.py            # CLI entry point
├── config/
│   ├── default.yaml   # Default configuration
│   └── loader.py      # Configuration loader
├── simulation/
│   ├── base_simulator.py    # Main simulator
│   └── factorial.py         # Factorial analysis
├── game/
│   ├── bayes_game.py        # Game theory implementation
│   ├── miner.py             # Miner model
│   └── strategy.py          # Strategy classes
├── metrics/
│   ├── base.py              # Abstract samplers
│   ├── price.py             # Price sampling
│   ├── cost.py              # Cost sampling
│   └── risk.py              # Risk sampling
├── sampler/
│   ├── rig.py               # Hardware sampling
│   └── network.py           # Network sampling
├── visualization/
│   ├── scatter.py           # Scatter plots
│   ├── distribution.py      # Distribution plots
│   └── violin.py            # Violin plots
├── utils/
│   └── logger.py            # Logging utilities
└── legacy/
    ├── *.py                 # Legacy simulation code
    └── *.xlsx              # Excel data files
```

## 🎯 Quick Navigation

| I want to... | Go to... |
|-------------|----------|
| Understand the system | [Main Documentation](./README.md) |
| See visual diagrams | [Architecture](./architecture.md) |
| Get started quickly | [Quick Start](./quickstart.md) |
| Extend/customize code | [API Reference](./api.md) |
| Configure the system | [Configuration](#configuration) |
| Run simulations | [Usage Examples](#usage-examples) |
| Troubleshoot issues | [Troubleshooting](#troubleshooting) |

## 📊 Key Topics

### System Overview
- [Architecture Overview](./README.md#architecture-overview)
- [Data Flow](./README.md#data-flow)
- [Core Components](./README.md#core-components)

### Configuration
- [YAML Configuration](./README.md#configuration-system)
- [Input Sources](./README.md#input-sources)
- [Parameter Reference](./README.md#yaml-configuration-structure)

### Usage
- [Command Line Interface](./quickstart.md#basic-usage)
- [Custom Configuration](./quickstart.md#using-custom-configuration)
- [Batch Processing](./quickstart.md#batch-processing)

### Development
- [API Reference](./api.md#core-classes)
- [Extension Points](./api.md#extension-points)
- [Custom Samplers](./api.md#creating-custom-samplers)

### Analysis
- [Output Formats](./README.md#output-formats)
- [Visualization Types](./README.md#visualization-system)
- [Result Interpretation](./quickstart.md#understanding-outputs)

## 🔗 Related Resources

- **Legacy Code**: Original simulation implementation in `legacy/` (deprecated, Excel files moved to `data/`)
- **Configuration**: Default settings in `config/default.yaml`
- **Examples**: Sample configurations and usage patterns
- **Outputs**: Generated results in `outputs/` directory

## 📞 Support

For questions or issues:
1. Check the [Troubleshooting](./quickstart.md#troubleshooting) section
2. Review the [Configuration](./README.md#configuration-system) documentation
3. Examine the [API Reference](./api.md) for extension guidance
4. Enable debug logging: `python main.py --log-level DEBUG`

## 📈 Version Information

- **Current Version**: 1.0.0
- **Python Version**: 3.9+
- **Last Updated**: September 12, 2025
- **Documentation**: Comprehensive system documentation

---

*This index provides navigation for the complete Bayesian Monte Carlo simulation documentation suite.*</content>
<parameter name="filePath">/Users/enricotedeschi/projects/bitcoin-mining-gametheory/data_and_scripts/enrico/bayesian_montecarlo/docs/index.md