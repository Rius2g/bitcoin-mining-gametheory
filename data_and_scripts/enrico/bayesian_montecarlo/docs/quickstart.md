# Quick Start Guide

## Installation

1. **Clone the repository** (if applicable)
2. **Navigate to the project directory**
   ```bash
   cd /path/to/bayesian_montecarlo
   ```

3. **Install dependencies**
   ```bash
   pip install numpy pandas scipy pyyaml matplotlib seaborn plotly openpyxl
   ```

4. **Verify installation**
   ```bash
   python main.py --help
   ```

## Basic Usage

### Run a Simple Simulation

```bash
# Run basic Monte Carlo simulation with default settings (1000 draws, 500 miners)
python main.py basic
```

### Customize Simulation Parameters

```bash
# Run with 500 draws
python main.py basic --draws 500

# Run with custom pool hash rate (50 EH/s)
python main.py basic --draws 1000 --pool-hash 50000000

# Record individual miner data (increases memory usage)
python main.py basic --draws 100 --record-miners

# Generate plots automatically
python main.py basic --draws 1000 --generate-plots
```

### Factorial Analysis

```bash
# Run analysis across different market contexts (good/average/bad)
python main.py factorial --draws 500
```

## Configuration

### Using Custom Configuration

```bash
# Use a custom configuration file
python main.py --config my_config.yaml basic --draws 1000
```

### Example Custom Configuration

Create `my_config.yaml`:

```yaml
simulation:
  draws: 2000
  miners: 1000
  max_iterations: 20

metrics:
  price:
    type: lognormal
    mean: 150000    # Higher average price
    sigma: 0.3      # Lower volatility
  cost:
    type: uniform
    low: 0.05       # Lower cost range
    high: 0.15
  risk:
    type: beta
    alpha: 2.0      # More risk-seeking miners
    beta: 5.0

network:
  hash_rate: 1200000000  # Higher network hash rate
  pool_fee: 0.02          # Lower pool fees
  discount_factor: 0.3    # Higher time preference
```

## Understanding Outputs

### Output Directory Structure

```
outputs/
├── data/
│   ├── simulation_summary.csv     # Summary statistics
│   ├── simulation_miners.csv      # Individual miner data (if --record-miners)
│   └── simulation_metadata.json   # Run metadata
└── plots/
    ├── scatter_cost_hash.png      # Cost vs hash rate scatter plot
    ├── distribution_strategies.png # Strategy distribution
    └── violin_risk_strategy.png   # Risk distribution by strategy
```

### Key Output Metrics

**Summary Statistics:**
- `pct_solo`: Percentage of miners choosing solo mining
- `pct_pool`: Percentage of miners choosing pool mining
- `pct_offline`: Percentage of miners staying offline
- `pct_active`: Percentage of miners with positive hash rate
- `total_hash_rate`: Total network hash rate from active miners
- `btc_price`: Average BTC price in simulation

### Interpreting Results

1. **High solo mining percentage**: Profitable conditions, low risk aversion
2. **High pool mining percentage**: High competition, risk-averse miners
3. **High offline percentage**: Unprofitable conditions, high costs
4. **Low convergence**: Complex equilibrium, may need more iterations

## Troubleshooting

### Common Issues and Solutions

#### 1. Configuration File Not Found
```bash
ERROR: Configuration file 'config/default.yaml' not found
```
**Solution**: Ensure you're running from the correct directory or specify full path:
```bash
python main.py --config /full/path/to/config/default.yaml basic
```

#### 2. Excel Files Not Found
```bash
ERROR: Excel file 'data/cost_rates.xlsx' not found
```
**Solution**: Verify Excel files exist in the `data/` directory

#### 3. Memory Issues
```bash
ERROR: Out of memory
```
**Solutions**:
- Reduce number of draws: `--draws 100`
- Disable miner recording: remove `--record-miners`
- Increase system memory or use smaller miner population

#### 4. Long Execution Times
**Solutions**:
- Reduce draws for testing: `--draws 100`
- Use fewer miners per draw (modify config)
- Monitor progress with debug logging: `--log-level DEBUG`

### Debug Mode

Enable detailed logging:
```bash
python main.py --log-level DEBUG basic --draws 10 --log-file debug.log
```

### Performance Optimization

1. **For development/testing**:
   ```bash
   python main.py basic --draws 50 --record-miners
   ```

2. **For production runs**:
   ```bash
   python main.py basic --draws 10000 --generate-plots
   ```

3. **For memory-constrained systems**:
   ```bash
   python main.py basic --draws 1000  # No --record-miners
   ```

## Advanced Usage

### Custom Analysis Scripts

```python
from config.loader import ConfigLoader
from simulation.base_simulator import MonteCarloSimulator

# Load configuration
config_loader = ConfigLoader('config/default.yaml')
config = config_loader.load()

# Create simulator
simulator = MonteCarloSimulator(config)

# Run custom analysis
result = simulator.run(draws=100, record_miners=True)

# Access results
summary_df = result.summary_df
miners_df = result.miners_df

# Custom analysis
print(f"Average solo mining: {summary_df['pct_solo'].mean():.1%}")
print(f"Average pool mining: {summary_df['pct_pool'].mean():.1%}")
```

### Batch Processing

```bash
# Run multiple scenarios
for scenario in good average bad; do
    echo "Running $scenario scenario..."
    python main.py --config config/${scenario}.yaml basic --draws 1000
done
```

## Next Steps

1. **Explore the full documentation**: See `docs/README.md`
2. **Modify configuration**: Adjust parameters in `config/default.yaml`
3. **Analyze results**: Examine CSV outputs and plots
4. **Extend the system**: Add custom samplers or strategies
5. **Contribute**: Report issues or suggest improvements

## Support

- **Documentation**: `docs/README.md` and `docs/architecture.md`
- **Configuration**: `config/default.yaml` with comments
- **Examples**: Command examples in this guide
- **Debugging**: Use `--log-level DEBUG` for detailed output</content>
<parameter name="filePath">/Users/enricotedeschi/projects/bitcoin-mining-gametheory/data_and_scripts/enrico/bayesian_montecarlo/docs/quickstart.md