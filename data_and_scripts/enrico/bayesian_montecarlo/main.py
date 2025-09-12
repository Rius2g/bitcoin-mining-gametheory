#!/usr/bin/env python3
"""
Bayesian Monte Carlo Simulation for Bitcoin Mining Game Theory

Main entrypoint for running simulations with configurable parameters.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config.loader import ConfigLoader
from simulation.base_simulator import MonteCarloSimulator
from simulation.factorial import FactorialMonteCarlo
from visualization.scatter import PlotScatter
from visualization.distribution import PlotDistribution
from visualization.violin import PlotViolin
from utils.logger import setup_logging, logger


def create_output_dirs(config):
    """Create output directories for results and plots."""
    output_dir = Path(config.visualization.output_dir)
    plots_dir = output_dir / config.visualization.plots_dir
    data_dir = output_dir / config.visualization.data_dir

    for dir_path in [output_dir, plots_dir, data_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return output_dir, plots_dir, data_dir


def run_basic_simulation(config, args):
    """Run basic Monte Carlo simulation."""
    logger.info("Running basic Monte Carlo simulation")

    # Initialize simulator
    simulator = MonteCarloSimulator(config)

    # Run simulation
    result = simulator.run(
        draws=args.draws or config.simulation.draws,
        pool_hash_override=args.pool_hash,
        record_miners=args.record_miners
    )

    # Create output directories
    output_dir, plots_dir, data_dir = create_output_dirs(config)

    # Save results
    prefix = f"basic_{args.pool_hash or 'endogenous'}"
    result.save(str(data_dir), prefix)

    # Generate plots
    if args.generate_plots:
        logger.info("Generating plots")

        # Initialize plotters
        scatter_plot = PlotScatter(config)
        dist_plot = PlotDistribution(config)
        violin_plot = PlotViolin(config)

        # Get sample data for plotting
        if result.miners_df is not None and not result.miners_df.empty:
            sample_df = result.miners_df[result.miners_df['draw_idx'] == 0].copy()
        else:
            # Run single draw for plotting
            sample_df = simulator.run_single(
                config.simulation.miners,
                args.pool_hash
            )

        # Generate plots
        scatter_plot.plot(
            sample_df,
            save_path=str(plots_dir / f"{prefix}_scatter.png")
        )

        dist_plot.plot_summary_distributions(
            result.summary_df,
            save_path=str(plots_dir / f"{prefix}_distributions.png")
        )

        violin_plot.plot_potential_vs_actual(
            sample_df,
            save_path=str(plots_dir / f"{prefix}_violin.png")
        )

    logger.info(f"Basic simulation completed. Results saved to {output_dir}")


def run_factorial_simulation(config, args):
    """Run factorial simulation across contexts."""
    logger.info("Running factorial simulation")

    # Initialize factorial simulator
    factorial_sim = FactorialMonteCarlo(config)

    # Run context sweep
    results = factorial_sim.run_context_sweep(
        draws_per_context=args.draws or config.simulation.draws
    )

    # Create output directories
    output_dir, plots_dir, data_dir = create_output_dirs(config)

    # Save results for each context
    for context_name, result in results.items():
        result.save(str(data_dir), f"factorial_{context_name}")

        if args.generate_plots and result.miners_df is not None:
            # Generate plots for each context
            sample_df = result.miners_df[result.miners_df['draw_idx'] == 0].copy()

            scatter_plot = PlotScatter(config)
            violin_plot = PlotViolin(config)

            scatter_plot.plot(
                sample_df,
                title_suffix=f" ({context_name})",
                save_path=str(plots_dir / f"factorial_{context_name}_scatter.png")
            )

            violin_plot.plot_potential_vs_actual(
                sample_df,
                title_suffix=f" ({context_name})",
                save_path=str(plots_dir / f"factorial_{context_name}_violin.png")
            )

    logger.info(f"Factorial simulation completed. Results saved to {output_dir}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Bayesian Monte Carlo Simulation for Bitcoin Mining"
    )

    # Global options
    parser.add_argument(
        "--config", "-c",
        default="config/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Basic simulation
    basic_parser = subparsers.add_parser("basic", help="Run basic Monte Carlo simulation")
    basic_parser.add_argument(
        "--draws", "-d", type=int,
        help="Number of simulation draws"
    )
    basic_parser.add_argument(
        "--pool-hash", type=float,
        help="Fixed pool hash rate (TH/s)"
    )
    basic_parser.add_argument(
        "--record-miners", action="store_true",
        help="Record individual miner data"
    )
    basic_parser.add_argument(
        "--generate-plots", action="store_true",
        help="Generate plots"
    )

    # Factorial simulation
    factorial_parser = subparsers.add_parser("factorial", help="Run factorial simulation")
    factorial_parser.add_argument(
        "--draws", "-d", type=int,
        help="Number of draws per context"
    )
    factorial_parser.add_argument(
        "--generate-plots", action="store_true",
        help="Generate plots"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config_loader = ConfigLoader(args.config)
        config = config_loader.load()

        # Log configuration summary
        config_summary = {
            "draws": config.simulation.draws,
            "miners": config.simulation.miners,
            "price_sampler": config.metrics.price.type,
            "cost_sampler": config.metrics.cost.type,
            "contexts": list(config.factorial.contexts.keys())
        }
        logger.log_simulation_start(config_summary)

        # Execute command
        if args.command == "basic":
            run_basic_simulation(config, args)
        elif args.command == "factorial":
            run_factorial_simulation(config, args)
        else:
            parser.print_help()

    except Exception as e:
        logger.log_error(e, "main execution")
        sys.exit(1)


if __name__ == "__main__":
    main()