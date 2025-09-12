import logging
import sys
from pathlib import Path
from typing import Optional


class Logger:
    """Structured logging utility for the simulation system."""

    def __init__(self, name: str = "bayesian_montecarlo",
                 level: str = "INFO",
                 log_file: Optional[str] = None,
                 console: bool = True):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))

        # Remove existing handlers
        self.logger.handlers.clear()

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def debug(self, message: str, *args, **kwargs):
        """Log debug message."""
        self.logger.debug(message, *args, **kwargs)

    def info(self, message: str, *args, **kwargs):
        """Log info message."""
        self.logger.info(message, *args, **kwargs)

    def warning(self, message: str, *args, **kwargs):
        """Log warning message."""
        self.logger.warning(message, *args, **kwargs)

    def error(self, message: str, *args, **kwargs):
        """Log error message."""
        self.logger.error(message, *args, **kwargs)

    def critical(self, message: str, *args, **kwargs):
        """Log critical message."""
        self.logger.critical(message, *args, **kwargs)

    def log_simulation_start(self, config_summary: dict):
        """Log simulation start with configuration summary."""
        self.info("Starting Bayesian Monte Carlo simulation")
        self.info(f"Configuration: {config_summary}")

    def log_simulation_progress(self, current: int, total: int, context: str = ""):
        """Log simulation progress."""
        progress = (current / total) * 100
        self.info(".1f")

    def log_simulation_end(self, results_summary: dict):
        """Log simulation end with results summary."""
        self.info("Simulation completed")
        self.info(f"Results summary: {results_summary}")

    def log_error(self, error: Exception, context: str = ""):
        """Log error with context."""
        self.error(f"Error in {context}: {str(error)}", exc_info=True)


# Global logger instance
logger = Logger()


def get_logger(name: str = "bayesian_montecarlo") -> Logger:
    """Get logger instance."""
    return Logger(name)


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> Logger:
    """Setup global logging configuration."""
    global logger
    logger = Logger("bayesian_montecarlo", level, log_file)
    return logger