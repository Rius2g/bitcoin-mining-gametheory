from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from .miner import MinerType

if TYPE_CHECKING:
    from .bayes_game import BayesianGame

# Use absolute imports to avoid relative import issues
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Strategy(ABC):
    """Abstract base class for mining strategies."""

    @abstractmethod
    def choose_action(self, miner: MinerType, game: 'BayesianGame') -> str:
        """Choose an action for the given miner in the game context."""
        pass

    @abstractmethod
    def get_name(self) -> str:
        """Return the strategy name."""
        pass


class SoloStrategy(Strategy):
    """Solo mining strategy - mine independently."""

    def choose_action(self, miner: MinerType, game: 'BayesianGame') -> str:
        """Always choose solo mining."""
        return 'solo'

    def get_name(self) -> str:
        return 'solo'


class PoolStrategy(Strategy):
    """Pool mining strategy - join a mining pool."""

    def choose_action(self, miner: MinerType, game: 'BayesianGame') -> str:
        """Always choose pool mining."""
        return 'pool'

    def get_name(self) -> str:
        return 'pool'


class OfflineStrategy(Strategy):
    """Offline strategy - don't mine."""

    def choose_action(self, miner: MinerType, game: 'BayesianGame') -> str:
        """Always choose offline."""
        return 'offline'

    def get_name(self) -> str:
        return 'offline'


class UtilityMaximizingStrategy(Strategy):
    """Strategy that chooses action based on utility maximization."""

    def choose_action(self, miner: MinerType, game: 'BayesianGame') -> str:
        """Choose action that maximizes expected utility."""
        return miner.choose_strategy()

    def get_name(self) -> str:
        return 'utility_maximizing'


# Strategy registry for easy access
STRATEGY_REGISTRY = {
    'solo': SoloStrategy,
    'pool': PoolStrategy,
    'offline': OfflineStrategy,
    'utility_maximizing': UtilityMaximizingStrategy,
}


def get_strategy(strategy_name: str) -> Strategy:
    """Factory function to get strategy instance by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return STRATEGY_REGISTRY[strategy_name]()
