"""Deep reinforcement learning algorithms in PyTorch."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deep-reinforcement-learning")
except PackageNotFoundError:
    __version__ = "0+unknown"
