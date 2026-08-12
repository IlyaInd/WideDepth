"""NumPy-first dataset loaders for WideDepth."""

from .benchmark import WideDepthBenchmarkDataset
from .train import WideDepthTrainDataset

__all__ = ["WideDepthBenchmarkDataset", "WideDepthTrainDataset"]
