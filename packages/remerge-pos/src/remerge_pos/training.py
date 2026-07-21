"""Provisional reproducible training and export tools.

This module is intentionally outside the stable inference compatibility
promise. Callers must pin package version, configuration, seed, and inputs.
"""

from ._perceptron import Config, Model, train, write_artifact


SELECTED_CONFIG = Config(epochs=6, feature_cutoff=1, feature_buckets=131_072)
SELECTED_SEED = 20_260_720

__all__ = [
    "Config",
    "Model",
    "SELECTED_CONFIG",
    "SELECTED_SEED",
    "train",
    "write_artifact",
]
