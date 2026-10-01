"""Data-independent tools for foot-trajectory motor-learning analysis."""

from .features import extract_kinematic_features
from .preprocessing import prepare_trajectory

__all__ = ["extract_kinematic_features", "prepare_trajectory"]

