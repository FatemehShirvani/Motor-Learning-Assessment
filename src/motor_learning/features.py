"""Kinematic feature extraction for uniformly sampled 2D trajectories."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def five_point_derivative(values: ArrayLike, step: float) -> NDArray:
    """Differentiate a signal with a five-point central stencil.

    Second-order one-sided gradients are used for the two samples at each edge.
    """

    value_array = np.asarray(values, dtype=float)
    if value_array.ndim not in (1, 2):
        raise ValueError("values must be a one- or two-dimensional array")
    if len(value_array) < 5:
        raise ValueError("the five-point stencil requires at least five samples")
    if step <= 0:
        raise ValueError("step must be positive")

    derivative = np.asarray(np.gradient(value_array, step, axis=0, edge_order=2))
    derivative[2:-2] = (
        value_array[:-4]
        - 8.0 * value_array[1:-3]
        + 8.0 * value_array[3:-1]
        - value_array[4:]
    ) / (12.0 * step)
    return derivative


def extract_kinematic_features(
    time: ArrayLike, coordinates: ArrayLike
) -> dict[str, float]:
    """Return duration and aggregate velocity, curvature, acceleration, and jerk.

    Inputs must already be uniformly sampled. The returned values summarize one
    trial and contain no participant identifiers.
    """

    time_array = np.asarray(time, dtype=float)
    coordinate_array = np.asarray(coordinates, dtype=float)
    if time_array.ndim != 1 or coordinate_array.shape != (len(time_array), 2):
        raise ValueError("coordinates must have shape (len(time), 2)")
    if len(time_array) < 5:
        raise ValueError("at least five uniformly sampled points are required")

    intervals = np.diff(time_array)
    if np.any(intervals <= 0) or not np.allclose(intervals, intervals[0], rtol=1e-4):
        raise ValueError("time must be strictly increasing and uniformly sampled")

    step = float(intervals[0])
    velocity = five_point_derivative(coordinate_array, step)
    acceleration = five_point_derivative(velocity, step)
    jerk = five_point_derivative(acceleration, step)

    speed = np.linalg.norm(velocity, axis=1)
    acceleration_magnitude = np.linalg.norm(acceleration, axis=1)
    jerk_magnitude = np.linalg.norm(jerk, axis=1)
    curvature_numerator = np.abs(velocity[:, 0] * acceleration[:, 1] - velocity[:, 1] * acceleration[:, 0])
    curvature = np.divide(
        curvature_numerator,
        speed**3,
        out=np.zeros_like(speed),
        where=speed > np.finfo(float).eps,
    )

    return {
        "movement_time": float(time_array[-1] - time_array[0]),
        "path_length": float(np.sum(np.linalg.norm(np.diff(coordinate_array, axis=0), axis=1))),
        "mean_velocity": float(np.mean(speed)),
        "mean_curvature": float(np.mean(curvature)),
        "mean_acceleration": float(np.mean(acceleration_magnitude)),
        "mean_jerk": float(np.mean(jerk_magnitude)),
    }

