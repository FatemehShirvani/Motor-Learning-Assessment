"""Trajectory preprocessing without assumptions about the private dataset format."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _validate_trajectory(time: ArrayLike, coordinates: ArrayLike) -> tuple[NDArray, NDArray]:
    time_array = np.asarray(time, dtype=float)
    coordinate_array = np.asarray(coordinates, dtype=float)

    if time_array.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_samples, 2)")
    if len(time_array) != len(coordinate_array):
        raise ValueError("time and coordinates must contain the same number of samples")
    if len(time_array) < 2:
        raise ValueError("at least two trajectory samples are required")
    if not np.all(np.isfinite(time_array)) or not np.all(np.isfinite(coordinate_array)):
        raise ValueError("trajectory values must be finite")
    if np.any(np.diff(time_array) < 0):
        raise ValueError("time values must be ordered")

    return time_array, coordinate_array


def remove_repeated_samples(
    time: ArrayLike, coordinates: ArrayLike
) -> tuple[NDArray, NDArray]:
    """Remove consecutive duplicate timestamps or coordinate samples.

    Quantized acquisition can produce consecutive copies of the same point. The
    first observation in each repeated run is retained.
    """

    time_array, coordinate_array = _validate_trajectory(time, coordinates)
    time_changed = np.r_[True, np.diff(time_array) > 0]
    position_changed = np.r_[True, np.any(np.diff(coordinate_array, axis=0) != 0, axis=1)]
    keep = time_changed & position_changed

    cleaned_time = time_array[keep]
    cleaned_coordinates = coordinate_array[keep]
    if len(cleaned_time) < 2:
        raise ValueError("too few distinct samples remain after duplicate removal")
    return cleaned_time, cleaned_coordinates


def resample_trajectory(
    time: ArrayLike, coordinates: ArrayLike, sampling_rate_hz: float = 200.0
) -> tuple[NDArray, NDArray]:
    """Linearly resample an ordered 2D trajectory at a fixed sampling rate."""

    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    time_array, coordinate_array = _validate_trajectory(time, coordinates)
    duration = time_array[-1] - time_array[0]
    if duration <= 0:
        raise ValueError("trajectory duration must be positive")

    sample_count = max(2, int(np.floor(duration * sampling_rate_hz)) + 1)
    uniform_time = np.linspace(time_array[0], time_array[-1], sample_count)
    uniform_coordinates = np.column_stack(
        [np.interp(uniform_time, time_array, coordinate_array[:, axis]) for axis in range(2)]
    )
    return uniform_time, uniform_coordinates


def moving_average(coordinates: ArrayLike, window: int = 7) -> NDArray:
    """Smooth a trajectory with an edge-padded centered moving average."""

    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] != 2:
        raise ValueError("coordinates must have shape (n_samples, 2)")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")
    if window > len(coordinate_array):
        raise ValueError("window cannot exceed the trajectory length")
    if window == 1:
        return coordinate_array.copy()

    padding = window // 2
    padded = np.pad(coordinate_array, ((padding, padding), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.column_stack(
        [np.convolve(padded[:, axis], kernel, mode="valid") for axis in range(2)]
    )


def prepare_trajectory(
    time: ArrayLike,
    coordinates: ArrayLike,
    sampling_rate_hz: float = 200.0,
    smoothing_window: int = 7,
) -> tuple[NDArray, NDArray]:
    """Remove repeats, resample, and smooth a 2D trajectory."""

    clean_time, clean_coordinates = remove_repeated_samples(time, coordinates)
    uniform_time, uniform_coordinates = resample_trajectory(
        clean_time, clean_coordinates, sampling_rate_hz
    )
    return uniform_time, moving_average(uniform_coordinates, smoothing_window)

