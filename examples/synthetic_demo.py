"""Run the public preprocessing and feature pipeline on a synthetic trajectory."""

from __future__ import annotations

import numpy as np

from motor_learning import extract_kinematic_features, prepare_trajectory


def main() -> None:
    rng = np.random.default_rng(7)
    time = np.linspace(0.0, 2.0, 151)
    angle = np.linspace(0.0, 2.0 * np.pi, len(time))
    coordinates = np.column_stack((np.cos(angle), 0.7 * np.sin(angle)))
    coordinates += rng.normal(scale=0.015, size=coordinates.shape)

    uniform_time, clean_coordinates = prepare_trajectory(time, coordinates)
    features = extract_kinematic_features(uniform_time, clean_coordinates)

    print("Synthetic trajectory features")
    for name, value in features.items():
        print(f"{name:>20}: {value:.4f}")


if __name__ == "__main__":
    main()

