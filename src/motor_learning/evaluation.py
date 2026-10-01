"""Participant-independent evaluation helpers for aggregate feature tables."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict


def leave_one_participant_out_accuracy(
    estimator: Any,
    features: ArrayLike,
    labels: ArrayLike,
    participant_groups: ArrayLike,
) -> float:
    """Evaluate an estimator while holding out every participant in turn.

    ``participant_groups`` should contain anonymized group labels used only to
    define folds; it must not be included among the predictive features.
    """

    feature_array = np.asarray(features, dtype=float)
    label_array = np.asarray(labels)
    group_array = np.asarray(participant_groups)

    if feature_array.ndim != 2:
        raise ValueError("features must be a two-dimensional matrix")
    if not (len(feature_array) == len(label_array) == len(group_array)):
        raise ValueError("features, labels, and groups must have equal lengths")
    if len(np.unique(group_array)) < 2:
        raise ValueError("at least two participant groups are required")

    predictions = cross_val_predict(
        clone(estimator),
        feature_array,
        label_array,
        groups=group_array,
        cv=LeaveOneGroupOut(),
    )
    return float(accuracy_score(label_array, predictions))

