"""
Tabular two-sided CUSUM control chart.

The reference value k and decision interval h are expressed
in the same units as the monitored subgroup means.
"""

import math

from dataclasses import dataclass
from typing import Sequence


Number = int | float


class CUSUMInputError(ValueError):
    """Invalid input supplied to CUSUM analysis."""


@dataclass(frozen=True)
class CUSUMResult:
    subgroup_means: tuple[float, ...]

    target_mean: float
    reference_value: float
    decision_interval: float

    positive_cusum: tuple[float, ...]
    negative_cusum: tuple[float, ...]

    positive_signal_indices: tuple[int, ...]
    negative_signal_indices: tuple[int, ...]


def _finite_number(
    value,
    *,
    label: str,
) -> float:

    try:
        numeric = float(value)

    except (TypeError, ValueError):
        raise CUSUMInputError(
            f"{label} must be numeric."
        )

    if not math.isfinite(numeric):
        raise CUSUMInputError(
            f"{label} must be finite."
        )

    return numeric


def _parse_subgroups(
    subgroups: Sequence[
        Sequence[Number]
    ],
) -> tuple[
    tuple[float, ...],
    ...
]:

    if isinstance(
        subgroups,
        (str, bytes),
    ):
        raise CUSUMInputError(
            "Subgroups must be a sequence."
        )

    parsed = []

    for subgroup_index, subgroup in enumerate(
        subgroups,
        start=1,
    ):
        if isinstance(
            subgroup,
            (str, bytes),
        ):
            raise CUSUMInputError(
                (
                    f"Subgroup {subgroup_index} "
                    "must be a sequence."
                )
            )

        values = tuple(
            _finite_number(
                value,
                label=(
                    f"Subgroup {subgroup_index} "
                    "observation"
                ),
            )
            for value in subgroup
        )

        if not values:
            raise CUSUMInputError(
                (
                    f"Subgroup {subgroup_index} "
                    "cannot be empty."
                )
            )

        parsed.append(values)

    if not parsed:
        raise CUSUMInputError(
            "At least one subgroup is required."
        )

    subgroup_sizes = {
        len(subgroup)
        for subgroup in parsed
    }

    if len(subgroup_sizes) != 1:
        raise CUSUMInputError(
            (
                "All subgroups must have "
                "the same size."
            )
        )

    return tuple(parsed)


def calculate_cusum(
    subgroups: Sequence[
        Sequence[Number]
    ],
    *,
    target_mean: Number | None = None,
    reference_value: Number,
    decision_interval: Number,
) -> CUSUMResult:

    groups = _parse_subgroups(
        subgroups
    )

    subgroup_means = tuple(
        sum(subgroup) / len(subgroup)
        for subgroup in groups
    )

    if target_mean is None:
        all_values = tuple(
            value
            for subgroup in groups
            for value in subgroup
        )

        target = (
            sum(all_values)
            / len(all_values)
        )

    else:
        target = _finite_number(
            target_mean,
            label="Target mean",
        )

    k = _finite_number(
        reference_value,
        label="Reference value",
    )

    h = _finite_number(
        decision_interval,
        label="Decision interval",
    )

    if k < 0:
        raise CUSUMInputError(
            (
                "Reference value must be "
                "greater than or equal to 0."
            )
        )

    if h <= 0:
        raise CUSUMInputError(
            (
                "Decision interval must be "
                "greater than 0."
            )
        )

    positive = []
    negative_magnitude = []

    previous_positive = 0.0
    previous_negative = 0.0

    positive_signals = []
    negative_signals = []

    for index, subgroup_mean in enumerate(
        subgroup_means,
        start=1,
    ):
        previous_positive = max(
            0.0,
            (
                previous_positive
                + subgroup_mean
                - target
                - k
            ),
        )

        previous_negative = max(
            0.0,
            (
                previous_negative
                + target
                - subgroup_mean
                - k
            ),
        )

        positive.append(
            previous_positive
        )

        # Store the lower CUSUM as a negative value so that
        # it can be plotted below zero naturally.
        negative_magnitude.append(
            -previous_negative
        )

        if previous_positive > h:
            positive_signals.append(
                index
            )

        if previous_negative > h:
            negative_signals.append(
                index
            )

    return CUSUMResult(
        subgroup_means=subgroup_means,

        target_mean=target,
        reference_value=k,
        decision_interval=h,

        positive_cusum=tuple(
            positive
        ),

        negative_cusum=tuple(
            negative_magnitude
        ),

        positive_signal_indices=tuple(
            positive_signals
        ),

        negative_signal_indices=tuple(
            negative_signals
        ),
    )