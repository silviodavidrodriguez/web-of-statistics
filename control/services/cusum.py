"""
CUSUM control-chart statistical engine.

Implemented methods:
    - Two-sided tabular CUSUM
    - Two-sided V-mask CUSUM

Both methods use the same basic parameters:

    target_mean:
        Target or reference process mean.

    reference_value (K):
        Reference value used to determine sensitivity
        to process shifts.

    decision_interval (H):
        Decision interval used to determine when a
        special-cause signal occurs.

For the V-mask representation, the lead distance is:

    d = H / K

The cumulative sum used by the V-mask is:

    S_i = sum(x_bar_j - target_mean)

with S_0 = 0.
"""

import math

from dataclasses import dataclass
from typing import Sequence


Number = int | float


# ================================================================
# Errors
# ================================================================


class CUSUMInputError(ValueError):
    """Invalid input supplied to CUSUM analysis."""


# ================================================================
# Tabular CUSUM result
# ================================================================


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


# ================================================================
# V-mask results
# ================================================================


@dataclass(frozen=True)
class VMaskSignal:
    """
    Signal detected by the V-mask.

    current_index:
        1-based subgroup index at which the signal
        is detected.

    direction:
        "upward" or "downward".

    violating_cumulative_indices:
        Indices of cumulative-sum points that cross
        the corresponding V-mask arm.

        Index 0 represents the cumulative origin S0 = 0.
    """

    current_index: int
    direction: str
    violating_cumulative_indices: tuple[int, ...]


@dataclass(frozen=True)
class VMaskCUSUMResult:
    subgroup_means: tuple[float, ...]

    # Includes S0 = 0.
    cumulative_sums: tuple[float, ...]

    target_mean: float
    reference_value: float
    decision_interval: float

    # d = H / K
    lead_distance: float

    # Geometry of the V-mask placed at the
    # final cumulative-sum point.
    final_vertex_x: float
    final_vertex_y: float

    final_mask_x: tuple[float, ...]
    final_upper_boundary: tuple[float, ...]
    final_lower_boundary: tuple[float, ...]

    positive_signal_indices: tuple[int, ...]
    negative_signal_indices: tuple[int, ...]

    signals: tuple[VMaskSignal, ...]


# ================================================================
# Validation helpers
# ================================================================


def _finite_number(
    value,
    *,
    label: str,
) -> float:
    """
    Parse a numeric value and require it to be finite.
    """

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
    """
    Validate and parse subgroups.

    All subgroups must:
        - contain numeric finite observations;
        - contain at least one observation;
        - have the same subgroup size.
    """

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

        parsed.append(
            values
        )

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

    return tuple(
        parsed
    )


def _calculate_subgroup_means(
    groups,
) -> tuple[float, ...]:
    """
    Calculate the arithmetic mean of each subgroup.
    """

    return tuple(
        sum(subgroup) / len(subgroup)
        for subgroup in groups
    )


def _resolve_target_mean(
    groups,
    target_mean,
) -> float:
    """
    Return the supplied target mean.

    If no target is supplied, use the overall arithmetic
    mean of all observations.
    """

    if target_mean is not None:
        return _finite_number(
            target_mean,
            label="Target mean",
        )

    all_values = tuple(
        value
        for subgroup in groups
        for value in subgroup
    )

    return (
        sum(all_values)
        / len(all_values)
    )


# ================================================================
# Tabular CUSUM
# ================================================================


def calculate_cusum(
    subgroups: Sequence[
        Sequence[Number]
    ],
    *,
    target_mean: Number | None = None,
    reference_value: Number,
    decision_interval: Number,
) -> CUSUMResult:
    """
    Calculate a two-sided tabular CUSUM.

    Positive CUSUM:

        C+_i = max(
            0,
            C+_(i-1) + x_bar_i - target - K
        )

    Negative CUSUM magnitude:

        C-_i = max(
            0,
            C-_(i-1) + target - x_bar_i - K
        )

    For convenient plotting, the lower CUSUM is returned
    as a negative value.

    A signal occurs whenever the corresponding magnitude
    exceeds H.
    """

    groups = _parse_subgroups(
        subgroups
    )

    subgroup_means = (
        _calculate_subgroup_means(
            groups
        )
    )

    target = _resolve_target_mean(
        groups,
        target_mean,
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
    negative = []

    previous_positive = 0.0
    previous_negative = 0.0

    positive_signals = []
    negative_signals = []

    for index, subgroup_mean in enumerate(
        subgroup_means,
        start=1,
    ):
        # --------------------------------------------------------
        # Upper / positive CUSUM
        # --------------------------------------------------------

        previous_positive = max(
            0.0,
            (
                previous_positive
                + subgroup_mean
                - target
                - k
            ),
        )

        # --------------------------------------------------------
        # Lower / negative CUSUM
        #
        # Internally it is maintained as a positive magnitude.
        # It is returned as a negative value for plotting below
        # the zero line.
        # --------------------------------------------------------

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

        negative.append(
            -previous_negative
        )

        # --------------------------------------------------------
        # Signals
        # --------------------------------------------------------

        if previous_positive > h:
            positive_signals.append(
                index
            )

        if previous_negative > h:
            negative_signals.append(
                index
            )

    return CUSUMResult(
        subgroup_means=(
            subgroup_means
        ),

        target_mean=(
            target
        ),

        reference_value=(
            k
        ),

        decision_interval=(
            h
        ),

        positive_cusum=tuple(
            positive
        ),

        negative_cusum=tuple(
            negative
        ),

        positive_signal_indices=tuple(
            positive_signals
        ),

        negative_signal_indices=tuple(
            negative_signals
        ),
    )


# ================================================================
# V-mask CUSUM
# ================================================================


def calculate_vmask_cusum(
    subgroups: Sequence[
        Sequence[Number]
    ],
    *,
    target_mean: Number | None = None,
    reference_value: Number,
    decision_interval: Number,
) -> VMaskCUSUMResult:
    """
    Calculate a two-sided CUSUM using the V-mask method.

    The cumulative sums are:

        S_0 = 0

        S_i = S_(i-1) + x_bar_i - target

    The V-mask lead distance is:

        d = H / K

    For each current cumulative-sum point S_i, the
    conceptual vertex of the mask is placed d units
    ahead of the current subgroup:

        vertex_x = i + d
        vertex_y = S_i

    For any previous cumulative point S_j:

        upper arm =
            S_i + K * (i + d - j)

        lower arm =
            S_i - K * (i + d - j)

    A previous cumulative point below the lower arm
    indicates an upward shift.

    A previous cumulative point above the upper arm
    indicates a downward shift.

    This construction is mathematically equivalent to
    the two-sided tabular CUSUM when K and H are the same.
    """

    groups = _parse_subgroups(
        subgroups
    )

    subgroup_means = (
        _calculate_subgroup_means(
            groups
        )
    )

    target = _resolve_target_mean(
        groups,
        target_mean,
    )

    k = _finite_number(
        reference_value,
        label="Reference value",
    )

    h = _finite_number(
        decision_interval,
        label="Decision interval",
    )

    # ------------------------------------------------------------
    # V-mask specifically requires K > 0 because d = H / K.
    # ------------------------------------------------------------

    if k <= 0:
        raise CUSUMInputError(
            (
                "Reference value must be "
                "greater than 0 for the "
                "V-mask method."
            )
        )

    if h <= 0:
        raise CUSUMInputError(
            (
                "Decision interval must be "
                "greater than 0."
            )
        )

    # ============================================================
    # Lead distance
    # ============================================================

    lead_distance = (
        h / k
    )

    # ============================================================
    # Cumulative sums
    #
    # Keep S0 = 0 explicitly.
    # This is important for the geometric equivalence between
    # V-mask and tabular CUSUM.
    # ============================================================

    cumulative_sums = [
        0.0
    ]

    running_sum = 0.0

    for subgroup_mean in (
        subgroup_means
    ):
        running_sum += (
            subgroup_mean
            - target
        )

        cumulative_sums.append(
            running_sum
        )

    # ============================================================
    # Signal detection
    # ============================================================

    positive_signals = []
    negative_signals = []

    signals = []

    # current_index runs from 1 to n.
    #
    # This directly corresponds to the user-facing subgroup
    # number, so signal indices are naturally 1-based.
    for current_index in range(
        1,
        len(cumulative_sums),
    ):
        current_sum = (
            cumulative_sums[
                current_index
            ]
        )

        lower_violations = []
        upper_violations = []

        # Compare the current V-mask against all earlier
        # cumulative-sum points, including S0.
        for previous_index in range(
            current_index
        ):
            previous_sum = (
                cumulative_sums[
                    previous_index
                ]
            )

            horizontal_distance = (
                current_index
                + lead_distance
                - previous_index
            )

            upper_boundary = (
                current_sum
                + k
                * horizontal_distance
            )

            lower_boundary = (
                current_sum
                - k
                * horizontal_distance
            )

            # ----------------------------------------------------
            # Upward shift
            #
            # If an earlier cumulative-sum point falls below
            # the lower arm of the V-mask, the process has
            # accumulated enough positive deviation to signal.
            # ----------------------------------------------------

            if (
                previous_sum
                < lower_boundary
            ):
                lower_violations.append(
                    previous_index
                )

            # ----------------------------------------------------
            # Downward shift
            #
            # If an earlier cumulative-sum point rises above
            # the upper arm, the process has accumulated enough
            # negative deviation to signal.
            # ----------------------------------------------------

            if (
                previous_sum
                > upper_boundary
            ):
                upper_violations.append(
                    previous_index
                )

        # --------------------------------------------------------
        # Positive / upward signal
        # --------------------------------------------------------

        if lower_violations:
            positive_signals.append(
                current_index
            )

            signals.append(
                VMaskSignal(
                    current_index=(
                        current_index
                    ),

                    direction=(
                        "upward"
                    ),

                    violating_cumulative_indices=tuple(
                        lower_violations
                    ),
                )
            )

        # --------------------------------------------------------
        # Negative / downward signal
        # --------------------------------------------------------

        if upper_violations:
            negative_signals.append(
                current_index
            )

            signals.append(
                VMaskSignal(
                    current_index=(
                        current_index
                    ),

                    direction=(
                        "downward"
                    ),

                    violating_cumulative_indices=tuple(
                        upper_violations
                    ),
                )
            )

    # ============================================================
    # Geometry of the FINAL V-mask
    #
    # The statistical engine calculates the geometry.
    #
    # Later Plotly only needs to draw these coordinates.
    # Plotly will NOT be responsible for deciding whether
    # the process has generated a signal.
    # ============================================================

    final_index = len(
        subgroup_means
    )

    final_sum = (
        cumulative_sums[
            final_index
        ]
    )

    final_vertex_x = (
        final_index
        + lead_distance
    )

    final_vertex_y = (
        final_sum
    )

    # Points along the historical cumulative-sum axis:
    #
    # 0, 1, 2, ..., n
    final_mask_x = tuple(
        float(index)
        for index in range(
            final_index + 1
        )
    )

    final_upper_boundary = []
    final_lower_boundary = []

    for cumulative_index in range(
        final_index + 1
    ):
        horizontal_distance = (
            final_vertex_x
            - cumulative_index
        )

        final_upper_boundary.append(
            final_sum
            + k
            * horizontal_distance
        )

        final_lower_boundary.append(
            final_sum
            - k
            * horizontal_distance
        )

    return VMaskCUSUMResult(
        subgroup_means=(
            subgroup_means
        ),

        cumulative_sums=tuple(
            cumulative_sums
        ),

        target_mean=(
            target
        ),

        reference_value=(
            k
        ),

        decision_interval=(
            h
        ),

        lead_distance=(
            lead_distance
        ),

        final_vertex_x=(
            final_vertex_x
        ),

        final_vertex_y=(
            final_vertex_y
        ),

        final_mask_x=(
            final_mask_x
        ),

        final_upper_boundary=tuple(
            final_upper_boundary
        ),

        final_lower_boundary=tuple(
            final_lower_boundary
        ),

        positive_signal_indices=tuple(
            positive_signals
        ),

        negative_signal_indices=tuple(
            negative_signals
        ),

        signals=tuple(
            signals
        ),
    )