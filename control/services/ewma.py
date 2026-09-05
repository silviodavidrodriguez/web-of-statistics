"""
Exponentially Weighted Moving Average (EWMA)
control-chart statistical engine.

For subgroup means:

    z_0 = target

    z_i = lambda * x_bar_i
          + (1 - lambda) * z_(i-1)

Dynamic control limits:

    sigma_z(i) =
        sigma / sqrt(n)
        * sqrt(
            lambda / (2 - lambda)
            * (
                1
                - (1 - lambda) ** (2 * i)
            )
        )

    UCL_i = target + L * sigma_z(i)
    LCL_i = target - L * sigma_z(i)

where:

    sigma:
        within-process standard deviation

    n:
        subgroup size

    L:
        control-limit multiplier,
        conventionally 3.

If sigma is not supplied, it is estimated using the
pooled within-subgroup sample variance.
"""

import math

from dataclasses import dataclass
from typing import Sequence


Number = int | float


# ================================================================
# Errors
# ================================================================


class EWMAInputError(ValueError):
    """Invalid input supplied to EWMA analysis."""


# ================================================================
# Result
# ================================================================


@dataclass(frozen=True)
class EWMAResult:
    subgroup_size: int

    subgroup_means: tuple[float, ...]
    ewma_values: tuple[float, ...]

    target_mean: float
    lambda_value: float
    control_limit_width: float

    process_sigma: float
    sigma_source: str

    ewma_standard_errors: tuple[float, ...]

    upper_control_limits: tuple[float, ...]
    lower_control_limits: tuple[float, ...]

    signal_indices: tuple[int, ...]


# ================================================================
# Validation
# ================================================================


def _finite_number(
    value,
    *,
    label: str,
) -> float:

    try:
        numeric = float(value)

    except (TypeError, ValueError):
        raise EWMAInputError(
            f"{label} must be numeric."
        )

    if not math.isfinite(numeric):
        raise EWMAInputError(
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
        raise EWMAInputError(
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
            raise EWMAInputError(
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
            raise EWMAInputError(
                (
                    f"Subgroup {subgroup_index} "
                    "cannot be empty."
                )
            )

        parsed.append(
            values
        )

    if len(parsed) < 2:
        raise EWMAInputError(
            (
                "At least two subgroups "
                "are required."
            )
        )

    subgroup_sizes = {
        len(subgroup)
        for subgroup in parsed
    }

    if len(subgroup_sizes) != 1:
        raise EWMAInputError(
            (
                "All subgroups must have "
                "the same size."
            )
        )

    return tuple(
        parsed
    )


# ================================================================
# Statistical helpers
# ================================================================


def _subgroup_mean(
    subgroup,
) -> float:

    return (
        sum(subgroup)
        / len(subgroup)
    )


def _sample_variance(
    values,
) -> float:

    n = len(values)

    if n < 2:
        raise EWMAInputError(
            (
                "At least two observations per subgroup "
                "are required to estimate process sigma."
            )
        )

    average = (
        sum(values)
        / n
    )

    return (
        sum(
            (
                value - average
            ) ** 2
            for value in values
        )
        / (n - 1)
    )


def _pooled_within_sigma(
    groups,
) -> float:
    """
    Estimate the within-process sigma from the pooled
    within-subgroup sample variances.
    """

    numerator = 0.0
    denominator = 0

    for subgroup in groups:
        degrees_of_freedom = (
            len(subgroup) - 1
        )

        subgroup_variance = (
            _sample_variance(
                subgroup
            )
        )

        numerator += (
            degrees_of_freedom
            * subgroup_variance
        )

        denominator += (
            degrees_of_freedom
        )

    pooled_variance = (
        numerator
        / denominator
    )

    sigma = math.sqrt(
        pooled_variance
    )

    if sigma <= 0:
        raise EWMAInputError(
            (
                "Process sigma cannot be estimated "
                "because there is no within-subgroup "
                "variation."
            )
        )

    return sigma


# ================================================================
# EWMA calculation
# ================================================================


def calculate_ewma(
    subgroups: Sequence[
        Sequence[Number]
    ],
    *,
    target_mean: Number | None = None,
    lambda_value: Number = 0.2,
    process_sigma: Number | None = None,
    control_limit_width: Number = 3.0,
) -> EWMAResult:

    groups = _parse_subgroups(
        subgroups
    )

    subgroup_size = len(
        groups[0]
    )

    subgroup_means = tuple(
        _subgroup_mean(
            subgroup
        )
        for subgroup in groups
    )

    # ------------------------------------------------------------
    # Target mean
    # ------------------------------------------------------------

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

    # ------------------------------------------------------------
    # Lambda
    # ------------------------------------------------------------

    lambda_parsed = _finite_number(
        lambda_value,
        label="Lambda",
    )

    if not (
        0.0
        < lambda_parsed
        <= 1.0
    ):
        raise EWMAInputError(
            (
                "Lambda must be greater than 0 "
                "and less than or equal to 1."
            )
        )

    # ------------------------------------------------------------
    # Control-limit width L
    # ------------------------------------------------------------

    limit_width = _finite_number(
        control_limit_width,
        label="Control-limit width",
    )

    if limit_width <= 0:
        raise EWMAInputError(
            (
                "Control-limit width must be "
                "greater than 0."
            )
        )

    # ------------------------------------------------------------
    # Process sigma
    # ------------------------------------------------------------

    if process_sigma is None:
        sigma = _pooled_within_sigma(
            groups
        )

        sigma_source = (
            "pooled_within_subgroup"
        )

    else:
        sigma = _finite_number(
            process_sigma,
            label="Process sigma",
        )

        if sigma <= 0:
            raise EWMAInputError(
                (
                    "Process sigma must be "
                    "greater than 0."
                )
            )

        sigma_source = (
            "provided"
        )

    # ------------------------------------------------------------
    # EWMA recurrence
    #
    # z0 = target
    #
    # z1 uses the FIRST subgroup.
    # ------------------------------------------------------------

    ewma_values = []

    previous_ewma = (
        target
    )

    for subgroup_mean in (
        subgroup_means
    ):
        current_ewma = (
            lambda_parsed
            * subgroup_mean
            + (
                1.0
                - lambda_parsed
            )
            * previous_ewma
        )

        ewma_values.append(
            current_ewma
        )

        previous_ewma = (
            current_ewma
        )

    # ------------------------------------------------------------
    # Dynamic control limits
    # ------------------------------------------------------------

    standard_errors = []
    upper_limits = []
    lower_limits = []

    sigma_xbar = (
        sigma
        / math.sqrt(
            subgroup_size
        )
    )

    for index in range(
        1,
        len(subgroup_means) + 1,
    ):
        transient_factor = math.sqrt(
            (
                lambda_parsed
                / (
                    2.0
                    - lambda_parsed
                )
            )
            * (
                1.0
                - (
                    1.0
                    - lambda_parsed
                ) ** (
                    2 * index
                )
            )
        )

        ewma_standard_error = (
            sigma_xbar
            * transient_factor
        )

        standard_errors.append(
            ewma_standard_error
        )

        upper_limits.append(
            target
            + limit_width
            * ewma_standard_error
        )

        lower_limits.append(
            target
            - limit_width
            * ewma_standard_error
        )

    # ------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------

    signal_indices = tuple(
        index
        for index, (
            ewma_value,
            lower_limit,
            upper_limit,
        ) in enumerate(
            zip(
                ewma_values,
                lower_limits,
                upper_limits,
            ),
            start=1,
        )
        if (
            ewma_value
            < lower_limit
            or ewma_value
            > upper_limit
        )
    )

    return EWMAResult(
        subgroup_size=(
            subgroup_size
        ),

        subgroup_means=(
            subgroup_means
        ),

        ewma_values=tuple(
            ewma_values
        ),

        target_mean=(
            target
        ),

        lambda_value=(
            lambda_parsed
        ),

        control_limit_width=(
            limit_width
        ),

        process_sigma=(
            sigma
        ),

        sigma_source=(
            sigma_source
        ),

        ewma_standard_errors=tuple(
            standard_errors
        ),

        upper_control_limits=tuple(
            upper_limits
        ),

        lower_control_limits=tuple(
            lower_limits
        ),

        signal_indices=(
            signal_indices
        ),
    )