"""
Statistical engine for attribute control charts.

Control-limit formulas are based on:
    Tables of Formulas for Control Charts
    Table 8C - Attribute Data
    Reference: AIAG manual for SPC
"""

import math

from dataclasses import dataclass
from typing import Sequence


Number = int | float


# ================================================================
# Errors
# ================================================================


class AttributeChartInputError(
    ValueError
):
    pass


# ================================================================
# Results
# ================================================================


@dataclass(frozen=True)
class PChartResult:
    sample_sizes: tuple[int, ...]
    defectives: tuple[int, ...]
    proportions: tuple[float, ...]
    centerline: float
    upper_control_limits: tuple[float, ...]
    lower_control_limits: tuple[float, ...]


@dataclass(frozen=True)
class NPChartResult:
    sample_size: int
    defectives: tuple[int, ...]
    p_bar: float
    centerline: float
    upper_control_limit: float
    lower_control_limit: float


@dataclass(frozen=True)
class CChartResult:
    counts: tuple[int, ...]
    centerline: float
    upper_control_limit: float
    lower_control_limit: float


@dataclass(frozen=True)
class UChartResult:
    sample_sizes: tuple[int, ...]
    incidences: tuple[int, ...]
    rates: tuple[float, ...]
    centerline: float
    upper_control_limits: tuple[float, ...]
    lower_control_limits: tuple[float, ...]


# ================================================================
# Validation
# ================================================================


def _integer_sequence(
    values: Sequence[Number],
    *,
    label: str,
    minimum: int,
) -> tuple[int, ...]:

    if isinstance(
        values,
        (str, bytes),
    ):
        raise AttributeChartInputError(
            f"{label} must be a sequence."
        )

    parsed = []

    for index, value in enumerate(
        values,
        start=1,
    ):
        try:
            numeric = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):
            raise AttributeChartInputError(
                (
                    f"{label} value "
                    f"{index} must be numeric."
                )
            )

        if not math.isfinite(
            numeric
        ):
            raise AttributeChartInputError(
                (
                    f"{label} value "
                    f"{index} must be finite."
                )
            )

        if not numeric.is_integer():
            raise AttributeChartInputError(
                (
                    f"{label} value "
                    f"{index} must be an integer."
                )
            )

        integer = int(
            numeric
        )

        if integer < minimum:
            raise AttributeChartInputError(
                (
                    f"{label} value "
                    f"{index} must be "
                    f"at least {minimum}."
                )
            )

        parsed.append(
            integer
        )

    if not parsed:
        raise AttributeChartInputError(
            f"{label} cannot be empty."
        )

    return tuple(
        parsed
    )


def _sample_sizes(
    values,
) -> tuple[int, ...]:

    return _integer_sequence(
        values,
        label="Sample size",
        minimum=1,
    )


def _counts(
    values,
    *,
    label,
) -> tuple[int, ...]:

    return _integer_sequence(
        values,
        label=label,
        minimum=0,
    )


def _require_same_length(
    first,
    second,
):
    if len(first) != len(second):
        raise AttributeChartInputError(
            (
                "Sample sizes and counts "
                "must contain the same "
                "number of observations."
            )
        )


# ================================================================
# p chart
# ================================================================


def calculate_p_chart(
    sample_sizes: Sequence[Number],
    defectives: Sequence[Number],
) -> PChartResult:

    sizes = _sample_sizes(
        sample_sizes
    )

    counts = _counts(
        defectives,
        label="Number of defectives",
    )

    _require_same_length(
        sizes,
        counts,
    )

    for index, (
        sample_size,
        defective_count,
    ) in enumerate(
        zip(
            sizes,
            counts,
        ),
        start=1,
    ):
        if (
            defective_count
            > sample_size
        ):
            raise AttributeChartInputError(
                (
                    "Number of defectives "
                    f"cannot exceed sample size "
                    f"in observation {index}."
                )
            )

    total_sample_size = sum(
        sizes
    )

    total_defectives = sum(
        counts
    )

    p_bar = (
        total_defectives
        / total_sample_size
    )

    proportions = tuple(
        defective_count
        / sample_size
        for sample_size,
        defective_count
        in zip(
            sizes,
            counts,
        )
    )

    upper_limits = []
    lower_limits = []

    for sample_size in sizes:

        standard_error = math.sqrt(
            (
                p_bar
                * (1.0 - p_bar)
            )
            / sample_size
        )

        upper_limits.append(
            p_bar
            + 3.0 * standard_error
        )

        lower_limits.append(
            p_bar
            - 3.0 * standard_error
        )

    return PChartResult(
        sample_sizes=sizes,
        defectives=counts,
        proportions=(
            proportions
        ),
        centerline=p_bar,
        upper_control_limits=tuple(
            upper_limits
        ),
        lower_control_limits=tuple(
            lower_limits
        ),
    )


# ================================================================
# np chart
# ================================================================


def calculate_np_chart(
    sample_sizes: Sequence[Number],
    defectives: Sequence[Number],
) -> NPChartResult:

    sizes = _sample_sizes(
        sample_sizes
    )

    counts = _counts(
        defectives,
        label="Number of defectives",
    )

    _require_same_length(
        sizes,
        counts,
    )

    unique_sizes = set(
        sizes
    )

    if len(unique_sizes) != 1:
        raise AttributeChartInputError(
            (
                "The np chart requires "
                "a constant sample size."
            )
        )

    sample_size = sizes[0]

    for index, defective_count in (
        enumerate(
            counts,
            start=1,
        )
    ):
        if (
            defective_count
            > sample_size
        ):
            raise AttributeChartInputError(
                (
                    "Number of defectives "
                    f"cannot exceed sample size "
                    f"in observation {index}."
                )
            )

    p_bar = (
        sum(counts)
        / (
            sample_size
            * len(counts)
        )
    )

    centerline = (
        sample_size
        * p_bar
    )

    standard_deviation = (
        math.sqrt(
            sample_size
            * p_bar
            * (1.0 - p_bar)
        )
    )

    upper_limit = (
        centerline
        + 3.0
        * standard_deviation
    )

    lower_limit = (
        centerline
        - 3.0
        * standard_deviation
    )

    return NPChartResult(
        sample_size=(
            sample_size
        ),
        defectives=counts,
        p_bar=p_bar,
        centerline=centerline,
        upper_control_limit=(
            upper_limit
        ),
        lower_control_limit=(
            lower_limit
        ),
    )


# ================================================================
# c chart
# ================================================================


def calculate_c_chart(
    incidences: Sequence[Number],
) -> CChartResult:

    counts = _counts(
        incidences,
        label="Number of incidences",
    )

    centerline = (
        sum(counts)
        / len(counts)
    )

    standard_deviation = (
        math.sqrt(
            centerline
        )
    )

    upper_limit = (
        centerline
        + 3.0
        * standard_deviation
    )

    lower_limit = (
        centerline
        - 3.0
        * standard_deviation
    )

    return CChartResult(
        counts=counts,
        centerline=centerline,
        upper_control_limit=(
            upper_limit
        ),
        lower_control_limit=(
            lower_limit
        ),
    )


# ================================================================
# u chart
# ================================================================


def calculate_u_chart(
    sample_sizes: Sequence[Number],
    incidences: Sequence[Number],
) -> UChartResult:

    sizes = _sample_sizes(
        sample_sizes
    )

    counts = _counts(
        incidences,
        label="Number of incidences",
    )

    _require_same_length(
        sizes,
        counts,
    )

    rates = tuple(
        incidence_count
        / sample_size
        for sample_size,
        incidence_count
        in zip(
            sizes,
            counts,
        )
    )

    # Pooled average number of
    # incidences per inspected unit.
    u_bar = (
        sum(counts)
        / sum(sizes)
    )

    upper_limits = []
    lower_limits = []

    for sample_size in sizes:

        standard_error = (
            math.sqrt(
                u_bar
                / sample_size
            )
        )

        upper_limits.append(
            u_bar
            + 3.0
            * standard_error
        )

        lower_limits.append(
            u_bar
            - 3.0
            * standard_error
        )

    return UChartResult(
        sample_sizes=sizes,
        incidences=counts,
        rates=rates,
        centerline=u_bar,
        upper_control_limits=tuple(
            upper_limits
        ),
        lower_control_limits=tuple(
            lower_limits
        ),
    )