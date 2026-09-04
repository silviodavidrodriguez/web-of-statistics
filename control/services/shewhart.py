"""
Statistical engine for Shewhart variable control charts.

Implemented charts:
    - X-bar and R
    - X-bar and s
    - Median and R
    - Individuals and Moving Range

Control-chart constants are defined centrally in
control.services.constants.
"""

import math

from dataclasses import dataclass
from statistics import median
from typing import Sequence

import numpy as np

from .constants import (
    INDIVIDUAL_MR_CONSTANTS,
    MEDIAN_R_CONSTANTS,
    XBAR_R_CONSTANTS,
    XBAR_S_CONSTANTS,
)


Number = int | float


# ================================================================
# Errors
# ================================================================


class VariableChartInputError(
    ValueError
):
    pass


# ================================================================
# Result objects
# ================================================================


@dataclass(frozen=True)
class XBarRResult:
    subgroup_size: int

    subgroup_means: tuple[float, ...]
    subgroup_ranges: tuple[float, ...]

    x_centerline: float
    x_upper_control_limit: float
    x_lower_control_limit: float

    range_centerline: float
    range_upper_control_limit: float
    range_lower_control_limit: float

    estimated_sigma: float

    A2: float
    d2: float
    D3: float
    D4: float


@dataclass(frozen=True)
class XBarSResult:
    subgroup_size: int

    subgroup_means: tuple[float, ...]
    subgroup_standard_deviations: tuple[float, ...]

    x_centerline: float
    x_upper_control_limit: float
    x_lower_control_limit: float

    s_centerline: float
    s_upper_control_limit: float
    s_lower_control_limit: float

    estimated_sigma: float

    A3: float
    c4: float
    B3: float
    B4: float


@dataclass(frozen=True)
class MedianRResult:
    subgroup_size: int

    subgroup_medians: tuple[float, ...]
    subgroup_ranges: tuple[float, ...]

    median_centerline: float
    median_upper_control_limit: float
    median_lower_control_limit: float

    range_centerline: float
    range_upper_control_limit: float
    range_lower_control_limit: float

    estimated_sigma: float

    A2_tilde: float
    d2: float
    D3: float
    D4: float


@dataclass(frozen=True)
class IndividualsMRResult:
    moving_range_length: int

    observations: tuple[float, ...]
    moving_ranges: tuple[float, ...]

    individuals_centerline: float
    individuals_upper_control_limit: float
    individuals_lower_control_limit: float

    moving_range_centerline: float
    moving_range_upper_control_limit: float
    moving_range_lower_control_limit: float

    estimated_sigma: float

    E2: float
    d2: float
    D3: float
    D4: float


# ================================================================
# Validation helpers
# ================================================================


def _finite_number(
    value,
    *,
    label,
) -> float:

    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        raise VariableChartInputError(
            f"{label} must be numeric."
        )

    if not math.isfinite(
        numeric
    ):
        raise VariableChartInputError(
            f"{label} must be finite."
        )

    return numeric


def _validate_subgroups(
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
        raise VariableChartInputError(
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
            raise VariableChartInputError(
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
            raise VariableChartInputError(
                (
                    f"Subgroup "
                    f"{subgroup_index} "
                    "cannot be empty."
                )
            )

        parsed.append(
            values
        )

    if len(parsed) < 2:
        raise VariableChartInputError(
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
        raise VariableChartInputError(
            (
                "All subgroups must have "
                "the same size."
            )
        )

    return tuple(
        parsed
    )


def _validate_supported_size(
    subgroup_size,
    constants,
    *,
    chart_name,
):
    if subgroup_size not in constants:
        supported = ", ".join(
            str(value)
            for value in constants
        )

        raise VariableChartInputError(
            (
                f"{chart_name} does not have "
                "a tabulated constant for "
                f"subgroup size {subgroup_size}. "
                "Supported sizes are: "
                f"{supported}."
            )
        )


def _subgroup_means(
    subgroups,
):
    return tuple(
        float(
            np.mean(
                subgroup
            )
        )
        for subgroup in subgroups
    )


def _subgroup_ranges(
    subgroups,
):
    return tuple(
        float(
            max(subgroup)
            - min(subgroup)
        )
        for subgroup in subgroups
    )


# ================================================================
# X-bar and R chart
# ================================================================


def calculate_xbar_r(
    subgroups: Sequence[
        Sequence[Number]
    ],
) -> XBarRResult:

    groups = _validate_subgroups(
        subgroups
    )

    subgroup_size = len(
        groups[0]
    )

    _validate_supported_size(
        subgroup_size,
        XBAR_R_CONSTANTS,
        chart_name="X-bar and R chart",
    )

    constants = (
        XBAR_R_CONSTANTS[
            subgroup_size
        ]
    )

    subgroup_means = (
        _subgroup_means(
            groups
        )
    )

    subgroup_ranges = (
        _subgroup_ranges(
            groups
        )
    )

    x_bar_bar = float(
        np.mean(
            subgroup_means
        )
    )

    r_bar = float(
        np.mean(
            subgroup_ranges
        )
    )

    A2 = constants["A2"]
    d2 = constants["d2"]
    D3 = constants["D3"]
    D4 = constants["D4"]

    return XBarRResult(
        subgroup_size=(
            subgroup_size
        ),

        subgroup_means=(
            subgroup_means
        ),

        subgroup_ranges=(
            subgroup_ranges
        ),

        x_centerline=(
            x_bar_bar
        ),

        x_upper_control_limit=(
            x_bar_bar
            + A2 * r_bar
        ),

        x_lower_control_limit=(
            x_bar_bar
            - A2 * r_bar
        ),

        range_centerline=(
            r_bar
        ),

        range_upper_control_limit=(
            D4 * r_bar
        ),

        range_lower_control_limit=(
            D3 * r_bar
        ),

        estimated_sigma=(
            r_bar / d2
        ),

        A2=A2,
        d2=d2,
        D3=D3,
        D4=D4,
    )


# ================================================================
# X-bar and s chart
# ================================================================


def calculate_xbar_s(
    subgroups: Sequence[
        Sequence[Number]
    ],
) -> XBarSResult:

    groups = _validate_subgroups(
        subgroups
    )

    subgroup_size = len(
        groups[0]
    )

    _validate_supported_size(
        subgroup_size,
        XBAR_S_CONSTANTS,
        chart_name="X-bar and s chart",
    )

    constants = (
        XBAR_S_CONSTANTS[
            subgroup_size
        ]
    )

    subgroup_means = (
        _subgroup_means(
            groups
        )
    )

    subgroup_standard_deviations = tuple(
        float(
            np.std(
                subgroup,
                ddof=1,
            )
        )
        for subgroup in groups
    )

    x_bar_bar = float(
        np.mean(
            subgroup_means
        )
    )

    s_bar = float(
        np.mean(
            subgroup_standard_deviations
        )
    )

    A3 = constants["A3"]
    c4 = constants["c4"]
    B3 = constants["B3"]
    B4 = constants["B4"]

    return XBarSResult(
        subgroup_size=(
            subgroup_size
        ),

        subgroup_means=(
            subgroup_means
        ),

        subgroup_standard_deviations=(
            subgroup_standard_deviations
        ),

        x_centerline=(
            x_bar_bar
        ),

        x_upper_control_limit=(
            x_bar_bar
            + A3 * s_bar
        ),

        x_lower_control_limit=(
            x_bar_bar
            - A3 * s_bar
        ),

        s_centerline=(
            s_bar
        ),

        s_upper_control_limit=(
            B4 * s_bar
        ),

        s_lower_control_limit=(
            B3 * s_bar
        ),

        estimated_sigma=(
            s_bar / c4
        ),

        A3=A3,
        c4=c4,
        B3=B3,
        B4=B4,
    )


# ================================================================
# Median and R chart
# ================================================================


def calculate_median_r(
    subgroups: Sequence[
        Sequence[Number]
    ],
) -> MedianRResult:

    groups = _validate_subgroups(
        subgroups
    )

    subgroup_size = len(
        groups[0]
    )

    _validate_supported_size(
        subgroup_size,
        MEDIAN_R_CONSTANTS,
        chart_name="Median and R chart",
    )

    constants = (
        MEDIAN_R_CONSTANTS[
            subgroup_size
        ]
    )

    subgroup_medians = tuple(
        float(
            median(
                subgroup
            )
        )
        for subgroup in groups
    )

    subgroup_ranges = (
        _subgroup_ranges(
            groups
        )
    )

    median_bar = float(
        np.mean(
            subgroup_medians
        )
    )

    r_bar = float(
        np.mean(
            subgroup_ranges
        )
    )

    A2_tilde = (
        constants[
            "A2_tilde"
        ]
    )

    d2 = constants["d2"]
    D3 = constants["D3"]
    D4 = constants["D4"]

    return MedianRResult(
        subgroup_size=(
            subgroup_size
        ),

        subgroup_medians=(
            subgroup_medians
        ),

        subgroup_ranges=(
            subgroup_ranges
        ),

        median_centerline=(
            median_bar
        ),

        median_upper_control_limit=(
            median_bar
            + A2_tilde * r_bar
        ),

        median_lower_control_limit=(
            median_bar
            - A2_tilde * r_bar
        ),

        range_centerline=(
            r_bar
        ),

        range_upper_control_limit=(
            D4 * r_bar
        ),

        range_lower_control_limit=(
            D3 * r_bar
        ),

        estimated_sigma=(
            r_bar / d2
        ),

        A2_tilde=(
            A2_tilde
        ),

        d2=d2,
        D3=D3,
        D4=D4,
    )


# ================================================================
# Individuals and Moving Range chart
# ================================================================


def calculate_individuals_mr(
    observations: Sequence[Number],
    *,
    moving_range_length: int = 2,
) -> IndividualsMRResult:

    if isinstance(
        observations,
        (str, bytes),
    ):
        raise VariableChartInputError(
            (
                "Observations must be "
                "a sequence."
            )
        )

    values = tuple(
        _finite_number(
            value,
            label="Observation",
        )
        for value in observations
    )

    if len(values) < 2:
        raise VariableChartInputError(
            (
                "At least two observations "
                "are required."
            )
        )

    if (
        isinstance(
            moving_range_length,
            bool,
        )
        or not isinstance(
            moving_range_length,
            int,
        )
    ):
        raise VariableChartInputError(
            (
                "Moving range length "
                "must be an integer."
            )
        )

    _validate_supported_size(
        moving_range_length,
        INDIVIDUAL_MR_CONSTANTS,
        chart_name=(
            "Individuals and Moving "
            "Range chart"
        ),
    )

    if (
        len(values)
        < moving_range_length
    ):
        raise VariableChartInputError(
            (
                "The number of observations "
                "must be at least as large as "
                "the moving range length."
            )
        )

    constants = (
        INDIVIDUAL_MR_CONSTANTS[
            moving_range_length
        ]
    )

    moving_ranges = tuple(
        float(
            max(
                values[
                    index
                    - moving_range_length
                    + 1:
                    index + 1
                ]
            )
            - min(
                values[
                    index
                    - moving_range_length
                    + 1:
                    index + 1
                ]
            )
        )
        for index in range(
            moving_range_length - 1,
            len(values),
        )
    )

    x_bar = float(
        np.mean(
            values
        )
    )

    mr_bar = float(
        np.mean(
            moving_ranges
        )
    )

    E2 = constants["E2"]
    d2 = constants["d2"]
    D3 = constants["D3"]
    D4 = constants["D4"]

    return IndividualsMRResult(
        moving_range_length=(
            moving_range_length
        ),

        observations=(
            values
        ),

        moving_ranges=(
            moving_ranges
        ),

        individuals_centerline=(
            x_bar
        ),

        individuals_upper_control_limit=(
            x_bar
            + E2 * mr_bar
        ),

        individuals_lower_control_limit=(
            x_bar
            - E2 * mr_bar
        ),

        moving_range_centerline=(
            mr_bar
        ),

        moving_range_upper_control_limit=(
            D4 * mr_bar
        ),

        moving_range_lower_control_limit=(
            D3 * mr_bar
        ),

        estimated_sigma=(
            mr_bar / d2
        ),

        E2=E2,
        d2=d2,
        D3=D3,
        D4=D4,
    )