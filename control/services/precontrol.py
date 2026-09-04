"""
Precontrol statistical engine.

The classic symmetric Precontrol chart divides the
specification interval into:

    Red | Yellow | Green | Yellow | Red

For a nominal value M and symmetric tolerance T:

    LSL = M - T
    USL = M + T

    Green lower = M - T / 2
    Green upper = M + T / 2

Setup qualification is evaluated sequentially:

    - Five consecutive green observations:
        process qualified.

    - One yellow observation:
        restart the green count.

    - Two consecutive yellow observations
      on the same side:
        stop and adjust.

    - Two consecutive yellow observations
      on opposite sides:
        stop and investigate excessive variation.

    - One red observation:
        stop and adjust.
"""

import math

from dataclasses import dataclass
from typing import Sequence


Number = int | float


# ================================================================
# Zone names
# ================================================================


GREEN = "green"

YELLOW_LOWER = "yellow_lower"
YELLOW_UPPER = "yellow_upper"

RED_LOWER = "red_lower"
RED_UPPER = "red_upper"


# ================================================================
# Errors
# ================================================================


class PrecontrolInputError(ValueError):
    """Invalid input supplied to Precontrol analysis."""


# ================================================================
# Results
# ================================================================


@dataclass(frozen=True)
class PrecontrolPoint:
    index: int
    value: float
    zone: str


@dataclass(frozen=True)
class PrecontrolDecision:
    status: str
    decision_index: int | None
    reason: str | None
    action: str


@dataclass(frozen=True)
class PrecontrolResult:
    observations: tuple[float, ...]

    nominal_value: float
    tolerance_value: float

    lower_spec_limit: float
    upper_spec_limit: float

    green_lower_limit: float
    green_upper_limit: float

    points: tuple[PrecontrolPoint, ...]

    green_count: int
    yellow_lower_count: int
    yellow_upper_count: int
    red_lower_count: int
    red_upper_count: int

    decision: PrecontrolDecision


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
        raise PrecontrolInputError(
            f"{label} must be numeric."
        )

    if not math.isfinite(numeric):
        raise PrecontrolInputError(
            f"{label} must be finite."
        )

    return numeric


def _parse_observations(
    observations: Sequence[Number],
) -> tuple[float, ...]:

    if isinstance(
        observations,
        (str, bytes),
    ):
        raise PrecontrolInputError(
            "Observations must be a sequence."
        )

    parsed = tuple(
        _finite_number(
            value,
            label=f"Observation {index}",
        )
        for index, value in enumerate(
            observations,
            start=1,
        )
    )

    if not parsed:
        raise PrecontrolInputError(
            "At least one observation is required."
        )

    return parsed


# ================================================================
# Zone classification
# ================================================================


def _classify_zone(
    value: float,
    *,
    lower_spec_limit: float,
    upper_spec_limit: float,
    green_lower_limit: float,
    green_upper_limit: float,
) -> str:

    if value < lower_spec_limit:
        return RED_LOWER

    if value > upper_spec_limit:
        return RED_UPPER

    if value < green_lower_limit:
        return YELLOW_LOWER

    if value > green_upper_limit:
        return YELLOW_UPPER

    return GREEN


# ================================================================
# Qualification rules
# ================================================================


def _evaluate_qualification(
    points: tuple[
        PrecontrolPoint,
        ...
    ],
) -> PrecontrolDecision:

    consecutive_green = 0

    previous_yellow_zone = None

    for point in points:

        # --------------------------------------------------------
        # Red
        # --------------------------------------------------------

        if point.zone in (
            RED_LOWER,
            RED_UPPER,
        ):
            return PrecontrolDecision(
                status="rejected",
                decision_index=point.index,
                reason="red_observation",
                action=(
                    "Stop and adjust the process. "
                    "The observation is outside "
                    "the specification limits."
                ),
            )

        # --------------------------------------------------------
        # Green
        # --------------------------------------------------------

        if point.zone == GREEN:
            consecutive_green += 1

            # A green observation breaks a sequence
            # of consecutive yellow observations.
            previous_yellow_zone = None

            if consecutive_green >= 5:
                return PrecontrolDecision(
                    status="qualified",
                    decision_index=point.index,
                    reason="five_consecutive_green",
                    action=(
                        "Setup qualified. "
                        "Five consecutive observations "
                        "are in the green zone."
                    ),
                )

            continue

        # --------------------------------------------------------
        # Yellow
        # --------------------------------------------------------

        consecutive_green = 0

        if point.zone in (
            YELLOW_LOWER,
            YELLOW_UPPER,
        ):

            if previous_yellow_zone is not None:

                # Same-side yellow pair:
                # process appears off-center.
                if (
                    previous_yellow_zone
                    == point.zone
                ):
                    return PrecontrolDecision(
                        status="rejected",
                        decision_index=point.index,
                        reason=(
                            "two_yellow_same_side"
                        ),
                        action=(
                            "Stop and adjust the process. "
                            "Two consecutive yellow "
                            "observations occurred on "
                            "the same side of the target."
                        ),
                    )

                # Opposite-side yellow pair:
                # process spread may be excessive.
                return PrecontrolDecision(
                    status="rejected",
                    decision_index=point.index,
                    reason=(
                        "two_yellow_opposite_sides"
                    ),
                    action=(
                        "Stop and investigate process "
                        "variation. Two consecutive "
                        "yellow observations occurred "
                        "on opposite sides of the target."
                    ),
                )

            previous_yellow_zone = (
                point.zone
            )

    return PrecontrolDecision(
        status="pending",
        decision_index=None,
        reason=None,
        action=(
            "More observations are required "
            "to reach a qualification decision."
        ),
    )


# ================================================================
# Main calculation
# ================================================================


def calculate_precontrol(
    observations: Sequence[Number],
    *,
    nominal_value: Number,
    tolerance_value: Number,
) -> PrecontrolResult:

    values = _parse_observations(
        observations
    )

    nominal = _finite_number(
        nominal_value,
        label="Nominal value",
    )

    tolerance = _finite_number(
        tolerance_value,
        label="Tolerance value",
    )

    if tolerance <= 0:
        raise PrecontrolInputError(
            (
                "Tolerance value must be "
                "greater than 0."
            )
        )

    # ============================================================
    # Specification limits
    # ============================================================

    lower_spec_limit = (
        nominal
        - tolerance
    )

    upper_spec_limit = (
        nominal
        + tolerance
    )

    # ============================================================
    # Precontrol reference limits
    #
    # Green zone = central 50% of tolerance band.
    # ============================================================

    green_lower_limit = (
        nominal
        - tolerance / 2.0
    )

    green_upper_limit = (
        nominal
        + tolerance / 2.0
    )

    # ============================================================
    # Classify every observation
    # ============================================================

    points = tuple(
        PrecontrolPoint(
            index=index,
            value=value,
            zone=_classify_zone(
                value,
                lower_spec_limit=(
                    lower_spec_limit
                ),
                upper_spec_limit=(
                    upper_spec_limit
                ),
                green_lower_limit=(
                    green_lower_limit
                ),
                green_upper_limit=(
                    green_upper_limit
                ),
            ),
        )
        for index, value in enumerate(
            values,
            start=1,
        )
    )

    decision = _evaluate_qualification(
        points
    )

    return PrecontrolResult(
        observations=values,

        nominal_value=nominal,
        tolerance_value=tolerance,

        lower_spec_limit=(
            lower_spec_limit
        ),

        upper_spec_limit=(
            upper_spec_limit
        ),

        green_lower_limit=(
            green_lower_limit
        ),

        green_upper_limit=(
            green_upper_limit
        ),

        points=points,

        green_count=sum(
            point.zone == GREEN
            for point in points
        ),

        yellow_lower_count=sum(
            point.zone == YELLOW_LOWER
            for point in points
        ),

        yellow_upper_count=sum(
            point.zone == YELLOW_UPPER
            for point in points
        ),

        red_lower_count=sum(
            point.zone == RED_LOWER
            for point in points
        ),

        red_upper_count=sum(
            point.zone == RED_UPPER
            for point in points
        ),

        decision=decision,
    )