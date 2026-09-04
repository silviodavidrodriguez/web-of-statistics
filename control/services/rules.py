"""
Nelson rules for detecting special-cause signals
in statistical process control charts.

The rules operate on standardized z-scores, where:

    z = (value - centerline) / sigma

Point indices returned to the user are 1-based.
"""

import math

from dataclasses import dataclass
from typing import Sequence


Number = int | float


class ControlRuleInputError(ValueError):
    """Invalid input supplied to the control-rule engine."""


@dataclass(frozen=True)
class ControlRuleSignal:
    rule: int
    name: str
    point_indices: tuple[int, ...]
    description: str


NELSON_RULE_NAMES = {
    1: "Point beyond 3σ",
    2: "Nine points on the same side",
    3: "Six-point trend",
    4: "Fourteen points alternating",
    5: "Two of three beyond 2σ",
    6: "Four of five beyond 1σ",
    7: "Fifteen points within 1σ",
    8: "Eight points outside 1σ",
}


NELSON_RULE_DESCRIPTIONS = {
    1: (
        "One point is more than 3 standard deviations "
        "from the centerline."
    ),
    2: (
        "Nine consecutive points are on the same side "
        "of the centerline."
    ),
    3: (
        "Six consecutive points are continuously "
        "increasing or decreasing."
    ),
    4: (
        "Fourteen consecutive points alternate "
        "up and down."
    ),
    5: (
        "At least two of three consecutive points are "
        "more than 2 standard deviations from the "
        "centerline on the same side."
    ),
    6: (
        "At least four of five consecutive points are "
        "more than 1 standard deviation from the "
        "centerline on the same side."
    ),
    7: (
        "Fifteen consecutive points are within "
        "1 standard deviation of the centerline."
    ),
    8: (
        "Eight consecutive points lie outside the "
        "1-sigma zone and occur on both sides of "
        "the centerline."
    ),
}


def _validate_z_scores(
    values: Sequence[Number],
) -> tuple[float, ...]:

    if isinstance(values, (str, bytes)):
        raise ControlRuleInputError(
            "Z-scores must be a sequence."
        )

    parsed = []

    for index, value in enumerate(values, start=1):
        try:
            numeric = float(value)

        except (TypeError, ValueError):
            raise ControlRuleInputError(
                f"Z-score {index} must be numeric."
            )

        if not math.isfinite(numeric):
            raise ControlRuleInputError(
                f"Z-score {index} must be finite."
            )

        parsed.append(numeric)

    if not parsed:
        raise ControlRuleInputError(
            "Z-scores cannot be empty."
        )

    return tuple(parsed)


def _make_signal(
    rule: int,
    indices,
) -> ControlRuleSignal:

    return ControlRuleSignal(
        rule=rule,
        name=NELSON_RULE_NAMES[rule],
        point_indices=tuple(
            index + 1
            for index in indices
        ),
        description=(
            NELSON_RULE_DESCRIPTIONS[rule]
        ),
    )


def _append_unique(
    signals,
    seen,
    signal,
):
    key = (
        signal.rule,
        signal.point_indices,
    )

    if key not in seen:
        seen.add(key)
        signals.append(signal)


def detect_nelson_rules(
    z_scores: Sequence[Number],
) -> tuple[ControlRuleSignal, ...]:

    z = _validate_z_scores(
        z_scores
    )

    signals = []
    seen = set()

    # ============================================================
    # Rule 1
    # One point beyond 3 sigma
    # ============================================================

    for index, value in enumerate(z):
        if abs(value) > 3.0:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    1,
                    [index],
                ),
            )

    # ============================================================
    # Rule 2
    # Nine consecutive points on the same side
    # ============================================================

    for start in range(
        len(z) - 8
    ):
        window = z[
            start:start + 9
        ]

        all_above = all(
            value > 0
            for value in window
        )

        all_below = all(
            value < 0
            for value in window
        )

        if all_above or all_below:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    2,
                    range(
                        start,
                        start + 9,
                    ),
                ),
            )

    # ============================================================
    # Rule 3
    # Six consecutive increasing or decreasing points
    # ============================================================

    for start in range(
        len(z) - 5
    ):
        window = z[
            start:start + 6
        ]

        increasing = all(
            window[index]
            < window[index + 1]
            for index in range(5)
        )

        decreasing = all(
            window[index]
            > window[index + 1]
            for index in range(5)
        )

        if increasing or decreasing:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    3,
                    range(
                        start,
                        start + 6,
                    ),
                ),
            )

    # ============================================================
    # Rule 4
    # Fourteen consecutive points alternating up/down
    # ============================================================

    for start in range(
        len(z) - 13
    ):
        window = z[
            start:start + 14
        ]

        differences = [
            window[index + 1]
            - window[index]
            for index in range(13)
        ]

        has_equal_points = any(
            difference == 0
            for difference in differences
        )

        alternating = all(
            differences[index]
            * differences[index + 1]
            < 0
            for index in range(12)
        )

        if (
            not has_equal_points
            and alternating
        ):
            _append_unique(
                signals,
                seen,
                _make_signal(
                    4,
                    range(
                        start,
                        start + 14,
                    ),
                ),
            )

    # ============================================================
    # Rule 5
    # Two of three beyond 2 sigma on same side
    # ============================================================

    for start in range(
        len(z) - 2
    ):
        window = z[
            start:start + 3
        ]

        above = [
            start + offset
            for offset, value
            in enumerate(window)
            if value > 2.0
        ]

        below = [
            start + offset
            for offset, value
            in enumerate(window)
            if value < -2.0
        ]

        if len(above) >= 2:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    5,
                    above,
                ),
            )

        if len(below) >= 2:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    5,
                    below,
                ),
            )

    # ============================================================
    # Rule 6
    # Four of five beyond 1 sigma on same side
    # ============================================================

    for start in range(
        len(z) - 4
    ):
        window = z[
            start:start + 5
        ]

        above = [
            start + offset
            for offset, value
            in enumerate(window)
            if value > 1.0
        ]

        below = [
            start + offset
            for offset, value
            in enumerate(window)
            if value < -1.0
        ]

        if len(above) >= 4:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    6,
                    above,
                ),
            )

        if len(below) >= 4:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    6,
                    below,
                ),
            )

    # ============================================================
    # Rule 7
    # Fifteen consecutive points within 1 sigma
    # ============================================================

    for start in range(
        len(z) - 14
    ):
        window = z[
            start:start + 15
        ]

        if all(
            abs(value) < 1.0
            for value in window
        ):
            _append_unique(
                signals,
                seen,
                _make_signal(
                    7,
                    range(
                        start,
                        start + 15,
                    ),
                ),
            )

    # ============================================================
    # Rule 8
    # Eight consecutive points outside 1 sigma
    # and on both sides of the centerline
    # ============================================================

    for start in range(
        len(z) - 7
    ):
        window = z[
            start:start + 8
        ]

        all_outside = all(
            abs(value) > 1.0
            for value in window
        )

        both_sides = (
            any(
                value > 0
                for value in window
            )
            and any(
                value < 0
                for value in window
            )
        )

        if all_outside and both_sides:
            _append_unique(
                signals,
                seen,
                _make_signal(
                    8,
                    range(
                        start,
                        start + 8,
                    ),
                ),
            )

    return tuple(
        sorted(
            signals,
            key=lambda signal: (
                signal.rule,
                signal.point_indices,
            ),
        )
    )


def detect_nelson_rules_for_values(
    values: Sequence[Number],
    *,
    centerline: Number,
    sigma: Number,
) -> tuple[ControlRuleSignal, ...]:

    try:
        center = float(
            centerline
        )

        standard_deviation = float(
            sigma
        )

    except (TypeError, ValueError):
        raise ControlRuleInputError(
            "Centerline and sigma must be numeric."
        )

    if (
        not math.isfinite(center)
        or not math.isfinite(
            standard_deviation
        )
    ):
        raise ControlRuleInputError(
            "Centerline and sigma must be finite."
        )

    if standard_deviation <= 0:
        raise ControlRuleInputError(
            "Sigma must be greater than 0."
        )

    standardized = []

    for index, value in enumerate(
        values,
        start=1,
    ):
        try:
            numeric = float(value)

        except (TypeError, ValueError):
            raise ControlRuleInputError(
                (
                    f"Observation {index} "
                    "must be numeric."
                )
            )

        if not math.isfinite(numeric):
            raise ControlRuleInputError(
                (
                    f"Observation {index} "
                    "must be finite."
                )
            )

        standardized.append(
            (
                numeric - center
            )
            / standard_deviation
        )

    return detect_nelson_rules(
        standardized
    )