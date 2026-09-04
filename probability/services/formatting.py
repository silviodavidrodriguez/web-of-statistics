import math

from dataclasses import dataclass

from probability.distributions import (
    get_distribution_spec,
)

from .calculator import CalculationResult


OPERATION_LABELS = {
    # Continuous
    "density": "Density",
    "left": "Left probability",
    "right": "Right probability",
    "between": "Interval probability",
    "outside": "Outside probability",
    "left_quantile": "Left quantile",
    "right_quantile": "Right quantile",
    "central_interval": "Central interval",

    # Discrete
    "mass": "Point probability",
    "less": "Probability below",
    "less_equal": "Cumulative probability",
    "greater": "Probability above",
    "greater_equal": "Upper-tail probability",
    "quantile": "Cumulative quantile",
}


@dataclass(frozen=True)
class FormattedCalculation:
    distribution_label: str
    parameterization: str
    operation_label: str

    expression: str

    result_label: str
    result_display: str

    complement_label: str | None
    complement_display: str | None

    interpretation: str

    parameter_summary: tuple[str, ...]


def format_number(
    value: int | float,
    *,
    decimals: int = 6,
) -> str:

    if isinstance(value, int):
        return str(value)

    value = float(value)

    if not math.isfinite(value):
        if math.isnan(value):
            return "Undefined"

        if value > 0:
            return "+∞"

        return "−∞"

    if value == 0:
        return "0"

    absolute_value = abs(value)

    if (
        absolute_value < 1e-4
        or absolute_value >= 1e6
    ):
        return f"{value:.6g}"

    formatted = (
        f"{value:.{decimals}f}"
        .rstrip("0")
        .rstrip(".")
    )

    if formatted == "-0":
        return "0"

    return formatted


def format_probability_percentage(
    probability: float,
) -> str:

    return (
        f"{probability * 100:.2f}%"
    )


def _parameter_summary(
    result: CalculationResult,
) -> tuple[str, ...]:

    spec = get_distribution_spec(
        result.distribution_key
    )

    summary = []

    for parameter in spec.parameters:

        if (
            parameter.name
            not in result.parameters
        ):
            continue

        value = result.parameters[
            parameter.name
        ]

        summary.append(
            (
                f"{parameter.label} "
                f"({parameter.symbol}) = "
                f"{format_number(value)}"
            )
        )

    return tuple(summary)


def _format_continuous(
    result: CalculationResult,
) -> tuple[
    str,
    str,
    str,
    str | None,
    str | None,
    str,
]:

    spec = get_distribution_spec(
        result.distribution_key
    )

    symbol = spec.variable_symbol
    operation = result.operation
    inputs = result.inputs

    # ------------------------------------------------------------
    # Density
    # ------------------------------------------------------------

    if operation == "density":

        x = format_number(
            inputs["x"]
        )

        density = format_number(
            result.value
        )

        expression = (
            f"f_{symbol}({x}) = {density}"
        )

        interpretation = (
            f"The probability density of "
            f"{symbol} at {x} is {density}. "
            f"For a continuous distribution, "
            f"this density is not itself a "
            f"probability."
        )

        return (
            expression,
            "Density",
            density,
            None,
            None,
            interpretation,
        )

    # ------------------------------------------------------------
    # Left probability
    # ------------------------------------------------------------

    if operation == "left":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} ≤ {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"Approximately "
            f"{format_probability_percentage(result.probability)} "
            f"of the distribution lies at or "
            f"below {x}."
        )

        return (
            expression,
            "Probability",
            probability,
            "Complementary probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    # ------------------------------------------------------------
    # Right probability
    # ------------------------------------------------------------

    if operation == "right":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} ≥ {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"Approximately "
            f"{format_probability_percentage(result.probability)} "
            f"of the distribution lies at or "
            f"above {x}."
        )

        return (
            expression,
            "Probability",
            probability,
            "Complementary probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    # ------------------------------------------------------------
    # Between
    # ------------------------------------------------------------

    if operation == "between":

        a = format_number(
            inputs["a"]
        )

        b = format_number(
            inputs["b"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({a} ≤ {symbol} ≤ {b}) = "
            f"{probability}"
        )

        interpretation = (
            f"Approximately "
            f"{format_probability_percentage(result.probability)} "
            f"of the distribution lies between "
            f"{a} and {b}."
        )

        return (
            expression,
            "Probability",
            probability,
            "Complementary probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    # ------------------------------------------------------------
    # Outside
    # ------------------------------------------------------------

    if operation == "outside":

        a = format_number(
            inputs["a"]
        )

        b = format_number(
            inputs["b"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} ≤ {a} or "
            f"{symbol} ≥ {b}) = "
            f"{probability}"
        )

        interpretation = (
            f"Approximately "
            f"{format_probability_percentage(result.probability)} "
            f"of the distribution lies outside "
            f"the interval from {a} to {b}."
        )

        return (
            expression,
            "Probability",
            probability,
            "Complementary probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    # ------------------------------------------------------------
    # Left quantile
    # ------------------------------------------------------------

    if operation == "left_quantile":

        p = format_number(
            inputs["p"]
        )

        x = format_number(
            result.value
        )

        expression = (
            f"P({symbol} ≤ x) = {p} "
            f"→ x = {x}"
        )

        interpretation = (
            f"The cumulative probability "
            f"{format_probability_percentage(inputs['p'])} "
            f"is reached at {symbol} = {x}."
        )

        return (
            expression,
            "Critical value",
            x,
            "1 − p",
            format_number(
                result.complement
            ),
            interpretation,
        )

    # ------------------------------------------------------------
    # Right quantile
    # ------------------------------------------------------------

    if operation == "right_quantile":

        p = format_number(
            inputs["p"]
        )

        x = format_number(
            result.value
        )

        expression = (
            f"P({symbol} ≥ x) = {p} "
            f"→ x = {x}"
        )

        interpretation = (
            f"The upper-tail probability "
            f"{format_probability_percentage(inputs['p'])} "
            f"begins at {symbol} = {x}."
        )

        return (
            expression,
            "Critical value",
            x,
            "1 − p",
            format_number(
                result.complement
            ),
            interpretation,
        )

    # ------------------------------------------------------------
    # Central interval
    # ------------------------------------------------------------

    if operation == "central_interval":

        p = inputs["p"]

        lower, upper = (
            result.value
        )

        lower_text = format_number(
            lower
        )

        upper_text = format_number(
            upper
        )

        probability = format_number(
            p
        )

        tail_probability = (
            1.0 - p
        ) / 2.0

        expression = (
            f"P({lower_text} ≤ "
            f"{symbol} ≤ {upper_text}) = "
            f"{probability}"
        )

        interpretation = (
            f"The central "
            f"{format_probability_percentage(p)} "
            f"of the distribution lies between "
            f"{lower_text} and {upper_text}. "
            f"Each tail contains "
            f"{format_probability_percentage(tail_probability)}."
        )

        return (
            expression,
            "Central interval",
            (
                f"[{lower_text}, "
                f"{upper_text}]"
            ),
            "Total tail probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    raise ValueError(
        (
            "Unsupported continuous "
            f"operation: {operation}"
        )
    )


def _format_discrete(
    result: CalculationResult,
) -> tuple[
    str,
    str,
    str,
    str | None,
    str | None,
    str,
]:

    spec = get_distribution_spec(
        result.distribution_key
    )

    symbol = spec.variable_symbol
    operation = result.operation
    inputs = result.inputs

    if operation == "mass":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} = {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing "
            f"exactly {x} is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            "P(X ≠ x)",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "less":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} < {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing a "
            f"value strictly below {x} is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            f"P({symbol} ≥ {x})",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "less_equal":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} ≤ {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing "
            f"{x} or fewer is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            f"P({symbol} > {x})",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "greater":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} > {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing a "
            f"value strictly above {x} is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            f"P({symbol} ≤ {x})",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "greater_equal":

        x = format_number(
            inputs["x"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} ≥ {x}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing "
            f"{x} or more is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            f"P({symbol} < {x})",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "between":

        a = format_number(
            inputs["a"]
        )

        b = format_number(
            inputs["b"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({a} ≤ {symbol} ≤ {b}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing an "
            f"integer value from {a} through "
            f"{b}, including both endpoints, is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            "Complementary probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "outside":

        a = format_number(
            inputs["a"]
        )

        b = format_number(
            inputs["b"]
        )

        probability = format_number(
            result.probability
        )

        expression = (
            f"P({symbol} ≤ {a} or "
            f"{symbol} ≥ {b}) = "
            f"{probability}"
        )

        interpretation = (
            f"The probability of observing "
            f"{a} or fewer, or {b} or more, is "
            f"{format_probability_percentage(result.probability)}."
        )

        return (
            expression,
            "Probability",
            probability,
            "Complementary probability",
            format_number(
                result.complement
            ),
            interpretation,
        )

    if operation == "quantile":

        p = format_number(
            inputs["p"]
        )

        x = format_number(
            result.value
        )

        expression = (
            f"Smallest x with "
            f"P({symbol} ≤ x) ≥ {p}: "
            f"x = {x}"
        )

        interpretation = (
            f"{x} is the smallest integer "
            f"whose cumulative probability is "
            f"at least "
            f"{format_probability_percentage(inputs['p'])}."
        )

        return (
            expression,
            "Quantile",
            x,
            "1 − p",
            format_number(
                result.complement
            ),
            interpretation,
        )

    raise ValueError(
        (
            "Unsupported discrete "
            f"operation: {operation}"
        )
    )


def format_calculation(
    result: CalculationResult,
) -> FormattedCalculation:

    spec = get_distribution_spec(
        result.distribution_key
    )

    if result.category == "continuous":

        (
            expression,
            result_label,
            result_display,
            complement_label,
            complement_display,
            interpretation,
        ) = _format_continuous(
            result
        )

    elif result.category == "discrete":

        (
            expression,
            result_label,
            result_display,
            complement_label,
            complement_display,
            interpretation,
        ) = _format_discrete(
            result
        )

    else:
        raise ValueError(
            (
                "Unsupported distribution "
                f"category: {result.category}"
            )
        )

    return FormattedCalculation(
        distribution_label=(
            spec.label
        ),
        parameterization=(
            spec.parameterization
        ),
        operation_label=(
            OPERATION_LABELS.get(
                result.operation,
                result.operation,
            )
        ),
        expression=expression,
        result_label=result_label,
        result_display=result_display,
        complement_label=(
            complement_label
        ),
        complement_display=(
            complement_display
        ),
        interpretation=interpretation,
        parameter_summary=(
            _parameter_summary(
                result
            )
        ),
    )