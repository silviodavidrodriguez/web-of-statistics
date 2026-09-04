import math

from dataclasses import dataclass
from typing import Any, Mapping

from probability.distributions import (
    create_distribution,
    get_distribution_spec,
)

from .validators import (
    require_valid_distribution_parameters,
)


# ================================================================
# Supported operations
# ================================================================

CONTINUOUS_OPERATIONS = (
    "density",
    "left",
    "right",
    "between",
    "outside",
    "left_quantile",
    "right_quantile",
    "central_interval",
)


DISCRETE_OPERATIONS = (
    "mass",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "between",
    "outside",
    "quantile",
)


# ================================================================
# Result object
# ================================================================

@dataclass(frozen=True)
class CalculationResult:
    distribution_key: str
    distribution_label: str
    category: str

    operation: str

    parameters: dict[str, int | float]
    inputs: dict[str, int | float]

    value: (
        int
        | float
        | tuple[int | float, int | float]
    )

    probability: float | None = None
    complement: float | None = None


# ================================================================
# Calculator input errors
# ================================================================

class CalculatorInputError(ValueError):

    def __init__(
        self,
        *,
        field_errors: dict[str, str] | None = None,
        non_field_errors: list[str] | None = None,
    ):
        self.field_errors = (
            field_errors or {}
        )

        self.non_field_errors = (
            non_field_errors or []
        )

        messages = list(
            self.field_errors.values()
        )

        messages.extend(
            self.non_field_errors
        )

        super().__init__(
            "; ".join(messages)
            or "Invalid calculation input."
        )


# ================================================================
# Input parsing
# ================================================================

def _parse_finite_number(
    raw_value: Any,
    *,
    field_name: str,
    label: str,
) -> float:

    if raw_value is None:
        raise CalculatorInputError(
            field_errors={
                field_name:
                    f"{label} is required."
            }
        )

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()

        if raw_value == "":
            raise CalculatorInputError(
                field_errors={
                    field_name:
                        f"{label} is required."
                }
            )

    if isinstance(raw_value, bool):
        raise CalculatorInputError(
            field_errors={
                field_name:
                    f"{label} must be numeric."
            }
        )

    try:
        value = float(raw_value)

    except (TypeError, ValueError):
        raise CalculatorInputError(
            field_errors={
                field_name:
                    f"{label} must be numeric."
            }
        )

    if not math.isfinite(value):
        raise CalculatorInputError(
            field_errors={
                field_name:
                    (
                        f"{label} must be a "
                        f"finite number."
                    )
            }
        )

    return value


def _parse_integer(
    raw_value: Any,
    *,
    field_name: str,
    label: str,
) -> int:

    value = _parse_finite_number(
        raw_value,
        field_name=field_name,
        label=label,
    )

    if not value.is_integer():
        raise CalculatorInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    return int(value)


def _parse_probability(
    raw_value: Any,
    *,
    field_name: str = "p",
    label: str = "Probability",
) -> float:

    value = _parse_finite_number(
        raw_value,
        field_name=field_name,
        label=label,
    )

    # Quantiles at exactly 0 or 1 often
    # produce +/- infinity or support-boundary
    # artifacts. The Functions calculator will
    # require a proper probability strictly
    # inside the unit interval.
    if not 0 < value < 1:
        raise CalculatorInputError(
            field_errors={
                field_name:
                    (
                        f"{label} must be greater "
                        f"than 0 and less than 1."
                    )
            }
        )

    return value


# ================================================================
# Helpers
# ================================================================

def _clean_probability(
    value: float,
) -> float:
    """
    Protect probability outputs from tiny
    floating-point excursions such as
    -1e-16 or 1.0000000000000002.
    """

    value = float(value)

    if value < 0:
        return 0.0

    if value > 1:
        return 1.0

    return value


def _clean_discrete_quantile(
    value: float,
) -> int | float:

    value = float(value)

    if (
        math.isfinite(value)
        and value.is_integer()
    ):
        return int(value)

    return value


def _validate_interval(
    a: int | float,
    b: int | float,
    *,
    strict: bool = False,
):

    if strict:

        if a >= b:
            raise CalculatorInputError(
                non_field_errors=[
                    (
                        "The upper bound must be "
                        "greater than the lower "
                        "bound."
                    )
                ]
            )

    else:

        if a > b:
            raise CalculatorInputError(
                non_field_errors=[
                    (
                        "The upper bound must be "
                        "greater than or equal to "
                        "the lower bound."
                    )
                ]
            )


# ================================================================
# Continuous distributions
# ================================================================

def _calculate_continuous(
    distribution,
    operation: str,
    raw_inputs: Mapping[str, Any],
):

    if operation not in CONTINUOUS_OPERATIONS:
        raise CalculatorInputError(
            non_field_errors=[
                (
                    f"Operation '{operation}' is "
                    f"not available for continuous "
                    f"distributions."
                )
            ]
        )

    # ------------------------------------------------------------
    # Density
    # ------------------------------------------------------------

    if operation == "density":

        x = _parse_finite_number(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        density = float(
            distribution.pdf(x)
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": density,
            "probability": None,
            "complement": None,
        }

    # ------------------------------------------------------------
    # Left probability
    #
    # P(X <= x)
    # ------------------------------------------------------------

    if operation == "left":

        x = _parse_finite_number(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.cdf(x)
        )

        complement = _clean_probability(
            distribution.sf(x)
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": complement,
        }

    # ------------------------------------------------------------
    # Right probability
    #
    # P(X >= x)
    #
    # For continuous distributions the endpoint
    # has probability zero, therefore this is
    # equivalent to P(X > x).
    # ------------------------------------------------------------

    if operation == "right":

        x = _parse_finite_number(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.sf(x)
        )

        complement = _clean_probability(
            distribution.cdf(x)
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": complement,
        }

    # ------------------------------------------------------------
    # Between
    #
    # P(a <= X <= b)
    # ------------------------------------------------------------

    if operation == "between":

        a = _parse_finite_number(
            raw_inputs.get("a"),
            field_name="a",
            label="Lower bound",
        )

        b = _parse_finite_number(
            raw_inputs.get("b"),
            field_name="b",
            label="Upper bound",
        )

        _validate_interval(
            a,
            b,
            strict=False,
        )

        probability = _clean_probability(
            distribution.cdf(b)
            - distribution.cdf(a)
        )

        complement = _clean_probability(
            1.0 - probability
        )

        return {
            "inputs": {
                "a": a,
                "b": b,
            },
            "value": probability,
            "probability": probability,
            "complement": complement,
        }

    # ------------------------------------------------------------
    # Outside
    #
    # P(X <= a OR X >= b)
    # ------------------------------------------------------------

    if operation == "outside":

        a = _parse_finite_number(
            raw_inputs.get("a"),
            field_name="a",
            label="Lower bound",
        )

        b = _parse_finite_number(
            raw_inputs.get("b"),
            field_name="b",
            label="Upper bound",
        )

        _validate_interval(
            a,
            b,
            strict=True,
        )

        probability = _clean_probability(
            distribution.cdf(a)
            + distribution.sf(b)
        )

        complement = _clean_probability(
            1.0 - probability
        )

        return {
            "inputs": {
                "a": a,
                "b": b,
            },
            "value": probability,
            "probability": probability,
            "complement": complement,
        }

    # ------------------------------------------------------------
    # Left quantile
    #
    # Find x such that:
    #
    # P(X <= x) = p
    # ------------------------------------------------------------

    if operation == "left_quantile":

        p = _parse_probability(
            raw_inputs.get("p"),
        )

        x = float(
            distribution.ppf(p)
        )

        return {
            "inputs": {
                "p": p,
            },
            "value": x,
            "probability": p,
            "complement": (
                1.0 - p
            ),
        }

    # ------------------------------------------------------------
    # Right quantile
    #
    # Find x such that:
    #
    # P(X >= x) = p
    #
    # Use ISF rather than PPF(1-p) for improved
    # numerical accuracy in extreme tails.
    # ------------------------------------------------------------

    if operation == "right_quantile":

        p = _parse_probability(
            raw_inputs.get("p"),
        )

        x = float(
            distribution.isf(p)
        )

        return {
            "inputs": {
                "p": p,
            },
            "value": x,
            "probability": p,
            "complement": (
                1.0 - p
            ),
        }

    # ------------------------------------------------------------
    # Central equal-tail interval
    #
    # P(a <= X <= b) = p
    #
    # with:
    #
    # P(X < a) = (1-p)/2
    # P(X > b) = (1-p)/2
    # ------------------------------------------------------------

    if operation == "central_interval":

        p = _parse_probability(
            raw_inputs.get("p"),
        )

        tail_probability = (
            1.0 - p
        ) / 2.0

        lower = float(
            distribution.ppf(
                tail_probability
            )
        )

        upper = float(
            distribution.isf(
                tail_probability
            )
        )

        return {
            "inputs": {
                "p": p,
            },
            "value": (
                lower,
                upper,
            ),
            "probability": p,
            "complement": (
                1.0 - p
            ),
        }

    raise CalculatorInputError(
        non_field_errors=[
            "Unsupported continuous operation."
        ]
    )


# ================================================================
# Discrete distributions
# ================================================================

def _calculate_discrete(
    distribution,
    operation: str,
    raw_inputs: Mapping[str, Any],
):

    if operation not in DISCRETE_OPERATIONS:
        raise CalculatorInputError(
            non_field_errors=[
                (
                    f"Operation '{operation}' is "
                    f"not available for discrete "
                    f"distributions."
                )
            ]
        )

    # ------------------------------------------------------------
    # Probability mass
    #
    # P(X = x)
    # ------------------------------------------------------------

    if operation == "mass":

        x = _parse_integer(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.pmf(x)
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                1.0 - probability
            ),
        }

    # ------------------------------------------------------------
    # Strictly less
    #
    # P(X < x)
    #
    # For integer-valued distributions:
    #
    # P(X < x) = P(X <= x - 1)
    # ------------------------------------------------------------

    if operation == "less":

        x = _parse_integer(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.cdf(
                x - 1
            )
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                1.0 - probability
            ),
        }

    # ------------------------------------------------------------
    # Less than or equal
    #
    # P(X <= x)
    # ------------------------------------------------------------

    if operation == "less_equal":

        x = _parse_integer(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.cdf(x)
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                distribution.sf(x)
            ),
        }

    # ------------------------------------------------------------
    # Strictly greater
    #
    # P(X > x)
    # ------------------------------------------------------------

    if operation == "greater":

        x = _parse_integer(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.sf(x)
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                distribution.cdf(x)
            ),
        }

    # ------------------------------------------------------------
    # Greater than or equal
    #
    # P(X >= x)
    #
    # Critical discrete distinction:
    #
    # P(X >= x) = P(X > x - 1)
    #
    # therefore:
    #
    # sf(x - 1)
    #
    # NOT sf(x).
    # ------------------------------------------------------------

    if operation == "greater_equal":

        x = _parse_integer(
            raw_inputs.get("x"),
            field_name="x",
            label="Value",
        )

        probability = _clean_probability(
            distribution.sf(
                x - 1
            )
        )

        return {
            "inputs": {
                "x": x,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                1.0 - probability
            ),
        }

    # ------------------------------------------------------------
    # Inclusive interval
    #
    # P(a <= X <= b)
    #
    # = F(b) - F(a - 1)
    # ------------------------------------------------------------

    if operation == "between":

        a = _parse_integer(
            raw_inputs.get("a"),
            field_name="a",
            label="Lower bound",
        )

        b = _parse_integer(
            raw_inputs.get("b"),
            field_name="b",
            label="Upper bound",
        )

        _validate_interval(
            a,
            b,
            strict=False,
        )

        probability = _clean_probability(
            distribution.cdf(b)
            - distribution.cdf(
                a - 1
            )
        )

        return {
            "inputs": {
                "a": a,
                "b": b,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                1.0 - probability
            ),
        }

    # ------------------------------------------------------------
    # Outside interval
    #
    # P(X <= a OR X >= b)
    #
    # = F(a) + P(X > b - 1)
    # = F(a) + sf(b - 1)
    # ------------------------------------------------------------

    if operation == "outside":

        a = _parse_integer(
            raw_inputs.get("a"),
            field_name="a",
            label="Lower bound",
        )

        b = _parse_integer(
            raw_inputs.get("b"),
            field_name="b",
            label="Upper bound",
        )

        _validate_interval(
            a,
            b,
            strict=True,
        )

        probability = _clean_probability(
            distribution.cdf(a)
            + distribution.sf(
                b - 1
            )
        )

        return {
            "inputs": {
                "a": a,
                "b": b,
            },
            "value": probability,
            "probability": probability,
            "complement": _clean_probability(
                1.0 - probability
            ),
        }

    # ------------------------------------------------------------
    # Cumulative quantile
    #
    # Smallest integer x for which:
    #
    # P(X <= x) >= p
    # ------------------------------------------------------------

    if operation == "quantile":

        p = _parse_probability(
            raw_inputs.get("p"),
        )

        x = _clean_discrete_quantile(
            distribution.ppf(p)
        )

        return {
            "inputs": {
                "p": p,
            },
            "value": x,
            "probability": p,
            "complement": (
                1.0 - p
            ),
        }

    raise CalculatorInputError(
        non_field_errors=[
            "Unsupported discrete operation."
        ]
    )


# ================================================================
# Public calculator
# ================================================================

def calculate(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
    operation: str,
    raw_inputs: Mapping[str, Any],
) -> CalculationResult:

    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    if spec.category == "continuous":

        raw_result = _calculate_continuous(
            distribution,
            operation,
            raw_inputs,
        )

    elif spec.category == "discrete":

        raw_result = _calculate_discrete(
            distribution,
            operation,
            raw_inputs,
        )

    else:
        raise CalculatorInputError(
            non_field_errors=[
                (
                    "Unsupported probability "
                    "distribution category."
                )
            ]
        )

    return CalculationResult(
        distribution_key=distribution_key,
        distribution_label=spec.label,
        category=spec.category,
        operation=operation,
        parameters=dict(parameters),
        inputs=raw_result["inputs"],
        value=raw_result["value"],
        probability=raw_result[
            "probability"
        ],
        complement=raw_result[
            "complement"
        ],
    )