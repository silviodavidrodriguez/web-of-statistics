import math

from dataclasses import dataclass, field
from typing import Any, Mapping

from probability.distributions.metadata import (
    Number,
    ParameterSpec,
)
from probability.distributions.registry import (
    get_distribution_spec,
)


@dataclass
class ValidationResult:
    values: dict[str, Number] = field(
        default_factory=dict
    )

    field_errors: dict[str, str] = field(
        default_factory=dict
    )

    non_field_errors: list[str] = field(
        default_factory=list
    )

    @property
    def is_valid(self) -> bool:
        return (
            not self.field_errors
            and not self.non_field_errors
        )


class DistributionValidationError(ValueError):

    def __init__(
        self,
        result: ValidationResult,
    ):
        self.result = result

        messages = list(
            result.field_errors.values()
        )
        messages.extend(
            result.non_field_errors
        )

        super().__init__(
            "; ".join(messages)
            or "Invalid distribution parameters."
        )


def _parse_numeric_value(
    parameter: ParameterSpec,
    raw_value: Any,
) -> tuple[Number | None, str | None]:

    if raw_value is None:
        return (
            None,
            f"{parameter.label} is required.",
        )

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()

        if raw_value == "":
            return (
                None,
                f"{parameter.label} is required.",
            )

    if isinstance(raw_value, bool):
        return (
            None,
            f"{parameter.label} must be numeric.",
        )

    try:
        numeric_value = float(raw_value)

    except (TypeError, ValueError):
        return (
            None,
            f"{parameter.label} must be numeric.",
        )

    if not math.isfinite(numeric_value):
        return (
            None,
            (
                f"{parameter.label} must be a "
                f"finite number."
            ),
        )

    if parameter.kind == "int":

        if not numeric_value.is_integer():
            return (
                None,
                (
                    f"{parameter.label} must be "
                    f"an integer."
                ),
            )

        value: Number = int(
            numeric_value
        )

    else:
        value = float(
            numeric_value
        )

    return value, None


def _validate_bounds(
    parameter: ParameterSpec,
    value: Number,
) -> str | None:

    if parameter.min_value is not None:

        if parameter.min_inclusive:
            invalid_min = (
                value < parameter.min_value
            )
        else:
            invalid_min = (
                value <= parameter.min_value
            )

        if invalid_min:

            operator = (
                "greater than or equal to"
                if parameter.min_inclusive
                else "greater than"
            )

            return (
                f"{parameter.label} must be "
                f"{operator} "
                f"{parameter.min_value}."
            )

    if parameter.max_value is not None:

        if parameter.max_inclusive:
            invalid_max = (
                value > parameter.max_value
            )
        else:
            invalid_max = (
                value >= parameter.max_value
            )

        if invalid_max:

            operator = (
                "less than or equal to"
                if parameter.max_inclusive
                else "less than"
            )

            return (
                f"{parameter.label} must be "
                f"{operator} "
                f"{parameter.max_value}."
            )

    return None


def validate_distribution_parameters(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
) -> ValidationResult:

    result = ValidationResult()

    try:
        spec = get_distribution_spec(
            distribution_key
        )

    except ValueError as exc:
        result.non_field_errors.append(
            str(exc)
        )
        return result

    for parameter in spec.parameters:

        raw_value = raw_parameters.get(
            parameter.name
        )

        value, parsing_error = (
            _parse_numeric_value(
                parameter,
                raw_value,
            )
        )

        if parsing_error:
            result.field_errors[
                parameter.name
            ] = parsing_error
            continue

        bounds_error = _validate_bounds(
            parameter,
            value,
        )

        if bounds_error:
            result.field_errors[
                parameter.name
            ] = bounds_error
            continue

        result.values[
            parameter.name
        ] = value

    # Cross-parameter validation is only
    # meaningful if every individual field
    # has already passed validation.
    if (
        not result.field_errors
        and spec.cross_validator is not None
    ):

        cross_errors = spec.cross_validator(
            result.values
        )

        result.non_field_errors.extend(
            cross_errors
        )

    return result


def require_valid_distribution_parameters(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
) -> dict[str, Number]:

    result = validate_distribution_parameters(
        distribution_key,
        raw_parameters,
    )

    if not result.is_valid:
        raise DistributionValidationError(
            result
        )

    return result.values


def get_default_parameters(
    distribution_key: str,
) -> dict[str, Number]:

    spec = get_distribution_spec(
        distribution_key
    )

    return {
        parameter.name:
            parameter.default
        for parameter in spec.parameters
    }