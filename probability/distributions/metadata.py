from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping


Number = int | float
ParameterKind = Literal["float", "int"]
DistributionCategory = Literal["continuous", "discrete"]


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    label: str
    symbol: str
    default: Number

    kind: ParameterKind = "float"

    min_value: Number | None = None
    max_value: Number | None = None

    min_inclusive: bool = True
    max_inclusive: bool = True

    help_text: str = ""


CrossValidator = Callable[
    [Mapping[str, Number]],
    list[str],
]


@dataclass(frozen=True)
class DistributionSpec:
    key: str
    label: str
    category: DistributionCategory

    variable_symbol: str

    parameters: tuple[ParameterSpec, ...]

    factory: Callable[..., Any]

    description: str = ""
    parameterization: str = ""

    cross_validator: CrossValidator | None = None

    supports_hazard: bool = True