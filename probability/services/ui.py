from probability.distributions import (
    DISTRIBUTIONS,
)


CONTINUOUS_OPERATION_UI = {
    "density": {
        "label": "Density",
        "expression": "f({symbol}) at x",
        "description": (
            "Evaluate the probability density "
            "function at a selected value."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Value",
                "symbol": "x",
                "kind": "float",
                "default": 0.0,
            },
        ),
    },
    "left": {
        "label": "Left probability",
        "expression": "P({symbol} ≤ x)",
        "description": (
            "Calculate the probability at or "
            "below a selected value."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Value",
                "symbol": "x",
                "kind": "float",
                "default": 0.0,
            },
        ),
    },
    "right": {
        "label": "Right probability",
        "expression": "P({symbol} ≥ x)",
        "description": (
            "Calculate the probability at or "
            "above a selected value."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Value",
                "symbol": "x",
                "kind": "float",
                "default": 0.0,
            },
        ),
    },
    "between": {
        "label": "Between",
        "expression": "P(a ≤ {symbol} ≤ b)",
        "description": (
            "Calculate the probability inside "
            "an interval."
        ),
        "inputs": (
            {
                "name": "a",
                "label": "Lower bound",
                "symbol": "a",
                "kind": "float",
                "default": -1.0,
            },
            {
                "name": "b",
                "label": "Upper bound",
                "symbol": "b",
                "kind": "float",
                "default": 1.0,
            },
        ),
    },
    "outside": {
        "label": "Outside",
        "expression": (
            "P({symbol} ≤ a or {symbol} ≥ b)"
        ),
        "description": (
            "Calculate the total probability "
            "outside two selected boundaries."
        ),
        "inputs": (
            {
                "name": "a",
                "label": "Lower boundary",
                "symbol": "a",
                "kind": "float",
                "default": -1.0,
            },
            {
                "name": "b",
                "label": "Upper boundary",
                "symbol": "b",
                "kind": "float",
                "default": 1.0,
            },
        ),
    },
    "left_quantile": {
        "label": "Left quantile",
        "expression": (
            "Find x from P({symbol} ≤ x) = p"
        ),
        "description": (
            "Find the value associated with a "
            "selected cumulative probability."
        ),
        "inputs": (
            {
                "name": "p",
                "label": "Probability",
                "symbol": "p",
                "kind": "float",
                "default": 0.95,
                "min_value": 0.0,
                "max_value": 1.0,
                "min_inclusive": False,
                "max_inclusive": False,
            },
        ),
    },
    "right_quantile": {
        "label": "Right quantile",
        "expression": (
            "Find x from P({symbol} ≥ x) = p"
        ),
        "description": (
            "Find the value associated with a "
            "selected upper-tail probability."
        ),
        "inputs": (
            {
                "name": "p",
                "label": "Upper-tail probability",
                "symbol": "p",
                "kind": "float",
                "default": 0.05,
                "min_value": 0.0,
                "max_value": 1.0,
                "min_inclusive": False,
                "max_inclusive": False,
            },
        ),
    },
    "central_interval": {
        "label": "Central interval",
        "expression": (
            "Central interval containing p"
        ),
        "description": (
            "Find an equal-tail central interval "
            "containing a selected probability."
        ),
        "inputs": (
            {
                "name": "p",
                "label": "Central probability",
                "symbol": "p",
                "kind": "float",
                "default": 0.95,
                "min_value": 0.0,
                "max_value": 1.0,
                "min_inclusive": False,
                "max_inclusive": False,
            },
        ),
    },
}


DISCRETE_OPERATION_UI = {
    "mass": {
        "label": "Point probability",
        "expression": "P({symbol} = x)",
        "description": (
            "Calculate the probability of exactly "
            "one integer value."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Integer value",
                "symbol": "x",
                "kind": "int",
                "default": 1,
            },
        ),
    },
    "less": {
        "label": "Strictly below",
        "expression": "P({symbol} < x)",
        "description": (
            "Calculate the probability of values "
            "strictly below x."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Integer value",
                "symbol": "x",
                "kind": "int",
                "default": 1,
            },
        ),
    },
    "less_equal": {
        "label": "At or below",
        "expression": "P({symbol} ≤ x)",
        "description": (
            "Calculate the cumulative probability "
            "through x."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Integer value",
                "symbol": "x",
                "kind": "int",
                "default": 1,
            },
        ),
    },
    "greater": {
        "label": "Strictly above",
        "expression": "P({symbol} > x)",
        "description": (
            "Calculate the probability of values "
            "strictly above x."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Integer value",
                "symbol": "x",
                "kind": "int",
                "default": 1,
            },
        ),
    },
    "greater_equal": {
        "label": "At or above",
        "expression": "P({symbol} ≥ x)",
        "description": (
            "Calculate the probability of x or "
            "larger integer values."
        ),
        "inputs": (
            {
                "name": "x",
                "label": "Integer value",
                "symbol": "x",
                "kind": "int",
                "default": 1,
            },
        ),
    },
    "between": {
        "label": "Between",
        "expression": "P(a ≤ {symbol} ≤ b)",
        "description": (
            "Calculate probability across an "
            "inclusive integer interval."
        ),
        "inputs": (
            {
                "name": "a",
                "label": "Lower integer",
                "symbol": "a",
                "kind": "int",
                "default": 1,
            },
            {
                "name": "b",
                "label": "Upper integer",
                "symbol": "b",
                "kind": "int",
                "default": 3,
            },
        ),
    },
    "outside": {
        "label": "Outside",
        "expression": (
            "P({symbol} ≤ a or {symbol} ≥ b)"
        ),
        "description": (
            "Calculate probability in the two "
            "inclusive outer regions."
        ),
        "inputs": (
            {
                "name": "a",
                "label": "Lower boundary",
                "symbol": "a",
                "kind": "int",
                "default": 1,
            },
            {
                "name": "b",
                "label": "Upper boundary",
                "symbol": "b",
                "kind": "int",
                "default": 3,
            },
        ),
    },
    "quantile": {
        "label": "Cumulative quantile",
        "expression": (
            "Smallest x with P({symbol} ≤ x) ≥ p"
        ),
        "description": (
            "Find the smallest integer reaching "
            "a selected cumulative probability."
        ),
        "inputs": (
            {
                "name": "p",
                "label": "Probability",
                "symbol": "p",
                "kind": "float",
                "default": 0.5,
                "min_value": 0.0,
                "max_value": 1.0,
                "min_inclusive": False,
                "max_inclusive": False,
            },
        ),
    },
}


def get_operation_registry(
    category,
):
    if category == "continuous":
        return CONTINUOUS_OPERATION_UI

    if category == "discrete":
        return DISCRETE_OPERATION_UI

    raise ValueError(
        f"Unsupported distribution category: "
        f"{category}"
    )


def get_default_operation(
    category,
):
    if category == "continuous":
        return "left"

    if category == "discrete":
        return "mass"

    raise ValueError(
        f"Unsupported distribution category: "
        f"{category}"
    )


def get_operation_ui(
    category,
    operation,
):
    registry = get_operation_registry(
        category
    )

    try:
        return registry[operation]

    except KeyError as exc:
        raise ValueError(
            (
                f"Unsupported operation "
                f"'{operation}' for {category} "
                f"distributions."
            )
        ) from exc


def get_default_operation_inputs(
    category,
    operation,
):
    operation_ui = get_operation_ui(
        category,
        operation,
    )

    return {
        item["name"]: item["default"]
        for item in operation_ui["inputs"]
    }


def _serialize_parameter(
    parameter,
):
    return {
        "name": parameter.name,
        "label": parameter.label,
        "symbol": parameter.symbol,
        "default": parameter.default,
        "kind": parameter.kind,
        "min_value": parameter.min_value,
        "max_value": parameter.max_value,
        "min_inclusive": (
            parameter.min_inclusive
        ),
        "max_inclusive": (
            parameter.max_inclusive
        ),
        "help_text": parameter.help_text,
    }


def build_probability_ui_config():
    distributions = {}

    for key, spec in (
        DISTRIBUTIONS.items()
    ):
        operation_registry = (
            get_operation_registry(
                spec.category
            )
        )

        distributions[key] = {
            "key": key,
            "label": spec.label,
            "category": spec.category,
            "variable_symbol": (
                spec.variable_symbol
            ),
            "description": (
                spec.description
            ),
            "parameterization": (
                spec.parameterization
            ),
            "supports_hazard": (
                spec.supports_hazard
            ),
            "parameters": [
                _serialize_parameter(
                    parameter
                )
                for parameter
                in spec.parameters
            ],
            "operations": list(
                operation_registry.keys()
            ),
            "default_operation": (
                get_default_operation(
                    spec.category
                )
            ),
        }

    return {
        "distributions": distributions,
        "operations": {
            "continuous":
                CONTINUOUS_OPERATION_UI,
            "discrete":
                DISCRETE_OPERATION_UI,
        },
    }