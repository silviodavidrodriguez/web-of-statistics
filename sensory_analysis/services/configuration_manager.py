from __future__ import annotations
from typing import Any
import pandas as pd

NUMERIC_SCALE_ROLES = {
    "liking",
    "jar",
    "purchase_intention",
}

def get_columns_by_role(
    mapping: dict[str, dict[str, str]],
    role: str,
) -> list[dict[str, str]]:
    """
    Returns every mapped column assigned to a specific semantic role.
    """
    variables = []

    for dataset_name, dataset_mapping in mapping.items():
        for column_name, assigned_role in dataset_mapping.items():
            if assigned_role == role:
                variables.append(
                    {
                        "dataset_name": dataset_name,
                        "column_name": column_name,
                    }
                )

    return variables

def get_series(
    datasets: dict[str, pd.DataFrame],
    dataset_name: str,
    column_name: str,
) -> pd.Series | None:
    dataframe = datasets.get(dataset_name)

    if dataframe is None:
        return None

    if column_name not in dataframe.columns:
        return None

    return dataframe[column_name]

def clean_unique_values(series: pd.Series | None) -> list[Any]:
    if series is None:
        return []

    values = []

    for value in series.dropna().unique().tolist():
        if isinstance(value, str):
            value = value.strip()

            if not value:
                continue

        values.append(value)

    return values

def infer_numeric_range(
    series: pd.Series | None,
) -> dict[str, float | int | None]:
    if series is None:
        return {
            "minimum": None,
            "maximum": None,
        }

    numeric = pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False),
        errors="coerce",
    ).dropna()

    if numeric.empty:
        return {
            "minimum": None,
            "maximum": None,
        }

    minimum = float(numeric.min())
    maximum = float(numeric.max())

    if minimum.is_integer():
        minimum = int(minimum)

    if maximum.is_integer():
        maximum = int(maximum)

    return {
        "minimum": minimum,
        "maximum": maximum,
    }

def suggest_scale(
    series: pd.Series | None,
    role: str,
) -> dict[str, Any]:
    """
    Suggests scale limits according to observed values and semantic role.
    """
    observed = infer_numeric_range(series)

    observed_minimum = observed["minimum"]
    observed_maximum = observed["maximum"]

    if role == "liking":
        if (
            observed_minimum is not None
            and observed_minimum >= 1
            and observed_maximum <= 9
        ):
            return {
                "minimum": 1,
                "maximum": 9,
            }

        return {
            "minimum": observed_minimum if observed_minimum is not None else 1,
            "maximum": observed_maximum if observed_maximum is not None else 9,
        }

    if role == "purchase_intention":
        if (
            observed_minimum is not None
            and observed_minimum >= 1
            and observed_maximum <= 5
        ):
            return {
                "minimum": 1,
                "maximum": 5,
            }

        if (
            observed_minimum is not None
            and observed_minimum >= 1
            and observed_maximum <= 7
        ):
            return {
                "minimum": 1,
                "maximum": 7,
            }

        return {
            "minimum": observed_minimum if observed_minimum is not None else 1,
            "maximum": observed_maximum if observed_maximum is not None else 5,
        }

    if role == "jar":
        values = clean_unique_values(series)

        numeric_values = []

        for value in values:
            try:
                numeric_values.append(
                    float(str(value).replace(",", "."))
                )
            except (TypeError, ValueError):
                continue

        value_set = set(numeric_values)

        if value_set and value_set.issubset({1.0, 2.0, 3.0}):
            return {
                "minimum": 1,
                "maximum": 3,
                "ideal_value": 2,
            }

        if value_set and value_set.issubset({-1.0, 0.0, 1.0}):
            return {
                "minimum": -1,
                "maximum": 1,
                "ideal_value": 0,
            }

        if value_set and value_set.issubset(
            {1.0, 2.0, 3.0, 4.0, 5.0}
        ):
            return {
                "minimum": 1,
                "maximum": 5,
                "ideal_value": 3,
            }

        minimum = observed_minimum if observed_minimum is not None else 1
        maximum = observed_maximum if observed_maximum is not None else 5

        ideal_value = None

        if isinstance(minimum, (int, float)) and isinstance(
            maximum,
            (int, float),
        ):
            midpoint = (minimum + maximum) / 2

            ideal_value = (
                int(midpoint)
                if float(midpoint).is_integer()
                else midpoint
            )

        return {
            "minimum": minimum,
            "maximum": maximum,
            "ideal_value": ideal_value,
        }

    return {
        "minimum": observed_minimum,
        "maximum": observed_maximum,
    }

def detect_cata_separator(
    series: pd.Series | None,
) -> str:
    if series is None:
        return ";"

    candidates = {
        ";": 0,
        ",": 0,
        "|": 0,
    }

    for value in series.dropna().astype(str).head(100):
        for separator in candidates:
            candidates[separator] += value.count(separator)

    detected = max(
        candidates,
        key=candidates.get,
    )

    if candidates[detected] == 0:
        return ";"

    return detected

def build_default_configuration(
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """
    Creates the initial study configuration from mapped variables.
    """
    configuration: dict[str, Any] = {
        "liking": {},
        "jar": {},
        "cata": {},
        "purchase_intention": {},
        "demographics": {},
        "general_questions": {},
        "segmentation": {},
    }

    for variable in get_columns_by_role(mapping, "liking"):
        dataset_name = variable["dataset_name"]
        column_name = variable["column_name"]

        series = get_series(
            datasets,
            dataset_name,
            column_name,
        )

        suggested = suggest_scale(series, "liking")

        configuration["liking"][column_name] = {
            "dataset": dataset_name,
            "minimum": suggested["minimum"],
            "maximum": suggested["maximum"],
            "treat_as": "numeric",
        }

    for variable in get_columns_by_role(mapping, "jar"):
        dataset_name = variable["dataset_name"]
        column_name = variable["column_name"]

        series = get_series(
            datasets,
            dataset_name,
            column_name,
        )

        suggested = suggest_scale(series, "jar")

        configuration["jar"][column_name] = {
            "dataset": dataset_name,
            "minimum": suggested["minimum"],
            "maximum": suggested["maximum"],
            "ideal_value": suggested["ideal_value"],
        }

    for variable in get_columns_by_role(mapping, "cata"):
        dataset_name = variable["dataset_name"]
        column_name = variable["column_name"]

        series = get_series(
            datasets,
            dataset_name,
            column_name,
        )

        configuration["cata"][column_name] = {
            "dataset": dataset_name,
            "separator": detect_cata_separator(series),
            "trim_spaces": True,
            "case_sensitive": False,
            "remove_empty_attributes": True,
        }

    for variable in get_columns_by_role(
        mapping,
        "purchase_intention",
    ):
        dataset_name = variable["dataset_name"]
        column_name = variable["column_name"]

        series = get_series(
            datasets,
            dataset_name,
            column_name,
        )

        suggested = suggest_scale(
            series,
            "purchase_intention",
        )

        configuration["purchase_intention"][column_name] = {
            "dataset": dataset_name,
            "minimum": suggested["minimum"],
            "maximum": suggested["maximum"],
            "treat_as": "ordinal",
        }

    for role, configuration_key in (
        ("demographic", "demographics"),
        ("segmentation", "segmentation"),
        ("general_question", "general_questions"),
    ):
        for variable in get_columns_by_role(mapping, role):
            dataset_name = variable["dataset_name"]
            column_name = variable["column_name"]

            series = get_series(
                datasets,
                dataset_name,
                column_name,
            )

            variable_type = infer_variable_treatment(
                series=series,
                column_name=column_name,
                role=role,
            )

            configuration[configuration_key][column_name] = {
                "dataset": dataset_name,
                "treatment": variable_type,
                "group_numeric": False,
                "number_of_groups": 4,
                "reference_level": "",
            }

    return configuration

def infer_variable_treatment(
    series: pd.Series | None,
    column_name: str = "",
    role: str = "",
) -> str:
    if series is None:
        return "categorical"

    non_blank = series.dropna()

    if non_blank.empty:
        return "categorical"

    normalized_name = (
        column_name.strip().lower().replace(" ", "_")
    )

    numeric = pd.to_numeric(
        non_blank.astype(str).str.replace(
            ",",
            ".",
            regex=False,
        ),
        errors="coerce",
    )

    numeric_ratio = numeric.notna().mean()
    unique_count = non_blank.nunique(dropna=True)

    continuous_names = {
        "age",
        "edad",
        "income",
        "ingreso",
        "weight",
        "peso",
        "height",
        "altura",
    }

    if normalized_name in continuous_names:
        return "continuous"

    if numeric_ratio == 1:
        if role == "general_question" and unique_count <= 7:
            return "ordinal"

        if role == "demographic":
            return "continuous"

        if unique_count <= 7:
            return "ordinal"

        return "continuous"

    return "categorical"

def merge_configuration(
    default_configuration: dict[str, Any],
    saved_configuration: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Preserves already saved configuration while adding newly mapped variables.
    """
    if not saved_configuration:
        return default_configuration

    merged = {}

    for section_name, default_section in default_configuration.items():
        saved_section = saved_configuration.get(
            section_name,
            {},
        )

        merged[section_name] = {}

        for variable_name, default_values in default_section.items():
            saved_values = saved_section.get(
                variable_name,
                {},
            )

            merged[section_name][variable_name] = {
                **default_values,
                **saved_values,
            }

    return merged

def validate_configuration(
    configuration: dict[str, Any],
) -> dict[str, list[str]]:
    errors = []
    warnings = []

    for column_name, values in configuration.get(
        "liking",
        {},
    ).items():
        minimum = values.get("minimum")
        maximum = values.get("maximum")

        if minimum is None or maximum is None:
            errors.append(
                f"{column_name}: define the minimum and maximum "
                "values of the liking scale."
            )
        elif minimum >= maximum:
            errors.append(
                f"{column_name}: the scale minimum must be lower "
                "than the maximum."
            )

    for column_name, values in configuration.get(
        "jar",
        {},
    ).items():
        minimum = values.get("minimum")
        maximum = values.get("maximum")
        ideal_value = values.get("ideal_value")

        if (
            minimum is None
            or maximum is None
            or ideal_value is None
        ):
            errors.append(
                f"{column_name}: complete the JAR scale configuration."
            )
            continue

        if minimum >= maximum:
            errors.append(
                f"{column_name}: the JAR scale minimum must be "
                "lower than the maximum."
            )

        if ideal_value < minimum or ideal_value > maximum:
            errors.append(
                f"{column_name}: the ideal value must be within "
                "the JAR scale range."
            )

    for column_name, values in configuration.get(
        "cata",
        {},
    ).items():
        separator = values.get("separator", "")

        if not separator:
            errors.append(
                f"{column_name}: define the CATA separator."
            )

        if len(separator) > 1:
            errors.append(
                f"{column_name}: the CATA separator must contain "
                "only one character."
            )

    for column_name, values in configuration.get(
        "purchase_intention",
        {},
    ).items():
        minimum = values.get("minimum")
        maximum = values.get("maximum")

        if minimum is None or maximum is None:
            errors.append(
                f"{column_name}: define the purchase intention scale."
            )
        elif minimum >= maximum:
            errors.append(
                f"{column_name}: the scale minimum must be lower "
                "than the maximum."
            )

    return {
        "errors": errors,
        "warnings": warnings,
    }