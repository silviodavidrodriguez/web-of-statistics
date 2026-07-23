from __future__ import annotations
from typing import Any
import pandas as pd

DEFAULT_JAR_MINIMUM = 1.0
DEFAULT_JAR_IDEAL = 3.0
DEFAULT_JAR_MAXIMUM = 5.0
DEFAULT_PENALTY_PERCENTAGE_THRESHOLD = 20.0
DEFAULT_PENALTY_VALUE_THRESHOLD = 1.0
DEFAULT_MINIMUM_GROUP_SIZE = 2

def get_mapped_columns(
    mapping: dict[str, dict[str, str]],
    *,
    dataset_name: str,
    role: str,
) -> list[str]:
    dataset_mapping = mapping.get(dataset_name, {})

    return [
        column_name
        for column_name, assigned_role in dataset_mapping.items()
        if assigned_role == role
    ]

def normalize_identifier_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

def normalize_numeric_series(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series.astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace("", pd.NA)
    )

    return pd.to_numeric(
        normalized,
        errors="coerce",
    )

def prepare_jar_data(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    selected_jar_variables: list[str] | None = None,
) -> dict[str, Any]:
    evaluations = datasets.get("evaluations")

    if evaluations is None or evaluations.empty:
        return {
            "is_valid": False,
            "error": "The evaluations dataset is empty.",
        }

    consumer_columns = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="consumer_id",
    )

    sample_columns = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="sample_id",
    )

    jar_columns = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="jar",
    )

    liking_columns = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="liking",
    )

    if len(consumer_columns) != 1:
        return {
            "is_valid": False,
            "error": (
                "Exactly one Consumer ID variable is required "
                "for JAR analysis."
            ),
        }

    if len(sample_columns) != 1:
        return {
            "is_valid": False,
            "error": (
                "Exactly one Sample ID variable is required "
                "for JAR analysis."
            ),
        }

    if not jar_columns:
        return {
            "is_valid": False,
            "error": "No JAR variables were mapped.",
        }
    
    if len(liking_columns) != 1:
        return {
            "is_valid": False,
            "error": (
                "Exactly one Overall Liking variable is required "
                "for JAR penalty analysis."
            ),
        }

    if selected_jar_variables is None:
        jar_variables = jar_columns
    else:
        jar_variables = [
            column_name
            for column_name in selected_jar_variables
            if column_name in jar_columns
        ]

    if not jar_variables:
        return {
            "is_valid": False,
            "error": "No valid JAR variables were selected.",
        }

    consumer_column = consumer_columns[0]
    sample_column = sample_columns[0]

    liking_column = liking_columns[0]

    required_columns = [
        consumer_column,
        sample_column,
        liking_column,
        *jar_variables,
    ]

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name not in evaluations.columns
    ]

    if missing_columns:
        return {
            "is_valid": False,
            "error": (
                "The following required columns are missing: "
                + ", ".join(missing_columns)
            ),
        }

    data = evaluations[required_columns].copy()

    data[consumer_column] = normalize_identifier_series(
        data[consumer_column]
    )

    data[sample_column] = normalize_identifier_series(
        data[sample_column]
    )

    for jar_variable in jar_variables:
        data[jar_variable] = normalize_numeric_series(
            data[jar_variable]
        )

    initial_row_count = len(data)

    data = data.dropna(
        subset=[
            consumer_column,
            sample_column,
        ]
    )

    excluded_identifier_rows = (
        initial_row_count - len(data)
    )

    duplicate_count = int(
        data.duplicated(
            subset=[
                consumer_column,
                sample_column,
            ],
            keep=False,
        ).sum()
    )

    data[liking_column] = normalize_numeric_series(
        data[liking_column]
    )

    return {
        "is_valid": True,
        "data": data,
        "consumer_column": consumer_column,
        "sample_column": sample_column,
        "liking_column": liking_column,
        "jar_variables": jar_variables,
        "available_jar_variables": jar_columns,
        "excluded_identifier_rows": excluded_identifier_rows,
        "duplicate_count": duplicate_count,
    }

def get_jar_variable_configuration(
    *,
    configuration: dict,
    jar_variable: str,
) -> dict[str, float]:
    configuration = configuration or {}

    jar_configuration = configuration.get(
        "jar",
        {},
    )

    variable_configuration = jar_configuration.get(
        jar_variable,
        {},
    )

    minimum = variable_configuration.get(
        "minimum",
        variable_configuration.get(
            "scale_minimum",
            DEFAULT_JAR_MINIMUM,
        ),
    )

    ideal = variable_configuration.get(
        "ideal",
        variable_configuration.get(
            "ideal_value",
            DEFAULT_JAR_IDEAL,
        ),
    )

    maximum = variable_configuration.get(
        "maximum",
        variable_configuration.get(
            "scale_maximum",
            DEFAULT_JAR_MAXIMUM,
        ),
    )

    try:
        minimum = float(minimum)
    except (TypeError, ValueError):
        minimum = DEFAULT_JAR_MINIMUM

    try:
        ideal = float(ideal)
    except (TypeError, ValueError):
        ideal = DEFAULT_JAR_IDEAL

    try:
        maximum = float(maximum)
    except (TypeError, ValueError):
        maximum = DEFAULT_JAR_MAXIMUM

    if not minimum < ideal < maximum:
        minimum = DEFAULT_JAR_MINIMUM
        ideal = DEFAULT_JAR_IDEAL
        maximum = DEFAULT_JAR_MAXIMUM

    return {
        "minimum": minimum,
        "ideal": ideal,
        "maximum": maximum,
    }

def classify_jar_response(
    value: float,
    *,
    ideal_value: float,
) -> str:
    if value < ideal_value:
        return "too_little"

    if value > ideal_value:
        return "too_much"

    return "jar"

def calculate_jar_distribution(
    *,
    data: pd.DataFrame,
    sample_column: str,
    jar_variables: list[str],
    configuration: dict,
) -> list[dict[str, Any]]:
    results = []

    for jar_variable in jar_variables:
        variable_configuration = (
            get_jar_variable_configuration(
                configuration=configuration,
                jar_variable=jar_variable,
            )
        )

        minimum = variable_configuration["minimum"]
        maximum = variable_configuration["maximum"]
        ideal_value = variable_configuration["ideal"]

        variable_data = data[
            [
                sample_column,
                jar_variable,
            ]
        ].dropna()

        variable_data = variable_data[
            variable_data[jar_variable].between(
                minimum,
                maximum,
                inclusive="both",
            )
        ]

        grouped = variable_data.groupby(
            sample_column,
            sort=False,
            observed=True,
        )

        for sample, group in grouped:
            values = group[jar_variable]

            categories = values.apply(
                lambda value: classify_jar_response(
                    float(value),
                    ideal_value=ideal_value,
                )
            )

            total_count = int(len(categories))

            too_little_count = int(
                (categories == "too_little").sum()
            )

            jar_count = int(
                (categories == "jar").sum()
            )

            too_much_count = int(
                (categories == "too_much").sum()
            )

            if total_count:
                too_little_percentage = (
                    too_little_count / total_count * 100
                )

                jar_percentage = (
                    jar_count / total_count * 100
                )

                too_much_percentage = (
                    too_much_count / total_count * 100
                )
            else:
                too_little_percentage = 0.0
                jar_percentage = 0.0
                too_much_percentage = 0.0

            results.append(
                {
                    "variable": jar_variable,
                    "sample": str(sample),
                    "count": total_count,
                    "too_little_count": too_little_count,
                    "jar_count": jar_count,
                    "too_much_count": too_much_count,
                    "too_little_percentage": (
                        too_little_percentage
                    ),
                    "jar_percentage": jar_percentage,
                    "too_much_percentage": (
                        too_much_percentage
                    ),
                    "minimum": (
                        variable_configuration["minimum"]
                    ),
                    "ideal": ideal_value,
                    "maximum": (
                        variable_configuration["maximum"]
                    ),
                }
            )

    return sorted(
        results,
        key=lambda item: (
            item["variable"],
            -item["jar_percentage"],
            item["sample"],
        ),
    )

def build_jar_variable_summary(
    distribution: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variables = {}

    for result in distribution:
        variable_name = result["variable"]

        if variable_name not in variables:
            variables[variable_name] = {
                "variable": variable_name,
                "products": [],
            }

        variables[variable_name]["products"].append(
            result
        )

    summaries = []

    for variable_name, variable_data in variables.items():
        products = sorted(
            variable_data["products"],
            key=lambda item: (
                -item["jar_percentage"],
                item["sample"],
            ),
        )

        best_product = products[0] if products else None
        lowest_product = products[-1] if products else None

        summaries.append(
            {
                "variable": variable_name,
                "product_count": len(products),
                "products": products,
                "best_product": best_product,
                "lowest_product": lowest_product,
            }
        )

    return summaries

def build_jar_chart_data(
    distribution: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    variable_names = []

    for result in distribution:
        variable_name = result["variable"]

        if variable_name not in variable_names:
            variable_names.append(variable_name)

    charts = []

    for variable_name in variable_names:
        variable_results = [
            result
            for result in distribution
            if result["variable"] == variable_name
        ]

        variable_results = sorted(
            variable_results,
            key=lambda item: item["sample"],
        )

        charts.append(
            {
                "variable": variable_name,
                "labels": [
                    result["sample"]
                    for result in variable_results
                ],
                "too_little": [
                    result["too_little_percentage"]
                    for result in variable_results
                ],
                "jar": [
                    result["jar_percentage"]
                    for result in variable_results
                ],
                "too_much": [
                    result["too_much_percentage"]
                    for result in variable_results
                ],
            }
        )

    return charts

def calculate_penalty_analysis(
    *,
    data: pd.DataFrame,
    sample_column: str,
    liking_column: str,
    jar_variables: list[str],
    configuration: dict,
) -> list[dict[str, Any]]:
    results = []

    penalty_configuration = (
        configuration
        .get("jar_penalty", {})
    )

    percentage_threshold = penalty_configuration.get(
        "percentage_threshold",
        DEFAULT_PENALTY_PERCENTAGE_THRESHOLD,
    )

    penalty_threshold = penalty_configuration.get(
        "penalty_threshold",
        DEFAULT_PENALTY_VALUE_THRESHOLD,
    )

    minimum_group_size = penalty_configuration.get(
        "minimum_group_size",
        DEFAULT_MINIMUM_GROUP_SIZE,
    )

    try:
        percentage_threshold = float(
            percentage_threshold
        )
    except (TypeError, ValueError):
        percentage_threshold = (
            DEFAULT_PENALTY_PERCENTAGE_THRESHOLD
        )

    try:
        penalty_threshold = float(
            penalty_threshold
        )
    except (TypeError, ValueError):
        penalty_threshold = (
            DEFAULT_PENALTY_VALUE_THRESHOLD
        )

    try:
        minimum_group_size = int(
            minimum_group_size
        )
    except (TypeError, ValueError):
        minimum_group_size = (
            DEFAULT_MINIMUM_GROUP_SIZE
        )

    for jar_variable in jar_variables:
        variable_configuration = (
            get_jar_variable_configuration(
                configuration=configuration,
                jar_variable=jar_variable,
            )
        )

        minimum = variable_configuration["minimum"]
        ideal_value = variable_configuration["ideal"]
        maximum = variable_configuration["maximum"]

        variable_data = data[
            [
                sample_column,
                liking_column,
                jar_variable,
            ]
        ].dropna()

        variable_data = variable_data[
            variable_data[jar_variable].between(
                minimum,
                maximum,
                inclusive="both",
            )
        ].copy()

        variable_data["jar_category"] = (
            variable_data[jar_variable].apply(
                lambda value: classify_jar_response(
                    float(value),
                    ideal_value=ideal_value,
                )
            )
        )

        grouped = variable_data.groupby(
            sample_column,
            sort=False,
            observed=True,
        )

        for sample, product_data in grouped:
            total_count = int(len(product_data))

            jar_group = product_data[
                product_data["jar_category"] == "jar"
            ]

            jar_count = int(len(jar_group))

            jar_mean = (
                float(jar_group[liking_column].mean())
                if jar_count
                else None
            )

            for direction, label in (
                ("too_little", "Too little"),
                ("too_much", "Too much"),
            ):
                deviation_group = product_data[
                    product_data["jar_category"]
                    == direction
                ]

                deviation_count = int(
                    len(deviation_group)
                )

                affected_percentage = (
                    deviation_count / total_count * 100
                    if total_count
                    else 0.0
                )

                deviation_mean = (
                    float(
                        deviation_group[
                            liking_column
                        ].mean()
                    )
                    if deviation_count
                    else None
                )

                available = bool(
                    jar_mean is not None
                    and deviation_mean is not None
                )

                mean_penalty = (
                    jar_mean - deviation_mean
                    if available
                    else None
                )

                sufficient_sample = bool(
                    jar_count >= minimum_group_size
                    and deviation_count
                    >= minimum_group_size
                )

                meets_priority_thresholds = bool(
                    available
                    and mean_penalty >= penalty_threshold
                    and affected_percentage >= percentage_threshold
                )

                reformulation_priority = bool(
                    meets_priority_thresholds
                    and sufficient_sample
                )

                potential_priority = bool(
                    meets_priority_thresholds
                    and not sufficient_sample
                )

                results.append(
                    {
                        "variable": jar_variable,
                        "sample": str(sample),
                        "direction": direction,
                        "direction_label": label,
                        "total_count": total_count,
                        "jar_count": jar_count,
                        "deviation_count": (
                            deviation_count
                        ),
                        "affected_percentage": (
                            affected_percentage
                        ),
                        "jar_mean_liking": jar_mean,
                        "deviation_mean_liking": (
                            deviation_mean
                        ),
                        "mean_penalty": mean_penalty,
                        "available": available,
                        "sufficient_sample": (
                            sufficient_sample
                        ),
                        "minimum_group_size": (
                            minimum_group_size
                        ),
                        "percentage_threshold": (
                            percentage_threshold
                        ),
                        "penalty_threshold": (
                            penalty_threshold
                        ),
                        "reformulation_priority": (
                            reformulation_priority
                        ),
                        "meets_priority_thresholds": meets_priority_thresholds,
                        "reformulation_priority": reformulation_priority,
                        "potential_priority": potential_priority,
                    }
                )

    return sorted(
        results,
        key=lambda item: (
            not item["reformulation_priority"],
            -(
                item["mean_penalty"]
                if item["mean_penalty"] is not None
                else -999
            ),
            -item["affected_percentage"],
            item["variable"],
            item["sample"],
            item["direction"],
        ),
    )

def build_penalty_summary(
    penalty_analysis: list[dict[str, Any]],
) -> dict[str, Any]:
    available_results = [
        result
        for result in penalty_analysis
        if result["available"]
    ]

    priority_results = [
        result
        for result in available_results
        if result["reformulation_priority"]
    ]

    insufficient_sample_results = [
        result
        for result in available_results
        if not result["sufficient_sample"]
    ]

    highest_penalty = None

    if available_results:
        highest_penalty = max(
            available_results,
            key=lambda result: result["mean_penalty"],
        )

    potential_priority_results = [
        result
        for result in available_results
        if result.get("potential_priority")
    ]

    return {
        "comparison_count": len(
            penalty_analysis
        ),
        "available_count": len(
            available_results
        ),
        "priority_count": len(
            priority_results
        ),
        "insufficient_sample_count": len(
            insufficient_sample_results
        ),
        "highest_penalty": highest_penalty,
        "potential_priority_count": len(
            potential_priority_results
        ),
    }

def build_penalty_chart_data(
    penalty_analysis: list[dict[str, Any]],
) -> dict[str, Any]:
    points = []

    for result in penalty_analysis:
        if not result["available"]:
            continue

        points.append(
            {
                "x": result[
                    "affected_percentage"
                ],
                "y": result["mean_penalty"],
                "variable": result["variable"],
                "sample": result["sample"],
                "direction": result[
                    "direction_label"
                ],
                "priority": result[
                    "reformulation_priority"
                ],
                "sufficient_sample": result[
                    "sufficient_sample"
                ],
                "jar_count": result["jar_count"],
                "deviation_count": result[
                    "deviation_count"
                ],
            }
        )

    percentage_threshold = (
        DEFAULT_PENALTY_PERCENTAGE_THRESHOLD
    )

    penalty_threshold = (
        DEFAULT_PENALTY_VALUE_THRESHOLD
    )

    if penalty_analysis:
        percentage_threshold = (
            penalty_analysis[0][
                "percentage_threshold"
            ]
        )

        penalty_threshold = (
            penalty_analysis[0][
                "penalty_threshold"
            ]
        )

    return {
        "points": points,
        "percentage_threshold": (
            percentage_threshold
        ),
        "penalty_threshold": penalty_threshold,
    }

def build_jar_interpretation(
    variable_summaries: list[dict[str, Any]],
    penalty_analysis: list[dict[str, Any]],
) -> dict[str, Any]:
    jar_summary_statements = []
    priority_statements = []
    deviation_statements = []
    methodology_statements = []

    # 1. Mejores y peores porcentajes JAR
    for variable_summary in variable_summaries:
        variable_name = variable_summary["variable"]

        best_product = variable_summary.get(
            "best_product"
        )

        lowest_product = variable_summary.get(
            "lowest_product"
        )

        if best_product:
            jar_summary_statements.append(
                (
                    f"For {variable_name}, product "
                    f"{best_product['sample']} obtained the "
                    f"highest JAR percentage "
                    f"({best_product['jar_percentage']:.1f}%)."
                )
            )

        if (
            lowest_product
            and best_product
            and lowest_product["sample"]
            != best_product["sample"]
        ):
            jar_summary_statements.append(
                (
                    f"Product {lowest_product['sample']} obtained "
                    f"the lowest JAR percentage for "
                    f"{variable_name} "
                    f"({lowest_product['jar_percentage']:.1f}%)."
                )
            )

    # 2. Prioridades confirmadas y potenciales
    priority_results = [
        result
        for result in penalty_analysis
        if result.get("reformulation_priority")
    ]

    potential_priority_results = [
        result
        for result in penalty_analysis
        if result.get("potential_priority")
    ]

    available_results = [
        result
        for result in penalty_analysis
        if result.get("available")
    ]

    if priority_results:
        for result in priority_results:
            priority_statements.append(
                (
                    f"For product {result['sample']}, "
                    f"{result['variable']} rated as "
                    f"{result['direction_label'].lower()} affected "
                    f"{result['affected_percentage']:.1f}% of valid "
                    f"responses and was associated with a "
                    f"{result['mean_penalty']:.2f}-point reduction "
                    f"in mean liking. This result qualifies as a "
                    f"confirmed reformulation priority."
                )
            )

    elif potential_priority_results:
        priority_statements.append(
            (
                "No JAR deviation qualified as a confirmed "
                "reformulation priority because the comparisons "
                "meeting the numerical thresholds did not satisfy "
                "the configured minimum group-size requirement."
            )
        )

    elif available_results:
        priority_statements.append(
            (
                "No JAR deviation met the configured consumer-"
                "percentage and mean-penalty thresholds for "
                "reformulation priority."
            )
        )

    else:
        priority_statements.append(
            (
                "Penalty analysis could not be calculated because "
                "the product-attribute comparisons lacked both JAR "
                "and non-JAR response groups."
            )
        )

    for result in potential_priority_results:
        priority_statements.append(
            (
                f"For product {result['sample']}, "
                f"{result['variable']} rated as "
                f"{result['direction_label'].lower()} affected "
                f"{result['affected_percentage']:.1f}% of valid "
                f"responses and was associated with a "
                f"{result['mean_penalty']:.2f}-point reduction "
                f"in mean liking. The numerical thresholds were met, "
                f"but this should be considered a potential priority "
                f"because the group size was below the configured "
                f"minimum."
            )
        )

    # 3. Desviaciones totales sin grupo JAR
    non_calculable_deviations = [
        result
        for result in penalty_analysis
        if (
            not result.get("available")
            and result.get("deviation_count", 0) > 0
            and result.get("jar_count", 0) == 0
        )
    ]

    for result in non_calculable_deviations:
        deviation_statements.append(
            (
                f"For product {result['sample']}, all valid responses "
                f"classified {result['variable']} as "
                f"{result['direction_label'].lower()}. A classical "
                f"mean-drop penalty could not be calculated because "
                f"no JAR reference group was available; however, "
                f"the result indicates a consistent directional "
                f"deviation."
            )
        )

    # 4. Advertencia metodológica final
    insufficient_results = [
        result
        for result in penalty_analysis
        if (
            result.get("available")
            and not result.get("sufficient_sample", False)
        )
    ]

    if insufficient_results:
        methodology_statements.append(
            (
                "Results based on groups smaller than the configured "
                "minimum should be interpreted as exploratory."
            )
        )

    statements = (
        jar_summary_statements
        + priority_statements
        + deviation_statements
        + methodology_statements
    )

    return {
        "statements": statements,
        "summary": " ".join(statements),
    }

def run_jar_analysis(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    configuration: dict | None = None,
    selected_jar_variables: list[str] | None = None,
) -> dict[str, Any]:
    configuration = configuration or {}

    prepared = prepare_jar_data(
        datasets=datasets,
        mapping=mapping,
        selected_jar_variables=selected_jar_variables,
    )

    if not prepared["is_valid"]:
        return prepared

    data = prepared["data"]
    sample_column = prepared["sample_column"]
    jar_variables = prepared["jar_variables"]
    liking_column = prepared["liking_column"]

    distribution = calculate_jar_distribution(
        data=data,
        sample_column=sample_column,
        jar_variables=jar_variables,
        configuration=configuration,
    )

    variable_summaries = build_jar_variable_summary(
        distribution
    )

    chart_data = build_jar_chart_data(
        distribution
    )

    penalty_analysis = calculate_penalty_analysis(
        data=data,
        sample_column=sample_column,
        liking_column=liking_column,
        jar_variables=jar_variables,
        configuration=configuration,
    )

    penalty_summary = build_penalty_summary(
        penalty_analysis
    )

    penalty_chart_data = build_penalty_chart_data(
        penalty_analysis
    )

    interpretation = build_jar_interpretation(
        variable_summaries,
        penalty_analysis,
    )

    return {
        "is_valid": True,
        "consumer_column": prepared["consumer_column"],
        "sample_column": sample_column,
        "liking_column": liking_column,
        "jar_variables": jar_variables,
        "available_jar_variables": prepared[
            "available_jar_variables"
        ],
        "excluded_identifier_rows": prepared[
            "excluded_identifier_rows"
        ],
        "duplicate_count": prepared[
            "duplicate_count"
        ],
        "distribution": distribution,
        "variable_summaries": variable_summaries,
        "chart_data": chart_data,
        "penalty_analysis": penalty_analysis,
        "penalty_summary": penalty_summary,
        "penalty_chart_data": penalty_chart_data,
        "interpretation": interpretation,
    }