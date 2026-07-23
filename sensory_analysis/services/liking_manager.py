from __future__ import annotations
import math
from itertools import combinations
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats

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

def prepare_liking_data(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    liking_variable: str | None = None,
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
                "for liking analysis."
            ),
        }

    if len(sample_columns) != 1:
        return {
            "is_valid": False,
            "error": (
                "Exactly one Sample ID variable is required "
                "for liking analysis."
            ),
        }

    if not liking_columns:
        return {
            "is_valid": False,
            "error": "No liking variable was mapped.",
        }

    if liking_variable is None:
        liking_variable = liking_columns[0]

    if liking_variable not in liking_columns:
        return {
            "is_valid": False,
            "error": (
                f"{liking_variable} is not mapped as an "
                "overall liking variable."
            ),
        }

    consumer_column = consumer_columns[0]
    sample_column = sample_columns[0]

    required_columns = [
        consumer_column,
        sample_column,
        liking_variable,
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

    liking_text = (
        data[liking_variable]
        .astype("string")
        .str.strip()
        .str.replace(",", ".", regex=False)
        .replace("", pd.NA)
    )

    data[liking_variable] = pd.to_numeric(
        liking_text,
        errors="coerce",
    )

    initial_row_count = len(data)

    data = data.dropna(
        subset=[
            consumer_column,
            sample_column,
            liking_variable,
        ]
    )

    excluded_row_count = initial_row_count - len(data)

    duplicate_count = int(
        data.duplicated(
            subset=[
                consumer_column,
                sample_column,
            ],
            keep=False,
        ).sum()
    )

    return {
        "is_valid": True,
        "data": data,
        "consumer_column": consumer_column,
        "sample_column": sample_column,
        "liking_variable": liking_variable,
        "liking_variables": liking_columns,
        "excluded_row_count": excluded_row_count,
        "duplicate_count": duplicate_count,
    }

def calculate_descriptive_statistics(
    *,
    data: pd.DataFrame,
    sample_column: str,
    liking_variable: str,
    confidence_level: float = 0.95,
) -> list[dict[str, Any]]:
    results = []

    alpha = 1 - confidence_level

    grouped = data.groupby(
        sample_column,
        sort=False,
        observed=True,
    )

    for sample, group in grouped:
        values = group[liking_variable].dropna()
        count = int(values.count())

        mean = float(values.mean()) if count else None
        standard_deviation = (
            float(values.std(ddof=1))
            if count > 1
            else None
        )
        standard_error = (
            standard_deviation / math.sqrt(count)
            if standard_deviation is not None
            else None
        )

        confidence_lower = None
        confidence_upper = None

        if count > 1 and standard_error is not None:
            critical_value = stats.t.ppf(
                1 - alpha / 2,
                df=count - 1,
            )

            confidence_lower = (
                mean - critical_value * standard_error
            )
            confidence_upper = (
                mean + critical_value * standard_error
            )

        results.append(
            {
                "sample": str(sample),
                "count": count,
                "mean": mean,
                "median": float(values.median()) if count else None,
                "standard_deviation": standard_deviation,
                "standard_error": standard_error,
                "minimum": float(values.min()) if count else None,
                "maximum": float(values.max()) if count else None,
                "confidence_lower": confidence_lower,
                "confidence_upper": confidence_upper,
            }
        )

    return sorted(
        results,
        key=lambda item: (
            item["mean"] is None,
            -(item["mean"] or 0),
        ),
    )

def build_product_ranking(
    descriptive_statistics: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    valid_products = [
        item
        for item in descriptive_statistics
        if item.get("mean") is not None
    ]

    ordered_products = sorted(
        valid_products,
        key=lambda item: item["mean"],
        reverse=True,
    )

    ranking = []
    previous_mean = None
    previous_rank = 0

    for index, item in enumerate(
        ordered_products,
        start=1,
    ):
        mean = item["mean"]

        if (
            previous_mean is not None
            and math.isclose(
                mean,
                previous_mean,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            rank = previous_rank
        else:
            rank = index

        ranking.append(
            {
                "rank": rank,
                "sample": item["sample"],
                "mean": mean,
                "median": item.get("median"),
                "count": item.get("count"),
            }
        )

        previous_mean = mean
        previous_rank = rank

    return ranking

def evaluate_liking_design(
    *,
    data: pd.DataFrame,
    consumer_column: str,
    sample_column: str,
) -> dict[str, Any]:
    consumer_sample_counts = (
        data.groupby(
            consumer_column,
            observed=True,
        )[sample_column]
        .nunique()
    )

    total_samples = int(
        data[sample_column].nunique()
    )
    total_consumers = int(
        data[consumer_column].nunique()
    )

    complete_consumers = int(
        (consumer_sample_counts == total_samples).sum()
    )

    incomplete_consumers = (
        total_consumers - complete_consumers
    )

    repeated_rows = int(
        data.duplicated(
            subset=[
                consumer_column,
                sample_column,
            ],
            keep=False,
        ).sum()
    )

    is_complete = (
        total_consumers > 0
        and total_samples > 1
        and incomplete_consumers == 0
        and repeated_rows == 0
    )

    return {
        "consumer_count": total_consumers,
        "sample_count": total_samples,
        "evaluation_count": int(len(data)),
        "complete_consumer_count": complete_consumers,
        "incomplete_consumer_count": incomplete_consumers,
        "repeated_row_count": repeated_rows,
        "is_complete_repeated_measures": is_complete,
    }

def build_liking_matrix(
    *,
    data: pd.DataFrame,
    consumer_column: str,
    sample_column: str,
    liking_variable: str,
) -> pd.DataFrame:
    return data.pivot_table(
        index=consumer_column,
        columns=sample_column,
        values=liking_variable,
        aggfunc="mean",
        observed=True,
    )

def describe_liking_design(
    design: dict[str, Any],
) -> dict[str, Any]:
    consumer_count = design.get(
        "consumer_count",
        0,
    )
    sample_count = design.get(
        "sample_count",
        0,
    )
    evaluation_count = design.get(
        "evaluation_count",
        0,
    )
    incomplete_count = design.get(
        "incomplete_consumer_count",
        0,
    )
    repeated_count = design.get(
        "repeated_row_count",
        0,
    )

    if design.get(
        "is_complete_repeated_measures"
    ):
        design_type = "Complete repeated-measures design"
        balance = "Balanced"
        description = (
            "Every consumer evaluated every product once."
        )

    elif repeated_count > 0:
        design_type = "Repeated evaluations detected"
        balance = "Requires review"
        description = (
            "At least one consumer-product combination "
            "contains more than one evaluation."
        )

    elif incomplete_count > 0:
        design_type = "Incomplete repeated-measures design"
        balance = "Unbalanced"
        description = (
            f"{incomplete_count} consumer(s) did not evaluate "
            "all products."
        )

    else:
        design_type = "Consumer-product evaluation design"
        balance = "Undetermined"
        description = (
            "The available data do not match a complete "
            "repeated-measures design."
        )

    return {
        "type": design_type,
        "balance": balance,
        "description": description,
        "consumer_count": consumer_count,
        "sample_count": sample_count,
        "evaluation_count": evaluation_count,
    }

def format_p_value(
    p_value: float | None,
) -> str:
    if p_value is None:
        return "not available"

    if p_value < 0.0001:
        return "< 0.0001"

    return f"{p_value:.4f}"


def build_liking_interpretation(
    *,
    ranking: list[dict[str, Any]],
    friedman_result: dict[str, Any],
    pairwise_results: list[dict[str, Any]],
    effect_size: dict[str, Any],
    design: dict[str, Any],
) -> dict[str, Any]:
    statements = []

    if ranking:
        best_product = ranking[0]

        statements.append(
            (
                f"Product {best_product['sample']} obtained the "
                f"highest average liking score "
                f"({best_product['mean']:.2f})."
            )
        )

        if len(ranking) > 1:
            lowest_product = ranking[-1]

            statements.append(
                (
                    f"Product {lowest_product['sample']} obtained "
                    f"the lowest average liking score "
                    f"({lowest_product['mean']:.2f})."
                )
            )

    if friedman_result.get("available"):
        p_value = friedman_result.get("p_value")
        formatted_p = format_p_value(p_value)

        if friedman_result.get("significant"):
            statements.append(
                (
                    "The Friedman test detected statistically "
                    "significant differences among products "
                    f"(p = {formatted_p})."
                )
            )
        else:
            statements.append(
                (
                    "The Friedman test did not detect statistically "
                    "significant differences among products "
                    f"(p = {formatted_p})."
                )
            )
    else:
        statements.append(
            (
                "The overall product comparison could not be "
                "performed with the available data."
            )
        )

    significant_comparisons = [
        result
        for result in pairwise_results
        if (
            result.get("available")
            and result.get("significant")
        )
    ]

    available_comparisons = [
        result
        for result in pairwise_results
        if result.get("available")
    ]

    if available_comparisons:
        if significant_comparisons:
            count = len(significant_comparisons)

            statements.append(
                (
                    f"{count} pairwise product comparison(s) "
                    "remained statistically significant after "
                    "Holm adjustment."
                )
            )

        elif friedman_result.get("significant"):
            statements.append(
                (
                    "The global test indicates that at least one "
                    "product differs, but no specific pair remained "
                    "significant after multiple-comparison adjustment."
                )
            )

        else:
            statements.append(
                (
                    "No individual pair of products was statistically "
                    "significant after Holm adjustment."
                )
            )

    if effect_size.get("available"):
        statements.append(
            (
                f"Kendall's W was "
                f"{effect_size['value']:.3f}, indicating "
                f"{effect_size['interpretation'].lower()} agreement "
                "in product ranking among complete consumers."
            )
        )

    consumer_count = design.get(
        "consumer_count",
        0,
    )

    if consumer_count < 30:
        statements.append(
            (
                "The number of consumers is small, so the results "
                "should be interpreted as exploratory and may have "
                "limited statistical power."
            )
        )

    return {
        "statements": statements,
        "summary": " ".join(statements),
    }

def build_liking_chart_data(
    *,
    descriptive_statistics: list[dict[str, Any]],
    scale: dict[str, float],
) -> dict[str, Any]:
    ordered = sorted(
        descriptive_statistics,
        key=lambda item: (
            item.get("mean") is None,
            -(item.get("mean") or 0),
        ),
    )

    scale_minimum = scale["minimum"]
    scale_maximum = scale["maximum"]

    points = []

    for item in ordered:
        mean = item.get("mean")
        confidence_lower = item.get("confidence_lower")
        confidence_upper = item.get("confidence_upper")

        if mean is None:
            continue

        points.append(
            {
                "sample": str(item["sample"]),
                "mean": float(mean),

                # Valores estadísticos completos.
                "confidence_lower": (
                    float(confidence_lower)
                    if confidence_lower is not None
                    else None
                ),
                "confidence_upper": (
                    float(confidence_upper)
                    if confidence_upper is not None
                    else None
                ),

                # Valores limitados únicamente para dibujar.
                "display_lower": (
                    max(
                        scale_minimum,
                        float(confidence_lower),
                    )
                    if confidence_lower is not None
                    else float(mean)
                ),
                "display_upper": (
                    min(
                        scale_maximum,
                        float(confidence_upper),
                    )
                    if confidence_upper is not None
                    else float(mean)
                ),
            }
        )

    return {
        "labels": [
            point["sample"]
            for point in points
        ],
        "means": [
            point["mean"]
            for point in points
        ],
        "confidence_lower": [
            point["confidence_lower"]
            for point in points
        ],
        "confidence_upper": [
            point["confidence_upper"]
            for point in points
        ],
        "display_lower": [
            point["display_lower"]
            for point in points
        ],
        "display_upper": [
            point["display_upper"]
            for point in points
        ],
        "scale_minimum": scale_minimum,
        "scale_maximum": scale_maximum,
    }

def run_friedman_test(
    liking_matrix: pd.DataFrame,
) -> dict[str, Any]:
    complete_matrix = liking_matrix.dropna(
        axis=0,
        how="any",
    )

    sample_names = [
        str(column)
        for column in complete_matrix.columns
    ]

    if len(sample_names) < 3:
        return {
            "available": False,
            "reason": (
                "The Friedman test requires at least "
                "three products."
            ),
        }

    if len(complete_matrix) < 2:
        return {
            "available": False,
            "reason": (
                "At least two consumers with complete evaluations "
                "are required."
            ),
        }

    groups = [
        complete_matrix[column].to_numpy()
        for column in complete_matrix.columns
    ]

    alpha = 0.05

    try:
        statistic, p_value = stats.friedmanchisquare(
            *groups
        )
    except ValueError as error:
        return {
            "available": False,
            "reason": str(error),
        }

    return {
        "available": True,
        "test": "Friedman test",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": alpha,
        "consumer_count": int(len(complete_matrix)),
        "sample_count": len(sample_names),
        "sample_names": sample_names,
        "significant": bool(p_value < alpha),
    }

def calculate_kendalls_w(
    *,
    friedman_statistic: float | None,
    consumer_count: int,
    sample_count: int,
) -> dict[str, Any]:
    if (
        friedman_statistic is None
        or consumer_count <= 0
        or sample_count <= 1
    ):
        return {
            "available": False,
            "value": None,
            "interpretation": None,
        }

    denominator = consumer_count * (sample_count - 1)

    if denominator <= 0:
        return {
            "available": False,
            "value": None,
            "interpretation": None,
        }

    value = friedman_statistic / denominator
    value = max(0.0, min(1.0, float(value)))

    if value < 0.10:
        interpretation = "Negligible"

    elif value < 0.30:
        interpretation = "Weak"

    elif value < 0.50:
        interpretation = "Moderate"

    else:
        interpretation = "Strong"

    return {
        "available": True,
        "name": "Kendall's W",
        "value": value,
        "interpretation": interpretation,
    }

def holm_adjustment(
    p_values: list[float],
) -> list[float]:
    number_of_tests = len(p_values)

    if number_of_tests == 0:
        return []

    order = np.argsort(p_values)
    adjusted = np.empty(number_of_tests, dtype=float)

    previous_value = 0.0

    for rank, original_index in enumerate(order):
        multiplier = number_of_tests - rank

        adjusted_value = min(
            1.0,
            p_values[original_index] * multiplier,
        )

        adjusted_value = max(
            adjusted_value,
            previous_value,
        )

        adjusted[original_index] = adjusted_value
        previous_value = adjusted_value

    return adjusted.tolist()

def run_pairwise_wilcoxon(
    liking_matrix: pd.DataFrame,
) -> list[dict[str, Any]]:
    raw_results = []

    for sample_a, sample_b in combinations(
        liking_matrix.columns,
        2,
    ):
        pair = liking_matrix[
            [sample_a, sample_b]
        ].dropna()

        if len(pair) < 2:
            raw_results.append(
                {
                    "sample_a": str(sample_a),
                    "sample_b": str(sample_b),
                    "available": False,
                    "reason": (
                        "Not enough paired observations."
                    ),
                }
            )
            continue

        differences = (
            pair[sample_a] - pair[sample_b]
        )

        if np.allclose(
            differences.to_numpy(),
            0,
        ):
            statistic = 0.0
            p_value = 1.0
        else:
            try:
                statistic, p_value = stats.wilcoxon(
                    pair[sample_a],
                    pair[sample_b],
                    zero_method="wilcox",
                    alternative="two-sided",
                )
            except ValueError as error:
                raw_results.append(
                    {
                        "sample_a": str(sample_a),
                        "sample_b": str(sample_b),
                        "available": False,
                        "reason": str(error),
                    }
                )
                continue

        raw_results.append(
            {
                "sample_a": str(sample_a),
                "sample_b": str(sample_b),
                "available": True,
                "consumer_count": int(len(pair)),
                "mean_a": float(pair[sample_a].mean()),
                "mean_b": float(pair[sample_b].mean()),
                "mean_difference": float(
                    differences.mean()
                ),
                "median_difference": float(
                    differences.median()
                ),
                "statistic": float(statistic),
                "raw_p_value": float(p_value),
            }
        )

    available_results = [
        result
        for result in raw_results
        if result.get("available")
    ]

    adjusted_values = holm_adjustment(
        [
            result["raw_p_value"]
            for result in available_results
        ]
    )

    for result, adjusted_p_value in zip(
        available_results,
        adjusted_values,
    ):
        result["adjusted_p_value"] = (
            float(adjusted_p_value)
        )
        result["significant"] = bool(
            adjusted_p_value < 0.05
        )

    return raw_results

def run_liking_analysis(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    configuration: dict | None = None,
    liking_variable: str | None = None,
) -> dict[str, Any]:
    
    configuration = configuration or {}
    
    prepared = prepare_liking_data(
        datasets=datasets,
        mapping=mapping,
        liking_variable=liking_variable,
    )

    if not prepared["is_valid"]:
        return prepared

    data = prepared["data"]
    consumer_column = prepared["consumer_column"]
    sample_column = prepared["sample_column"]
    liking_variable = prepared["liking_variable"]

    descriptive_statistics = (
        calculate_descriptive_statistics(
            data=data,
            sample_column=sample_column,
            liking_variable=liking_variable,
        )
    )

    design = evaluate_liking_design(
        data=data,
        consumer_column=consumer_column,
        sample_column=sample_column,
    )

    design_summary = describe_liking_design(
        design
    )

    liking_matrix = build_liking_matrix(
        data=data,
        consumer_column=consumer_column,
        sample_column=sample_column,
        liking_variable=liking_variable,
    )

    friedman_result = run_friedman_test(
        liking_matrix
    )

    pairwise_results = run_pairwise_wilcoxon(
        liking_matrix
    )

    friedman_statistic = None
    complete_consumer_count = 0
    friedman_sample_count = 0

    if friedman_result.get("available"):
        friedman_statistic = friedman_result.get(
            "statistic"
        )
        complete_consumer_count = friedman_result.get(
            "consumer_count",
            0,
        )
        friedman_sample_count = friedman_result.get(
            "sample_count",
            0,
        )

    effect_size = calculate_kendalls_w(
        friedman_statistic=friedman_statistic,
        consumer_count=complete_consumer_count,
        sample_count=friedman_sample_count,
    )

    ranking = build_product_ranking(
        descriptive_statistics
    )

    interpretation = build_liking_interpretation(
        ranking=ranking,
        friedman_result=friedman_result,
        pairwise_results=pairwise_results,
        effect_size=effect_size,
        design=design,
    )

    liking_scale = get_liking_scale(
        configuration=configuration,
        liking_variable=liking_variable,
    )

    chart_data = build_liking_chart_data(
        descriptive_statistics=descriptive_statistics,
        scale=liking_scale,
    )

    return {
        "is_valid": True,

        "variables": {
            "consumer": consumer_column,
            "sample": sample_column,
            "liking": liking_variable,
            "available_liking_variables": prepared[
                "liking_variables"
            ],
        },

        "consumer_column": consumer_column,
        "sample_column": sample_column,
        "liking_variable": liking_variable,
        "liking_variables": prepared[
            "liking_variables"
        ],

        "data_quality": {
            "excluded_row_count": prepared[
                "excluded_row_count"
            ],
            "duplicate_count": prepared[
                "duplicate_count"
            ],
        },

        "excluded_row_count": prepared[
            "excluded_row_count"
        ],
        "duplicate_count": prepared[
            "duplicate_count"
        ],

        "design": design,
        "design_summary": design_summary,
        "descriptive_statistics": descriptive_statistics,
        "ranking": ranking,
        "friedman": friedman_result,
        "effect_size": effect_size,
        "pairwise_comparisons": pairwise_results,
        "interpretation": interpretation,
        "scale": liking_scale,
        "chart_data": chart_data,
    }

def get_liking_scale(
    *,
    configuration: dict,
    liking_variable: str,
) -> dict[str, float]:
    default_scale = {
        "minimum": 1.0,
        "maximum": 9.0,
    }

    liking_configuration = configuration.get(
        "liking",
        {},
    )

    variable_configuration = liking_configuration.get(
        liking_variable,
        {},
    )

    scale_minimum = variable_configuration.get(
        "scale_minimum",
        variable_configuration.get(
            "minimum",
            default_scale["minimum"],
        ),
    )

    scale_maximum = variable_configuration.get(
        "scale_maximum",
        variable_configuration.get(
            "maximum",
            default_scale["maximum"],
        ),
    )

    try:
        scale_minimum = float(scale_minimum)
    except (TypeError, ValueError):
        scale_minimum = default_scale["minimum"]

    try:
        scale_maximum = float(scale_maximum)
    except (TypeError, ValueError):
        scale_maximum = default_scale["maximum"]

    if scale_minimum >= scale_maximum:
        return default_scale

    return {
        "minimum": scale_minimum,
        "maximum": scale_maximum,
    }

