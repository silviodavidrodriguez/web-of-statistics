from __future__ import annotations
from collections import defaultdict
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2, ttest_ind

DEFAULT_CATA_SEPARATOR = ";"
EMPTY_CATA_VALUES = {
    "",
    "nan",
    "none",
    "null",
    "na",
    "n/a",
}

def get_mapped_columns(
    mapping: dict[str, dict[str, str]],
    *,
    dataset_name: str,
    role: str,
) -> list[str]:

    dataset_mapping = mapping.get(
        dataset_name,
        {},
    )

    return [
        column_name
        for column_name, assigned_role
        in dataset_mapping.items()
        if assigned_role == role
    ]

def normalize_identifier_series(
    series: pd.Series,
) -> pd.Series:

    normalized = (
        series
        .astype("string")
        .str.strip()
    )

    normalized = normalized.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "None": pd.NA,
            "none": pd.NA,
            "NULL": pd.NA,
            "null": pd.NA,
        }
    )

    return normalized

def normalize_numeric_series(
    series: pd.Series,
) -> pd.Series:

    normalized = (
        series
        .astype("string")
        .str.strip()
        .str.replace(
            ",",
            ".",
            regex=False,
        )
        .replace(
            {
                "": pd.NA,
                "nan": pd.NA,
                "NaN": pd.NA,
                "None": pd.NA,
                "none": pd.NA,
                "null": pd.NA,
            }
        )
    )

    return pd.to_numeric(
        normalized,
        errors="coerce",
    )

def normalize_attribute_name(
    value: Any,
) -> str | None:

    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass

    normalized = " ".join(
        str(value).strip().split()
    )

    if normalized.casefold() in EMPTY_CATA_VALUES:
        return None

    return normalized

def split_cata_response(
    value: Any,
    separator: str = DEFAULT_CATA_SEPARATOR,
) -> list[str]:

    if value is None:
        return []

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    raw_value = str(value).strip()

    if raw_value.casefold() in EMPTY_CATA_VALUES:
        return []

    attributes = []
    normalized_keys = set()

    for item in raw_value.split(separator):
        normalized = normalize_attribute_name(item)

        if not normalized:
            continue

        normalized_key = normalized.casefold()

        if normalized_key in normalized_keys:
            continue

        normalized_keys.add(normalized_key)
        attributes.append(normalized)

    return attributes

def consolidate_cata_responses(
    *,
    response_data: pd.DataFrame,
    liking_available: bool,
) -> pd.DataFrame:

    if response_data.empty:
        columns = [
            "consumer",
            "sample",
            "selected_attributes",
            "attribute_count",
            "source_row_count",
        ]

        if liking_available:
            columns.append("liking")

        return pd.DataFrame(
            columns=columns
        )

    consolidated_rows = []

    grouped = response_data.groupby(
        [
            "consumer",
            "sample",
        ],
        sort=False,
        dropna=False,
        observed=True,
    )

    for (
        consumer,
        sample,
    ), group in grouped:
        selected_attributes = []
        selected_keys = set()

        for attribute_list in group[
            "selected_attributes"
        ]:
            for attribute in attribute_list:
                attribute_key = (
                    attribute.casefold()
                )

                if attribute_key in selected_keys:
                    continue

                selected_keys.add(
                    attribute_key
                )

                selected_attributes.append(
                    attribute
                )

        consolidated_row = {
            "consumer": consumer,
            "sample": sample,
            "selected_attributes": (
                selected_attributes
            ),
            "attribute_count": len(
                selected_attributes
            ),
            "source_row_count": int(
                len(group)
            ),
        }

        if liking_available:
            liking_values = pd.to_numeric(
                group["liking"],
                errors="coerce",
            ).dropna()

            consolidated_row["liking"] = (
                float(liking_values.mean())
                if not liking_values.empty
                else None
            )

        consolidated_rows.append(
            consolidated_row
        )

    consolidated_columns = [
        "consumer",
        "sample",
        "selected_attributes",
        "attribute_count",
        "source_row_count",
    ]

    if liking_available:
        consolidated_columns.append(
            "liking"
        )

    consolidated_data = pd.DataFrame(
        consolidated_rows,
        columns=consolidated_columns,
    )

    return (
        consolidated_data
        .sort_values(
            by=[
                "sample",
                "consumer",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

def prepare_cata_data(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    selected_cata_variables: list[str] | None = None,
    separator: str = DEFAULT_CATA_SEPARATOR,
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

    cata_columns = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="cata",
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
                "for CATA analysis."
            ),
        }

    if len(sample_columns) != 1:
        return {
            "is_valid": False,
            "error": (
                "Exactly one Sample ID variable is required "
                "for CATA analysis."
            ),
        }

    if not cata_columns:
        return {
            "is_valid": False,
            "error": "No CATA variables were mapped.",
        }

    if selected_cata_variables is None:
        cata_variables = cata_columns
    else:
        cata_variables = [
            column_name
            for column_name in selected_cata_variables
            if column_name in cata_columns
        ]

    if not cata_variables:
        return {
            "is_valid": False,
            "error": "No valid CATA variables were selected.",
        }

    consumer_column = consumer_columns[0]
    sample_column = sample_columns[0]

    liking_column = (
        liking_columns[0]
        if len(liking_columns) == 1
        else None
    )

    required_columns = [
        consumer_column,
        sample_column,
        *cata_variables,
    ]

    if liking_column:
        required_columns.append(
            liking_column
        )

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

    data = evaluations[
        required_columns
    ].copy()

    data[consumer_column] = normalize_identifier_series(
        data[consumer_column]
    )

    data[sample_column] = normalize_identifier_series(
        data[sample_column]
    )

    initial_row_count = len(data)

    data = data.dropna(
        subset=[
            consumer_column,
            sample_column,
        ]
    ).copy()

    excluded_identifier_rows = (
        initial_row_count - len(data)
    )

    duplicate_mask = data.duplicated(
        subset=[
            consumer_column,
            sample_column,
        ],
        keep=False,
    )

    duplicate_count = int(
        duplicate_mask.sum()
    )

    duplicate_pair_count = int(
        data.loc[
            duplicate_mask,
            [
                consumer_column,
                sample_column,
            ],
        ]
        .drop_duplicates()
        .shape[0]
    )

    if liking_column:
        data[liking_column] = normalize_numeric_series(
            data[liking_column]
        )

    response_rows = []
    attribute_display_names = {}

    for row_index, row in data.iterrows():
        consumer = row[consumer_column]
        sample = row[sample_column]

        selected_attributes = []
        selected_attribute_keys = set()

        for cata_variable in cata_variables:
            variable_attributes = split_cata_response(
                row[cata_variable],
                separator=separator,
            )

            for attribute in variable_attributes:
                attribute_key = attribute.casefold()

                if attribute_key in selected_attribute_keys:
                    continue

                selected_attribute_keys.add(
                    attribute_key
                )

                attribute_display_names.setdefault(
                    attribute_key,
                    attribute,
                )

                selected_attributes.append(
                    attribute_display_names[
                        attribute_key
                    ]
                )

        response_row = {
            "row_index": row_index,
            "consumer": consumer,
            "sample": sample,
            "selected_attributes": (
                selected_attributes
            ),
            "attribute_count": len(
                selected_attributes
            ),
        }

        if liking_column:
            response_row["liking"] = row[
                liking_column
            ]

        response_rows.append(
            response_row
        )

    response_columns = [
        "row_index",
        "consumer",
        "sample",
        "selected_attributes",
        "attribute_count",
    ]

    if liking_column:
        response_columns.append(
            "liking"
        )

    response_data = pd.DataFrame(
        response_rows,
        columns=response_columns,
    )

    raw_response_data = response_data.copy()
    response_data = consolidate_cata_responses(
        response_data=raw_response_data,
        liking_available=bool(
            liking_column
        ),
    )

    consolidated_row_count = int(
        len(raw_response_data)
        - len(response_data)
    )

    consolidated_pair_count = int(
        (
            response_data[
                "source_row_count"
            ]
            > 1
        ).sum()
    ) if not response_data.empty else 0

    attribute_keys = sorted(
        attribute_display_names,
        key=lambda value: (
            attribute_display_names[
                value
            ].casefold()
        ),
    )

    attributes = [
        attribute_display_names[
            attribute_key
        ]
        for attribute_key in attribute_keys
    ]

    binary_rows = []
    for _, response_row in (response_data.iterrows()):
        selected_keys = {
            attribute.casefold()
            for attribute in response_row[
                "selected_attributes"
            ]
        }

        binary_row = {
            "consumer": response_row[
                "consumer"
            ],
            "sample": response_row[
                "sample"
            ],
            "source_row_count": int(
                response_row[
                    "source_row_count"
                ]
            ),
        }

        if liking_column:
            binary_row["liking"] = (
                response_row.get("liking")
            )

        for attribute_key in attribute_keys:
            attribute_name = (
                attribute_display_names[
                    attribute_key
                ]
            )

            binary_row[attribute_name] = int(
                attribute_key in selected_keys
            )

        binary_rows.append(
            binary_row
        )

    binary_columns = [
        "consumer",
        "sample",
        "source_row_count",
    ]

    if liking_column:
        binary_columns.append(
            "liking"
        )

    binary_columns.extend(
        attributes
    )

    binary_data = pd.DataFrame(
        binary_rows,
        columns=binary_columns,
    )

    if not binary_data.empty:
        binary_data = (
            binary_data
            .sort_values(
                by=[
                    "sample",
                    "consumer",
                ],
                kind="stable",
            )
            .reset_index(drop=True)
        )

    products = (
        data[sample_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    products = sorted(
        products,
        key=lambda value: value.casefold(),
    )

    consumers = (
        data[consumer_column]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    consumers = sorted(
        consumers,
        key=lambda value: value.casefold(),
    )

    empty_cata_responses = int(
        (
            response_data[
                "attribute_count"
            ]
            == 0
        ).sum()
    ) if not response_data.empty else 0

    return {
        "is_valid": True,
        "data": data,
        "raw_response_data": (
            raw_response_data
        ),
        "response_data": response_data,
        "binary_data": binary_data,
        "consumer_column": consumer_column,
        "sample_column": sample_column,
        "liking_column": liking_column,
        "cata_variables": cata_variables,
        "available_cata_variables": (
            cata_columns
        ),
        "attributes": attributes,
        "products": products,
        "consumers": consumers,
        "original_valid_evaluations": int(
            len(raw_response_data)
        ),
        "consolidated_evaluations": int(
            len(response_data)
        ),
        "consolidated_row_count": (
            consolidated_row_count
        ),
        "consolidated_pair_count": (
            consolidated_pair_count
        ),
        "excluded_identifier_rows": (
            excluded_identifier_rows
        ),
        "duplicate_count": duplicate_count,
        "duplicate_pair_count": (
            duplicate_pair_count
        ),
        "empty_cata_responses": (
            empty_cata_responses
        ),
        "separator": separator,
    }

def calculate_cata_frequencies(
    *,
    binary_data: pd.DataFrame,
    attributes: list[str],
) -> list[dict[str, Any]]:

    if binary_data.empty or not attributes:
        return []

    results = []

    grouped = binary_data.groupby(
        "sample",
        sort=False,
        dropna=False,
        observed=True,
    )

    for sample, product_data in grouped:
        valid_evaluations = int(
            len(product_data)
        )

        valid_consumers = int(
            product_data[
                "consumer"
            ].nunique()
        )

        for attribute in attributes:
            mentions = int(
                product_data[
                    attribute
                ].sum()
            )

            selection_percentage = (
                mentions
                / valid_evaluations
                * 100
                if valid_evaluations > 0
                else 0.0
            )

            results.append(
                {
                    "sample": str(sample),
                    "attribute": attribute,
                    "mentions": mentions,
                    "valid_evaluations": (
                        valid_evaluations
                    ),
                    "valid_consumers": (
                        valid_consumers
                    ),
                    "selection_percentage": (
                        selection_percentage
                    ),
                }
            )

    return sorted(
        results,
        key=lambda result: (
            result["attribute"].casefold(),
            -result[
                "selection_percentage"
            ],
            result["sample"].casefold(),
        ),
    )

def _format_rank_groups(
    ranked_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Group equal percentages and build a compact product ranking."""

    groups: list[dict[str, Any]] = []

    for result in ranked_results:
        percentage = float(result["selection_percentage"])

        if (
            groups
            and abs(groups[-1]["percentage"] - percentage) < 1e-12
        ):
            groups[-1]["products"].append(result["sample"])
        else:
            groups.append(
                {
                    "percentage": percentage,
                    "products": [result["sample"]],
                }
            )

    ranking_parts = []
    for group in groups:
        products_text = " = ".join(group["products"])
        ranking_parts.append(
            f"{products_text} ({group['percentage']:.1f}%)"
        )

    return groups, " > ".join(ranking_parts)


def build_product_summary(
    frequency_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    grouped_results = defaultdict(list)

    for result in frequency_results:
        grouped_results[result["sample"]].append(result)

    summaries = []

    for sample, product_results in grouped_results.items():
        mentioned_results = [
            result
            for result in product_results
            if result["mentions"] > 0
        ]

        ranked_results = sorted(
            mentioned_results,
            key=lambda result: (
                -result["selection_percentage"],
                -result["mentions"],
                result["attribute"].casefold(),
            ),
        )

        highest_result = ranked_results[0] if ranked_results else None
        total_mentions = sum(
            result["mentions"]
            for result in product_results
        )
        valid_evaluations = max(
            (
                result["valid_evaluations"]
                for result in product_results
            ),
            default=0,
        )
        average_attributes_per_response = (
            total_mentions / valid_evaluations
            if valid_evaluations > 0
            else 0.0
        )

        top_attributes = [
            {
                "attribute": result["attribute"],
                "mentions": result["mentions"],
                "selection_percentage": result[
                    "selection_percentage"
                ],
            }
            for result in ranked_results[:3]
        ]

        top_attribute_text = ", ".join(
            (
                f"{result['attribute']} "
                f"({result['selection_percentage']:.1f}%)"
            )
            for result in top_attributes
        )

        summaries.append(
            {
                "sample": sample,
                "valid_evaluations": valid_evaluations,
                "most_selected_attribute": (
                    highest_result["attribute"]
                    if highest_result
                    else None
                ),
                "highest_selection_percentage": (
                    highest_result["selection_percentage"]
                    if highest_result
                    else 0.0
                ),
                "attributes_mentioned": len(mentioned_results),
                "total_mentions": total_mentions,
                "average_attributes_per_response": (
                    average_attributes_per_response
                ),
                "top_attributes": top_attributes,
                "top_attribute_text": top_attribute_text,
            }
        )

    return sorted(
        summaries,
        key=lambda result: result["sample"].casefold(),
    )


def build_attribute_summary(
    frequency_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    grouped_results = defaultdict(list)

    for result in frequency_results:
        grouped_results[result["attribute"]].append(result)

    summaries = []

    for attribute, attribute_results in grouped_results.items():
        ranked_results = sorted(
            attribute_results,
            key=lambda result: (
                -result["selection_percentage"],
                -result["mentions"],
                result["sample"].casefold(),
            ),
        )
        reverse_ranked_results = list(reversed(ranked_results))
        highest_result = ranked_results[0] if ranked_results else None
        lowest_result = reverse_ranked_results[0] if ranked_results else None
        total_mentions = sum(
            result["mentions"]
            for result in attribute_results
        )
        ranking_groups, ranking_text = _format_rank_groups(
            ranked_results
        )

        summaries.append(
            {
                "attribute": attribute,
                "total_mentions": total_mentions,
                "highest_product": (
                    highest_result["sample"]
                    if highest_result
                    else None
                ),
                "highest_percentage": (
                    highest_result["selection_percentage"]
                    if highest_result
                    else 0.0
                ),
                "lowest_product": (
                    lowest_result["sample"]
                    if lowest_result
                    else None
                ),
                "lowest_percentage": (
                    lowest_result["selection_percentage"]
                    if lowest_result
                    else 0.0
                ),
                "product_results": ranked_results,
                "ranking_groups": ranking_groups,
                "ranking_text": ranking_text,
            }
        )

    return sorted(
        summaries,
        key=lambda result: (
            -result["total_mentions"],
            result["attribute"].casefold(),
        ),
    )


def build_cata_chart_data(
    frequency_results: list[dict[str, Any]],
) -> dict[str, Any]:

    products = sorted(
        {result["sample"] for result in frequency_results},
        key=lambda value: value.casefold(),
    )
    attributes = sorted(
        {result["attribute"] for result in frequency_results},
        key=lambda value: value.casefold(),
    )
    percentage_lookup = {
        (result["sample"], result["attribute"]): result[
            "selection_percentage"
        ]
        for result in frequency_results
    }
    mention_lookup = {
        (result["sample"], result["attribute"]): result[
            "mentions"
        ]
        for result in frequency_results
    }

    datasets = []
    for attribute in attributes:
        datasets.append(
            {
                "label": attribute,
                "data": [
                    percentage_lookup.get((sample, attribute), 0.0)
                    for sample in products
                ],
                "mentions": [
                    mention_lookup.get((sample, attribute), 0)
                    for sample in products
                ],
            }
        )

    heatmap_rows = []
    for attribute in attributes:
        cells = []
        for sample in products:
            value = float(
                percentage_lookup.get((sample, attribute), 0.0)
            )
            opacity = 0.08 + 0.72 * (value / 100.0)
            cells.append(
                {
                    "sample": sample,
                    "value": value,
                    "mentions": int(
                        mention_lookup.get((sample, attribute), 0)
                    ),
                    "style": (
                        "background-color: rgba(52, 64, 84, "
                        f"{opacity:.3f}); "
                        f"color: {'#ffffff' if value >= 55 else '#101828'};"
                    ),
                }
            )
        heatmap_rows.append(
            {
                "attribute": attribute,
                "cells": cells,
            }
        )

    return {
        "products": products,
        "attributes": attributes,
        "datasets": datasets,
        "heatmap_rows": heatmap_rows,
    }


def _holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm step-down adjusted p-values in original order."""

    count = len(p_values)
    if count == 0:
        return []

    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [1.0] * count
    running_max = 0.0

    for rank, original_index in enumerate(order):
        candidate = min(
            1.0,
            (count - rank) * float(p_values[original_index]),
        )
        running_max = max(running_max, candidate)
        adjusted[original_index] = running_max

    return adjusted


def calculate_cochran_q(
    *,
    binary_data: pd.DataFrame,
    attributes: list[str],
    products: list[str],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Calculate Cochran's Q for each CATA attribute."""

    results = []
    product_count = len(products)

    for attribute in attributes:
        pivot = binary_data.pivot_table(
            index="consumer",
            columns="sample",
            values=attribute,
            aggfunc="max",
        )
        pivot = pivot.reindex(columns=products)
        complete = pivot.dropna()
        n_complete = int(len(complete))

        result = {
            "attribute": attribute,
            "calculable": False,
            "complete_consumers": n_complete,
            "product_count": product_count,
            "degrees_of_freedom": max(product_count - 1, 0),
            "q_statistic": None,
            "p_value": None,
            "significant": False,
            "small_sample": n_complete < 10,
            "reason": None,
        }

        if product_count < 3:
            result["reason"] = (
                "At least three products are required for Cochran's Q."
            )
            results.append(result)
            continue

        if n_complete < 2:
            result["reason"] = (
                "At least two consumers with complete product data are required."
            )
            results.append(result)
            continue

        matrix = complete.to_numpy(dtype=float)
        column_totals = matrix.sum(axis=0)
        row_totals = matrix.sum(axis=1)
        total = float(column_totals.sum())
        denominator = (
            product_count * total
            - float(np.square(row_totals).sum())
        )

        if denominator <= 0:
            q_statistic = 0.0
            p_value = 1.0
        else:
            numerator = (
                (product_count - 1)
                * (
                    product_count
                    * float(np.square(column_totals).sum())
                    - total**2
                )
            )
            q_statistic = max(0.0, numerator / denominator)
            p_value = float(
                chi2.sf(q_statistic, product_count - 1)
            )

        result.update(
            {
                "calculable": True,
                "q_statistic": float(q_statistic),
                "p_value": p_value,
                "significant": bool(p_value < alpha),
            }
        )
        results.append(result)

    return results


def calculate_pairwise_mcnemar(
    *,
    binary_data: pd.DataFrame,
    cochran_results: list[dict[str, Any]],
    products: list[str],
    alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Run exact McNemar tests for attributes significant by Cochran Q."""

    all_results = []

    for cochran_result in cochran_results:
        if not cochran_result.get("significant"):
            continue

        attribute = cochran_result["attribute"]
        pivot = binary_data.pivot_table(
            index="consumer",
            columns="sample",
            values=attribute,
            aggfunc="max",
        ).reindex(columns=products)

        attribute_results = []
        raw_p_values = []

        for product_a, product_b in combinations(products, 2):
            pair = pivot[[product_a, product_b]].dropna()
            n_paired = int(len(pair))

            if n_paired == 0:
                result = {
                    "attribute": attribute,
                    "product_a": product_a,
                    "product_b": product_b,
                    "paired_consumers": 0,
                    "a_yes_b_no": 0,
                    "a_no_b_yes": 0,
                    "discordant_pairs": 0,
                    "p_value": None,
                    "adjusted_p_value": None,
                    "significant": False,
                    "direction": "Not calculable",
                }
                attribute_results.append(result)
                continue

            a_yes_b_no = int(
                ((pair[product_a] == 1) & (pair[product_b] == 0)).sum()
            )
            a_no_b_yes = int(
                ((pair[product_a] == 0) & (pair[product_b] == 1)).sum()
            )
            discordant = a_yes_b_no + a_no_b_yes

            if discordant == 0:
                p_value = 1.0
            else:
                p_value = float(
                    binomtest(
                        min(a_yes_b_no, a_no_b_yes),
                        n=discordant,
                        p=0.5,
                        alternative="two-sided",
                    ).pvalue
                )

            if a_yes_b_no > a_no_b_yes:
                direction = f"{product_a} > {product_b}"
            elif a_no_b_yes > a_yes_b_no:
                direction = f"{product_b} > {product_a}"
            else:
                direction = "No directional difference"

            result = {
                "attribute": attribute,
                "product_a": product_a,
                "product_b": product_b,
                "paired_consumers": n_paired,
                "a_yes_b_no": a_yes_b_no,
                "a_no_b_yes": a_no_b_yes,
                "discordant_pairs": discordant,
                "p_value": p_value,
                "adjusted_p_value": None,
                "significant": False,
                "direction": direction,
            }
            attribute_results.append(result)
            raw_p_values.append(p_value)

        calculable_results = [
            result
            for result in attribute_results
            if result["p_value"] is not None
        ]
        adjusted = _holm_adjust(raw_p_values)

        for result, adjusted_p in zip(calculable_results, adjusted):
            result["adjusted_p_value"] = float(adjusted_p)
            result["significant"] = bool(adjusted_p < alpha)

        all_results.extend(attribute_results)

    return all_results


def summarize_cata_statistics(
    *,
    cochran_results: list[dict[str, Any]],
    mcnemar_results: list[dict[str, Any]],
) -> dict[str, Any]:

    calculable = [
        result
        for result in cochran_results
        if result["calculable"]
    ]
    significant = [
        result
        for result in calculable
        if result["significant"]
    ]
    significant_pairwise = [
        result
        for result in mcnemar_results
        if result["significant"]
    ]

    return {
        "attributes_analyzed": len(cochran_results),
        "attributes_calculable": len(calculable),
        "attributes_significant": len(significant),
        "attributes_not_significant": (
            len(calculable) - len(significant)
        ),
        "attributes_not_calculable": (
            len(cochran_results) - len(calculable)
        ),
        "significant_attribute_names": [
            result["attribute"]
            for result in significant
        ],
        "pairwise_comparisons": len(mcnemar_results),
        "significant_pairwise_comparisons": len(
            significant_pairwise
        ),
    }


def calculate_correspondence_analysis(
    *,
    binary_data: pd.DataFrame,
    attributes: list[str],
    products: list[str],
) -> dict[str, Any]:
    """Calculate a two-dimensional correspondence analysis map."""

    if binary_data.empty or len(products) < 2 or len(attributes) < 2:
        return {
            "available": False,
            "reason": "At least two products and two attributes are required.",
            "product_coordinates": [],
            "attribute_coordinates": [],
            "dimensions": [],
        }

    contingency = (
        binary_data.groupby("sample", observed=True)[attributes]
        .sum()
        .reindex(products)
        .fillna(0.0)
    )

    positive_rows = contingency.sum(axis=1) > 0
    positive_columns = contingency.sum(axis=0) > 0
    reduced = contingency.loc[positive_rows, positive_columns]

    if reduced.shape[0] < 2 or reduced.shape[1] < 2:
        return {
            "available": False,
            "reason": (
                "The contingency table does not contain enough non-zero "
                "rows and columns."
            ),
            "product_coordinates": [],
            "attribute_coordinates": [],
            "dimensions": [],
        }

    matrix = reduced.to_numpy(dtype=float)
    grand_total = float(matrix.sum())

    if grand_total <= 0:
        return {
            "available": False,
            "reason": "No positive CATA mentions are available.",
            "product_coordinates": [],
            "attribute_coordinates": [],
            "dimensions": [],
        }

    proportions = matrix / grand_total
    row_masses = proportions.sum(axis=1)
    column_masses = proportions.sum(axis=0)
    expected = np.outer(row_masses, column_masses)
    standardized = (
        (proportions - expected)
        / np.sqrt(expected)
    )

    u_matrix, singular_values, vt_matrix = np.linalg.svd(
        standardized,
        full_matrices=False,
    )
    eigenvalues = np.square(singular_values)
    total_inertia = float(eigenvalues.sum())
    max_dimensions = min(2, len(singular_values))

    if total_inertia <= 1e-15 or max_dimensions == 0:
        return {
            "available": False,
            "reason": "The profiles contain no correspondence variation.",
            "product_coordinates": [],
            "attribute_coordinates": [],
            "dimensions": [],
        }

    row_coordinates = (
        np.diag(1.0 / np.sqrt(row_masses))
        @ u_matrix[:, :max_dimensions]
        @ np.diag(singular_values[:max_dimensions])
    )
    column_coordinates = (
        np.diag(1.0 / np.sqrt(column_masses))
        @ vt_matrix.T[:, :max_dimensions]
        @ np.diag(singular_values[:max_dimensions])
    )

    dimensions = []
    for index in range(max_dimensions):
        dimensions.append(
            {
                "dimension": index + 1,
                "eigenvalue": float(eigenvalues[index]),
                "explained_inertia": float(
                    eigenvalues[index] / total_inertia * 100
                ),
            }
        )

    def coordinate_value(array: np.ndarray, row: int, col: int) -> float:
        if col >= array.shape[1]:
            return 0.0
        return float(array[row, col])

    product_coordinates = [
        {
            "label": str(label),
            "x": coordinate_value(row_coordinates, index, 0),
            "y": coordinate_value(row_coordinates, index, 1),
            "type": "product",
        }
        for index, label in enumerate(reduced.index)
    ]
    attribute_coordinates = [
        {
            "label": str(label),
            "x": coordinate_value(column_coordinates, index, 0),
            "y": coordinate_value(column_coordinates, index, 1),
            "type": "attribute",
        }
        for index, label in enumerate(reduced.columns)
    ]

    return {
        "available": True,
        "reason": None,
        "total_inertia": total_inertia,
        "dimensions": dimensions,
        "dimension_1_percentage": dimensions[0]["explained_inertia"],
        "dimension_2_percentage": (
            dimensions[1]["explained_inertia"]
            if len(dimensions) > 1
            else 0.0
        ),
        "displayed_inertia": float(
            sum(
                dimension["explained_inertia"]
                for dimension in dimensions[:2]
            )
        ),
        "interpretation_note": (
            "The first two dimensions explain "
            f"{sum(dimension['explained_inertia'] for dimension in dimensions[:2]):.1f}% "
            "of the total inertia, providing a complete two-dimensional "
            "representation of the product–attribute structure."
            if sum(dimension["explained_inertia"] for dimension in dimensions[:2]) >= 99.95
            else
            "The first two dimensions explain "
            f"{sum(dimension['explained_inertia'] for dimension in dimensions[:2]):.1f}% "
            "of the total inertia. The map therefore represents most, but not all, "
            "of the product–attribute structure."
        ),
        "product_coordinates": product_coordinates,
        "attribute_coordinates": attribute_coordinates,
        "excluded_products": [
            str(label)
            for label in contingency.index[~positive_rows]
        ],
        "excluded_attributes": [
            str(label)
            for label in contingency.columns[~positive_columns]
        ],
        "contingency_table": [
            {
                "sample": str(sample),
                "values": [
                    {
                        "attribute": str(attribute),
                        "mentions": int(contingency.loc[sample, attribute]),
                    }
                    for attribute in contingency.columns
                ],
            }
            for sample in contingency.index
        ],
    }



def calculate_cata_liking_drivers(
    *,
    binary_data: pd.DataFrame,
    attributes: list[str],
    alpha: float = 0.05,
    minimum_group_size: int = 2,
) -> dict[str, Any]:
    """Estimate CATA drivers of liking from selected vs not-selected groups.

    The lift is mean liking when the attribute was selected minus mean liking
    when it was not selected. Welch tests are descriptive/exploratory because
    CATA observations are repeated within consumers and products. Holm
    adjustment is applied across calculable attributes.
    """

    if "liking" not in binary_data.columns:
        return {
            "available": False,
            "reason": (
                "Map exactly one Overall Liking variable to calculate "
                "CATA drivers of liking."
            ),
            "results": [],
            "chart_data": {},
            "summary": {},
        }

    data = binary_data.copy()
    data["liking"] = pd.to_numeric(data["liking"], errors="coerce")
    data = data.dropna(subset=["liking"])

    if data.empty:
        return {
            "available": False,
            "reason": "No valid liking values are available for CATA analysis.",
            "results": [],
            "chart_data": {},
            "summary": {},
        }

    rows: list[dict[str, Any]] = []
    calculable_indices: list[int] = []
    raw_p_values: list[float] = []

    for attribute in attributes:
        selected = data.loc[data[attribute] == 1, "liking"].dropna()
        not_selected = data.loc[data[attribute] == 0, "liking"].dropna()
        selected_n = int(len(selected))
        not_selected_n = int(len(not_selected))
        total_n = selected_n + not_selected_n
        selected_mean = float(selected.mean()) if selected_n else None
        not_selected_mean = float(not_selected.mean()) if not_selected_n else None
        lift = (
            selected_mean - not_selected_mean
            if selected_mean is not None and not_selected_mean is not None
            else None
        )
        selected_percentage = (selected_n / total_n * 100) if total_n else 0.0
        calculable = (
            selected_n >= minimum_group_size
            and not_selected_n >= minimum_group_size
            and lift is not None
        )
        p_value = None
        reason = None
        if calculable:
            test = ttest_ind(
                selected.to_numpy(dtype=float),
                not_selected.to_numpy(dtype=float),
                equal_var=False,
                nan_policy="omit",
            )
            p_value = float(test.pvalue) if np.isfinite(test.pvalue) else 1.0
        else:
            reason = (
                f"At least {minimum_group_size} liking observations are required "
                "in both the selected and not-selected groups."
            )

        if lift is None:
            direction = "Not estimable"
        elif not calculable:
            direction = "Insufficient observations"
        elif lift > 0:
            direction = "Positive association"
        elif lift < 0:
            direction = "Negative association"
        else:
            direction = "Neutral association"

        row = {
            "attribute": attribute,
            "selected_n": selected_n,
            "not_selected_n": not_selected_n,
            "total_n": total_n,
            "selected_percentage": float(selected_percentage),
            "selected_mean_liking": selected_mean,
            "not_selected_mean_liking": not_selected_mean,
            "liking_lift": float(lift) if lift is not None else None,
            "liking_lift_display": (
                f"{float(lift):+.2f}".replace("-", "−")
                if lift is not None else None
            ),
            "absolute_lift": abs(float(lift)) if lift is not None else None,
            "direction": direction,
            "calculable": calculable,
            "p_value": p_value,
            "adjusted_p_value": None,
            "significant": False,
            "reason": reason,
        }
        rows.append(row)
        if calculable and p_value is not None:
            calculable_indices.append(len(rows) - 1)
            raw_p_values.append(p_value)

    adjusted = _holm_adjust(raw_p_values)
    for index, adjusted_p in zip(calculable_indices, adjusted):
        row = rows[index]
        row["adjusted_p_value"] = float(adjusted_p)
        row["significant"] = bool(adjusted_p < alpha)

        if row["significant"]:
            if row["liking_lift"] > 0:
                row["direction"] = "Significant positive driver"
            elif row["liking_lift"] < 0:
                row["direction"] = "Significant negative driver"
            else:
                row["direction"] = "No directional association"

    rows.sort(
        key=lambda row: (
            -(row["absolute_lift"] if row["absolute_lift"] is not None else -1),
            row["attribute"].casefold(),
        )
    )
    positive = [
        row for row in rows
        if row["liking_lift"] is not None and row["liking_lift"] > 0
    ]
    negative = [
        row for row in rows
        if row["liking_lift"] is not None and row["liking_lift"] < 0
    ]
    significant = [row for row in rows if row["significant"]]
    significant_positive = [
        row for row in significant if row["liking_lift"] > 0
    ]
    significant_negative = [
        row for row in significant if row["liking_lift"] < 0
    ]

    chart_rows = [row for row in rows if row["liking_lift"] is not None]
    chart_data = {
        "points": [
            {
                "x": row["selected_percentage"],
                "y": row["liking_lift"],
                "label": row["attribute"],
                "direction": row["direction"],
                "selected_n": row["selected_n"],
                "not_selected_n": row["not_selected_n"],
                "adjusted_p_value": row["adjusted_p_value"],
            }
            for row in chart_rows
        ]
    }

    return {
        "available": True,
        "reason": None,
        "results": rows,
        "chart_data": chart_data,
        "summary": {
            "attributes_analyzed": len(rows),
            "attributes_evaluated": len(rows),
            "attributes_calculable": len(calculable_indices),
            "attributes_with_test": len(calculable_indices),
            "positive_associations": len(positive),
            "negative_associations": len(negative),
            "positive_drivers": len(significant_positive),
            "negative_drivers": len(significant_negative),
            "significant_drivers": len(significant),
            "insufficient_observations": sum(
                1 for row in rows if not row["calculable"]
            ),
            "strongest_positive": positive[0] if positive else None,
            "strongest_negative": negative[0] if negative else None,
            "strongest_significant_positive": (
                significant_positive[0] if significant_positive else None
            ),
            "strongest_significant_negative": (
                significant_negative[0] if significant_negative else None
            ),
        },
        "method_note": (
            "Liking lift equals mean liking when selected minus mean liking when "
            "not selected. Welch tests and Holm-adjusted p-values are exploratory "
            "because repeated observations are not modeled in this simple driver view."
        ),
    }

def build_cata_interpretation(
    *,
    product_summaries: list[dict[str, Any]],
    attribute_summaries: list[dict[str, Any]],
    response_data: pd.DataFrame,
    statistics_summary: dict[str, Any],
    cochran_results: list[dict[str, Any]],
    mcnemar_results: list[dict[str, Any]],
    duplicate_count: int = 0,
    consolidated_row_count: int = 0,
    liking_drivers: dict[str, Any] | None = None,
) -> dict[str, Any]:

    profile_statements = []
    statistical_statements = []
    methodology_statements = []

    def natural_language_list(items: list[str]) -> str:
        """Join labels using natural English punctuation."""
        cleaned = [str(item) for item in items if str(item).strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) == 2:
            return f"{cleaned[0]} and {cleaned[1]}"
        return ", ".join(cleaned[:-1]) + f", and {cleaned[-1]}"

    for summary in product_summaries:
        top = summary.get("top_attributes", [])
        sample = summary["sample"]

        if not top:
            profile_statements.append(
                f"No CATA attributes were selected for product {sample}."
            )
            continue

        primary = top[0]
        secondary = top[1:]
        statement = (
            f"Product {sample} was primarily associated with "
            f"{primary['attribute']} "
            f"({primary['selection_percentage']:.1f}%)"
        )

        if secondary:
            secondary_text = " and ".join(
                item["attribute"]
                for item in secondary
            )
            statement += (
                f", with {secondary_text} as secondary descriptors"
            )

        profile_statements.append(statement + ".")

    significant_attributes = statistics_summary[
        "significant_attribute_names"
    ]

    if significant_attributes:
        attributes_text = natural_language_list(significant_attributes)
        significant_pair_count = int(
            statistics_summary.get(
                "significant_pairwise_comparisons",
                0,
            )
        )

        if significant_pair_count == 0:
            statistical_statements.append(
                f"{attributes_text} showed significant overall differences "
                "among products according to Cochran's Q. However, no "
                "individual product comparison remained significant after "
                "Holm adjustment. This pattern should be interpreted "
                "cautiously when the complete sample is small."
            )
        else:
            statistical_statements.append(
                f"{len(significant_attributes)} of "
                f"{statistics_summary['attributes_calculable']} calculable "
                "attributes showed significant overall product differences: "
                f"{attributes_text}."
            )

            for attribute in significant_attributes:
                significant_pairs = [
                    result
                    for result in mcnemar_results
                    if (
                        result["attribute"] == attribute
                        and result["significant"]
                    )
                ]
                if significant_pairs:
                    pair_text = "; ".join(
                        (
                            f"{result['direction']} "
                            f"(Holm-adjusted p="
                            f"{result['adjusted_p_value']:.4f})"
                        )
                        for result in significant_pairs
                    )
                    statistical_statements.append(
                        f"For {attribute}, significant pairwise differences "
                        f"were: {pair_text}."
                    )
                else:
                    statistical_statements.append(
                        f"{attribute} differed overall by Cochran's Q, but no "
                        "individual product pair remained significant after "
                        "Holm adjustment."
                    )
    elif statistics_summary["attributes_calculable"] > 0:
        statistical_statements.append(
            "No CATA attribute showed a statistically significant "
            "difference among products at the 5% level."
        )
    else:
        statistical_statements.append(
            "Cochran's Q could not be calculated for the available data."
        )

    if duplicate_count > 0 and consolidated_row_count > 0:
        methodology_statements.append(
            f"{consolidated_row_count} repeated rows were consolidated "
            "by consumer and product using a logical OR rule for CATA "
            "attributes."
        )

    if not response_data.empty:
        empty_count = int((response_data["attribute_count"] == 0).sum())
        if empty_count > 0:
            methodology_statements.append(
                f"{empty_count} consolidated responses contained no "
                "selected CATA attributes."
            )

    small_sample_attributes = [
        result["attribute"]
        for result in cochran_results
        if result.get("calculable") and result.get("small_sample")
    ]
    if small_sample_attributes:
        methodology_statements.append(
            "Cochran's Q results are exploratory for attributes based on "
            "fewer than 10 complete consumers: "
            + ", ".join(small_sample_attributes)
            + "."
        )

    if liking_drivers and liking_drivers.get("available"):
        driver_summary = liking_drivers.get("summary", {})
        strongest_positive = driver_summary.get(
            "strongest_significant_positive"
        ) or driver_summary.get("strongest_positive")
        strongest_negative = driver_summary.get(
            "strongest_significant_negative"
        ) or driver_summary.get("strongest_negative")

        if strongest_positive:
            if strongest_positive.get("significant"):
                statistical_statements.append(
                    f"{strongest_positive['attribute']} showed the strongest "
                    "significant positive association with liking and remained "
                    "significant after Holm adjustment, with an estimated liking "
                    f"lift of {strongest_positive['liking_lift']:+.2f} points."
                )
            else:
                statistical_statements.append(
                    f"The strongest positive descriptive liking association "
                    f"was {strongest_positive['attribute']} "
                    f"(lift {strongest_positive['liking_lift']:+.2f}), but it "
                    "did not remain significant after Holm adjustment."
                )

        if strongest_negative:
            if strongest_negative.get("significant"):
                statistical_statements.append(
                    f"{strongest_negative['attribute']} showed the strongest "
                    "significant negative association with liking, with an "
                    "estimated penalty of "
                    f"{strongest_negative['liking_lift']:.2f}".replace("-", "−")
                    + " points."
                )
            else:
                statistical_statements.append(
                    f"The strongest negative descriptive liking association "
                    f"was {strongest_negative['attribute']} "
                    f"(lift {strongest_negative['liking_lift']:+.2f}), but it "
                    "did not remain significant after Holm adjustment."
                )

        methodology_statements.append(liking_drivers.get("method_note", ""))

    methodology_statements.append(
        "Cochran's Q used consumers with complete observations across all "
        "products for each attribute. Exact McNemar tests were performed "
        "only after a significant omnibus test, with Holm adjustment "
        "within each attribute."
    )

    statements = (
        profile_statements
        + statistical_statements
        + methodology_statements
    )

    return {
        "profile_statements": profile_statements,
        "statistical_statements": statistical_statements,
        "methodology_statements": methodology_statements,
        "statements": statements,
        "summary": " ".join(statements),
    }


def run_cata_analysis(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    configuration: dict | None = None,
    selected_cata_variables: list[str] | None = None,
) -> dict[str, Any]:

    configuration = configuration or {}
    cata_configuration = configuration.get("cata", {})
    separator = cata_configuration.get(
        "separator",
        DEFAULT_CATA_SEPARATOR,
    )
    if not isinstance(separator, str) or not separator:
        separator = DEFAULT_CATA_SEPARATOR

    alpha = cata_configuration.get("alpha", 0.05)
    try:
        alpha = float(alpha)
    except (TypeError, ValueError):
        alpha = 0.05
    if not 0 < alpha < 1:
        alpha = 0.05

    prepared = prepare_cata_data(
        datasets=datasets,
        mapping=mapping,
        selected_cata_variables=selected_cata_variables,
        separator=separator,
    )
    if not prepared["is_valid"]:
        return prepared

    binary_data = prepared["binary_data"]
    attributes = prepared["attributes"]
    products = prepared["products"]

    frequency_results = calculate_cata_frequencies(
        binary_data=binary_data,
        attributes=attributes,
    )
    product_summaries = build_product_summary(frequency_results)
    attribute_summaries = build_attribute_summary(frequency_results)
    chart_data = build_cata_chart_data(frequency_results)

    cochran_results = calculate_cochran_q(
        binary_data=binary_data,
        attributes=attributes,
        products=products,
        alpha=alpha,
    )
    mcnemar_results = calculate_pairwise_mcnemar(
        binary_data=binary_data,
        cochran_results=cochran_results,
        products=products,
        alpha=alpha,
    )
    statistics_summary = summarize_cata_statistics(
        cochran_results=cochran_results,
        mcnemar_results=mcnemar_results,
    )
    correspondence_analysis = calculate_correspondence_analysis(
        binary_data=binary_data,
        attributes=attributes,
        products=products,
    )
    liking_drivers = calculate_cata_liking_drivers(
        binary_data=binary_data,
        attributes=attributes,
        alpha=alpha,
    )

    interpretation = build_cata_interpretation(
        product_summaries=product_summaries,
        attribute_summaries=attribute_summaries,
        response_data=prepared["response_data"],
        statistics_summary=statistics_summary,
        cochran_results=cochran_results,
        mcnemar_results=mcnemar_results,
        duplicate_count=prepared["duplicate_count"],
        consolidated_row_count=prepared["consolidated_row_count"],
        liking_drivers=liking_drivers,
    )

    valid_evaluations = int(len(prepared["response_data"]))
    valid_consumers = int(len(prepared["consumers"]))
    total_mentions = int(
        sum(result["mentions"] for result in frequency_results)
    )
    average_attributes_per_evaluation = (
        total_mentions / valid_evaluations
        if valid_evaluations > 0
        else 0.0
    )

    return {
        "is_valid": True,
        "alpha": alpha,
        "consumer_column": prepared["consumer_column"],
        "sample_column": prepared["sample_column"],
        "liking_column": prepared["liking_column"],
        "cata_variables": prepared["cata_variables"],
        "available_cata_variables": prepared[
            "available_cata_variables"
        ],
        "separator": prepared["separator"],
        "attributes": attributes,
        "products": products,
        "consumers": prepared["consumers"],
        "attribute_count": int(len(attributes)),
        "product_count": int(len(products)),
        "consumer_count": valid_consumers,
        "valid_evaluations": valid_evaluations,
        "total_mentions": total_mentions,
        "average_attributes_per_evaluation": (
            average_attributes_per_evaluation
        ),
        "frequency_results": frequency_results,
        "product_summaries": product_summaries,
        "attribute_summaries": attribute_summaries,
        "chart_data": chart_data,
        "cochran_results": cochran_results,
        "mcnemar_results": mcnemar_results,
        "statistics_summary": statistics_summary,
        "correspondence_analysis": correspondence_analysis,
        "liking_drivers": liking_drivers,
        "original_valid_evaluations": prepared[
            "original_valid_evaluations"
        ],
        "consolidated_evaluations": prepared[
            "consolidated_evaluations"
        ],
        "consolidated_row_count": prepared[
            "consolidated_row_count"
        ],
        "consolidated_pair_count": prepared[
            "consolidated_pair_count"
        ],
        "interpretation": interpretation,
        "excluded_identifier_rows": prepared[
            "excluded_identifier_rows"
        ],
        "duplicate_count": prepared["duplicate_count"],
        "duplicate_pair_count": prepared["duplicate_pair_count"],
        "empty_cata_responses": prepared["empty_cata_responses"],
        "data_quality": {
            "excluded_identifier_rows": prepared[
                "excluded_identifier_rows"
            ],
            "duplicate_consumer_product_rows": prepared[
                "duplicate_count"
            ],
            "duplicate_consumer_product_pairs": prepared[
                "duplicate_pair_count"
            ],
            "consolidated_rows": prepared[
                "consolidated_row_count"
            ],
            "consolidated_pairs": prepared[
                "consolidated_pair_count"
            ],
            "empty_cata_responses": prepared[
                "empty_cata_responses"
            ],
        },
    }