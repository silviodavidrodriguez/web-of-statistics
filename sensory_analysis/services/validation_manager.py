from __future__ import annotations
from typing import Any
import pandas as pd

BLOCKING_LEVEL = "error"
WARNING_LEVEL = "warning"
INFO_LEVEL = "info"

def add_validation_item(
    items: list[dict[str, Any]],
    *,
    code: str,
    title: str,
    message: str,
    level: str,
    dataset: str | None = None,
    variable: str | None = None,
    count: int | None = None,
) -> None:
    items.append(
        {
            "code": code,
            "title": title,
            "message": message,
            "level": level,
            "dataset": dataset,
            "variable": variable,
            "count": count,
        }
    )

def get_mapped_column(
    mapping: dict[str, dict[str, str]],
    dataset_name: str,
    role: str,
) -> str | None:
    dataset_mapping = mapping.get(dataset_name, {})

    matches = [
        column_name
        for column_name, assigned_role in dataset_mapping.items()
        if assigned_role == role
    ]

    if len(matches) == 1:
        return matches[0]

    return None

def get_columns_by_role(
    mapping: dict[str, dict[str, str]],
    role: str,
) -> list[dict[str, str]]:
    results = []

    for dataset_name, dataset_mapping in mapping.items():
        for column_name, assigned_role in dataset_mapping.items():
            if assigned_role == role:
                results.append(
                    {
                        "dataset": dataset_name,
                        "column": column_name,
                    }
                )

    return results

def normalize_identifier_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

def validate_required_datasets(
    datasets: dict[str, pd.DataFrame],
    items: list[dict[str, Any]],
) -> None:
    evaluations = datasets.get("evaluations")

    if evaluations is None:
        add_validation_item(
            items,
            code="missing_evaluations",
            title="Sensory evaluations are missing",
            message=(
                "The sensory evaluations dataset is required "
                "to continue."
            ),
            level=BLOCKING_LEVEL,
            dataset="evaluations",
        )
        return

    if evaluations.empty:
        add_validation_item(
            items,
            code="empty_evaluations",
            title="Sensory evaluations are empty",
            message=(
                "The sensory evaluations dataset does not contain "
                "any records."
            ),
            level=BLOCKING_LEVEL,
            dataset="evaluations",
        )

def validate_required_roles(
    mapping: dict[str, dict[str, str]],
    items: list[dict[str, Any]],
) -> None:
    consumer_column = get_mapped_column(
        mapping,
        "evaluations",
        "consumer_id",
    )

    sample_column = get_mapped_column(
        mapping,
        "evaluations",
        "sample_id",
    )

    if not consumer_column:
        add_validation_item(
            items,
            code="missing_consumer_id",
            title="Consumer identifier is not defined",
            message=(
                "Assign exactly one Consumer ID variable in the "
                "sensory evaluations dataset."
            ),
            level=BLOCKING_LEVEL,
            dataset="evaluations",
        )

    if not sample_column:
        add_validation_item(
            items,
            code="missing_sample_id",
            title="Sample identifier is not defined",
            message=(
                "Assign exactly one Sample ID variable in the "
                "sensory evaluations dataset."
            ),
            level=BLOCKING_LEVEL,
            dataset="evaluations",
        )

    response_roles = {
        "liking",
        "jar",
        "cata",
        "purchase_intention",
    }

    mapped_response_variables = []

    for role in response_roles:
        mapped_response_variables.extend(
            get_columns_by_role(mapping, role)
        )

    if not mapped_response_variables:
        add_validation_item(
            items,
            code="missing_response_variables",
            title="No sensory response variables were found",
            message=(
                "Map at least one liking, JAR, CATA or purchase "
                "intention variable."
            ),
            level=BLOCKING_LEVEL,
            dataset="evaluations",
        )

def validate_missing_identifiers(
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    items: list[dict[str, Any]],
) -> None:
    evaluations = datasets.get("evaluations")

    if evaluations is None or evaluations.empty:
        return

    for role, label in (
        ("consumer_id", "consumer"),
        ("sample_id", "sample"),
    ):
        column_name = get_mapped_column(
            mapping,
            "evaluations",
            role,
        )

        if not column_name or column_name not in evaluations.columns:
            continue

        normalized = normalize_identifier_series(
            evaluations[column_name]
        )

        missing_count = int(normalized.isna().sum())

        if missing_count:
            add_validation_item(
                items,
                code=f"missing_{role}_values",
                title=f"Missing {label} identifiers",
                message=(
                    f"{missing_count} sensory evaluation records "
                    f"do not contain a valid {label} identifier."
                ),
                level=BLOCKING_LEVEL,
                dataset="evaluations",
                variable=column_name,
                count=missing_count,
            )

def validate_duplicate_evaluations(
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    items: list[dict[str, Any]],
) -> None:
    evaluations = datasets.get("evaluations")

    if evaluations is None or evaluations.empty:
        return

    consumer_column = get_mapped_column(
        mapping,
        "evaluations",
        "consumer_id",
    )
    sample_column = get_mapped_column(
        mapping,
        "evaluations",
        "sample_id",
    )

    if not consumer_column or not sample_column:
        return

    if (
        consumer_column not in evaluations.columns
        or sample_column not in evaluations.columns
    ):
        return

    duplicate_mask = evaluations.duplicated(
        subset=[
            consumer_column,
            sample_column,
        ],
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count:
        add_validation_item(
            items,
            code="duplicate_consumer_sample",
            title="Repeated consumer–sample evaluations",
            message=(
                f"{duplicate_count} rows share the same consumer "
                "and sample combination. Confirm whether repeated "
                "evaluations are intentional."
            ),
            level=WARNING_LEVEL,
            dataset="evaluations",
            count=duplicate_count,
        )

def validate_numeric_scale(
    dataframe: pd.DataFrame,
    column_name: str,
    minimum: int | float | None,
    maximum: int | float | None,
    items: list[dict[str, Any]],
    *,
    dataset_name: str,
    role_label: str,
) -> None:
    if column_name not in dataframe.columns:
        add_validation_item(
            items,
            code="configured_column_missing",
            title="Configured variable is missing",
            message=(
                f"The configured variable {column_name} was not "
                f"found in {dataset_name}."
            ),
            level=BLOCKING_LEVEL,
            dataset=dataset_name,
            variable=column_name,
        )
        return

    numeric = pd.to_numeric(
        dataframe[column_name]
        .astype("string")
        .str.replace(",", ".", regex=False),
        errors="coerce",
    )

    original_non_blank = (
        dataframe[column_name]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
        .notna()
    )

    invalid_numeric_count = int(
        (original_non_blank & numeric.isna()).sum()
    )

    if invalid_numeric_count:
        add_validation_item(
            items,
            code="non_numeric_scale_values",
            title=f"Non-numeric values in {role_label}",
            message=(
                f"{invalid_numeric_count} values in {column_name} "
                "could not be interpreted as numbers."
            ),
            level=BLOCKING_LEVEL,
            dataset=dataset_name,
            variable=column_name,
            count=invalid_numeric_count,
        )

    if minimum is None or maximum is None:
        return

    outside_mask = numeric.notna() & (
        (numeric < minimum)
        | (numeric > maximum)
    )

    outside_count = int(outside_mask.sum())

    if outside_count:
        add_validation_item(
            items,
            code="values_outside_scale",
            title=f"Values outside the configured {role_label} scale",
            message=(
                f"{outside_count} values in {column_name} are "
                f"outside the configured range "
                f"{minimum}–{maximum}."
            ),
            level=BLOCKING_LEVEL,
            dataset=dataset_name,
            variable=column_name,
            count=outside_count,
        )

def validate_configured_scales(
    datasets: dict[str, pd.DataFrame],
    configuration: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    for section_name, role_label in (
        ("liking", "liking"),
        ("jar", "JAR"),
        ("purchase_intention", "purchase intention"),
    ):
        for column_name, values in configuration.get(
            section_name,
            {},
        ).items():
            dataset_name = values.get(
                "dataset",
                "evaluations",
            )

            dataframe = datasets.get(dataset_name)

            if dataframe is None:
                continue

            validate_numeric_scale(
                dataframe=dataframe,
                column_name=column_name,
                minimum=values.get("minimum"),
                maximum=values.get("maximum"),
                items=items,
                dataset_name=dataset_name,
                role_label=role_label,
            )

def validate_jar_ideal_values(
    configuration: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
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
            continue

        if not minimum <= ideal_value <= maximum:
            add_validation_item(
                items,
                code="jar_ideal_outside_range",
                title="Invalid JAR ideal value",
                message=(
                    f"The ideal value configured for {column_name} "
                    "is outside its scale range."
                ),
                level=BLOCKING_LEVEL,
                dataset=values.get("dataset"),
                variable=column_name,
            )

def validate_cata_variables(
    datasets: dict[str, pd.DataFrame],
    configuration: dict[str, Any],
    items: list[dict[str, Any]],
) -> None:
    for column_name, values in configuration.get(
        "cata",
        {},
    ).items():
        dataset_name = values.get(
            "dataset",
            "evaluations",
        )
        dataframe = datasets.get(dataset_name)

        if dataframe is None or column_name not in dataframe.columns:
            continue

        separator = values.get("separator", ";")

        non_blank = (
            dataframe[column_name]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
            .dropna()
        )

        if non_blank.empty:
            add_validation_item(
                items,
                code="empty_cata_variable",
                title="CATA variable contains no selections",
                message=(
                    f"The variable {column_name} has no usable "
                    "attribute selections."
                ),
                level=WARNING_LEVEL,
                dataset=dataset_name,
                variable=column_name,
            )
            continue

        rows_with_separator = non_blank.str.contains(
            separator,
            regex=False,
        ).sum()

        unique_values = non_blank.nunique()

        if rows_with_separator == 0 and unique_values > 1:
            add_validation_item(
                items,
                code="cata_separator_not_found",
                title="CATA separator was not detected",
                message=(
                    f"The separator '{separator}' was not found in "
                    f"{column_name}. Verify that the selected "
                    "separator is correct."
                ),
                level=WARNING_LEVEL,
                dataset=dataset_name,
                variable=column_name,
            )

def validate_consumer_dataset_links(
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    items: list[dict[str, Any]],
) -> None:
    evaluations = datasets.get("evaluations")
    consumers = datasets.get("consumers")

    if (
        evaluations is None
        or consumers is None
        or consumers.empty
    ):
        return

    evaluations_consumer_column = get_mapped_column(
        mapping,
        "evaluations",
        "consumer_id",
    )
    consumers_consumer_column = get_mapped_column(
        mapping,
        "consumers",
        "consumer_id",
    )

    if not evaluations_consumer_column or not consumers_consumer_column:
        return

    evaluation_ids = set(
        normalize_identifier_series(
            evaluations[evaluations_consumer_column]
        ).dropna()
    )

    consumer_ids = normalize_identifier_series(
        consumers[consumers_consumer_column]
    )

    duplicate_consumer_count = int(
        consumer_ids.duplicated(keep=False).sum()
    )

    if duplicate_consumer_count:
        add_validation_item(
            items,
            code="duplicate_consumer_information",
            title="Repeated consumers in consumer information",
            message=(
                f"{duplicate_consumer_count} rows contain repeated "
                "consumer identifiers. Consumer information should "
                "normally contain one row per consumer."
            ),
            level=BLOCKING_LEVEL,
            dataset="consumers",
            variable=consumers_consumer_column,
            count=duplicate_consumer_count,
        )

    consumer_id_set = set(consumer_ids.dropna())

    missing_information_ids = (
        evaluation_ids - consumer_id_set
    )

    if missing_information_ids:
        add_validation_item(
            items,
            code="consumers_without_information",
            title="Consumers without consumer information",
            message=(
                f"{len(missing_information_ids)} consumers present "
                "in evaluations are missing from the consumer "
                "information dataset."
            ),
            level=WARNING_LEVEL,
            dataset="consumers",
            count=len(missing_information_ids),
        )

    unused_information_ids = (
        consumer_id_set - evaluation_ids
    )

    if unused_information_ids:
        add_validation_item(
            items,
            code="unused_consumer_information",
            title="Unused consumer information records",
            message=(
                f"{len(unused_information_ids)} consumers in the "
                "consumer information dataset do not appear in "
                "sensory evaluations."
            ),
            level=INFO_LEVEL,
            dataset="consumers",
            count=len(unused_information_ids),
        )

def validate_general_dataset_links(
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    items: list[dict[str, Any]],
) -> None:
    evaluations = datasets.get("evaluations")
    general = datasets.get("general")

    if (
        evaluations is None
        or general is None
        or general.empty
    ):
        return

    evaluations_consumer_column = get_mapped_column(
        mapping,
        "evaluations",
        "consumer_id",
    )
    general_consumer_column = get_mapped_column(
        mapping,
        "general",
        "consumer_id",
    )

    if not evaluations_consumer_column or not general_consumer_column:
        return

    evaluation_ids = set(
        normalize_identifier_series(
            evaluations[evaluations_consumer_column]
        ).dropna()
    )

    general_ids = normalize_identifier_series(
        general[general_consumer_column]
    )

    duplicate_count = int(
        general_ids.duplicated(keep=False).sum()
    )

    if duplicate_count:
        add_validation_item(
            items,
            code="duplicate_general_responses",
            title="Repeated consumers in general questions",
            message=(
                f"{duplicate_count} general-question rows contain "
                "repeated consumer identifiers."
            ),
            level=WARNING_LEVEL,
            dataset="general",
            variable=general_consumer_column,
            count=duplicate_count,
        )

    missing_ids = evaluation_ids - set(
        general_ids.dropna()
    )

    if missing_ids:
        add_validation_item(
            items,
            code="consumers_without_general_responses",
            title="Consumers without general-question responses",
            message=(
                f"{len(missing_ids)} consumers evaluated products "
                "but do not have general study responses."
            ),
            level=INFO_LEVEL,
            dataset="general",
            count=len(missing_ids),
        )

def build_analysis_availability(
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    configuration: dict[str, Any],
) -> list[dict[str, Any]]:
    analyses = []

    liking_variables = get_columns_by_role(
        mapping,
        "liking",
    )
    jar_variables = get_columns_by_role(
        mapping,
        "jar",
    )
    cata_variables = get_columns_by_role(
        mapping,
        "cata",
    )
    purchase_variables = get_columns_by_role(
        mapping,
        "purchase_intention",
    )

    analyses.append(
        {
            "name": "Liking analysis",
            "available": bool(liking_variables),
            "reason": (
                "At least one overall liking variable was mapped."
                if liking_variables
                else "No overall liking variable was mapped."
            ),
        }
    )

    analyses.append(
        {
            "name": "JAR penalty analysis",
            "available": bool(
                jar_variables and liking_variables
            ),
            "reason": (
                "JAR and liking variables are available."
                if jar_variables and liking_variables
                else "JAR penalty analysis requires both JAR and liking."
            ),
        }
    )

    analyses.append(
        {
            "name": "CATA analysis",
            "available": bool(cata_variables),
            "reason": (
                "At least one CATA variable was mapped."
                if cata_variables
                else "No CATA variable was mapped."
            ),
        }
    )

    analyses.append(
        {
            "name": "Purchase intention analysis",
            "available": bool(purchase_variables),
            "reason": (
                "A purchase intention variable was mapped."
                if purchase_variables
                else "No purchase intention variable was mapped."
            ),
        }
    )

    return analyses

def run_study_validation(
    *,
    datasets: dict[str, pd.DataFrame],
    mapping: dict[str, dict[str, str]],
    configuration: dict[str, Any],
) -> dict[str, Any]:
    items: list[dict[str, Any]] = []

    validate_required_datasets(
        datasets,
        items,
    )
    validate_required_roles(
        mapping,
        items,
    )
    validate_missing_identifiers(
        datasets,
        mapping,
        items,
    )
    validate_duplicate_evaluations(
        datasets,
        mapping,
        items,
    )
    validate_configured_scales(
        datasets,
        configuration,
        items,
    )
    validate_jar_ideal_values(
        configuration,
        items,
    )
    validate_cata_variables(
        datasets,
        configuration,
        items,
    )
    validate_consumer_dataset_links(
        datasets,
        mapping,
        items,
    )
    validate_general_dataset_links(
        datasets,
        mapping,
        items,
    )

    errors = [
        item for item in items
        if item["level"] == BLOCKING_LEVEL
    ]
    warnings = [
        item for item in items
        if item["level"] == WARNING_LEVEL
    ]
    information = [
        item for item in items
        if item["level"] == INFO_LEVEL
    ]

    analyses = build_analysis_availability(
        datasets=datasets,
        mapping=mapping,
        configuration=configuration,
    )

    return {
        "items": items,
        "errors": errors,
        "warnings": warnings,
        "information": information,
        "analyses": analyses,
        "is_valid": not errors,
        "summary": {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "information_count": len(information),
            "available_analysis_count": sum(
                1
                for analysis in analyses
                if analysis["available"]
            ),
        },
    }