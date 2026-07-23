from __future__ import annotations
import re
from typing import Any

ROLE_CHOICES = [
    ("consumer_id", "Consumer ID"),
    ("sample_id", "Sample ID"),
    ("order", "Presentation order"),
    ("position", "Presentation position"),
    ("liking", "Overall liking"),
    ("jar", "JAR variable"),
    ("cata", "CATA variable"),
    ("purchase_intention", "Purchase intention"),
    ("demographic", "Demographic variable"),
    ("general_question", "General study question"),
    ("segmentation", "Segmentation variable"),
    ("additional", "Additional variable"),
    ("ignore", "Ignore"),
]

ROLE_LABELS = dict(ROLE_CHOICES)

def normalize_column_name(column_name: str) -> str:
    """
    Converts a column name into a normalized form for automatic detection.
    """
    normalized = str(column_name).strip().lower()
    normalized = re.sub(r"[\s\-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_]", "", normalized)

    return normalized

def suggest_role(
    dataset_name: str,
    column_name: str,
) -> str:
    """
    Suggests a semantic role using the dataset and column name.
    """
    normalized = normalize_column_name(column_name)

    consumer_names = {
        "consumer",
        "consumer_id",
        "consumerid",
        "participant",
        "participant_id",
        "respondent",
        "respondent_id",
        "panelist",
        "panelist_id",
        "subject",
        "subject_id",
    }

    sample_names = {
        "sample",
        "sample_id",
        "sampleid",
        "product",
        "product_id",
        "productid",
        "code",
        "sample_code",
        "product_code",
    }

    order_names = {
        "order",
        "presentation_order",
        "serving_order",
        "sequence",
    }

    position_names = {
        "position",
        "presentation_position",
        "serving_position",
    }

    liking_names = {
        "liking",
        "overall_liking",
        "acceptability",
        "acceptance",
        "overall_acceptability",
        "hedonic",
        "hedonic_score",
    }

    cata_names = {
        "cata",
        "check_all_that_apply",
        "attributes",
        "selected_attributes",
    }

    purchase_names = {
        "purchase_intention",
        "purchase",
        "buying_intention",
        "intention_to_purchase",
        "purchase_likelihood",
    }

    if normalized in consumer_names:
        return "consumer_id"

    if dataset_name == "evaluations":
        if normalized in sample_names:
            return "sample_id"

        if normalized in order_names:
            return "order"

        if normalized in position_names:
            return "position"

        if normalized in liking_names:
            return "liking"

        if normalized in cata_names or "cata" in normalized:
            return "cata"

        if normalized in purchase_names:
            return "purchase_intention"

        if (
            normalized.endswith("_jar")
            or normalized.startswith("jar_")
            or "_jar_" in normalized
        ):
            return "jar"

        if "liking" in normalized or "acceptability" in normalized:
            return "liking"

        if "purchase" in normalized:
            return "purchase_intention"

        return "additional"

    if dataset_name == "consumers":
        return "demographic"

    if dataset_name == "general":
        return "general_question"

    return "additional"

def build_suggested_mapping(
    datasets: dict[str, Any],
) -> dict[str, dict[str, str]]:
    """
    Builds an automatic mapping for every dataset column.
    """
    mapping: dict[str, dict[str, str]] = {}

    for dataset_name, dataframe in datasets.items():
        mapping[dataset_name] = {}

        for column_name in dataframe.columns:
            mapping[dataset_name][str(column_name)] = suggest_role(
                dataset_name=dataset_name,
                column_name=str(column_name),
            )

    return mapping

def validate_mapping(
    datasets: dict[str, Any],
    mapping: dict[str, dict[str, str]],
) -> dict[str, list[str]]:
    """
    Validates the structural rules required before moving to Step 3.
    """
    errors: list[str] = []
    warnings: list[str] = []

    evaluations_mapping = mapping.get("evaluations", {})

    evaluation_roles = list(evaluations_mapping.values())

    consumer_id_count = evaluation_roles.count("consumer_id")
    sample_id_count = evaluation_roles.count("sample_id")

    if consumer_id_count == 0:
        errors.append(
            "Sensory evaluations must contain one Consumer ID variable."
        )
    elif consumer_id_count > 1:
        errors.append(
            "Sensory evaluations can contain only one Consumer ID variable."
        )

    if sample_id_count == 0:
        errors.append(
            "Sensory evaluations must contain one Sample ID variable."
        )
    elif sample_id_count > 1:
        errors.append(
            "Sensory evaluations can contain only one Sample ID variable."
        )

    for dataset_name in ("consumers", "general"):
        if dataset_name not in datasets:
            continue

        dataset_mapping = mapping.get(dataset_name, {})
        roles = list(dataset_mapping.values())
        dataset_consumer_count = roles.count("consumer_id")

        if dataset_consumer_count == 0:
            errors.append(
                f"The {dataset_name} dataset must contain one Consumer ID "
                "variable so it can be linked to sensory evaluations."
            )
        elif dataset_consumer_count > 1:
            errors.append(
                f"The {dataset_name} dataset can contain only one "
                "Consumer ID variable."
            )

    if evaluation_roles.count("liking") == 0:
        warnings.append(
            "No overall liking variable was assigned."
        )

    if all(
        role == "ignore"
        for dataset_mapping in mapping.values()
        for role in dataset_mapping.values()
    ):
        errors.append(
            "At least one variable must be included in the study."
        )

    return {
        "errors": errors,
        "warnings": warnings,
    }