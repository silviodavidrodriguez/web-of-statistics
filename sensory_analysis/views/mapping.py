from __future__ import annotations
from django.contrib import messages
from django.urls import reverse
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from sensory_analysis.services.mapping_manager import (
    ROLE_CHOICES,
    ROLE_LABELS,
    build_suggested_mapping,
    validate_mapping,
)
from sensory_analysis.services.study_manager import (
    get_study,
    get_study_datasets,
    save_study,
    update_study_mapping,
)

DATASET_TITLES = {
    "evaluations": "Sensory evaluations",
    "consumers": "Consumer information",
    "general": "General study questions",
}

def variable_mapping(request: HttpRequest) -> HttpResponse:
    study_id = request.session.get("sensory_study_id")
    study = get_study(study_id)

    if not study:
        messages.warning(
            request,
            "Load the study datasets before configuring variables.",
        )
        return redirect("sensory_variable_mapping")

    datasets = get_study_datasets(study)

    if not datasets:
        messages.warning(
            request,
            "No datasets were found for this study.",
        )
        return redirect("sensory_analysis")

    current_mapping = study.get("mapping") or build_suggested_mapping(
        datasets
    )

    validation_result = {
        "errors": [],
        "warnings": [],
    }

    if request.method == "POST":
        action = request.POST.get("action", "save_mapping")

        if action == "back_to_data":
            return redirect("/sensory_analysis/?tab=setup")

        submitted_mapping = extract_mapping_from_post(
            request=request,
            datasets=datasets,
        )

        validation_result = validate_mapping(
            datasets=datasets,
            mapping=submitted_mapping,
        )

        current_mapping = submitted_mapping

        if not validation_result["errors"]:
            study = update_study_mapping(
                study=study,
                mapping=submitted_mapping,
            )

            save_study(study)

            messages.success(
                request,
                "Variable mapping was saved successfully.",
            )

            configuration_url = reverse(
                "sensory_study_configuration"
            )

            return redirect(
                f"{configuration_url}?tab=setup"
            )

    mapping_sections = build_mapping_sections(
        datasets=datasets,
        mapping=current_mapping,
    )

    context = {
        "active_tab": request.GET.get("tab", "setup"),
        "setup_step": "variables",
        "study": study,
        "study_status": study.get("status", "new"),
        "mapping_sections": mapping_sections,
        "role_choices": ROLE_CHOICES,
        "validation_result": validation_result,
    }

    return render(
        request,
        "sensory_analysis/setup/step_mapping.html",
        context,
    )

def extract_mapping_from_post(
    request: HttpRequest,
    datasets: dict,
) -> dict[str, dict[str, str]]:
    mapping: dict[str, dict[str, str]] = {}

    valid_roles = {
        role_value
        for role_value, _role_label in ROLE_CHOICES
    }

    for dataset_name, dataframe in datasets.items():
        mapping[dataset_name] = {}

        for column_index, column_name in enumerate(dataframe.columns):
            field_name = (
                f"role__{dataset_name}__{column_index}"
            )

            selected_role = request.POST.get(
                field_name,
                "additional",
            )

            if selected_role not in valid_roles:
                selected_role = "additional"

            mapping[dataset_name][str(column_name)] = selected_role

    return mapping

def build_mapping_sections(
    datasets: dict,
    mapping: dict[str, dict[str, str]],
) -> list[dict]:
    sections: list[dict] = []

    for dataset_name, dataframe in datasets.items():
        variables = []

        dataset_mapping = mapping.get(dataset_name, {})

        for column_index, column_name in enumerate(dataframe.columns):
            column_name = str(column_name)

            selected_role = dataset_mapping.get(
                column_name,
                "additional",
            )

            variables.append(
                {
                    "column_index": column_index,
                    "column_name": column_name,
                    "field_name": (
                        f"role__{dataset_name}__{column_index}"
                    ),
                    "selected_role": selected_role,
                    "selected_role_label": ROLE_LABELS.get(
                        selected_role,
                        selected_role,
                    ),
                    "data_type": infer_display_data_type(
                        dataframe[column_name]
                    ),
                    "non_blank_count": int(
                        dataframe[column_name].notna().sum()
                    ),
                    "unique_count": int(
                        dataframe[column_name].nunique(
                            dropna=True
                        )
                    ),
                    "examples": get_example_values(
                        dataframe[column_name]
                    ),
                }
            )

        sections.append(
            {
                "dataset_name": dataset_name,
                "title": DATASET_TITLES.get(
                    dataset_name,
                    dataset_name.replace("_", " ").title(),
                ),
                "row_count": len(dataframe.index),
                "column_count": len(dataframe.columns),
                "variables": variables,
            }
        )

    return sections

def infer_display_data_type(series) -> str:
    if series.empty:
        return "Empty"

    numeric_series = series.dropna()

    if numeric_series.empty:
        return "Empty"

    converted = numeric_series.astype(str).str.strip()

    numeric_values = converted.str.match(
        r"^-?\d+(?:[\.,]\d+)?$"
    )

    if numeric_values.all():
        return "Numeric"

    unique_count = numeric_series.nunique(dropna=True)
    row_count = len(numeric_series.index)

    if unique_count <= min(20, max(5, row_count // 2)):
        return "Categorical"

    return "Text"

def get_example_values(series, limit: int = 3) -> list[str]:
    values = []

    for value in series.dropna().unique():
        text = str(value).strip()

        if not text:
            continue

        values.append(text)

        if len(values) >= limit:
            break

    return values