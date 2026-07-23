from __future__ import annotations
from typing import Any
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from sensory_analysis.services.configuration_manager import (
    build_default_configuration,
    merge_configuration,
    validate_configuration,
)
from sensory_analysis.services.study_manager import (
    get_study,
    get_study_datasets,
    save_study,
    update_study_configuration,
)

def prepare_configuration_for_template(
    configuration: dict[str, Any],
) -> dict[str, Any]:
    prepared = {}

    for section_name, variables in configuration.items():
        prepared[section_name] = []

        for column_name, values in variables.items():
            prepared[section_name].append(
                {
                    "column_name": column_name,
                    "field_prefix": build_field_prefix(
                        section_name,
                        column_name,
                    ),
                    **values,
                }
            )

    return prepared

def study_configuration(
    request: HttpRequest,
) -> HttpResponse:
    study_id = request.session.get("sensory_study_id")
    study = get_study(study_id)

    if not study:
        messages.warning(
            request,
            "Load the study datasets before configuring the study.",
        )

        setup_url = reverse("sensory_analysis")

        return redirect(
            f"{setup_url}?tab=setup"
        )

    mapping = study.get("mapping", {})

    if not mapping:
        messages.warning(
            request,
            "Complete the variable mapping before configuring the study.",
        )

        mapping_url = reverse(
            "sensory_variable_mapping"
        )

        return redirect(
            f"{mapping_url}?tab=setup"
        )

    datasets = get_study_datasets(study)

    default_configuration = build_default_configuration(
        datasets=datasets,
        mapping=mapping,
    )

    configuration = merge_configuration(
        default_configuration=default_configuration,
        saved_configuration=study.get("configuration"),
    )

    validation_result = {
        "errors": [],
        "warnings": [],
    }

    if request.method == "POST":
        action = request.POST.get(
            "action",
            "save_configuration",
        )

        if action == "back_to_mapping":
            mapping_url = reverse(
                "sensory_variable_mapping"
            )

            return redirect(
                f"{mapping_url}?tab=setup"
            )

        configuration = extract_configuration_from_post(
            request=request,
            configuration_structure=configuration,
        )

        validation_result = validate_configuration(
            configuration
        )

        if not validation_result["errors"]:
            study = update_study_configuration(
                study=study,
                configuration=configuration,
            )

            save_study(study)

            messages.success(
                request,
                "The study configuration was saved successfully.",
            )

            validation_url = reverse(
                "sensory_study_validation"
            )

            return redirect(
                f"{validation_url}?tab=setup"
            )

    context = {
        "active_tab": "setup",
        "setup_step": "configuration",
        "study": study,
        "study_status": study.get("status", "new"),
        "configuration": prepare_configuration_for_template(configuration),
        "validation_result": validation_result,
    }

    return render(
        request,
        "sensory_analysis/setup/step_configuration.html",
        context,
    )

def extract_configuration_from_post(
    request: HttpRequest,
    configuration_structure: dict[str, Any],
) -> dict[str, Any]:
    configuration = {
        "liking": {},
        "jar": {},
        "cata": {},
        "purchase_intention": {},
        "demographics": {},
        "general_questions": {},
        "segmentation": {},
    }

    for section_name in (
        "liking",
        "purchase_intention",
    ):
        for column_name, current_values in configuration_structure.get(
            section_name,
            {},
        ).items():
            field_prefix = build_field_prefix(
                section_name,
                column_name,
            )

            configuration[section_name][column_name] = {
                "dataset": current_values.get("dataset"),
                "minimum": parse_number(
                    request.POST.get(
                        f"{field_prefix}__minimum"
                    )
                ),
                "maximum": parse_number(
                    request.POST.get(
                        f"{field_prefix}__maximum"
                    )
                ),
                "treat_as": request.POST.get(
                    f"{field_prefix}__treat_as",
                    current_values.get(
                        "treat_as",
                        "numeric",
                    ),
                ),
            }

    for column_name, current_values in configuration_structure.get(
        "jar",
        {},
    ).items():
        field_prefix = build_field_prefix(
            "jar",
            column_name,
        )

        configuration["jar"][column_name] = {
            "dataset": current_values.get("dataset"),
            "minimum": parse_number(
                request.POST.get(
                    f"{field_prefix}__minimum"
                )
            ),
            "maximum": parse_number(
                request.POST.get(
                    f"{field_prefix}__maximum"
                )
            ),
            "ideal_value": parse_number(
                request.POST.get(
                    f"{field_prefix}__ideal_value"
                )
            ),
        }

    for column_name, current_values in configuration_structure.get(
        "cata",
        {},
    ).items():
        field_prefix = build_field_prefix(
            "cata",
            column_name,
        )

        configuration["cata"][column_name] = {
            "dataset": current_values.get("dataset"),
            "separator": request.POST.get(
                f"{field_prefix}__separator",
                ";",
            ),
            "trim_spaces": request.POST.get(
                f"{field_prefix}__trim_spaces"
            ) == "on",
            "case_sensitive": request.POST.get(
                f"{field_prefix}__case_sensitive"
            ) == "on",
            "remove_empty_attributes": request.POST.get(
                f"{field_prefix}__remove_empty_attributes"
            ) == "on",
        }

    for section_name in (
        "demographics",
        "general_questions",
        "segmentation",
    ):
        for column_name, current_values in configuration_structure.get(
            section_name,
            {},
        ).items():
            field_prefix = build_field_prefix(
                section_name,
                column_name,
            )

            configuration[section_name][column_name] = {
                "dataset": current_values.get("dataset"),
                "treatment": request.POST.get(
                    f"{field_prefix}__treatment",
                    current_values.get(
                        "treatment",
                        "categorical",
                    ),
                ),
                "group_numeric": request.POST.get(
                    f"{field_prefix}__group_numeric"
                ) == "on",
                "number_of_groups": parse_integer(
                    request.POST.get(
                        f"{field_prefix}__number_of_groups"
                    ),
                    default=4,
                ),
                "reference_level": request.POST.get(
                    f"{field_prefix}__reference_level",
                    "",
                ).strip(),
            }

    return configuration

def build_field_prefix(
    section_name: str,
    column_name: str,
) -> str:
    safe_column_name = (
        column_name
        .replace(" ", "_")
        .replace("-", "_")
    )

    return f"config__{section_name}__{safe_column_name}"

def parse_number(
    value: str | None,
) -> int | float | None:
    if value is None:
        return None

    value = value.strip().replace(",", ".")

    if not value:
        return None

    try:
        parsed = float(value)
    except ValueError:
        return None

    if parsed.is_integer():
        return int(parsed)

    return parsed

def parse_integer(
    value: str | None,
    default: int,
) -> int:
    if value is None:
        return default

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default

    return max(parsed, 2)