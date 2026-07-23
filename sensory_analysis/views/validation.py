from __future__ import annotations
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from sensory_analysis.services.study_manager import (
    get_study,
    get_study_datasets,
    save_study,
    update_study_validation,
)
from sensory_analysis.services.validation_manager import (
    run_study_validation,
)

def study_validation(
    request: HttpRequest,
) -> HttpResponse:
    study_id = request.session.get(
        "sensory_study_id"
    )
    study = get_study(study_id)

    if not study:
        messages.warning(
            request,
            "Load the study datasets before validating the study.",
        )

        setup_url = reverse(
            "sensory_analysis"
        )

        return redirect(
            f"{setup_url}?tab=setup"
        )

    if not study.get("mapping"):
        messages.warning(
            request,
            "Complete the variable mapping before validation.",
        )

        mapping_url = reverse(
            "sensory_variable_mapping"
        )

        return redirect(
            f"{mapping_url}?tab=setup"
        )

    if not study.get("configuration"):
        messages.warning(
            request,
            "Complete the study configuration before validation.",
        )

        configuration_url = reverse(
            "sensory_study_configuration"
        )

        return redirect(
            f"{configuration_url}?tab=setup"
        )

    datasets = get_study_datasets(study)

    validation_result = run_study_validation(
        datasets=datasets,
        mapping=study.get("mapping", {}),
        configuration=study.get(
            "configuration",
            {},
        ),
    )

    study = update_study_validation(
        study=study,
        validation=validation_result,
    )

    save_study(study)

    if request.method == "POST":
        action = request.POST.get(
            "action"
        )

        if action == "back_to_configuration":
            configuration_url = reverse(
                "sensory_study_configuration"
            )

            return redirect(
                f"{configuration_url}?tab=setup"
            )

        if action == "revalidate":
            messages.success(
                request,
                "The study was validated again.",
            )

            validation_url = reverse(
                "sensory_study_validation"
            )

            return redirect(
                f"{validation_url}?tab=setup"
            )

        if action == "continue_to_analyses":
            if not validation_result["is_valid"]:
                messages.error(
                    request,
                    "Resolve the blocking validation errors before "
                    "continuing to analyses.",
                )
            else:
                messages.success(
                    request,
                    "The study is ready for analysis.",
                )

                analyses_url = reverse(
                    "sensory_analysis"
                )

                return redirect(
                    f"{analyses_url}?tab=liking"
                )

    context = {
        "active_tab": "setup",
        "setup_step": "validation",
        "study": study,
        "study_status": study.get(
            "status",
            "configured",
        ),
        "validation_result": validation_result,
    }

    return render(
        request,
        "sensory_analysis/setup/step_validation.html",
        context,
    )