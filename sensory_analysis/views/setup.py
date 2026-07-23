from django.contrib import messages
from django.shortcuts import redirect, render
from django.urls import reverse
from sensory_analysis.forms import StudySetupForm
from sensory_analysis.services.study_manager import (
    build_new_study,
    delete_study,
    get_study,
    get_study_datasets,
    save_study,
    update_study_analysis,
)
from sensory_analysis.services.normalizer import build_dataset_preview
from sensory_analysis.services.parser import parse_pasted_table
from sensory_analysis.services.session_manager import clear_sensory_session
from sensory_analysis.services.validator import validate_dataset_structure
from sensory_analysis.services.liking_manager import (
    get_mapped_columns,
    run_liking_analysis,
)
from sensory_analysis.services.jar_manager import (
    run_jar_analysis,
)
from sensory_analysis.services.cata_manager import (
    run_cata_analysis,
)
from sensory_analysis.services.purchase_intention_manager import (
    run_purchase_intention_analysis,
)
from sensory_analysis.services.segmentation_manager import (
    run_segmentation_analysis,
)

DATASET_CONFIG = {
    "evaluations": {
        "data_field": "evaluations_data",
        "header_field": "evaluations_has_header",
        "required": True,
        "title": "Sensory evaluations",
    },
    "consumers": {
        "data_field": "consumers_data",
        "header_field": "consumers_has_header",
        "required": False,
        "title": "Consumer information",
    },
    "general": {
        "data_field": "general_data",
        "header_field": "general_has_header",
        "required": False,
        "title": "General study questions",
    },
}

ALLOWED_TABS = {
    "setup",
    "liking",
    "jar",
    "cata",
    "purchase_intention",
    "segmentation",
}

def build_previews_from_study(
    study: dict | None,
) -> dict:
    if not study:
        return {}

    datasets = get_study_datasets(study)
    dataset_previews = {}

    for dataset_name, dataframe in datasets.items():
        config = DATASET_CONFIG.get(dataset_name, {})

        dataset_previews[dataset_name] = {
            "title": config.get(
                "title",
                dataset_name.replace("_", " ").title(),
            ),
            "summary": build_dataset_preview(dataframe),
            "warnings": [],
        }

    return dataset_previews

def dataframe_to_paste_text(dataframe) -> str:
    if dataframe is None or dataframe.empty:
        return ""

    return dataframe.to_csv(
        sep="\t",
        index=False,
        lineterminator="\n",
    )

def build_setup_form_from_study(
    study: dict | None,
) -> StudySetupForm:
    if not study:
        return StudySetupForm()

    datasets = get_study_datasets(study)

    initial_data = {}

    evaluations = datasets.get("evaluations")

    if evaluations is not None:
        initial_data["evaluations_data"] = (
            dataframe_to_paste_text(evaluations)
        )
        initial_data["evaluations_has_header"] = True

    consumers = datasets.get("consumers")

    if consumers is not None:
        initial_data["consumers_data"] = (
            dataframe_to_paste_text(consumers)
        )
        initial_data["consumers_has_header"] = True

    general = datasets.get("general")

    if general is not None:
        initial_data["general_data"] = (
            dataframe_to_paste_text(general)
        )
        initial_data["general_has_header"] = True

    return StudySetupForm(initial=initial_data)

def build_liking_context(
    *,
    study: dict | None,
    datasets: dict,
) -> dict:
    context = {
        "liking_available": False,
        "liking_variables": [],
        "liking_result": None,
    }

    if not study:
        return context

    mapping = study.get("mapping", {})

    liking_variables = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="liking",
    )

    context["liking_variables"] = liking_variables
    context["liking_available"] = bool(
        study.get("status") == "ready"
        and liking_variables
    )

    saved_result = (
        study
        .get("analyses", {})
        .get("liking")
    )

    context["liking_result"] = saved_result

    return context

def build_jar_context(
    *,
    study: dict | None,
    datasets: dict,
) -> dict:
    context = {
        "jar_available": False,
        "jar_variables": [],
        "jar_result": None,
    }

    if not study:
        return context

    mapping = study.get("mapping", {})

    jar_variables = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="jar",
    )

    context["jar_variables"] = jar_variables

    context["jar_available"] = bool(
        study.get("status") == "ready"
        and jar_variables
    )

    context["jar_result"] = (
        study
        .get("analyses", {})
        .get("jar")
    )

    return context

def build_cata_context(
    *,
    study: dict | None,
    datasets: dict,
) -> dict:
    """
    Build the context required by the CATA analysis tab.
    """

    context = {
        "cata_available": False,
        "cata_variables": [],
        "cata_result": None,
    }

    if not study:
        return context

    mapping = study.get(
        "mapping",
        {},
    )

    cata_variables = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="cata",
    )

    context["cata_variables"] = (
        cata_variables
    )

    context["cata_available"] = bool(
        study.get("status") == "ready"
        and cata_variables
    )

    context["cata_result"] = (
        study
        .get("analyses", {})
        .get("cata")
    )

    return context


def build_purchase_context(
    *,
    study: dict | None,
    datasets: dict,
) -> dict:
    context = {
        "purchase_available": False,
        "purchase_variables": [],
        "purchase_result": None,
    }

    if not study:
        return context

    mapping = study.get("mapping", {})
    purchase_variables = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="purchase_intention",
    )

    context["purchase_variables"] = purchase_variables
    context["purchase_available"] = bool(
        study.get("status") == "ready"
        and purchase_variables
    )
    context["purchase_result"] = (
        study.get("analyses", {}).get("purchase_intention")
    )

    return context


def build_segmentation_context(
    *,
    study: dict | None,
    datasets: dict,
) -> dict:
    context = {
        "segmentation_available": False,
        "segmentation_result": None,
        "segmentation_purchase_available": False,
    }

    if not study:
        return context

    mapping = study.get("mapping", {})
    liking_variables = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="liking",
    )
    purchase_variables = get_mapped_columns(
        mapping,
        dataset_name="evaluations",
        role="purchase_intention",
    )

    context["segmentation_available"] = bool(
        study.get("status") == "ready"
        and liking_variables
    )
    context["segmentation_purchase_available"] = bool(
        purchase_variables
    )
    context["segmentation_result"] = (
        study.get("analyses", {}).get("segmentation")
    )
    return context

def process_segmentation_analysis(request):
    study_id = request.session.get("sensory_study_id")
    study = get_study(study_id)

    if not study:
        messages.error(request, "No active sensory study was found.")
        return redirect(f"{reverse('sensory_analysis')}?tab=setup")

    if study.get("status") != "ready":
        messages.error(
            request,
            "Validate the study before running consumer segmentation.",
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=segmentation"
        )

    include_purchase = request.POST.get("include_purchase") == "on"
    requested_clusters_text = request.POST.get("cluster_count", "auto")
    requested_clusters = None
    if requested_clusters_text not in {"", "auto"}:
        try:
            requested_clusters = int(requested_clusters_text)
        except (TypeError, ValueError):
            requested_clusters = None

    datasets = get_study_datasets(study)
    result = run_segmentation_analysis(
        datasets=datasets,
        mapping=study.get("mapping", {}),
        configuration=study.get("configuration", {}),
        include_purchase=include_purchase,
        requested_clusters=requested_clusters,
        maximum_clusters=6,
    )

    if not result.get("is_valid"):
        messages.error(
            request,
            result.get(
                "error",
                "Consumer segmentation could not be completed.",
            ),
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=segmentation"
        )

    study = update_study_analysis(
        study,
        analysis_name="segmentation",
        result=result,
    )
    save_study(study)
    messages.success(
        request,
        "Consumer segmentation was completed.",
    )
    return redirect(
        f"{reverse('sensory_analysis')}?tab=segmentation"
    )


def process_purchase_intention_analysis(request):
    study_id = request.session.get("sensory_study_id")
    study = get_study(study_id)

    if not study:
        messages.error(request, "No active sensory study was found.")
        return redirect(f"{reverse('sensory_analysis')}?tab=setup")

    if study.get("status") != "ready":
        messages.error(
            request,
            "Validate the study before running the purchase intention analysis.",
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=purchase_intention"
        )

    datasets = get_study_datasets(study)
    purchase_variable = request.POST.get("purchase_variable") or None

    result = run_purchase_intention_analysis(
        datasets=datasets,
        mapping=study.get("mapping", {}),
        configuration=study.get("configuration", {}),
        purchase_variable=purchase_variable,
    )

    if not result.get("is_valid"):
        messages.error(
            request,
            result.get(
                "error",
                "The purchase intention analysis could not be completed.",
            ),
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=purchase_intention"
        )

    study = update_study_analysis(
        study,
        analysis_name="purchase_intention",
        result=result,
    )
    save_study(study)
    messages.success(
        request,
        "The purchase intention analysis was completed.",
    )
    return redirect(
        f"{reverse('sensory_analysis')}?tab=purchase_intention"
    )

def process_liking_analysis(request):
    study_id = request.session.get(
        "sensory_study_id"
    )
    study = get_study(study_id)

    if not study:
        messages.error(
            request,
            "No active sensory study was found.",
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=setup"
        )

    if study.get("status") != "ready":
        messages.error(
            request,
            "Validate the study before running the analysis.",
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=liking"
        )

    liking_variable = (
        request.POST.get("liking_variable")
        or None
    )

    datasets = get_study_datasets(study)

    result = run_liking_analysis(
        datasets=datasets,
        mapping=study.get("mapping", {}),
        configuration=study.get("configuration", {}),
        liking_variable=liking_variable,
    )

    if not result.get("is_valid"):
        messages.error(
            request,
            result.get(
                "error",
                "The liking analysis could not be completed.",
            ),
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=liking"
        )

    study = update_study_analysis(
        study,
        analysis_name="liking",
        result=result,
    )

    save_study(study)

    messages.success(
        request,
        "The overall liking analysis was completed.",
    )

    return redirect(
        f"{reverse('sensory_analysis')}?tab=liking"
    )

def process_jar_analysis(request):
    study_id = request.session.get(
        "sensory_study_id"
    )

    study = get_study(study_id)

    if not study:
        messages.error(
            request,
            "No active sensory study was found.",
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=setup"
        )

    if study.get("status") != "ready":
        messages.error(
            request,
            "Validate the study before running the JAR analysis.",
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=jar"
        )

    datasets = get_study_datasets(study)

    selected_variables = request.POST.getlist(
        "jar_variables"
    )

    result = run_jar_analysis(
        datasets=datasets,
        mapping=study.get("mapping", {}),
        configuration=study.get(
            "configuration",
            {},
        ),
        selected_jar_variables=selected_variables,
    )

    if not result.get("is_valid"):
        messages.error(
            request,
            result.get(
                "error",
                "The JAR analysis could not be completed.",
            ),
        )
        return redirect(
            f"{reverse('sensory_analysis')}?tab=jar"
        )

    study = update_study_analysis(
        study,
        analysis_name="jar",
        result=result,
    )

    save_study(study)

    messages.success(
        request,
        "The JAR analysis was completed.",
    )

    return redirect(
        f"{reverse('sensory_analysis')}?tab=jar"
    )

def process_cata_analysis(request):
    """
    Validate the active study, run the CATA analysis
    and save its results.
    """

    study_id = request.session.get(
        "sensory_study_id"
    )

    study = get_study(
        study_id
    )

    if not study:
        messages.error(
            request,
            "No active sensory study was found.",
        )

        return redirect(
            f"{reverse('sensory_analysis')}?tab=setup"
        )

    if study.get("status") != "ready":
        messages.error(
            request,
            (
                "Validate the study before running "
                "the CATA analysis."
            ),
        )

        return redirect(
            f"{reverse('sensory_analysis')}?tab=cata"
        )

    datasets = get_study_datasets(
        study
    )

    selected_variables = (
        request.POST.getlist(
            "cata_variables"
        )
    )

    result = run_cata_analysis(
        datasets=datasets,
        mapping=study.get(
            "mapping",
            {},
        ),
        configuration=study.get(
            "configuration",
            {},
        ),
        selected_cata_variables=(
            selected_variables
        ),
    )

    if not result.get("is_valid"):
        messages.error(
            request,
            result.get(
                "error",
                (
                    "The CATA analysis could not "
                    "be completed."
                ),
            ),
        )

        return redirect(
            f"{reverse('sensory_analysis')}?tab=cata"
        )

    study = update_study_analysis(
        study,
        analysis_name="cata",
        result=result,
    )

    save_study(
        study
    )

    messages.success(
        request,
        "The CATA analysis was completed.",
    )

    return redirect(
        f"{reverse('sensory_analysis')}?tab=cata"
    )

def sensory_analysis(request):
    active_tab = request.GET.get("tab", "setup")

    if active_tab not in ALLOWED_TABS:
        active_tab = "setup"

    if request.method == "POST":
        action = request.POST.get("action")

        if action == "clear_study":
            return clear_study(request)

        if active_tab == "setup":
            return process_study_setup(
                request,
                active_tab,
            )
        
        if (
            active_tab == "liking"
            and action == "run_liking"
        ):
            return process_liking_analysis(request)
        
        if (
            active_tab == "jar"
            and action == "run_jar_analysis"
        ):
            return process_jar_analysis(request)
        
        if (
            active_tab == "cata"
            and action == "run_cata_analysis"
        ):
            return process_cata_analysis(
                request
            )

        if (
            active_tab == "purchase_intention"
            and action == "run_purchase_intention"
        ):
            return process_purchase_intention_analysis(request)

        if (
            active_tab == "segmentation"
            and action == "run_segmentation"
        ):
            return process_segmentation_analysis(request)

        messages.error(
            request,
            "The requested action is not available.",
        )

        return redirect(
            f"{reverse('sensory_analysis')}?tab={active_tab}"
        )

    study_id = request.session.get("sensory_study_id")
    study = get_study(study_id)

    datasets = {}
    if study:
        datasets = get_study_datasets(study)

    dataset_previews = build_previews_from_study(study)
    setup_form = build_setup_form_from_study(study)

    context = build_base_context(
        active_tab=active_tab,
        form=setup_form,
        dataset_previews=dataset_previews,
        study=study,
    )

    if active_tab == "liking":
        context.update(
            build_liking_context(
                study=study,
                datasets=datasets,
            )
        )

    elif active_tab == "jar":
        context.update(
            build_jar_context(
                study=study,
                datasets=datasets,
            )
        )

    elif active_tab == "cata":
        context.update(
            build_cata_context(
                study=study,
                datasets=datasets,
            )
        )

    elif active_tab == "purchase_intention":
        context.update(
            build_purchase_context(
                study=study,
                datasets=datasets,
            )
        )

    elif active_tab == "segmentation":
        context.update(
            build_segmentation_context(
                study=study,
                datasets=datasets,
            )
        )

    return render(
        request,
        "sensory_analysis/sensory_analysis.html",
        context,
    )

def build_base_context(
    active_tab: str = "setup",
    form: StudySetupForm | None = None,
    dataset_previews: dict | None = None,
    study: dict | None = None,
):
    study_status = study.get("status") if study else "new"

    return {
        "active_tab": active_tab,
        "setup_step": "data",
        "setup_form": form or StudySetupForm(),
        "dataset_previews": dataset_previews or {},
        "study": study,
        "study_loaded": bool(study and study.get("datasets")),
        "study_status": study_status,
        "study_ready": study_status == "ready",
        "validation": study.get("validation", {}) if study else {},
        "configuration": study.get("configuration", {}) if study else {},
    }

def process_study_setup(request, active_tab: str):
    form = StudySetupForm(request.POST)

    study_id = request.session.get("sensory_study_id")
    current_study = get_study(study_id)

    if not form.is_valid():
        current_previews = build_previews_from_study(
            current_study
        )

        context = build_base_context(
            active_tab=active_tab,
            form=form,
            dataset_previews=current_previews,
            study=current_study,
        )

        return render(
            request,
            "sensory_analysis/sensory_analysis.html",
            context,
        )

    datasets = {}
    dataset_previews = {}
    all_errors = []

    for dataset_name, config in DATASET_CONFIG.items():
        pasted_data = form.cleaned_data.get(
            config["data_field"],
            "",
        )

        has_header = form.cleaned_data.get(
            config["header_field"],
            True,
        )

        if not pasted_data.strip():
            if config["required"]:
                all_errors.append(
                    f"{config['title']}: this dataset is required."
                )

            continue

        try:
            dataframe = parse_pasted_table(
                pasted_data=pasted_data,
                has_header=has_header,
            )
        except ValueError as exc:
            all_errors.append(
                f"{config['title']}: {exc}"
            )
            continue

        validation_result = validate_dataset_structure(
            dataframe
        )

        validation_errors = validation_result.get(
            "errors",
            [],
        )

        validation_warnings = validation_result.get(
            "warnings",
            [],
        )

        for error in validation_errors:
            all_errors.append(
                f"{config['title']}: {error}"
            )

        if validation_errors:
            continue

        datasets[dataset_name] = dataframe

        dataset_previews[dataset_name] = {
            "title": config["title"],
            "summary": build_dataset_preview(
                dataframe
            ),
            "warnings": validation_warnings,
        }

    if all_errors:
        context = build_base_context(
            active_tab=active_tab,
            form=form,
            dataset_previews=dataset_previews,
            study=current_study,
        )

        context["setup_errors"] = all_errors

        return render(
            request,
            "sensory_analysis/sensory_analysis.html",
            context,
        )

    study = build_new_study(
        datasets=datasets,
    )

    save_study(study)

    request.session["sensory_study_id"] = study["id"]
    request.session.modified = True

    messages.success(
        request,
        "The study data were processed successfully.",
    )

    context = build_base_context(
        active_tab=active_tab,
        form=form,
        dataset_previews=dataset_previews,
        study=study,
    )

    return render(
        request,
        "sensory_analysis/sensory_analysis.html",
        context,
    )

def clear_study(request):
    study_id = request.session.get("sensory_study_id")
    delete_study(study_id)
    clear_sensory_session(request)

    messages.info(
        request,
        "The current study was cleared.",
    )

    setup_url = reverse("sensory_analysis")

    return redirect(f"{setup_url}?tab=setup")