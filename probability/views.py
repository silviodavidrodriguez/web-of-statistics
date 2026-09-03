from django.shortcuts import render

from probability.distributions import (
    get_continuous_distributions,
    get_discrete_distributions,
    get_distribution_spec,
)

from probability.services import (
    CalculatorInputError,
    DistributionValidationError,
    build_probability_ui_config,
    calculate,
    calculation_figure_html,
    format_calculation,
    get_default_operation,
    get_default_operation_inputs,
    get_default_parameters,
    get_operation_ui,
)


VALID_TABS = {
    "functions",
    "explorer",
    "simulation",
    "sampling",
}


LEGACY_TAB_MAP = {
    "density": "functions",
    "normal_table": "functions",
    "t_table": "functions",
    "chi_table": "functions",
    "f_table": "functions",
}


def _resolve_tab(request):
    requested = request.GET.get(
        "tab",
        "functions",
    )

    requested = LEGACY_TAB_MAP.get(
        requested,
        requested,
    )

    if requested not in VALID_TABS:
        return "functions"

    return requested


def _resolve_distribution(
    distribution_key,
):
    try:
        return get_distribution_spec(
            distribution_key
        )

    except ValueError:
        return get_distribution_spec(
            "standard_normal"
        )


def _resolve_operation(
    spec,
    requested_operation,
):
    default_operation = (
        get_default_operation(
            spec.category
        )
    )

    if not requested_operation:
        return default_operation

    try:
        get_operation_ui(
            spec.category,
            requested_operation,
        )

    except ValueError:
        return default_operation

    return requested_operation


def _parameter_state_from_post(
    request,
    spec,
):
    return {
        parameter.name:
            request.POST.get(
                f"param_{parameter.name}",
                "",
            )
        for parameter in spec.parameters
    }


def _input_state_from_post(
    request,
    operation_ui,
):
    return {
        item["name"]:
            request.POST.get(
                f"input_{item['name']}",
                "",
            )
        for item in operation_ui["inputs"]
    }


def probability(request):
    active_tab = _resolve_tab(
        request
    )

    selected_distribution = (
        "standard_normal"
    )

    selected_operation = "left"

    parameter_state = {}
    input_state = {}

    parameter_errors = {}
    input_errors = {}
    general_errors = []

    formatted_result = None
    chart_html = None

    if (
        request.method == "POST"
        and active_tab == "functions"
    ):
        requested_distribution = (
            request.POST.get(
                "distribution",
                "standard_normal",
            )
        )

        try:
            spec = get_distribution_spec(
                requested_distribution
            )

            selected_distribution = (
                requested_distribution
            )

        except ValueError:
            spec = get_distribution_spec(
                "standard_normal"
            )

            selected_distribution = (
                "standard_normal"
            )

            general_errors.append(
                "The selected distribution "
                "is not available."
            )

        requested_operation = (
            request.POST.get(
                "operation",
                "",
            )
        )

        selected_operation = (
            _resolve_operation(
                spec,
                requested_operation,
            )
        )

        if (
            requested_operation
            and requested_operation
            != selected_operation
        ):
            general_errors.append(
                "The selected operation is not "
                "available for this distribution."
            )

        operation_ui = get_operation_ui(
            spec.category,
            selected_operation,
        )

        parameter_state = (
            _parameter_state_from_post(
                request,
                spec,
            )
        )

        input_state = (
            _input_state_from_post(
                request,
                operation_ui,
            )
        )

        try:
            calculation = calculate(
                selected_distribution,
                parameter_state,
                selected_operation,
                input_state,
            )

            formatted_result = (
                format_calculation(
                    calculation
                )
            )

            chart_html = (
                calculation_figure_html(
                    calculation
                )
            )

        except DistributionValidationError as exc:
            parameter_errors.update(
                exc.result.field_errors
            )

            general_errors.extend(
                exc.result.non_field_errors
            )

        except CalculatorInputError as exc:
            input_errors.update(
                exc.field_errors
            )

            general_errors.extend(
                exc.non_field_errors
            )

    else:
        spec = _resolve_distribution(
            selected_distribution
        )

        selected_operation = (
            get_default_operation(
                spec.category
            )
        )

        parameter_state = (
            get_default_parameters(
                selected_distribution
            )
        )

        input_state = (
            get_default_operation_inputs(
                spec.category,
                selected_operation,
            )
        )

    context = {
        "segment": "probability",
        "active_tab": active_tab,

        "continuous_distributions":
            get_continuous_distributions(),

        "discrete_distributions":
            get_discrete_distributions(),

        "selected_distribution":
            selected_distribution,

        "selected_operation":
            selected_operation,

        "probability_ui_config":
            build_probability_ui_config(),

        "probability_form_state": {
            "distribution":
                selected_distribution,
            "operation":
                selected_operation,
            "parameters":
                parameter_state,
            "inputs":
                input_state,
        },

        "probability_field_errors": {
            "parameters":
                parameter_errors,
            "inputs":
                input_errors,
        },

        "general_errors":
            general_errors,

        "formatted_result":
            formatted_result,

        "chart_html":
            chart_html,
    }

    return render(
        request,
        "probability/probability.html",
        context,
    )