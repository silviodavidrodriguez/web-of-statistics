from django.shortcuts import render

from probability.distributions import (
    get_continuous_distributions,
    get_discrete_distributions,
    get_distribution_spec,
)

from probability.services import (
    CONTINUOUS_EXPLORER_VIEWS,
    DISCRETE_EXPLORER_VIEWS,
    CalculatorInputError,
    DistributionValidationError,
    ExplorerError,
    build_probability_ui_config,
    calculate,
    calculation_figure_html,
    explorer_figure_html,
    format_calculation,
    format_number,
    get_default_operation,
    get_default_operation_inputs,
    get_default_parameters,
    get_distribution_properties,
    get_operation_ui,
    ComparisonCurve,
    comparison_figure_html,
    validate_distribution_parameters,
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


EXPLORER_VIEW_LABELS = {
    "pdf": "Probability density (PDF)",
    "pmf": "Probability mass (PMF)",
    "cdf": "Cumulative distribution (CDF)",
    "survival": "Survival function",
    "hazard": "Hazard function",
}


EXPLORER_MODES = {
    "single",
    "compare",
}


COMPARISON_CATEGORIES = {
    "continuous",
    "discrete",
}


MIN_COMPARISON_CURVES = 2
MAX_COMPARISON_CURVES = 5


DEFAULT_COMPARISON_CURVES = {
    "continuous": (
        {
            "distribution": "student_t",
            "label": "t, df = 2",
            "parameters": {
                "df": 2.0,
            },
        },
        {
            "distribution": "student_t",
            "label": "t, df = 5",
            "parameters": {
                "df": 5.0,
            },
        },
        {
            "distribution": "student_t",
            "label": "t, df = 30",
            "parameters": {
                "df": 30.0,
            },
        },
        {
            "distribution":
                "standard_normal",
            "label": "Standard Normal",
            "parameters": {},
        },
    ),

    "discrete": (
        {
            "distribution": "binomial",
            "label": "Binomial(10, 0.5)",
            "parameters": {
                "n": 10,
                "p": 0.5,
            },
        },
        {
            "distribution": "poisson",
            "label": "Poisson(5)",
            "parameters": {
                "rate": 5.0,
            },
        },
    ),
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
    *,
    prefix="param_",
):
    return {
        parameter.name:
            request.POST.get(
                f"{prefix}{parameter.name}",
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


def _explorer_views_for_spec(
    spec,
):
    if spec.category == "continuous":
        views = list(
            CONTINUOUS_EXPLORER_VIEWS
        )

        if not spec.supports_hazard:
            views = [
                view
                for view in views
                if view != "hazard"
            ]

    else:
        views = list(
            DISCRETE_EXPLORER_VIEWS
        )

    return views


def _resolve_explorer_view(
    spec,
    requested_view,
):
    valid_views = (
        _explorer_views_for_spec(
            spec
        )
    )

    default_view = (
        "pdf"
        if spec.category == "continuous"
        else "pmf"
    )

    if requested_view in valid_views:
        return requested_view

    return default_view


def _format_optional_property(
    value,
):
    if value is None:
        return "Undefined"

    return format_number(
        value
    )


def _build_property_rows(
    properties,
):
    return (
        {
            "label": "Mean",
            "value": _format_optional_property(
                properties.mean
            ),
        },
        {
            "label": "Median",
            "value": _format_optional_property(
                properties.median
            ),
        },
        {
            "label": "Variance",
            "value": _format_optional_property(
                properties.variance
            ),
        },
        {
            "label": "Standard deviation",
            "value": _format_optional_property(
                properties.standard_deviation
            ),
        },
        {
            "label": "Skewness",
            "value": _format_optional_property(
                properties.skewness
            ),
        },
        {
            "label": "Excess kurtosis",
            "value": _format_optional_property(
                properties.excess_kurtosis
            ),
        },
    )


def _build_quantile_rows(
    properties,
):
    rows = []

    for probability, value in (
        properties.quantiles.items()
    ):
        rows.append(
            {
                "probability":
                    format_number(
                        probability
                    ),
                "percentage":
                    (
                        f"{probability * 100:g}%"
                    ),
                "value":
                    format_number(
                        value
                    ),
            }
        )

    return rows


def _resolve_explorer_mode(
    request,
):
    if request.method == "POST":
        requested = request.POST.get(
            "explorer_mode",
            "single",
        )

    else:
        requested = request.GET.get(
            "mode",
            "single",
        )

    if requested not in EXPLORER_MODES:
        return "single"

    return requested


def _comparison_fallback_distribution(
    category,
):
    if category == "continuous":
        return "standard_normal"

    return "binomial"


def _comparison_view_keys(
    category,
):
    if category == "continuous":
        return list(
            CONTINUOUS_EXPLORER_VIEWS
        )

    return list(
        DISCRETE_EXPLORER_VIEWS
    )


def _resolve_comparison_view(
    category,
    requested_view,
):
    valid_views = (
        _comparison_view_keys(
            category
        )
    )

    default_view = (
        "pdf"
        if category == "continuous"
        else "pmf"
    )

    if requested_view in valid_views:
        return requested_view

    return default_view


def _default_comparison_state(
    category,
):
    return [
        {
            "distribution":
                item["distribution"],

            "label":
                item["label"],

            "parameters":
                dict(
                    item["parameters"]
                ),
        }
        for item in (
            DEFAULT_COMPARISON_CURVES[
                category
            ]
        )
    ]


def _build_comparison_summary(
    spec,
    label,
    parameters,
):
    parameter_parts = []

    for parameter in spec.parameters:

        if parameter.name not in parameters:
            continue

        parameter_parts.append(
            (
                f"{parameter.symbol} = "
                f"{format_number(
                    parameters[
                        parameter.name
                    ]
                )}"
            )
        )

    return {
        "label":
            label or spec.label,

        "distribution":
            spec.label,

        "parameters":
            ", ".join(
                parameter_parts
            ) or "No adjustable parameters",
    }


def probability(request):
    active_tab = _resolve_tab(
        request
    )

    explorer_mode = (
        _resolve_explorer_mode(
            request
        )
        if active_tab == "explorer"
        else "single"
    )

    # ============================================================
    # Functions state
    # ============================================================

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

    # ============================================================
    # Explorer state
    # ============================================================

    explorer_distribution = (
        "standard_normal"
    )

    explorer_view = "pdf"

    explorer_parameter_state = {}
    explorer_parameter_errors = {}
    explorer_general_errors = []

    explorer_properties = None
    explorer_property_rows = ()
    explorer_quantile_rows = ()
    explorer_support = None
    explorer_chart_html = None

    # ============================================================
    # Explorer comparison state
    # ============================================================

    comparison_category = "continuous"
    comparison_view = "pdf"

    comparison_curve_state = []
    comparison_curve_summaries = []

    comparison_field_errors = {}
    comparison_general_errors = []

    comparison_chart_html = None

    # ============================================================
    # Functions
    # ============================================================

    if active_tab == "functions":

        if request.method == "POST":

            requested_distribution = (
                request.POST.get(
                    "distribution",
                    "standard_normal",
                )
            )

            try:
                spec = (
                    get_distribution_spec(
                        requested_distribution
                    )
                )

                selected_distribution = (
                    requested_distribution
                )

            except ValueError:
                spec = (
                    get_distribution_spec(
                        "standard_normal"
                    )
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
                    "The selected operation is "
                    "not available for this "
                    "distribution."
                )

            operation_ui = (
                get_operation_ui(
                    spec.category,
                    selected_operation,
                )
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

            except (
                DistributionValidationError
            ) as exc:
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

    # ============================================================
    # Distribution Explorer
    # ============================================================

    if (
        active_tab == "explorer"
        and explorer_mode == "single"
    ):

        if request.method == "POST":
            requested_distribution = (
                request.POST.get(
                    "explorer_distribution",
                    "standard_normal",
                )
            )
        else:
            requested_distribution = (
                request.GET.get(
                    "distribution",
                    "standard_normal",
                )
            )

        try:
            explorer_spec = (
                get_distribution_spec(
                    requested_distribution
                )
            )

            explorer_distribution = (
                requested_distribution
            )

        except ValueError:
            explorer_spec = (
                get_distribution_spec(
                    "standard_normal"
                )
            )

            explorer_distribution = (
                "standard_normal"
            )

            explorer_general_errors.append(
                "The selected distribution "
                "is not available."
            )

        if request.method == "POST":
            requested_view = (
                request.POST.get(
                    "explorer_view",
                    "",
                )
            )
        else:
            requested_view = (
                request.GET.get(
                    "view",
                    "",
                )
            )

        explorer_view = (
            _resolve_explorer_view(
                explorer_spec,
                requested_view,
            )
        )

        if (
            request.method == "POST"
            and requested_view
            and requested_view
            != explorer_view
        ):
            explorer_general_errors.append(
                "The selected visualization is "
                "not available for this "
                "distribution."
            )

        if request.method == "POST":
            explorer_parameter_state = (
                _parameter_state_from_post(
                    request,
                    explorer_spec,
                    prefix="explorer_param_",
                )
            )
        else:
            explorer_parameter_state = (
                get_default_parameters(
                    explorer_distribution
                )
            )

        try:
            explorer_properties = (
                get_distribution_properties(
                    explorer_distribution,
                    explorer_parameter_state,
                )
            )

            explorer_property_rows = (
                _build_property_rows(
                    explorer_properties
                )
            )

            explorer_quantile_rows = (
                _build_quantile_rows(
                    explorer_properties
                )
            )

            explorer_support = (
                "["
                + format_number(
                    explorer_properties
                    .support_lower
                )
                + ", "
                + format_number(
                    explorer_properties
                    .support_upper
                )
                + "]"
            )

            explorer_chart_html = (
                explorer_figure_html(
                    explorer_distribution,
                    explorer_parameter_state,
                    view=explorer_view,
                )
            )

        except (
            DistributionValidationError
        ) as exc:
            explorer_parameter_errors.update(
                exc.result.field_errors
            )

            explorer_general_errors.extend(
                exc.result.non_field_errors
            )

        except ExplorerError as exc:
            explorer_general_errors.append(
                str(exc)
            )

    # ============================================================
    # Distribution comparison
    # ============================================================

    if (
        active_tab == "explorer"
        and explorer_mode == "compare"
    ):

        valid_curves = []

        # --------------------------------------------------------
        # POST
        # --------------------------------------------------------

        if request.method == "POST":

            requested_category = (
                request.POST.get(
                    "comparison_category",
                    "continuous",
                )
            )

            if (
                requested_category
                in COMPARISON_CATEGORIES
            ):
                comparison_category = (
                    requested_category
                )

            else:
                comparison_category = (
                    "continuous"
                )

                comparison_general_errors.append(
                    (
                        "The selected comparison "
                        "category is not available."
                    )
                )

            requested_view = (
                request.POST.get(
                    "comparison_view",
                    "",
                )
            )

            comparison_view = (
                _resolve_comparison_view(
                    comparison_category,
                    requested_view,
                )
            )

            if (
                requested_view
                and requested_view
                != comparison_view
            ):
                comparison_general_errors.append(
                    (
                        "The selected comparison "
                        "function is not available."
                    )
                )

            raw_count = request.POST.get(
                "comparison_count",
                str(MIN_COMPARISON_CURVES),
            )

            try:
                comparison_count = int(
                    raw_count
                )

            except (TypeError, ValueError):
                comparison_count = (
                    MIN_COMPARISON_CURVES
                )

                comparison_general_errors.append(
                    (
                        "The number of comparison "
                        "curves is invalid."
                    )
                )

            if (
                comparison_count
                < MIN_COMPARISON_CURVES
                or comparison_count
                > MAX_COMPARISON_CURVES
            ):
                comparison_general_errors.append(
                    (
                        "Comparison requires between "
                        f"{MIN_COMPARISON_CURVES} and "
                        f"{MAX_COMPARISON_CURVES} "
                        "curves."
                    )
                )

                comparison_count = min(
                    max(
                        comparison_count,
                        MIN_COMPARISON_CURVES,
                    ),
                    MAX_COMPARISON_CURVES,
                )

            for index in range(
                comparison_count
            ):

                requested_key = (
                    request.POST.get(
                        (
                            f"compare_{index}_"
                            "distribution"
                        ),
                        _comparison_fallback_distribution(
                            comparison_category
                        ),
                    )
                )

                try:
                    curve_spec = (
                        get_distribution_spec(
                            requested_key
                        )
                    )

                except ValueError:
                    comparison_general_errors.append(
                        (
                            f"Curve {index + 1}: "
                            "the selected distribution "
                            "is not available."
                        )
                    )

                    requested_key = (
                        _comparison_fallback_distribution(
                            comparison_category
                        )
                    )

                    curve_spec = (
                        get_distribution_spec(
                            requested_key
                        )
                    )

                category_is_valid = (
                    curve_spec.category
                    == comparison_category
                )

                if not category_is_valid:
                    comparison_general_errors.append(
                        (
                            f"Curve {index + 1}: "
                            f"{curve_spec.label} is not "
                            f"a {comparison_category} "
                            "distribution."
                        )
                    )

                raw_parameters = {
                    parameter.name:
                        request.POST.get(
                            (
                                f"compare_{index}_"
                                f"param_{parameter.name}"
                            ),
                            "",
                        )
                    for parameter
                    in curve_spec.parameters
                }

                label = (
                    request.POST.get(
                        f"compare_{index}_label",
                        "",
                    )
                    .strip()
                )

                if not label:
                    label = curve_spec.label

                comparison_curve_state.append(
                    {
                        "distribution":
                            requested_key,

                        "label":
                            label,

                        "parameters":
                            raw_parameters,
                    }
                )

                validation = (
                    validate_distribution_parameters(
                        requested_key,
                        raw_parameters,
                    )
                )

                if validation.field_errors:
                    comparison_field_errors[
                        str(index)
                    ] = dict(
                        validation.field_errors
                    )

                for error in (
                    validation.non_field_errors
                ):
                    comparison_general_errors.append(
                        (
                            f"Curve {index + 1}: "
                            f"{error}"
                        )
                    )

                if (
                    validation.is_valid
                    and category_is_valid
                ):
                    valid_curves.append(
                        ComparisonCurve(
                            distribution_key=(
                                requested_key
                            ),
                            parameters=(
                                validation.values
                            ),
                            label=label,
                        )
                    )

                    comparison_curve_summaries.append(
                        _build_comparison_summary(
                            curve_spec,
                            label,
                            validation.values,
                        )
                    )

            has_field_errors = any(
                comparison_field_errors.values()
            )

            if (
                not comparison_general_errors
                and not has_field_errors
                and len(valid_curves)
                >= MIN_COMPARISON_CURVES
            ):
                try:
                    comparison_chart_html = (
                        comparison_figure_html(
                            valid_curves,
                            view=(
                                comparison_view
                            ),
                        )
                    )

                except (
                    ExplorerError,
                    DistributionValidationError,
                ) as exc:
                    comparison_general_errors.append(
                        str(exc)
                    )

        # --------------------------------------------------------
        # GET defaults
        # --------------------------------------------------------

        else:
            comparison_category = (
                "continuous"
            )

            comparison_view = "pdf"

            comparison_curve_state = (
                _default_comparison_state(
                    comparison_category
                )
            )

            for item in (
                comparison_curve_state
            ):
                curve_spec = (
                    get_distribution_spec(
                        item["distribution"]
                    )
                )

                validation = (
                    validate_distribution_parameters(
                        item["distribution"],
                        item["parameters"],
                    )
                )

                if not validation.is_valid:
                    continue

                valid_curves.append(
                    ComparisonCurve(
                        distribution_key=(
                            item[
                                "distribution"
                            ]
                        ),
                        parameters=(
                            validation.values
                        ),
                        label=item["label"],
                    )
                )

                comparison_curve_summaries.append(
                    _build_comparison_summary(
                        curve_spec,
                        item["label"],
                        validation.values,
                    )
                )

            comparison_chart_html = (
                comparison_figure_html(
                    valid_curves,
                    view=comparison_view,
                )
            )

    # ============================================================
    # Shared context
    # ============================================================

    context = {
        "segment":
            "probability",

        "active_tab":
            active_tab,

        "continuous_distributions":
            get_continuous_distributions(),

        "discrete_distributions":
            get_discrete_distributions(),

        "probability_ui_config":
            build_probability_ui_config(),

        # Functions
        "selected_distribution":
            selected_distribution,

        "selected_operation":
            selected_operation,

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

        # Explorer
        "explorer_distribution":
            explorer_distribution,

        "explorer_view":
            explorer_view,

        "explorer_view_choices": [
            {
                "key": view,
                "label":
                    EXPLORER_VIEW_LABELS[
                        view
                    ],
            }
            for view in (
                _explorer_views_for_spec(
                    _resolve_distribution(
                        explorer_distribution
                    )
                )
            )
        ],

        "explorer_form_state": {
            "distribution":
                explorer_distribution,
            "view":
                explorer_view,
            "parameters":
                explorer_parameter_state,
        },

        "explorer_field_errors": {
            "parameters":
                explorer_parameter_errors,
        },

        "explorer_general_errors":
            explorer_general_errors,

        "explorer_properties":
            explorer_properties,

        "explorer_property_rows":
            explorer_property_rows,

        "explorer_quantile_rows":
            explorer_quantile_rows,

        "explorer_support":
            explorer_support,

        "explorer_chart_html":
            explorer_chart_html,

        "explorer_mode":
            explorer_mode,

        "comparison_category":
            comparison_category,

        "comparison_view":
            comparison_view,

        "comparison_view_choices": [
            {
                "key": view,
                "label":
                    EXPLORER_VIEW_LABELS[
                        view
                    ],
            }
            for view in (
                _comparison_view_keys(
                    comparison_category
                )
            )
        ],

        "comparison_form_state": {
            "category":
                comparison_category,

            "view":
                comparison_view,

            "curves":
                comparison_curve_state,

            "max_curves":
                MAX_COMPARISON_CURVES,

            "min_curves":
                MIN_COMPARISON_CURVES,

            "defaults": {
                "continuous":
                    _default_comparison_state(
                        "continuous"
                    ),

                "discrete":
                    _default_comparison_state(
                        "discrete"
                    ),
            },
        },

        "comparison_field_errors":
            comparison_field_errors,

        "comparison_general_errors":
            comparison_general_errors,

        "comparison_curve_summaries":
            comparison_curve_summaries,

        "comparison_chart_html":
            comparison_chart_html,
    }

    return render(
        request,
        "probability/probability.html",
        context,
    )