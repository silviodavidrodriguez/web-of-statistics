from django.shortcuts import render

from probability.distributions import (
    get_continuous_distributions,
    get_discrete_distributions,
    get_distribution_spec,
)

from django.http import HttpResponse
from django.views.decorators.http import require_POST
import math
import re

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
    SimulationInputError,
    simulate_distribution,
    simulation_figures_html,
    simulation_to_csv,
    SAMPLING_STATISTICS,
    SamplingInputError,
    clt_figure_html,
    lln_figure_html,
    sampling_distribution_figure_html,
    simulate_clt,
    simulate_lln,
    simulate_sampling_distribution,
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


SIMULATION_STATISTICS = (
    (
        "mean",
        "Mean",
    ),
    (
        "variance",
        "Variance",
    ),
    (
        "standard_deviation",
        "Standard deviation",
    ),
    (
        "skewness",
        "Skewness",
    ),
    (
        "excess_kurtosis",
        "Excess kurtosis",
    ),
)


def _simulation_stat_rows(
    result,
):
    rows = []

    for attribute, label in (
        SIMULATION_STATISTICS
    ):
        theoretical = getattr(
            result.theoretical,
            attribute,
        )

        simulated = getattr(
            result.simulated,
            attribute,
        )

        difference = None

        if (
            theoretical is not None
            and simulated is not None
        ):
            difference = (
                simulated
                - theoretical
            )

        rows.append(
            {
                "label":
                    label,

                "theoretical":
                    _format_optional_property(
                        theoretical
                    ),

                "simulated":
                    _format_optional_property(
                        simulated
                    ),

                "difference":
                    _format_optional_property(
                        difference
                    ),
            }
        )

    return rows


VALID_SAMPLING_LABS = {
    "distribution",
    "clt",
    "lln",
}


def _resolve_sampling_lab(
    request,
):
    if request.method == "POST":
        requested = request.POST.get(
            "sampling_lab",
            "distribution",
        )

    else:
        requested = request.GET.get(
            "lab",
            "distribution",
        )

    if requested not in VALID_SAMPLING_LABS:
        return "distribution"

    return requested


def _split_clt_sample_sizes(
    raw_value,
):
    if raw_value is None:
        return []

    return [
        token
        for token in re.split(
            r"[,;\s]+",
            str(raw_value).strip(),
        )
        if token
    ]


def _sampling_summary_rows(
    result,
):
    rows = [
        {
            "label":
                "Mean of simulated statistic",
            "value":
                format_number(
                    result.empirical_mean
                ),
        },
        {
            "label":
                "SD of simulated statistic",
            "value":
                format_number(
                    result
                    .empirical_standard_deviation
                ),
        },
    ]

    if (
        result.theoretical_reference
        is not None
    ):
        rows.append(
            {
                "label":
                    result
                    .theoretical_reference_label,
                "value":
                    format_number(
                        result
                        .theoretical_reference
                    ),
            }
        )

    if (
        result.normal_approximation_sd
        is not None
    ):
        rows.append(
            {
                "label":
                    "Normal approximation SD",
                "value":
                    format_number(
                        result
                        .normal_approximation_sd
                    ),
            }
        )

    return rows


def _clt_summary_rows(
    result,
):
    rows = []

    for sample_size in (
        result.sample_sizes
    ):
        values = (
            result.means_by_size[
                sample_size
            ]
        )

        empirical_mean = float(
            values.mean()
        )

        empirical_sd = float(
            values.std(
                ddof=1
            )
        )

        theoretical_sd = None

        if (
            result.classical_clt_available
            and result.source_variance
            is not None
            and result.source_variance >= 0
        ):
            theoretical_sd = math.sqrt(
                result.source_variance
                / sample_size
            )

        rows.append(
            {
                "sample_size":
                    sample_size,

                "empirical_mean":
                    format_number(
                        empirical_mean
                    ),

                "empirical_sd":
                    format_number(
                        empirical_sd
                    ),

                "theoretical_mean":
                    _format_optional_property(
                        result.source_mean
                    ),

                "theoretical_sd":
                    _format_optional_property(
                        theoretical_sd
                    ),
            }
        )

    return rows


def _lln_summary_rows(
    result,
):
    rows = []

    for index, final_mean in enumerate(
        result.final_means,
        start=1,
    ):
        rows.append(
            {
                "path":
                    index,

                "final_mean":
                    format_number(
                        final_mean
                    ),

                "absolute_error":
                    format_number(
                        abs(
                            final_mean
                            - result
                            .theoretical_mean
                        )
                    ),
            }
        )

    return rows


def probability(request):
    active_tab = _resolve_tab(
        request
    )

    sampling_lab = (
        _resolve_sampling_lab(
            request
        )
        if active_tab == "sampling"
        else "distribution"
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
    # Simulation state
    # ============================================================

    simulation_distribution = (
        "standard_normal"
    )

    simulation_parameter_state = {}

    simulation_sample_size = "1000"
    simulation_seed = ""

    simulation_parameter_errors = {}
    simulation_input_errors = {}
    simulation_general_errors = []

    simulation_result = None
    simulation_stat_rows = ()
    simulation_chart_html = {}

    simulation_export_state = None

    # ============================================================
    # Sampling Distribution state
    # ============================================================

    sampling_distribution = (
        "standard_normal"
    )

    sampling_statistic = "mean"

    sampling_parameter_state = {}
    sampling_parameter_errors = {}

    sampling_sample_size = "30"
    sampling_repetitions = "5000"
    sampling_seed = ""

    sampling_input_errors = {}
    sampling_general_errors = []

    sampling_result = None
    sampling_summary_rows = ()
    sampling_chart_html = None


    # ============================================================
    # CLT state
    # ============================================================

    clt_distribution = "exponential"

    clt_parameter_state = {}
    clt_parameter_errors = {}

    clt_sample_sizes = "1, 5, 30, 100"
    clt_repetitions = "3000"
    clt_seed = ""

    clt_input_errors = {}
    clt_general_errors = []

    clt_result = None
    clt_summary_rows = ()
    clt_chart_html = None


    # ============================================================
    # LLN state
    # ============================================================

    lln_distribution = "exponential"

    lln_parameter_state = {}
    lln_parameter_errors = {}

    lln_max_sample_size = "5000"
    lln_paths = "5"
    lln_seed = ""

    lln_input_errors = {}
    lln_general_errors = []

    lln_result = None
    lln_summary_rows = ()
    lln_chart_html = None

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
    # Simulation
    # ============================================================

    if active_tab == "simulation":

        # --------------------------------------------------------
        # POST
        # --------------------------------------------------------

        if request.method == "POST":

            requested_distribution = (
                request.POST.get(
                    "simulation_distribution",
                    "standard_normal",
                )
            )

            try:
                simulation_spec = (
                    get_distribution_spec(
                        requested_distribution
                    )
                )

                simulation_distribution = (
                    requested_distribution
                )

            except ValueError:
                simulation_spec = (
                    get_distribution_spec(
                        "standard_normal"
                    )
                )

                simulation_distribution = (
                    "standard_normal"
                )

                simulation_general_errors.append(
                    (
                        "The selected distribution "
                        "is not available."
                    )
                )

            simulation_parameter_state = (
                _parameter_state_from_post(
                    request,
                    simulation_spec,
                    prefix="simulation_param_",
                )
            )

            simulation_sample_size = (
                request.POST.get(
                    "simulation_sample_size",
                    "1000",
                )
            )

            simulation_seed = (
                request.POST.get(
                    "simulation_seed",
                    "",
                )
            )

            try:
                simulation_result = (
                    simulate_distribution(
                        simulation_distribution,
                        simulation_parameter_state,
                        sample_size=(
                            simulation_sample_size
                        ),
                        seed=simulation_seed,
                    )
                )

                # Store normalized values and the
                # effective seed actually used.
                simulation_parameter_state = (
                    dict(
                        simulation_result.parameters
                    )
                )

                simulation_sample_size = str(
                    simulation_result.sample_size
                )

                simulation_seed = str(
                    simulation_result.seed
                )

                simulation_stat_rows = (
                    _simulation_stat_rows(
                        simulation_result
                    )
                )

                simulation_chart_html = (
                    simulation_figures_html(
                        simulation_result
                    )
                )

                simulation_export_state = {
                    "distribution":
                        simulation_result
                        .distribution_key,

                    "parameters":
                        dict(
                            simulation_result
                            .parameters
                        ),

                    "sample_size":
                        simulation_result
                        .sample_size,

                    "seed":
                        simulation_result.seed,
                }

            except (
                DistributionValidationError
            ) as exc:
                simulation_parameter_errors.update(
                    exc.result.field_errors
                )

                simulation_general_errors.extend(
                    exc.result.non_field_errors
                )

            except SimulationInputError as exc:
                simulation_input_errors.update(
                    exc.field_errors
                )

                simulation_general_errors.extend(
                    exc.non_field_errors
                )

        # --------------------------------------------------------
        # GET defaults
        # --------------------------------------------------------

        else:
            simulation_spec = (
                get_distribution_spec(
                    simulation_distribution
                )
            )

            simulation_parameter_state = (
                get_default_parameters(
                    simulation_distribution
                )
            )

    # ============================================================
    # Sampling Distribution
    # ============================================================

    if (
        active_tab == "sampling"
        and sampling_lab == "distribution"
    ):

        if request.method == "POST":

            requested_distribution = (
                request.POST.get(
                    "sampling_distribution",
                    "standard_normal",
                )
            )

            try:
                sampling_spec = (
                    get_distribution_spec(
                        requested_distribution
                    )
                )

                sampling_distribution = (
                    requested_distribution
                )

            except ValueError:
                sampling_spec = (
                    get_distribution_spec(
                        "standard_normal"
                    )
                )

                sampling_distribution = (
                    "standard_normal"
                )

                sampling_general_errors.append(
                    (
                        "The selected distribution "
                        "is not available."
                    )
                )

            sampling_parameter_state = (
                _parameter_state_from_post(
                    request,
                    sampling_spec,
                    prefix="sampling_param_",
                )
            )

            sampling_statistic = (
                request.POST.get(
                    "sampling_statistic",
                    "mean",
                )
            )

            sampling_sample_size = (
                request.POST.get(
                    "sampling_sample_size",
                    "30",
                )
            )

            sampling_repetitions = (
                request.POST.get(
                    "sampling_repetitions",
                    "5000",
                )
            )

            sampling_seed = (
                request.POST.get(
                    "sampling_seed",
                    "",
                )
            )

            try:
                sampling_result = (
                    simulate_sampling_distribution(
                        sampling_distribution,
                        sampling_parameter_state,
                        statistic=(
                            sampling_statistic
                        ),
                        sample_size=(
                            sampling_sample_size
                        ),
                        repetitions=(
                            sampling_repetitions
                        ),
                        seed=sampling_seed,
                    )
                )

                sampling_parameter_state = dict(
                    sampling_result.parameters
                )

                sampling_statistic = (
                    sampling_result.statistic
                )

                sampling_sample_size = str(
                    sampling_result.sample_size
                )

                sampling_repetitions = str(
                    sampling_result.repetitions
                )

                sampling_seed = str(
                    sampling_result.seed
                )

                sampling_summary_rows = (
                    _sampling_summary_rows(
                        sampling_result
                    )
                )

                sampling_chart_html = (
                    sampling_distribution_figure_html(
                        sampling_result
                    )
                )

            except (
                DistributionValidationError
            ) as exc:
                sampling_parameter_errors.update(
                    exc.result.field_errors
                )

                sampling_general_errors.extend(
                    exc.result.non_field_errors
                )

            except SamplingInputError as exc:
                sampling_input_errors.update(
                    exc.field_errors
                )

                sampling_general_errors.extend(
                    exc.non_field_errors
                )

        else:
            sampling_parameter_state = (
                get_default_parameters(
                    sampling_distribution
                )
            )

    # ============================================================
    # Central Limit Theorem
    # ============================================================

    if (
        active_tab == "sampling"
        and sampling_lab == "clt"
    ):

        if request.method == "POST":

            requested_distribution = (
                request.POST.get(
                    "clt_distribution",
                    "exponential",
                )
            )

            try:
                clt_spec = (
                    get_distribution_spec(
                        requested_distribution
                    )
                )

                clt_distribution = (
                    requested_distribution
                )

            except ValueError:
                clt_spec = (
                    get_distribution_spec(
                        "exponential"
                    )
                )

                clt_distribution = (
                    "exponential"
                )

                clt_general_errors.append(
                    (
                        "The selected distribution "
                        "is not available."
                    )
                )

            clt_parameter_state = (
                _parameter_state_from_post(
                    request,
                    clt_spec,
                    prefix="clt_param_",
                )
            )

            clt_sample_sizes = (
                request.POST.get(
                    "clt_sample_sizes",
                    "1, 5, 30, 100",
                )
            )

            clt_repetitions = (
                request.POST.get(
                    "clt_repetitions",
                    "3000",
                )
            )

            clt_seed = (
                request.POST.get(
                    "clt_seed",
                    "",
                )
            )

            try:
                clt_result = simulate_clt(
                    clt_distribution,
                    clt_parameter_state,
                    sample_sizes=(
                        _split_clt_sample_sizes(
                            clt_sample_sizes
                        )
                    ),
                    repetitions=(
                        clt_repetitions
                    ),
                    seed=clt_seed,
                )

                clt_parameter_state = dict(
                    clt_result.parameters
                )

                clt_sample_sizes = ", ".join(
                    str(value)
                    for value
                    in clt_result.sample_sizes
                )

                clt_repetitions = str(
                    clt_result.repetitions
                )

                clt_seed = str(
                    clt_result.seed
                )

                clt_summary_rows = (
                    _clt_summary_rows(
                        clt_result
                    )
                )

                clt_chart_html = (
                    clt_figure_html(
                        clt_result
                    )
                )

            except (
                DistributionValidationError
            ) as exc:
                clt_parameter_errors.update(
                    exc.result.field_errors
                )

                clt_general_errors.extend(
                    exc.result.non_field_errors
                )

            except SamplingInputError as exc:
                clt_input_errors.update(
                    exc.field_errors
                )

                clt_general_errors.extend(
                    exc.non_field_errors
                )

        else:
            clt_parameter_state = (
                get_default_parameters(
                    clt_distribution
                )
            )

    # ============================================================
    # Law of Large Numbers
    # ============================================================

    if (
        active_tab == "sampling"
        and sampling_lab == "lln"
    ):

        if request.method == "POST":

            requested_distribution = (
                request.POST.get(
                    "lln_distribution",
                    "exponential",
                )
            )

            try:
                lln_spec = (
                    get_distribution_spec(
                        requested_distribution
                    )
                )

                lln_distribution = (
                    requested_distribution
                )

            except ValueError:
                lln_spec = (
                    get_distribution_spec(
                        "exponential"
                    )
                )

                lln_distribution = (
                    "exponential"
                )

                lln_general_errors.append(
                    (
                        "The selected distribution "
                        "is not available."
                    )
                )

            lln_parameter_state = (
                _parameter_state_from_post(
                    request,
                    lln_spec,
                    prefix="lln_param_",
                )
            )

            lln_max_sample_size = (
                request.POST.get(
                    "lln_max_sample_size",
                    "5000",
                )
            )

            lln_paths = (
                request.POST.get(
                    "lln_paths",
                    "5",
                )
            )

            lln_seed = (
                request.POST.get(
                    "lln_seed",
                    "",
                )
            )

            try:
                lln_result = simulate_lln(
                    lln_distribution,
                    lln_parameter_state,
                    max_sample_size=(
                        lln_max_sample_size
                    ),
                    paths=lln_paths,
                    seed=lln_seed,
                )

                lln_parameter_state = dict(
                    lln_result.parameters
                )

                lln_max_sample_size = str(
                    lln_result.max_sample_size
                )

                lln_paths = str(
                    lln_result.paths
                )

                lln_seed = str(
                    lln_result.seed
                )

                lln_summary_rows = (
                    _lln_summary_rows(
                        lln_result
                    )
                )

                lln_chart_html = (
                    lln_figure_html(
                        lln_result
                    )
                )

            except (
                DistributionValidationError
            ) as exc:
                lln_parameter_errors.update(
                    exc.result.field_errors
                )

                lln_general_errors.extend(
                    exc.result.non_field_errors
                )

            except SamplingInputError as exc:
                lln_input_errors.update(
                    exc.field_errors
                )

                lln_general_errors.extend(
                    exc.non_field_errors
                )

        else:
            lln_parameter_state = (
                get_default_parameters(
                    lln_distribution
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

        "simulation_distribution":
            simulation_distribution,

        "simulation_form_state": {
            "distribution":
                simulation_distribution,

            "parameters":
                simulation_parameter_state,

            "sample_size":
                simulation_sample_size,

            "seed":
                simulation_seed,
        },

        "simulation_field_errors": {
            "parameters":
                simulation_parameter_errors,

            "inputs":
                simulation_input_errors,
        },

        "simulation_general_errors":
            simulation_general_errors,

        "simulation_result":
            simulation_result,

        "simulation_stat_rows":
            simulation_stat_rows,

        "simulation_chart_html":
            simulation_chart_html,

        "simulation_export_state":
            simulation_export_state,

        "sampling_lab":
            sampling_lab,

        "sampling_statistic_choices": [
            {
                "key": key,
                "label": label,
            }
            for key, label
            in SAMPLING_STATISTICS.items()
        ],


        # Sampling Distribution
        "sampling_distribution":
            sampling_distribution,

        "sampling_form_state": {
            "distribution":
                sampling_distribution,
            "parameters":
                sampling_parameter_state,
            "statistic":
                sampling_statistic,
            "sample_size":
                sampling_sample_size,
            "repetitions":
                sampling_repetitions,
            "seed":
                sampling_seed,
        },

        "sampling_field_errors": {
            "parameters":
                sampling_parameter_errors,
            "inputs":
                sampling_input_errors,
        },

        "sampling_general_errors":
            sampling_general_errors,

        "sampling_result":
            sampling_result,

        "sampling_summary_rows":
            sampling_summary_rows,

        "sampling_chart_html":
            sampling_chart_html,


        # CLT
        "clt_distribution":
            clt_distribution,

        "clt_form_state": {
            "distribution":
                clt_distribution,
            "parameters":
                clt_parameter_state,
            "sample_sizes":
                clt_sample_sizes,
            "repetitions":
                clt_repetitions,
            "seed":
                clt_seed,
        },

        "clt_field_errors": {
            "parameters":
                clt_parameter_errors,
            "inputs":
                clt_input_errors,
        },

        "clt_general_errors":
            clt_general_errors,

        "clt_result":
            clt_result,

        "clt_summary_rows":
            clt_summary_rows,

        "clt_chart_html":
            clt_chart_html,


        # LLN
        "lln_distribution":
            lln_distribution,

        "lln_form_state": {
            "distribution":
                lln_distribution,
            "parameters":
                lln_parameter_state,
            "max_sample_size":
                lln_max_sample_size,
            "paths":
                lln_paths,
            "seed":
                lln_seed,
        },

        "lln_field_errors": {
            "parameters":
                lln_parameter_errors,
            "inputs":
                lln_input_errors,
        },

        "lln_general_errors":
            lln_general_errors,

        "lln_result":
            lln_result,

        "lln_summary_rows":
            lln_summary_rows,

        "lln_chart_html":
            lln_chart_html,
    }

    return render(
        request,
        "probability/probability.html",
        context,
    )


@require_POST
def probability_simulation_export(
    request,
):
    distribution_key = (
        request.POST.get(
            "distribution",
            "standard_normal",
        )
    )

    try:
        spec = get_distribution_spec(
            distribution_key
        )

    except ValueError:
        return HttpResponse(
            "Invalid distribution.",
            status=400,
            content_type=(
                "text/plain; charset=utf-8"
            ),
        )

    parameters = {
        parameter.name:
            request.POST.get(
                f"param_{parameter.name}",
                "",
            )
        for parameter in spec.parameters
    }

    try:
        result = simulate_distribution(
            distribution_key,
            parameters,
            sample_size=(
                request.POST.get(
                    "sample_size",
                    ""
                )
            ),
            seed=(
                request.POST.get(
                    "seed",
                    ""
                )
            ),
        )

    except (
        DistributionValidationError,
        SimulationInputError,
    ) as exc:
        return HttpResponse(
            str(exc),
            status=400,
            content_type=(
                "text/plain; charset=utf-8"
            ),
        )

    content = simulation_to_csv(
        result
    )

    action = request.POST.get(
        "action",
        "download",
    )

    if action == "copy":
        return HttpResponse(
            content,
            content_type=(
                "text/plain; charset=utf-8"
            ),
        )

    response = HttpResponse(
        content,
        content_type=(
            "text/csv; charset=utf-8"
        ),
    )

    filename = (
        "probability-simulation-"
        f"{distribution_key}-"
        f"seed-{result.seed}.csv"
    )

    response[
        "Content-Disposition"
    ] = (
        f'attachment; filename="{filename}"'
    )

    return response