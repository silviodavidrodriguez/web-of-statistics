import math

import numpy as np
import plotly.graph_objects as go

from probability.distributions import (
    create_distribution,
    get_distribution_spec,
)

from .calculator import CalculationResult
from .formatting import (
    format_calculation,
    format_number,
)


PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "probability-distribution",
        "scale": 2,
    },
}


BASE_LINE_COLOR = "#2563eb"
SELECTED_COLOR = "rgba(37, 99, 235, 0.28)"
SELECTED_BAR_COLOR = "#2563eb"
UNSELECTED_BAR_COLOR = "#cbd5e1"
BOUNDARY_COLOR = "#475569"


# ================================================================
# Generic helpers
# ================================================================


def _finite_float(
    value,
    fallback=None,
):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return fallback

    if not math.isfinite(value):
        return fallback

    return value


def _operation_anchor_values(
    result: CalculationResult,
) -> list[float]:

    values = []

    for key in ("x", "a", "b"):
        if key in result.inputs:
            value = _finite_float(
                result.inputs[key]
            )

            if value is not None:
                values.append(value)

    if isinstance(
        result.value,
        tuple,
    ):
        for value in result.value:
            value = _finite_float(value)

            if value is not None:
                values.append(value)

    elif result.operation in {
        "left_quantile",
        "right_quantile",
        "quantile",
    }:
        value = _finite_float(
            result.value
        )

        if value is not None:
            values.append(value)

    return values


# ================================================================
# Continuous plot range
# ================================================================


def _continuous_plot_range(
    distribution,
    result: CalculationResult,
) -> tuple[float, float]:

    support_low, support_high = (
        distribution.support()
    )

    support_low = _finite_float(
        support_low
    )

    support_high = _finite_float(
        support_high
    )

    try:
        quantile_low = _finite_float(
            distribution.ppf(0.001)
        )

        quantile_high = _finite_float(
            distribution.ppf(0.999)
        )

    except Exception:
        quantile_low = None
        quantile_high = None

    if (
        quantile_low is None
        or quantile_high is None
        or quantile_low >= quantile_high
    ):
        try:
            mean = _finite_float(
                distribution.mean()
            )

            sd = _finite_float(
                distribution.std()
            )

        except Exception:
            mean = None
            sd = None

        if (
            mean is not None
            and sd is not None
            and sd > 0
        ):
            quantile_low = (
                mean - 4.0 * sd
            )

            quantile_high = (
                mean + 4.0 * sd
            )

        else:
            quantile_low = -10.0
            quantile_high = 10.0

    lower = quantile_low
    upper = quantile_high

    if support_low is not None:
        lower = max(
            lower,
            support_low,
        )

    if support_high is not None:
        upper = min(
            upper,
            support_high,
        )

    anchors = _operation_anchor_values(
        result
    )

    if anchors:
        lower = min(
            lower,
            min(anchors),
        )

        upper = max(
            upper,
            max(anchors),
        )

    if lower == upper:
        padding = (
            abs(lower) * 0.1
            or 1.0
        )

        lower -= padding
        upper += padding

    width = upper - lower

    if width <= 0:
        lower = -10.0
        upper = 10.0
        width = 20.0

    padding = width * 0.06

    if support_low is None:
        lower -= padding

    else:
        lower = max(
            support_low,
            lower - padding,
        )

    if support_high is None:
        upper += padding

    else:
        upper = min(
            support_high,
            upper + padding,
        )

    return (
        float(lower),
        float(upper),
    )


def _continuous_grid(
    distribution,
    result: CalculationResult,
):
    lower, upper = (
        _continuous_plot_range(
            distribution,
            result,
        )
    )

    x = np.linspace(
        lower,
        upper,
        800,
    )

    y = np.asarray(
        distribution.pdf(x),
        dtype=float,
    )

    # Some distributions have infinite density
    # exactly at a support boundary, for example
    # Beta(alpha < 1). Plotly should not receive
    # infinite coordinates.
    y[~np.isfinite(y)] = np.nan

    return x, y


# ================================================================
# Continuous highlighting
# ================================================================


def _add_continuous_region(
    figure,
    x,
    y,
    mask,
    *,
    name="Selected probability",
):
    selected_x = x[mask]
    selected_y = y[mask]

    if len(selected_x) == 0:
        return

    figure.add_trace(
        go.Scatter(
            x=selected_x,
            y=selected_y,
            mode="lines",
            line={
                "width": 0,
            },
            fill="tozeroy",
            fillcolor=SELECTED_COLOR,
            name=name,
            hoverinfo="skip",
        )
    )


def _add_boundary(
    figure,
    value,
    *,
    label=None,
):
    if value is None:
        return

    figure.add_vline(
        x=value,
        line_width=1.4,
        line_dash="dash",
        line_color=BOUNDARY_COLOR,
    )

    if label:
        figure.add_annotation(
            x=value,
            y=1,
            yref="paper",
            text=label,
            showarrow=False,
            yshift=10,
            font={
                "size": 11,
            },
        )


def _build_continuous_figure(
    distribution,
    result: CalculationResult,
):
    x, y = _continuous_grid(
        distribution,
        result,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            name="Probability density",
            line={
                "color": BASE_LINE_COLOR,
                "width": 2.4,
            },
            hovertemplate=(
                "x = %{x:.6g}"
                "<br>Density = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    operation = result.operation
    inputs = result.inputs

    # ------------------------------------------------------------
    # Density
    # ------------------------------------------------------------

    if operation == "density":

        x_value = inputs["x"]

        y_value = _finite_float(
            distribution.pdf(
                x_value
            ),
            0.0,
        )

        _add_boundary(
            figure,
            x_value,
            label=(
                f"x = "
                f"{format_number(x_value)}"
            ),
        )

        if y_value is not None:
            figure.add_trace(
                go.Scatter(
                    x=[x_value],
                    y=[y_value],
                    mode="markers",
                    name="Selected value",
                    marker={
                        "size": 10,
                        "color": BASE_LINE_COLOR,
                    },
                    hovertemplate=(
                        "x = %{x:.6g}"
                        "<br>Density = %{y:.6g}"
                        "<extra></extra>"
                    ),
                )
            )

    # ------------------------------------------------------------
    # Left
    # ------------------------------------------------------------

    elif operation == "left":

        boundary = inputs["x"]

        _add_continuous_region(
            figure,
            x,
            y,
            x <= boundary,
        )

        _add_boundary(
            figure,
            boundary,
            label=format_number(
                boundary
            ),
        )

    # ------------------------------------------------------------
    # Right
    # ------------------------------------------------------------

    elif operation == "right":

        boundary = inputs["x"]

        _add_continuous_region(
            figure,
            x,
            y,
            x >= boundary,
        )

        _add_boundary(
            figure,
            boundary,
            label=format_number(
                boundary
            ),
        )

    # ------------------------------------------------------------
    # Between
    # ------------------------------------------------------------

    elif operation == "between":

        lower = inputs["a"]
        upper = inputs["b"]

        _add_continuous_region(
            figure,
            x,
            y,
            (
                (x >= lower)
                & (x <= upper)
            ),
        )

        _add_boundary(
            figure,
            lower,
            label=format_number(
                lower
            ),
        )

        _add_boundary(
            figure,
            upper,
            label=format_number(
                upper
            ),
        )

    # ------------------------------------------------------------
    # Outside
    # ------------------------------------------------------------

    elif operation == "outside":

        lower = inputs["a"]
        upper = inputs["b"]

        _add_continuous_region(
            figure,
            x,
            y,
            x <= lower,
            name="Left region",
        )

        _add_continuous_region(
            figure,
            x,
            y,
            x >= upper,
            name="Right region",
        )

        _add_boundary(
            figure,
            lower,
            label=format_number(
                lower
            ),
        )

        _add_boundary(
            figure,
            upper,
            label=format_number(
                upper
            ),
        )

    # ------------------------------------------------------------
    # Left quantile
    # ------------------------------------------------------------

    elif operation == "left_quantile":

        boundary = float(
            result.value
        )

        _add_continuous_region(
            figure,
            x,
            y,
            x <= boundary,
        )

        _add_boundary(
            figure,
            boundary,
            label=format_number(
                boundary
            ),
        )

    # ------------------------------------------------------------
    # Right quantile
    # ------------------------------------------------------------

    elif operation == "right_quantile":

        boundary = float(
            result.value
        )

        _add_continuous_region(
            figure,
            x,
            y,
            x >= boundary,
        )

        _add_boundary(
            figure,
            boundary,
            label=format_number(
                boundary
            ),
        )

    # ------------------------------------------------------------
    # Central interval
    # ------------------------------------------------------------

    elif operation == "central_interval":

        lower, upper = result.value

        _add_continuous_region(
            figure,
            x,
            y,
            (
                (x >= lower)
                & (x <= upper)
            ),
        )

        _add_boundary(
            figure,
            lower,
            label=format_number(
                lower
            ),
        )

        _add_boundary(
            figure,
            upper,
            label=format_number(
                upper
            ),
        )

    return figure


# ================================================================
# Discrete plotting range
# ================================================================


def _discrete_quantile_range(
    distribution,
):
    quantile_pairs = (
        (0.0005, 0.9995),
        (0.001, 0.999),
        (0.005, 0.995),
        (0.01, 0.99),
        (0.025, 0.975),
    )

    for low_p, high_p in quantile_pairs:

        low = _finite_float(
            distribution.ppf(
                low_p
            )
        )

        high = _finite_float(
            distribution.ppf(
                high_p
            )
        )

        if (
            low is None
            or high is None
        ):
            continue

        low = int(
            math.floor(low)
        )

        high = int(
            math.ceil(high)
        )

        if high < low:
            continue

        if (
            high - low + 1
            <= 220
        ):
            return low, high

    return low, high


def _discrete_support(
    distribution,
    result: CalculationResult,
):
    support_low, support_high = (
        distribution.support()
    )

    support_low = _finite_float(
        support_low
    )

    support_high = _finite_float(
        support_high
    )

    low, high = (
        _discrete_quantile_range(
            distribution
        )
    )

    if support_low is not None:
        low = max(
            low,
            int(
                math.ceil(
                    support_low
                )
            ),
        )

    if support_high is not None:
        high = min(
            high,
            int(
                math.floor(
                    support_high
                )
            ),
        )

    anchors = [
        int(round(value))
        for value in
        _operation_anchor_values(
            result
        )
    ]

    if anchors:
        low = min(
            low,
            min(anchors),
        )

        high = max(
            high,
            max(anchors),
        )

    # Protect Plotly from pathological or
    # extremely heavy-tailed discrete ranges.
    max_points = 240

    if (
        high - low + 1
        > max_points
    ):
        center = _finite_float(
            distribution.ppf(0.5),
            0.0,
        )

        center = int(
            round(center)
        )

        half = (
            max_points // 2
        )

        display_low = (
            center - half
        )

        display_high = (
            display_low
            + max_points
            - 1
        )

        if anchors:
            anchor_low = min(
                anchors
            )

            anchor_high = max(
                anchors
            )

            if anchor_low < display_low:
                shift = (
                    display_low
                    - anchor_low
                )

                display_low -= shift
                display_high -= shift

            if anchor_high > display_high:
                shift = (
                    anchor_high
                    - display_high
                )

                display_low += shift
                display_high += shift

        if support_low is not None:
            if display_low < support_low:
                shift = (
                    int(support_low)
                    - display_low
                )

                display_low += shift
                display_high += shift

        if support_high is not None:
            if display_high > support_high:
                shift = (
                    display_high
                    - int(support_high)
                )

                display_low -= shift
                display_high -= shift

        low = max(
            low,
            display_low,
        )

        high = min(
            high,
            display_high,
        )

    if high < low:
        high = low

    return np.arange(
        low,
        high + 1,
        dtype=int,
    )


# ================================================================
# Discrete highlighting
# ================================================================


def _discrete_selected_mask(
    values,
    result: CalculationResult,
):
    operation = result.operation
    inputs = result.inputs

    if operation == "mass":
        return (
            values
            == inputs["x"]
        )

    if operation == "less":
        return (
            values
            < inputs["x"]
        )

    if operation == "less_equal":
        return (
            values
            <= inputs["x"]
        )

    if operation == "greater":
        return (
            values
            > inputs["x"]
        )

    if operation == "greater_equal":
        return (
            values
            >= inputs["x"]
        )

    if operation == "between":
        return (
            (values >= inputs["a"])
            & (values <= inputs["b"])
        )

    if operation == "outside":
        return (
            (values <= inputs["a"])
            | (values >= inputs["b"])
        )

    if operation == "quantile":
        return (
            values
            <= result.value
        )

    return np.zeros(
        len(values),
        dtype=bool,
    )


def _build_discrete_figure(
    distribution,
    result: CalculationResult,
):
    x = _discrete_support(
        distribution,
        result,
    )

    probabilities = np.asarray(
        distribution.pmf(x),
        dtype=float,
    )

    probabilities[
        ~np.isfinite(probabilities)
    ] = 0.0

    selected = (
        _discrete_selected_mask(
            x,
            result,
        )
    )

    colors = [
        (
            SELECTED_BAR_COLOR
            if is_selected
            else UNSELECTED_BAR_COLOR
        )
        for is_selected in selected
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=x,
            y=probabilities,
            marker={
                "color": colors,
            },
            name="Probability mass",
            hovertemplate=(
                "X = %{x}"
                "<br>P(X = %{x}) = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    if result.operation == "quantile":

        _add_boundary(
            figure,
            result.value,
            label=(
                f"x = "
                f"{format_number(result.value)}"
            ),
        )

    return figure


# ================================================================
# Shared layout
# ================================================================


def _apply_probability_layout(
    figure,
    result: CalculationResult,
):
    spec = get_distribution_spec(
        result.distribution_key
    )

    formatted = format_calculation(
        result
    )

    y_title = (
        "Density"
        if result.category
        == "continuous"
        else "Probability"
    )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": formatted.expression,
            "x": 0.5,
            "xanchor": "center",
            "font": {
                "size": 18,
            },
        },
        xaxis_title=(
            spec.variable_symbol
        ),
        yaxis_title=y_title,
        hovermode="closest",
        margin={
            "l": 60,
            "r": 30,
            "t": 80,
            "b": 60,
        },
        height=520,
        showlegend=(
            result.category
            == "continuous"
        ),
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    figure.update_xaxes(
        showgrid=True,
        zeroline=False,
    )

    figure.update_yaxes(
        showgrid=True,
        zeroline=False,
        rangemode="tozero",
    )

    if result.category == "discrete":

        x_values = (
            figure.data[0].x
            if figure.data
            else []
        )

        if len(x_values) <= 30:
            figure.update_xaxes(
                dtick=1
            )

    return figure


# ================================================================
# Public API
# ================================================================


def build_calculation_figure(
    result: CalculationResult,
):
    """
    Build the interactive Plotly figure for a
    completed probability calculation.

    The CalculationResult is the single source
    of truth for both the numerical result and
    the graphical highlighted region.
    """

    distribution = create_distribution(
        result.distribution_key,
        result.parameters,
    )

    if result.category == "continuous":

        figure = (
            _build_continuous_figure(
                distribution,
                result,
            )
        )

    elif result.category == "discrete":

        figure = (
            _build_discrete_figure(
                distribution,
                result,
            )
        )

    else:
        raise ValueError(
            (
                "Unsupported distribution "
                f"category: {result.category}"
            )
        )

    return _apply_probability_layout(
        figure,
        result,
    )


def calculation_figure_html(
    result: CalculationResult,
) -> str:

    figure = build_calculation_figure(
        result
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
        div_id="probability-calculation-chart",
    )