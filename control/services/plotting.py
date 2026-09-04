"""Plotly visualizations for the Statistical Process Control app."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import plotly.graph_objects as go


PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {
        "format": "png",
        "filename": "spc-chart",
        "scale": 2,
    },
}


PROCESS_COLOR = "#2563eb"
CENTERLINE_COLOR = "#475569"
LIMIT_COLOR = "#dc2626"
SIGNAL_COLOR = "#b91c1c"
SECONDARY_COLOR = "#7c3aed"
MASK_COLOR = "#ea580c"
GREEN_FILL = "rgba(34, 197, 94, 0.13)"
YELLOW_FILL = "rgba(234, 179, 8, 0.13)"
RED_FILL = "rgba(239, 68, 68, 0.10)"


def _base_layout(figure: go.Figure, *, title: str, y_title: str) -> None:
    figure.update_layout(
        title={"text": title, "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        hovermode="x unified",
        margin={"l": 60, "r": 24, "t": 62, "b": 52},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={
            "title": "Observation / subgroup",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={"title": y_title, "zeroline": False},
    )


def _to_html(figure: go.Figure, *, filename: str) -> str:
    config = dict(PLOT_CONFIG)
    config["toImageButtonOptions"] = {
        **PLOT_CONFIG["toImageButtonOptions"],
        "filename": filename,
    }
    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=config,
    )


def _as_series(value, n: int) -> list[float]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [float(item) for item in value]
        if len(values) != n:
            raise ValueError("Control-limit series length does not match values.")
        return values
    return [float(value)] * n


def control_chart_html(
    *,
    x,
    values,
    centerline,
    upper_control_limit,
    lower_control_limit,
    title: str,
    y_title: str,
    signal_indices=(),
    filename: str = "spc-control-chart",
) -> str:
    x_values = list(x)
    y_values = [float(value) for value in values]
    n = len(y_values)

    center = _as_series(centerline, n)
    upper = _as_series(upper_control_limit, n)
    lower = _as_series(lower_control_limit, n)

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="lines+markers",
            name="Observed",
            line={"color": PROCESS_COLOR, "width": 2},
            marker={"size": 7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=center,
            mode="lines",
            name="Centerline",
            line={"color": CENTERLINE_COLOR, "dash": "dash", "width": 1.7},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=upper,
            mode="lines",
            name="UCL",
            line={"color": LIMIT_COLOR, "dash": "dot", "width": 1.6},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x_values,
            y=lower,
            mode="lines",
            name="LCL",
            line={"color": LIMIT_COLOR, "dash": "dot", "width": 1.6},
        )
    )

    signal_set = {int(index) for index in signal_indices}
    signal_x = []
    signal_y = []
    for position, (x_value, y_value) in enumerate(zip(x_values, y_values), start=1):
        if position in signal_set or int(x_value) in signal_set:
            signal_x.append(x_value)
            signal_y.append(y_value)

    if signal_x:
        figure.add_trace(
            go.Scatter(
                x=signal_x,
                y=signal_y,
                mode="markers",
                name="Special-cause signal",
                marker={
                    "color": SIGNAL_COLOR,
                    "size": 11,
                    "symbol": "circle-open",
                    "line": {"width": 2.4, "color": SIGNAL_COLOR},
                },
            )
        )

    _base_layout(figure, title=title, y_title=y_title)
    return _to_html(figure, filename=filename)


def tabular_cusum_html(result) -> str:
    x = list(range(1, len(result.subgroup_means) + 1))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=result.positive_cusum,
            mode="lines+markers",
            name="C+",
            line={"color": PROCESS_COLOR, "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=x,
            y=result.negative_cusum,
            mode="lines+markers",
            name="C−",
            line={"color": SECONDARY_COLOR, "width": 2},
        )
    )
    figure.add_hline(
        y=result.decision_interval,
        line={"color": LIMIT_COLOR, "dash": "dot"},
        annotation_text="+H",
    )
    figure.add_hline(
        y=-result.decision_interval,
        line={"color": LIMIT_COLOR, "dash": "dot"},
        annotation_text="−H",
    )
    figure.add_hline(y=0, line={"color": CENTERLINE_COLOR, "dash": "dash"})

    signal_indices = set(result.positive_signal_indices) | set(result.negative_signal_indices)
    if signal_indices:
        sx, sy = [], []
        for index in sorted(signal_indices):
            sx.append(index)
            if index in result.positive_signal_indices:
                sy.append(result.positive_cusum[index - 1])
            else:
                sy.append(result.negative_cusum[index - 1])
        figure.add_trace(
            go.Scatter(
                x=sx,
                y=sy,
                mode="markers",
                name="Signal",
                marker={"color": SIGNAL_COLOR, "size": 11, "symbol": "diamond"},
            )
        )

    _base_layout(figure, title="Tabular CUSUM", y_title="CUSUM")
    return _to_html(figure, filename="cusum-tabular")


def vmask_cusum_html(result) -> str:
    x = list(range(0, len(result.cumulative_sums)))
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=x,
            y=result.cumulative_sums,
            mode="lines+markers",
            name="Cumulative sum",
            line={"color": PROCESS_COLOR, "width": 2},
            marker={"size": 7},
        )
    )

    mask_x = list(result.final_mask_x) + [result.final_vertex_x]
    upper = list(result.final_upper_boundary) + [result.final_vertex_y]
    lower = list(result.final_lower_boundary) + [result.final_vertex_y]

    figure.add_trace(
        go.Scatter(
            x=mask_x,
            y=upper,
            mode="lines",
            name="Upper V-mask arm",
            line={"color": MASK_COLOR, "dash": "dash", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=mask_x,
            y=lower,
            mode="lines",
            name="Lower V-mask arm",
            line={"color": MASK_COLOR, "dash": "dash", "width": 2},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[result.final_vertex_x],
            y=[result.final_vertex_y],
            mode="markers",
            name="V-mask vertex",
            marker={"color": MASK_COLOR, "size": 10, "symbol": "diamond"},
        )
    )

    signal_indices = set(result.positive_signal_indices) | set(result.negative_signal_indices)
    if signal_indices:
        figure.add_trace(
            go.Scatter(
                x=sorted(signal_indices),
                y=[result.cumulative_sums[index] for index in sorted(signal_indices)],
                mode="markers",
                name="Signal",
                marker={"color": SIGNAL_COLOR, "size": 12, "symbol": "circle-open"},
            )
        )

    _base_layout(figure, title="CUSUM with V-mask", y_title="Cumulative deviation")
    figure.update_xaxes(range=[-0.2, result.final_vertex_x + 0.4])
    return _to_html(figure, filename="cusum-vmask")


def ewma_html(result) -> str:
    x = list(range(1, len(result.ewma_values) + 1))
    return control_chart_html(
        x=x,
        values=result.ewma_values,
        centerline=result.target_mean,
        upper_control_limit=result.upper_control_limits,
        lower_control_limit=result.lower_control_limits,
        title="EWMA Chart",
        y_title="EWMA",
        signal_indices=result.signal_indices,
        filename="ewma-chart",
    )


def precontrol_html(result) -> str:
    x = list(range(1, len(result.observations) + 1))
    values = list(result.observations)
    low = min(values + [result.lower_spec_limit])
    high = max(values + [result.upper_spec_limit])
    span = max(high - low, result.tolerance_value * 2)
    plot_low = min(low, result.lower_spec_limit) - 0.12 * span
    plot_high = max(high, result.upper_spec_limit) + 0.12 * span

    figure = go.Figure()
    figure.add_hrect(y0=plot_low, y1=result.lower_spec_limit, fillcolor=RED_FILL, line_width=0)
    figure.add_hrect(y0=result.lower_spec_limit, y1=result.green_lower_limit, fillcolor=YELLOW_FILL, line_width=0)
    figure.add_hrect(y0=result.green_lower_limit, y1=result.green_upper_limit, fillcolor=GREEN_FILL, line_width=0)
    figure.add_hrect(y0=result.green_upper_limit, y1=result.upper_spec_limit, fillcolor=YELLOW_FILL, line_width=0)
    figure.add_hrect(y0=result.upper_spec_limit, y1=plot_high, fillcolor=RED_FILL, line_width=0)

    figure.add_trace(
        go.Scatter(
            x=x,
            y=values,
            mode="lines+markers",
            name="Observation",
            line={"color": PROCESS_COLOR, "width": 2},
            marker={"size": 8},
        )
    )
    figure.add_hline(y=result.nominal_value, line={"color": CENTERLINE_COLOR, "dash": "dash"}, annotation_text="Nominal")
    figure.add_hline(y=result.lower_spec_limit, line={"color": LIMIT_COLOR, "dash": "dot"}, annotation_text="LSL")
    figure.add_hline(y=result.upper_spec_limit, line={"color": LIMIT_COLOR, "dash": "dot"}, annotation_text="USL")
    figure.add_hline(y=result.green_lower_limit, line={"color": "#ca8a04", "dash": "dash"}, annotation_text="Lower PC")
    figure.add_hline(y=result.green_upper_limit, line={"color": "#ca8a04", "dash": "dash"}, annotation_text="Upper PC")

    _base_layout(figure, title="Precontrol Chart", y_title="Measurement")
    figure.update_yaxes(range=[plot_low, plot_high])
    return _to_html(figure, filename="precontrol-chart")


def capability_html(result, observations) -> str:
    values = np.asarray(list(observations), dtype=float)
    figure = go.Figure()
    figure.add_trace(
        go.Histogram(
            x=values,
            histnorm="probability density",
            name="Observed distribution",
            opacity=0.58,
            marker={"color": PROCESS_COLOR},
        )
    )

    if result.overall_sigma > 0:
        lower = float(np.min(values))
        upper = float(np.max(values))
        if result.lsl is not None:
            lower = min(lower, result.lsl)
        if result.usl is not None:
            upper = max(upper, result.usl)
        width = upper - lower or 1.0
        grid = np.linspace(lower - 0.1 * width, upper + 0.1 * width, 500)
        sigma = result.overall_sigma
        density = (
            1.0
            / (sigma * math.sqrt(2.0 * math.pi))
            * np.exp(-0.5 * ((grid - result.mean) / sigma) ** 2)
        )
        figure.add_trace(
            go.Scatter(
                x=grid,
                y=density,
                mode="lines",
                name="Normal reference (overall σ)",
                line={"color": SECONDARY_COLOR, "width": 2},
            )
        )

    figure.add_vline(x=result.mean, line={"color": CENTERLINE_COLOR, "dash": "dash"}, annotation_text="Mean")
    if result.lsl is not None:
        figure.add_vline(x=result.lsl, line={"color": LIMIT_COLOR, "dash": "dot"}, annotation_text="LSL")
    if result.usl is not None:
        figure.add_vline(x=result.usl, line={"color": LIMIT_COLOR, "dash": "dot"}, annotation_text="USL")

    figure.update_layout(
        title={"text": "Process capability overview", "x": 0.01, "xanchor": "left"},
        template="plotly_white",
        margin={"l": 60, "r": 24, "t": 62, "b": 52},
        legend={"orientation": "h", "y": 1.08, "x": 0},
        xaxis={"title": "Measurement"},
        yaxis={"title": "Density"},
        bargap=0.06,
    )
    return _to_html(figure, filename="process-capability")
