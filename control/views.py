from __future__ import annotations

import math
import re

from django.shortcuts import render

from control.services import (
    calculate_c_chart,
    calculate_cusum,
    calculate_ewma,
    calculate_individuals_mr,
    calculate_median_r,
    calculate_np_chart,
    calculate_p_chart,
    calculate_precontrol,
    calculate_process_capability,
    calculate_u_chart,
    calculate_vmask_cusum,
    calculate_xbar_r,
    calculate_xbar_s,
    detect_nelson_rules,
    detect_nelson_rules_for_values,
)
from control.services.plotting import (
    capability_html,
    control_chart_html,
    ewma_html,
    precontrol_html,
    tabular_cusum_html,
    vmask_cusum_html,
)


TAB_TOOLS = {
    "variables": (
        ("xbar_r", "X̄-R"),
        ("xbar_s", "X̄-S"),
        ("median_r", "Median-R"),
        ("individuals_mr", "Individuals-MR"),
    ),
    "attributes": (
        ("p_chart", "p Chart"),
        ("np_chart", "np Chart"),
        ("c_chart", "c Chart"),
        ("u_chart", "u Chart"),
    ),
    "advanced": (
        ("cusum", "CUSUM"),
        ("ewma", "EWMA"),
    ),
    "precontrol": (("precontrol", "Precontrol"),),
    "capability": (("capability", "Process Capability"),),
}

DEFAULT_TOOL = {
    tab: tools[0][0]
    for tab, tools in TAB_TOOLS.items()
}

TOOL_META = {
    "xbar_r": {
        "title": "X̄-R Chart",
        "description": "Monitor the process mean and within-subgroup range for rational subgroups.",
        "data_help": "Paste one subgroup per row and one observation per column. Supported subgroup sizes: 2–10, 15 and 25.",
    },
    "xbar_s": {
        "title": "X̄-S Chart",
        "description": "Monitor the process mean and within-subgroup standard deviation.",
        "data_help": "Paste one subgroup per row and one observation per column. Supported subgroup sizes: 2–10, 15 and 25.",
    },
    "median_r": {
        "title": "Median-R Chart",
        "description": "Monitor subgroup medians together with subgroup ranges.",
        "data_help": "Paste one subgroup per row. Supported subgroup sizes: 2–10.",
    },
    "individuals_mr": {
        "title": "Individuals-MR Chart",
        "description": "Monitor individual observations and their moving ranges when rational subgroups are unavailable.",
        "data_help": "Paste observations in one column, row, or rectangular block. Values are read in order.",
    },
    "p_chart": {
        "title": "p Chart",
        "description": "Monitor the proportion of defective units with constant or varying sample sizes.",
        "data_help": "Paste two columns: sample size and number of defective units.",
    },
    "np_chart": {
        "title": "np Chart",
        "description": "Monitor the number of defective units when the sample size is constant.",
        "data_help": "Paste two columns: sample size and number of defective units. Sample size must be constant.",
    },
    "c_chart": {
        "title": "c Chart",
        "description": "Monitor the number of incidences or nonconformities per inspection unit.",
        "data_help": "Paste counts in one column, row, or rectangular block. Values are read in order.",
    },
    "u_chart": {
        "title": "u Chart",
        "description": "Monitor incidences per unit when the number of inspected units can vary.",
        "data_help": "Paste two columns: sample size and number of incidences.",
    },
    "cusum": {
        "title": "CUSUM",
        "description": "Detect small sustained shifts using either tabular CUSUM or the equivalent V-mask representation.",
        "data_help": "Paste one subgroup per row. K and H must be expressed in the same units as the subgroup means.",
    },
    "ewma": {
        "title": "EWMA Chart",
        "description": "Detect gradual process shifts using exponentially weighted subgroup means and dynamic control limits.",
        "data_help": "Paste one subgroup per row and one observation per column.",
    },
    "precontrol": {
        "title": "Precontrol",
        "description": "Classify consecutive measurements into green, yellow and red zones and evaluate setup qualification.",
        "data_help": "Paste individual measurements in the order they were produced.",
    },
    "capability": {
        "title": "Process Capability",
        "description": "Estimate Cp, Cpk, Pp and Ppk from specification limits and observed process variation.",
        "data_help": "Paste observations. For X̄-R or X̄-S within-sigma estimation, use one subgroup per row.",
    },
}

LEGACY_TABS = {
    "shew_1": "variables",
    "shew_2": "attributes",
    "ewma_cusum": "advanced",
    "precontrol_chart": "precontrol",
}

LEGACY_TOOLS = {
    "xr_chart": "xbar_r",
    "xs_chart": "xbar_s",
    "mr_chart": "median_r",
    "individual_chart": "individuals_mr",
    "p_chart": "p_chart",
    "np_chart": "np_chart",
    "c_chart": "c_chart",
    "u_chart": "u_chart",
    "cusum_chart": "cusum",
    "ewma_chart": "ewma",
}


def _resolve_navigation(request):
    requested_tab = request.GET.get("tab", "variables")
    tab = LEGACY_TABS.get(requested_tab, requested_tab)
    if tab not in TAB_TOOLS:
        tab = "variables"

    requested_tool = request.GET.get("tool") or request.GET.get("subtab")
    tool = LEGACY_TOOLS.get(requested_tool, requested_tool)
    valid_tools = {key for key, _ in TAB_TOOLS[tab]}
    if tool not in valid_tools:
        tool = DEFAULT_TOOL[tab]

    return tab, tool


def _split_row(row: str):
    """Split one pasted data row.

    Supported input includes:
      - Excel / Google Sheets tab-separated rows
      - semicolon-separated rows
      - whitespace-separated rows
      - Markdown tables using pipe characters
    """
    row = row.strip()

    if "|" in row:
        # Markdown table rows often begin and end with a pipe.
        row = row.strip("|").strip()
        cells = row.split("|")
    elif "\t" in row:
        cells = row.split("\t")
    elif ";" in row:
        cells = row.split(";")
    else:
        cells = re.split(r"\s+", row)

    return [cell.strip() for cell in cells if cell.strip() != ""]


def _is_markdown_separator_row(cells):
    """Return True for rows such as ``| --: | :--- |``."""
    if not cells:
        return False

    return all(
        re.fullmatch(r":?-{2,}:?", cell.strip())
        for cell in cells
    )


def _parse_matrix(text: str):
    if text is None or not text.strip():
        raise ValueError("Please enter data before calculating.")

    rows = []
    for line_number, raw_line in enumerate(text.replace("\r", "").split("\n"), start=1):
        if not raw_line.strip():
            continue

        cells = _split_row(raw_line)
        if not cells:
            continue

        # Ignore the alignment row automatically added by Markdown tables.
        if _is_markdown_separator_row(cells):
            continue

        try:
            rows.append([float(cell) for cell in cells])
        except ValueError as exc:
            raise ValueError(f"Non-numeric value detected on row {line_number}.") from exc

    if not rows:
        raise ValueError("Please enter at least one numeric observation.")

    return rows


def _input_data_table(text: str):
    """Build a normalized table for the pasted dataset.

    The raw text remains in the form so the user can edit or recalculate,
    while the interface can show a clean server-rendered preview after a
    successful calculation.
    """
    rows = _parse_matrix(text)
    column_count = max(len(row) for row in rows)

    return {
        "headers": [str(index) for index in range(1, column_count + 1)],
        "rows": [
            [
                _fmt(value)
                for value in row
            ]
            for row in rows
        ],
        "row_count": len(rows),
        "column_count": column_count,
    }


def _parse_equal_subgroups(text: str):
    rows = _parse_matrix(text)
    sizes = {len(row) for row in rows}
    if len(sizes) != 1:
        raise ValueError("All subgroups must have the same size.")
    return rows


def _parse_vector(text: str):
    return [value for row in _parse_matrix(text) for value in row]


def _parse_two_columns(text: str):
    rows = _parse_matrix(text)
    if any(len(row) != 2 for row in rows):
        raise ValueError("The dataset must contain exactly two columns.")
    return [row[0] for row in rows], [row[1] for row in rows]


def _optional_float(post, name):
    raw = (post.get(name) or "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name.replace('_', ' ').title()} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name.replace('_', ' ').title()} must be finite.")
    return value


def _required_float(post, name, label=None, default=None):
    raw = (post.get(name) or "").strip()
    if not raw and default is not None:
        return float(default)
    if not raw:
        raise ValueError(f"{label or name.replace('_', ' ').title()} is required.")
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{label or name.replace('_', ' ').title()} must be numeric.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label or name.replace('_', ' ').title()} must be finite.")
    return value


def _required_int(post, name, label=None, default=None):
    value = _required_float(post, name, label=label, default=default)
    if not float(value).is_integer():
        raise ValueError(f"{label or name.replace('_', ' ').title()} must be an integer.")
    return int(value)


def _fmt(value, digits=6):
    if value is None:
        return "—"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(numeric):
        return "Undefined"
    if abs(numeric) >= 10000 or (0 < abs(numeric) < 0.0001):
        return f"{numeric:.5e}"
    text = f"{numeric:.{digits}f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _metric(label, value, note=None):
    return {"label": label, "value": _fmt(value), "note": note or ""}


def _signals_to_context(signals):
    items = []
    for signal in signals:
        items.append(
            {
                "rule": signal.rule,
                "name": signal.name,
                "points": ", ".join(str(index) for index in signal.point_indices),
                "description": signal.description,
            }
        )
    return items


def _signal_point_union(signals):
    points = set()
    for signal in signals:
        points.update(signal.point_indices)
    return tuple(sorted(points))


def _nelson_from_constant_limits(values, centerline, upper, lower):
    sigma = max(abs(float(upper) - float(centerline)), abs(float(centerline) - float(lower))) / 3.0
    if sigma <= 0:
        return ()
    return detect_nelson_rules_for_values(values, centerline=centerline, sigma=sigma)


def _nelson_from_variable_standard_errors(values, centerline, standard_errors):
    z_scores = []
    for value, standard_error in zip(values, standard_errors):
        if standard_error and standard_error > 0:
            z_scores.append((float(value) - float(centerline)) / float(standard_error))
        else:
            z_scores.append(0.0)
    return detect_nelson_rules(z_scores)


def _chart(title, html):
    return {"title": title, "html": html}


def _build_xbar_r(data_text):
    groups = _parse_equal_subgroups(data_text)
    calc = calculate_xbar_r(groups)
    signals = _nelson_from_constant_limits(
        calc.subgroup_means,
        calc.x_centerline,
        calc.x_upper_control_limit,
        calc.x_lower_control_limit,
    )
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.subgroup_means) + 1))

    charts = [
        _chart(
            "X̄ Chart",
            control_chart_html(
                x=x,
                values=calc.subgroup_means,
                centerline=calc.x_centerline,
                upper_control_limit=calc.x_upper_control_limit,
                lower_control_limit=calc.x_lower_control_limit,
                title="X̄ Chart",
                y_title="Subgroup mean",
                signal_indices=signal_points,
                filename="xbar-chart",
            ),
        ),
        _chart(
            "R Chart",
            control_chart_html(
                x=x,
                values=calc.subgroup_ranges,
                centerline=calc.range_centerline,
                upper_control_limit=calc.range_upper_control_limit,
                lower_control_limit=calc.range_lower_control_limit,
                title="Range Chart",
                y_title="Range",
                filename="range-chart",
            ),
        ),
    ]

    return {
        "metrics": [
            _metric("X̄̄", calc.x_centerline),
            _metric("R̄", calc.range_centerline),
            _metric("Estimated σ", calc.estimated_sigma, "R̄ / d2"),
            _metric("Signals", len(signals), "Nelson rules on X̄"),
        ],
        "charts": charts,
        "signals": _signals_to_context(signals),
        "table_headers": ["Subgroup", "Mean", "Range"],
        "table_rows": [
            [index, _fmt(mean), _fmt(rng)]
            for index, (mean, rng) in enumerate(zip(calc.subgroup_means, calc.subgroup_ranges), start=1)
        ],
        "details": [
            ("UCL X̄", _fmt(calc.x_upper_control_limit)),
            ("LCL X̄", _fmt(calc.x_lower_control_limit)),
            ("UCL R", _fmt(calc.range_upper_control_limit)),
            ("LCL R", _fmt(calc.range_lower_control_limit)),
            ("A2", _fmt(calc.A2)),
            ("d2", _fmt(calc.d2)),
            ("D3", _fmt(calc.D3)),
            ("D4", _fmt(calc.D4)),
        ],
    }


def _build_xbar_s(data_text):
    groups = _parse_equal_subgroups(data_text)
    calc = calculate_xbar_s(groups)
    signals = _nelson_from_constant_limits(
        calc.subgroup_means,
        calc.x_centerline,
        calc.x_upper_control_limit,
        calc.x_lower_control_limit,
    )
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.subgroup_means) + 1))

    return {
        "metrics": [
            _metric("X̄̄", calc.x_centerline),
            _metric("s̄", calc.s_centerline),
            _metric("Estimated σ", calc.estimated_sigma, "s̄ / c4"),
            _metric("Signals", len(signals), "Nelson rules on X̄"),
        ],
        "charts": [
            _chart(
                "X̄ chart",
                control_chart_html(
                    x=x,
                    values=calc.subgroup_means,
                    centerline=calc.x_centerline,
                    upper_control_limit=calc.x_upper_control_limit,
                    lower_control_limit=calc.x_lower_control_limit,
                    title="X̄ Chart",
                    y_title="Subgroup mean",
                    signal_indices=signal_points,
                    filename="xbar-s-mean",
                ),
            ),
            _chart(
                "s chart",
                control_chart_html(
                    x=x,
                    values=calc.subgroup_standard_deviations,
                    centerline=calc.s_centerline,
                    upper_control_limit=calc.s_upper_control_limit,
                    lower_control_limit=calc.s_lower_control_limit,
                    title="Standard Deviation Chart",
                    y_title="Sample standard deviation",
                    filename="xbar-s-std",
                ),
            ),
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Subgroup", "Mean", "Std. dev."],
        "table_rows": [
            [index, _fmt(mean), _fmt(std)]
            for index, (mean, std) in enumerate(zip(calc.subgroup_means, calc.subgroup_standard_deviations), start=1)
        ],
        "details": [
            ("UCL X̄", _fmt(calc.x_upper_control_limit)),
            ("LCL X̄", _fmt(calc.x_lower_control_limit)),
            ("UCL s", _fmt(calc.s_upper_control_limit)),
            ("LCL s", _fmt(calc.s_lower_control_limit)),
            ("A3", _fmt(calc.A3)),
            ("c4", _fmt(calc.c4)),
            ("B3", _fmt(calc.B3)),
            ("B4", _fmt(calc.B4)),
        ],
    }


def _build_median_r(data_text):
    groups = _parse_equal_subgroups(data_text)
    calc = calculate_median_r(groups)
    signals = _nelson_from_constant_limits(
        calc.subgroup_medians,
        calc.median_centerline,
        calc.median_upper_control_limit,
        calc.median_lower_control_limit,
    )
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.subgroup_medians) + 1))

    return {
        "metrics": [
            _metric("Median center", calc.median_centerline),
            _metric("R̄", calc.range_centerline),
            _metric("Estimated σ", calc.estimated_sigma, "R̄ / d2"),
            _metric("Signals", len(signals), "Nelson rules on median chart"),
        ],
        "charts": [
            _chart(
                "Median chart",
                control_chart_html(
                    x=x,
                    values=calc.subgroup_medians,
                    centerline=calc.median_centerline,
                    upper_control_limit=calc.median_upper_control_limit,
                    lower_control_limit=calc.median_lower_control_limit,
                    title="Median Chart",
                    y_title="Subgroup median",
                    signal_indices=signal_points,
                    filename="median-chart",
                ),
            ),
            _chart(
                "R chart",
                control_chart_html(
                    x=x,
                    values=calc.subgroup_ranges,
                    centerline=calc.range_centerline,
                    upper_control_limit=calc.range_upper_control_limit,
                    lower_control_limit=calc.range_lower_control_limit,
                    title="Range Chart",
                    y_title="Range",
                    filename="median-range-chart",
                ),
            ),
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Subgroup", "Median", "Range"],
        "table_rows": [
            [index, _fmt(med), _fmt(rng)]
            for index, (med, rng) in enumerate(zip(calc.subgroup_medians, calc.subgroup_ranges), start=1)
        ],
        "details": [
            ("UCL median", _fmt(calc.median_upper_control_limit)),
            ("LCL median", _fmt(calc.median_lower_control_limit)),
            ("UCL R", _fmt(calc.range_upper_control_limit)),
            ("LCL R", _fmt(calc.range_lower_control_limit)),
            ("Ã2", _fmt(calc.A2_tilde)),
            ("d2", _fmt(calc.d2)),
            ("D3", _fmt(calc.D3)),
            ("D4", _fmt(calc.D4)),
        ],
    }


def _build_individuals_mr(data_text, post):
    observations = _parse_vector(data_text)
    range_length = _required_int(post, "moving_range_length", "Moving range length", default=2)
    calc = calculate_individuals_mr(observations, moving_range_length=range_length)
    signals = detect_nelson_rules_for_values(
        calc.observations,
        centerline=calc.individuals_centerline,
        sigma=calc.estimated_sigma,
    ) if calc.estimated_sigma > 0 else ()
    signal_points = _signal_point_union(signals)

    x_ind = list(range(1, len(calc.observations) + 1))
    x_mr = list(range(calc.moving_range_length, len(calc.observations) + 1))

    return {
        "metrics": [
            _metric("X̄", calc.individuals_centerline),
            _metric("MR̄", calc.moving_range_centerline),
            _metric("Estimated σ", calc.estimated_sigma, "MR̄ / d2"),
            _metric("Signals", len(signals), "Nelson rules"),
        ],
        "charts": [
            _chart(
                "Individuals chart",
                control_chart_html(
                    x=x_ind,
                    values=calc.observations,
                    centerline=calc.individuals_centerline,
                    upper_control_limit=calc.individuals_upper_control_limit,
                    lower_control_limit=calc.individuals_lower_control_limit,
                    title="Individuals Chart",
                    y_title="Observation",
                    signal_indices=signal_points,
                    filename="individuals-chart",
                ),
            ),
            _chart(
                "Moving range chart",
                control_chart_html(
                    x=x_mr,
                    values=calc.moving_ranges,
                    centerline=calc.moving_range_centerline,
                    upper_control_limit=calc.moving_range_upper_control_limit,
                    lower_control_limit=calc.moving_range_lower_control_limit,
                    title="Moving Range Chart",
                    y_title="Moving range",
                    filename="moving-range-chart",
                ),
            ),
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Observation", "Value"],
        "table_rows": [[index, _fmt(value)] for index, value in enumerate(calc.observations, start=1)],
        "details": [
            ("MR length", calc.moving_range_length),
            ("UCL X", _fmt(calc.individuals_upper_control_limit)),
            ("LCL X", _fmt(calc.individuals_lower_control_limit)),
            ("UCL MR", _fmt(calc.moving_range_upper_control_limit)),
            ("LCL MR", _fmt(calc.moving_range_lower_control_limit)),
            ("E2", _fmt(calc.E2)),
            ("d2", _fmt(calc.d2)),
        ],
    }


def _build_p_chart(data_text):
    sizes, counts = _parse_two_columns(data_text)
    calc = calculate_p_chart(sizes, counts)
    ses = [
        math.sqrt(calc.centerline * (1.0 - calc.centerline) / n) if 0 < calc.centerline < 1 else 0.0
        for n in calc.sample_sizes
    ]
    signals = _nelson_from_variable_standard_errors(calc.proportions, calc.centerline, ses)
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.proportions) + 1))

    return {
        "metrics": [
            _metric("p̄", calc.centerline),
            _metric("Total units", sum(calc.sample_sizes)),
            _metric("Total defectives", sum(calc.defectives)),
            _metric("Signals", len(signals), "Nelson rules"),
        ],
        "charts": [
            _chart(
                "p chart",
                control_chart_html(
                    x=x,
                    values=calc.proportions,
                    centerline=calc.centerline,
                    upper_control_limit=calc.upper_control_limits,
                    lower_control_limit=calc.lower_control_limits,
                    title="p Chart",
                    y_title="Proportion defective",
                    signal_indices=signal_points,
                    filename="p-chart",
                ),
            )
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Sample", "n", "Defectives", "p", "LCL", "UCL"],
        "table_rows": [
            [i, n, d, _fmt(p), _fmt(lcl), _fmt(ucl)]
            for i, (n, d, p, lcl, ucl) in enumerate(
                zip(calc.sample_sizes, calc.defectives, calc.proportions, calc.lower_control_limits, calc.upper_control_limits),
                start=1,
            )
        ],
        "details": [],
    }


def _build_np_chart(data_text):
    sizes, counts = _parse_two_columns(data_text)
    calc = calculate_np_chart(sizes, counts)
    signals = _nelson_from_constant_limits(
        calc.defectives,
        calc.centerline,
        calc.upper_control_limit,
        calc.lower_control_limit,
    )
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.defectives) + 1))

    return {
        "metrics": [
            _metric("np̄", calc.centerline),
            _metric("p̄", calc.p_bar),
            _metric("Sample size", calc.sample_size),
            _metric("Signals", len(signals), "Nelson rules"),
        ],
        "charts": [
            _chart(
                "np chart",
                control_chart_html(
                    x=x,
                    values=calc.defectives,
                    centerline=calc.centerline,
                    upper_control_limit=calc.upper_control_limit,
                    lower_control_limit=calc.lower_control_limit,
                    title="np Chart",
                    y_title="Number defective",
                    signal_indices=signal_points,
                    filename="np-chart",
                ),
            )
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Sample", "Defectives"],
        "table_rows": [[i, value] for i, value in enumerate(calc.defectives, start=1)],
        "details": [
            ("UCL", _fmt(calc.upper_control_limit)),
            ("LCL", _fmt(calc.lower_control_limit)),
        ],
    }


def _build_c_chart(data_text):
    counts = _parse_vector(data_text)
    calc = calculate_c_chart(counts)
    signals = _nelson_from_constant_limits(
        calc.counts,
        calc.centerline,
        calc.upper_control_limit,
        calc.lower_control_limit,
    ) if calc.centerline > 0 else ()
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.counts) + 1))

    return {
        "metrics": [
            _metric("c̄", calc.centerline),
            _metric("UCL", calc.upper_control_limit),
            _metric("LCL", calc.lower_control_limit),
            _metric("Signals", len(signals), "Nelson rules"),
        ],
        "charts": [
            _chart(
                "c chart",
                control_chart_html(
                    x=x,
                    values=calc.counts,
                    centerline=calc.centerline,
                    upper_control_limit=calc.upper_control_limit,
                    lower_control_limit=calc.lower_control_limit,
                    title="c Chart",
                    y_title="Incidences",
                    signal_indices=signal_points,
                    filename="c-chart",
                ),
            )
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Sample", "Incidences"],
        "table_rows": [[i, value] for i, value in enumerate(calc.counts, start=1)],
        "details": [],
    }


def _build_u_chart(data_text):
    sizes, counts = _parse_two_columns(data_text)
    calc = calculate_u_chart(sizes, counts)
    ses = [math.sqrt(calc.centerline / n) if calc.centerline > 0 else 0.0 for n in calc.sample_sizes]
    signals = _nelson_from_variable_standard_errors(calc.rates, calc.centerline, ses)
    signal_points = _signal_point_union(signals)
    x = list(range(1, len(calc.rates) + 1))

    return {
        "metrics": [
            _metric("ū", calc.centerline),
            _metric("Total units", sum(calc.sample_sizes)),
            _metric("Total incidences", sum(calc.incidences)),
            _metric("Signals", len(signals), "Nelson rules"),
        ],
        "charts": [
            _chart(
                "u chart",
                control_chart_html(
                    x=x,
                    values=calc.rates,
                    centerline=calc.centerline,
                    upper_control_limit=calc.upper_control_limits,
                    lower_control_limit=calc.lower_control_limits,
                    title="u Chart",
                    y_title="Incidences per unit",
                    signal_indices=signal_points,
                    filename="u-chart",
                ),
            )
        ],
        "signals": _signals_to_context(signals),
        "table_headers": ["Sample", "n", "Incidences", "u", "LCL", "UCL"],
        "table_rows": [
            [i, n, c, _fmt(u), _fmt(lcl), _fmt(ucl)]
            for i, (n, c, u, lcl, ucl) in enumerate(
                zip(calc.sample_sizes, calc.incidences, calc.rates, calc.lower_control_limits, calc.upper_control_limits),
                start=1,
            )
        ],
        "details": [],
    }


def _build_cusum(data_text, post):
    groups = _parse_equal_subgroups(data_text)
    method = (post.get("cusum_method") or "tabular").strip().lower()
    if method not in {"tabular", "vmask"}:
        method = "tabular"
    target = _optional_float(post, "target_mean")
    k = _required_float(post, "reference_value", "Reference value K")
    h = _required_float(post, "decision_interval", "Decision interval H")

    if method == "vmask":
        calc = calculate_vmask_cusum(
            groups,
            target_mean=target,
            reference_value=k,
            decision_interval=h,
        )
        signals = [
            {
                "rule": "V-mask",
                "name": f"{signal.direction.title()} shift",
                "points": str(signal.current_index),
                "description": "V-mask boundary crossed by previous cumulative-sum point(s): "
                + ", ".join(str(index) for index in signal.violating_cumulative_indices),
            }
            for signal in calc.signals
        ]
        return {
            "metrics": [
                _metric("Target", calc.target_mean),
                _metric("K", calc.reference_value),
                _metric("H", calc.decision_interval),
                _metric("Lead distance d", calc.lead_distance, "H / K"),
            ],
            "charts": [_chart("CUSUM V-mask", vmask_cusum_html(calc))],
            "signals": signals,
            "table_headers": ["Subgroup", "Mean", "Cumulative sum"],
            "table_rows": [
                [i, _fmt(mean), _fmt(calc.cumulative_sums[i])]
                for i, mean in enumerate(calc.subgroup_means, start=1)
            ],
            "details": [
                ("Upward signal indices", ", ".join(map(str, calc.positive_signal_indices)) or "None"),
                ("Downward signal indices", ", ".join(map(str, calc.negative_signal_indices)) or "None"),
            ],
        }

    calc = calculate_cusum(
        groups,
        target_mean=target,
        reference_value=k,
        decision_interval=h,
    )
    signal_indices = sorted(set(calc.positive_signal_indices) | set(calc.negative_signal_indices))
    signals = []
    for index in signal_indices:
        direction = "Upward" if index in calc.positive_signal_indices else "Downward"
        signals.append(
            {
                "rule": "CUSUM",
                "name": f"{direction} shift",
                "points": str(index),
                "description": "Decision interval H was exceeded.",
            }
        )

    return {
        "metrics": [
            _metric("Target", calc.target_mean),
            _metric("K", calc.reference_value),
            _metric("H", calc.decision_interval),
            _metric("Signals", len(signal_indices)),
        ],
        "charts": [_chart("Tabular CUSUM", tabular_cusum_html(calc))],
        "signals": signals,
        "table_headers": ["Subgroup", "Mean", "C+", "C−"],
        "table_rows": [
            [i, _fmt(mean), _fmt(cplus), _fmt(cminus)]
            for i, (mean, cplus, cminus) in enumerate(
                zip(calc.subgroup_means, calc.positive_cusum, calc.negative_cusum),
                start=1,
            )
        ],
        "details": [],
    }


def _build_ewma(data_text, post):
    groups = _parse_equal_subgroups(data_text)
    target = _optional_float(post, "target_mean")
    lambda_value = _required_float(post, "lambda_value", "Lambda", default=0.2)
    process_sigma = _optional_float(post, "process_sigma")
    control_limit_width = _required_float(post, "control_limit_width", "Control-limit width L", default=3.0)

    calc = calculate_ewma(
        groups,
        target_mean=target,
        lambda_value=lambda_value,
        process_sigma=process_sigma,
        control_limit_width=control_limit_width,
    )

    signals = [
        {
            "rule": "EWMA",
            "name": "Control limit exceeded",
            "points": str(index),
            "description": "The EWMA statistic is outside its dynamic control limits.",
        }
        for index in calc.signal_indices
    ]

    return {
        "metrics": [
            _metric("Target", calc.target_mean),
            _metric("λ", calc.lambda_value),
            _metric("Process σ", calc.process_sigma, calc.sigma_source),
            _metric("Signals", len(calc.signal_indices)),
        ],
        "charts": [_chart("EWMA", ewma_html(calc))],
        "signals": signals,
        "table_headers": ["Subgroup", "Mean", "EWMA", "LCL", "UCL"],
        "table_rows": [
            [i, _fmt(mean), _fmt(ewma), _fmt(lcl), _fmt(ucl)]
            for i, (mean, ewma, lcl, ucl) in enumerate(
                zip(calc.subgroup_means, calc.ewma_values, calc.lower_control_limits, calc.upper_control_limits),
                start=1,
            )
        ],
        "details": [
            ("L", _fmt(calc.control_limit_width)),
            ("Subgroup size", calc.subgroup_size),
        ],
    }


def _build_precontrol(data_text, post):
    observations = _parse_vector(data_text)
    nominal = _required_float(post, "nominal_value", "Nominal value")
    tolerance = _required_float(post, "tolerance_value", "Tolerance value")
    calc = calculate_precontrol(
        observations,
        nominal_value=nominal,
        tolerance_value=tolerance,
    )

    status_class = {
        "qualified": "success",
        "rejected": "danger",
        "pending": "warning",
    }.get(calc.decision.status, "info")

    return {
        "metrics": [
            _metric("Nominal", calc.nominal_value),
            _metric("Tolerance", calc.tolerance_value),
            _metric("Green", calc.green_count),
            _metric("Yellow / Red", calc.yellow_lower_count + calc.yellow_upper_count + calc.red_lower_count + calc.red_upper_count),
        ],
        "charts": [_chart("Precontrol", precontrol_html(calc))],
        "signals": [],
        "status": {
            "class": status_class,
            "title": calc.decision.status.title(),
            "text": calc.decision.action,
            "detail": (
                f"Decision at observation {calc.decision.decision_index}."
                if calc.decision.decision_index is not None
                else "No final decision yet."
            ),
        },
        "table_headers": ["Observation", "Value", "Zone"],
        "table_rows": [[point.index, _fmt(point.value), point.zone.replace("_", " ").title()] for point in calc.points],
        "details": [
            ("LSL", _fmt(calc.lower_spec_limit)),
            ("Green lower", _fmt(calc.green_lower_limit)),
            ("Green upper", _fmt(calc.green_upper_limit)),
            ("USL", _fmt(calc.upper_spec_limit)),
        ],
    }


def _capability_interpretation(cpk):
    if cpk is None:
        return "Capability index unavailable without a within-process sigma estimate."
    if cpk < 1.0:
        return "The within-process spread is not capable of consistently meeting the specification limits."
    if cpk < 1.33:
        return "The process is marginally capable; improvement or closer monitoring is advisable."
    if cpk < 1.67:
        return "The process shows good capability relative to the specification limits."
    return "The process shows strong capability relative to the specification limits."


def _build_capability(data_text, post):
    method = (post.get("sigma_method") or "none").strip().lower()
    groups = _parse_matrix(data_text)
    observations = [value for row in groups for value in row]
    lsl = _optional_float(post, "lsl")
    usl = _optional_float(post, "usl")

    within_sigma = None
    sigma_note = "Not supplied"

    if method == "provided":
        within_sigma = _required_float(post, "within_sigma", "Within-process sigma")
        sigma_note = "Provided"
    elif method == "xbar_r":
        equal_groups = _parse_equal_subgroups(data_text)
        within_sigma = calculate_xbar_r(equal_groups).estimated_sigma
        sigma_note = "Estimated from X̄-R (R̄ / d2)"
    elif method == "xbar_s":
        equal_groups = _parse_equal_subgroups(data_text)
        within_sigma = calculate_xbar_s(equal_groups).estimated_sigma
        sigma_note = "Estimated from X̄-S (s̄ / c4)"
    elif method == "imr":
        within_sigma = calculate_individuals_mr(observations, moving_range_length=2).estimated_sigma
        sigma_note = "Estimated from Individuals-MR"
    elif method != "none":
        raise ValueError("Unknown within-sigma method.")

    calc = calculate_process_capability(
        observations,
        lsl=lsl,
        usl=usl,
        within_sigma=within_sigma,
    )

    main_index = calc.cpk if calc.cpk is not None else calc.ppk
    return {
        "metrics": [
            _metric("Cp", calc.cp),
            _metric("Cpk", calc.cpk),
            _metric("Pp", calc.pp),
            _metric("Ppk", calc.ppk),
        ],
        "charts": [_chart("Capability overview", capability_html(calc, observations))],
        "signals": [],
        "status": {
            "class": "info",
            "title": "Capability summary",
            "text": _capability_interpretation(calc.cpk),
            "detail": f"Within sigma: {sigma_note}. Overall sample sigma: {_fmt(calc.overall_sigma)}.",
        },
        "table_headers": [],
        "table_rows": [],
        "details": [
            ("Sample size", calc.sample_size),
            ("Mean", _fmt(calc.mean)),
            ("LSL", _fmt(calc.lsl)),
            ("USL", _fmt(calc.usl)),
            ("Within σ", _fmt(calc.within_sigma)),
            ("Overall σ", _fmt(calc.overall_sigma)),
            ("Cpl", _fmt(calc.cpl)),
            ("Cpu", _fmt(calc.cpu)),
            ("Ppl", _fmt(calc.ppl)),
            ("Ppu", _fmt(calc.ppu)),
            ("Primary index", _fmt(main_index)),
        ],
    }


BUILDERS = {
    "xbar_r": lambda data, post: _build_xbar_r(data),
    "xbar_s": lambda data, post: _build_xbar_s(data),
    "median_r": lambda data, post: _build_median_r(data),
    "individuals_mr": _build_individuals_mr,
    "p_chart": lambda data, post: _build_p_chart(data),
    "np_chart": lambda data, post: _build_np_chart(data),
    "c_chart": lambda data, post: _build_c_chart(data),
    "u_chart": lambda data, post: _build_u_chart(data),
    "cusum": _build_cusum,
    "ewma": _build_ewma,
    "precontrol": _build_precontrol,
    "capability": _build_capability,
}


def _initial_form_state(tool):
    return {
        "data": "",
        "moving_range_length": "2",
        "cusum_method": "tabular",
        "target_mean": "",
        "reference_value": "",
        "decision_interval": "",
        "lambda_value": "0.2",
        "process_sigma": "",
        "control_limit_width": "3",
        "nominal_value": "",
        "tolerance_value": "",
        "lsl": "",
        "usl": "",
        "sigma_method": "none",
        "within_sigma": "",
    }


def control(request):
    active_tab, active_tool = _resolve_navigation(request)
    form_state = _initial_form_state(active_tool)
    result = None
    error = None
    input_data_table = None

    if request.method == "POST":
        for key in form_state:
            if key in request.POST:
                form_state[key] = request.POST.get(key, "")

        if request.POST.get("action") == "reset":
            form_state = _initial_form_state(active_tool)
        else:
            try:
                result = BUILDERS[active_tool](form_state["data"], request.POST)
                input_data_table = _input_data_table(form_state["data"])
            except (ValueError, TypeError, ArithmeticError) as exc:
                error = str(exc)

    context = {
        "segment": "control",
        "active_tab": active_tab,
        "active_tool": active_tool,
        "tab_tools": TAB_TOOLS,
        "tool_options": TAB_TOOLS[active_tab],
        "tool_meta": TOOL_META[active_tool],
        "form_state": form_state,
        "result": result,
        "error": error,
        "input_data_table": input_data_table,
    }
    return render(request, "control/control.html", context)
