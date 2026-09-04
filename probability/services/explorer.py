import math

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from probability.distributions import (
    create_distribution,
    get_distribution_spec,
)

from .validators import (
    require_valid_distribution_parameters,
)

from .plotting import PLOT_CONFIG


CONTINUOUS_EXPLORER_VIEWS = (
    "pdf",
    "cdf",
    "survival",
    "hazard",
)

DISCRETE_EXPLORER_VIEWS = (
    "pmf",
    "cdf",
    "survival",
)


@dataclass(frozen=True)
class DistributionProperties:
    distribution_key: str
    distribution_label: str
    category: str

    parameters: dict[str, int | float]

    mean: float | None
    median: float | None
    variance: float | None
    standard_deviation: float | None
    skewness: float | None
    excess_kurtosis: float | None

    support_lower: float
    support_upper: float

    quantiles: dict[float, float]


@dataclass(frozen=True)
class ComparisonCurve:
    distribution_key: str
    parameters: Mapping[str, Any]
    label: str


class ExplorerError(ValueError):
    pass


def _optional_finite(
    value,
) -> float | None:

    try:
        value = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _support_value(
    value,
) -> float:

    value = float(value)

    if math.isnan(value):
        raise ExplorerError(
            "The distribution returned an "
            "undefined support boundary."
        )

    return value


def get_distribution_properties(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
    *,
    quantile_probabilities: Sequence[float] = (
        0.01,
        0.025,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        0.975,
        0.99,
    ),
) -> DistributionProperties:

    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    try:
        mean, variance, skewness, kurtosis = (
            distribution.stats(
                moments="mvsk"
            )
        )
    except Exception:
        mean = distribution.mean()
        variance = distribution.var()
        skewness = distribution.stats(
            moments="s"
        )
        kurtosis = distribution.stats(
            moments="k"
        )

    mean = _optional_finite(mean)
    variance = _optional_finite(
        variance
    )
    skewness = _optional_finite(
        skewness
    )
    kurtosis = _optional_finite(
        kurtosis
    )

    try:
        median = _optional_finite(
            distribution.median()
        )
    except Exception:
        median = None

    standard_deviation = None

    if (
        variance is not None
        and variance >= 0
    ):
        standard_deviation = (
            math.sqrt(variance)
        )

    support_lower, support_upper = (
        distribution.support()
    )

    quantiles = {}

    for probability in (
        quantile_probabilities
    ):
        probability = float(
            probability
        )

        if not 0 < probability < 1:
            raise ExplorerError(
                (
                    "Explorer quantile "
                    "probabilities must be "
                    "strictly between 0 and 1."
                )
            )

        value = distribution.ppf(
            probability
        )

        value = _optional_finite(
            value
        )

        if value is not None:
            quantiles[
                probability
            ] = value

    return DistributionProperties(
        distribution_key=distribution_key,
        distribution_label=spec.label,
        category=spec.category,
        parameters=dict(parameters),
        mean=mean,
        median=median,
        variance=variance,
        standard_deviation=(
            standard_deviation
        ),
        skewness=skewness,
        excess_kurtosis=kurtosis,
        support_lower=_support_value(
            support_lower
        ),
        support_upper=_support_value(
            support_upper
        ),
        quantiles=quantiles,
    )


def _continuous_range(
    distribution,
) -> tuple[float, float]:

    lower = _optional_finite(
        distribution.ppf(0.001)
    )

    upper = _optional_finite(
        distribution.ppf(0.999)
    )

    if (
        lower is not None
        and upper is not None
        and lower < upper
    ):
        return lower, upper

    mean = _optional_finite(
        distribution.mean()
    )

    sd = _optional_finite(
        distribution.std()
    )

    if (
        mean is not None
        and sd is not None
        and sd > 0
    ):
        return (
            mean - 4.0 * sd,
            mean + 4.0 * sd,
        )

    support_lower, support_upper = (
        distribution.support()
    )

    support_lower = _optional_finite(
        support_lower
    )

    support_upper = _optional_finite(
        support_upper
    )

    if (
        support_lower is not None
        and support_upper is not None
        and support_lower < support_upper
    ):
        return (
            support_lower,
            support_upper,
        )

    return -10.0, 10.0


def _continuous_grid(
    distribution,
):
    lower, upper = (
        _continuous_range(
            distribution
        )
    )

    return np.linspace(
        lower,
        upper,
        900,
    )


def _discrete_support_grid(
    distribution,
):
    support_lower, support_upper = (
        distribution.support()
    )

    finite_lower = _optional_finite(
        support_lower
    )

    finite_upper = _optional_finite(
        support_upper
    )

    # For finite discrete distributions,
    # show the entire support whenever it is
    # reasonably small.
    if (
        finite_lower is not None
        and finite_upper is not None
    ):
        low = int(
            math.ceil(
                finite_lower
            )
        )

        high = int(
            math.floor(
                finite_upper
            )
        )

        if (
            high >= low
            and high - low + 1 <= 220
        ):
            return np.arange(
                low,
                high + 1,
                dtype=int,
            )

    low = _optional_finite(
        distribution.ppf(
            0.001
        )
    )

    high = _optional_finite(
        distribution.ppf(
            0.999
        )
    )

    if low is None:
        low = 0

    if high is None:
        high = low + 100

    low = int(
        math.floor(low)
    )

    high = int(
        math.ceil(high)
    )

    if finite_lower is not None:
        low = max(
            low,
            int(
                math.ceil(
                    finite_lower
                )
            ),
        )

    if finite_upper is not None:
        high = min(
            high,
            int(
                math.floor(
                    finite_upper
                )
            ),
        )

    if high < low:
        high = low

    if high - low + 1 > 220:
        median = _optional_finite(
            distribution.ppf(
                0.5
            )
        )

        if median is None:
            median = (
                low + high
            ) / 2.0

        center = int(
            round(median)
        )

        low = center - 109
        high = center + 110

        if finite_lower is not None:
            if low < finite_lower:
                shift = (
                    int(
                        math.ceil(
                            finite_lower
                        )
                    )
                    - low
                )

                low += shift
                high += shift

        if finite_upper is not None:
            if high > finite_upper:
                shift = (
                    high
                    - int(
                        math.floor(
                            finite_upper
                        )
                    )
                )

                low -= shift
                high -= shift

    return np.arange(
        low,
        high + 1,
        dtype=int,
    )


def _validate_view(
    spec,
    view: str,
):

    if spec.category == "continuous":
        valid_views = (
            CONTINUOUS_EXPLORER_VIEWS
        )
    else:
        valid_views = (
            DISCRETE_EXPLORER_VIEWS
        )

    if view not in valid_views:
        raise ExplorerError(
            (
                f"View '{view}' is not "
                f"available for "
                f"{spec.category} "
                f"distributions."
            )
        )

    if (
        view == "hazard"
        and not spec.supports_hazard
    ):
        raise ExplorerError(
            (
                "The hazard function is not "
                "available for this "
                "distribution."
            )
        )


def _continuous_values(
    distribution,
    x,
    view,
):

    if view == "pdf":
        values = distribution.pdf(x)

    elif view == "cdf":
        values = distribution.cdf(x)

    elif view == "survival":
        values = distribution.sf(x)

    elif view == "hazard":
        density = np.asarray(
            distribution.pdf(x),
            dtype=float,
        )

        survival = np.asarray(
            distribution.sf(x),
            dtype=float,
        )

        values = np.full(
            density.shape,
            np.nan,
            dtype=float,
        )

        valid = (
            np.isfinite(density)
            & np.isfinite(survival)
            & (survival > 1e-14)
        )

        values[valid] = (
            density[valid]
            / survival[valid]
        )

    else:
        raise ExplorerError(
            f"Unknown explorer view: {view}"
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    values[
        ~np.isfinite(values)
    ] = np.nan

    return values


def _discrete_values(
    distribution,
    x,
    view,
):

    if view == "pmf":
        values = distribution.pmf(x)

    elif view == "cdf":
        values = distribution.cdf(x)

    elif view == "survival":
        values = distribution.sf(x)

    else:
        raise ExplorerError(
            f"Unknown explorer view: {view}"
        )

    values = np.asarray(
        values,
        dtype=float,
    )

    values[
        ~np.isfinite(values)
    ] = np.nan

    return values


def _view_label(
    category,
    view,
):
    labels = {
        "pdf": "Probability density",
        "pmf": "Probability mass",
        "cdf": "Cumulative probability",
        "survival": "Survival probability",
        "hazard": "Hazard",
    }

    return labels.get(
        view,
        view,
    )


def build_explorer_figure(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
    *,
    view: str | None = None,
):
    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    if view is None:
        view = (
            "pdf"
            if spec.category
            == "continuous"
            else "pmf"
        )

    _validate_view(
        spec,
        view,
    )

    figure = go.Figure()

    if spec.category == "continuous":

        x = _continuous_grid(
            distribution
        )

        y = _continuous_values(
            distribution,
            x,
            view,
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=spec.label,
                hovertemplate=(
                    "x = %{x:.6g}"
                    "<br>Value = %{y:.6g}"
                    "<extra></extra>"
                ),
            )
        )

    else:

        x = _discrete_support_grid(
            distribution
        )

        y = _discrete_values(
            distribution,
            x,
            view,
        )

        if view == "pmf":
            figure.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    name=spec.label,
                    hovertemplate=(
                        "X = %{x}"
                        "<br>P(X = %{x}) = "
                        "%{y:.6g}"
                        "<extra></extra>"
                    ),
                )
            )

        else:
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    name=spec.label,
                    hovertemplate=(
                        "X = %{x}"
                        "<br>Value = %{y:.6g}"
                        "<extra></extra>"
                    ),
                )
            )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"{spec.label} — "
                f"{_view_label(spec.category, view)}"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=(
            spec.variable_symbol
        ),
        yaxis_title=(
            _view_label(
                spec.category,
                view,
            )
        ),
        hovermode="closest",
        height=520,
        margin={
            "l": 60,
            "r": 30,
            "t": 75,
            "b": 60,
        },
    )

    figure.update_yaxes(
        rangemode="tozero"
    )

    if (
        spec.category == "discrete"
        and len(x) <= 30
    ):
        figure.update_xaxes(
            dtick=1
        )

    return figure


def build_comparison_figure(
    curves: Sequence[ComparisonCurve],
    *,
    view: str | None = None,
):
    if not curves:
        raise ExplorerError(
            (
                "At least one comparison "
                "curve is required."
            )
        )

    prepared = []

    categories = set()

    for curve in curves:

        spec = get_distribution_spec(
            curve.distribution_key
        )

        parameters = (
            require_valid_distribution_parameters(
                curve.distribution_key,
                curve.parameters,
            )
        )

        distribution = create_distribution(
            curve.distribution_key,
            parameters,
        )

        categories.add(
            spec.category
        )

        prepared.append(
            (
                curve,
                spec,
                distribution,
            )
        )

    if len(categories) != 1:
        raise ExplorerError(
            (
                "Comparison curves must all "
                "belong to the same distribution "
                "category."
            )
        )

    category = next(
        iter(categories)
    )

    if view is None:
        view = (
            "pdf"
            if category == "continuous"
            else "pmf"
        )

    for _, spec, _ in prepared:
        _validate_view(
            spec,
            view,
        )

    figure = go.Figure()

    if category == "continuous":

        ranges = [
            _continuous_range(
                distribution
            )
            for _, _, distribution
            in prepared
        ]

        lower = min(
            item[0]
            for item in ranges
        )

        upper = max(
            item[1]
            for item in ranges
        )

        x = np.linspace(
            lower,
            upper,
            1000,
        )

        for (
            curve,
            spec,
            distribution,
        ) in prepared:

            y = _continuous_values(
                distribution,
                x,
                view,
            )

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=curve.label,
                    hovertemplate=(
                        f"{curve.label}"
                        "<br>x = %{x:.6g}"
                        "<br>Value = %{y:.6g}"
                        "<extra></extra>"
                    ),
                )
            )

    else:

        support_sets = [
            _discrete_support_grid(
                distribution
            )
            for _, _, distribution
            in prepared
        ]

        low = min(
            int(values[0])
            for values in support_sets
        )

        high = max(
            int(values[-1])
            for values in support_sets
        )

        if high - low + 1 > 220:
            raise ExplorerError(
                (
                    "The combined discrete "
                    "comparison range is too "
                    "large to display clearly."
                )
            )

        x = np.arange(
            low,
            high + 1,
            dtype=int,
        )

        for (
            curve,
            spec,
            distribution,
        ) in prepared:

            y = _discrete_values(
                distribution,
                x,
                view,
            )

            if view == "pmf":
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        name=curve.label,
                    )
                )

            else:
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        name=curve.label,
                    )
                )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "Distribution comparison — "
                f"{_view_label(category, view)}"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="X",
        yaxis_title=(
            _view_label(
                category,
                view,
            )
        ),
        hovermode="closest",
        height=520,
        margin={
            "l": 60,
            "r": 30,
            "t": 75,
            "b": 60,
        },
    )

    figure.update_yaxes(
        rangemode="tozero"
    )

    return figure


def explorer_figure_html(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
    *,
    view: str | None = None,
) -> str:

    figure = build_explorer_figure(
        distribution_key,
        raw_parameters,
        view=view,
    )

    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
        div_id="probability-explorer-chart",
    )