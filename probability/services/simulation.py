import csv
import io
import math

import secrets
import warnings

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import plotly.graph_objects as go

from scipy.stats import (
    kurtosis as sample_kurtosis,
    skew as sample_skew,
)

from probability.distributions import (
    create_distribution,
    get_distribution_spec,
)

from .validators import (
    require_valid_distribution_parameters,
)

from .plotting import PLOT_CONFIG


MIN_SAMPLE_SIZE = 2
MAX_SAMPLE_SIZE = 100_000

MAX_ECDF_POINTS = 2_000
MAX_QQ_POINTS = 1_500
MAX_DISCRETE_POINTS = 220


# ================================================================
# Results
# ================================================================


@dataclass(frozen=True)
class SimulationStatistics:
    mean: float | None
    variance: float | None
    standard_deviation: float | None
    skewness: float | None
    excess_kurtosis: float | None


@dataclass
class SimulationResult:
    distribution_key: str
    distribution_label: str
    category: str

    parameters: dict[str, int | float]

    sample_size: int
    seed: int | None

    sample: np.ndarray

    theoretical: SimulationStatistics
    simulated: SimulationStatistics


# ================================================================
# Errors
# ================================================================


class SimulationInputError(ValueError):

    def __init__(
        self,
        *,
        field_errors: dict[str, str] | None = None,
        non_field_errors: list[str] | None = None,
    ):
        self.field_errors = (
            field_errors or {}
        )

        self.non_field_errors = (
            non_field_errors or []
        )

        messages = list(
            self.field_errors.values()
        )

        messages.extend(
            self.non_field_errors
        )

        super().__init__(
            "; ".join(messages)
            or "Invalid simulation input."
        )


# ================================================================
# Input validation
# ================================================================


def _parse_integer(
    raw_value: Any,
    *,
    field_name: str,
    label: str,
) -> int:

    if raw_value is None:
        raise SimulationInputError(
            field_errors={
                field_name:
                    f"{label} is required."
            }
        )

    if isinstance(raw_value, str):
        raw_value = raw_value.strip()

        if raw_value == "":
            raise SimulationInputError(
                field_errors={
                    field_name:
                        f"{label} is required."
                }
            )

    if isinstance(raw_value, bool):
        raise SimulationInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    try:
        numeric_value = float(
            raw_value
        )

    except (TypeError, ValueError):
        raise SimulationInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    if (
        not math.isfinite(
            numeric_value
        )
        or not numeric_value.is_integer()
    ):
        raise SimulationInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    return int(
        numeric_value
    )


def _parse_sample_size(
    raw_value: Any,
) -> int:

    value = _parse_integer(
        raw_value,
        field_name="sample_size",
        label="Sample size",
    )

    if value < MIN_SAMPLE_SIZE:
        raise SimulationInputError(
            field_errors={
                "sample_size":
                    (
                        "Sample size must be at "
                        f"least {MIN_SAMPLE_SIZE}."
                    )
            }
        )

    if value > MAX_SAMPLE_SIZE:
        raise SimulationInputError(
            field_errors={
                "sample_size":
                    (
                        "Sample size cannot exceed "
                        f"{MAX_SAMPLE_SIZE:,}."
                    )
            }
        )

    return value


def _parse_seed(
    raw_value: Any,
) -> int | None:

    if raw_value is None:
        return None

    if (
        isinstance(raw_value, str)
        and raw_value.strip() == ""
    ):
        return None

    value = _parse_integer(
        raw_value,
        field_name="seed",
        label="Random seed",
    )

    if value < 0:
        raise SimulationInputError(
            field_errors={
                "seed":
                    (
                        "Random seed must be "
                        "greater than or equal to 0."
                    )
            }
        )

    if value > 4_294_967_295:
        raise SimulationInputError(
            field_errors={
                "seed":
                    (
                        "Random seed must be less "
                        "than or equal to "
                        "4294967295."
                    )
            }
        )

    return value


# ================================================================
# Statistics
# ================================================================


def _optional_finite(
    value,
) -> float | None:

    try:
        value = float(
            value
        )

    except (TypeError, ValueError):
        return None

    if not math.isfinite(value):
        return None

    return value


def _theoretical_statistics(
    distribution,
) -> SimulationStatistics:

    # Some SciPy distributions evaluate
    # mathematically valid limiting cases through
    # formulas that temporarily produce 0/0 or
    # similar intermediate values. SciPy resolves
    # the final result correctly, but may emit a
    # RuntimeWarning while doing so.
    #
    # We suppress RuntimeWarning only inside this
    # narrowly scoped theoretical-moment call.
    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            RuntimeWarning,
        )

        try:
            (
                mean,
                variance,
                skewness,
                kurtosis,
            ) = distribution.stats(
                moments="mvsk"
            )

        except Exception:
            mean = distribution.mean()

            variance = (
                distribution.var()
            )

            skewness = (
                distribution.stats(
                    moments="s"
                )
            )

            kurtosis = (
                distribution.stats(
                    moments="k"
                )
            )

    mean = _optional_finite(
        mean
    )

    variance = _optional_finite(
        variance
    )

    skewness = _optional_finite(
        skewness
    )

    kurtosis = _optional_finite(
        kurtosis
    )

    standard_deviation = None

    if (
        variance is not None
        and variance >= 0
    ):
        standard_deviation = (
            math.sqrt(
                variance
            )
        )

    return SimulationStatistics(
        mean=mean,
        variance=variance,
        standard_deviation=(
            standard_deviation
        ),
        skewness=skewness,
        excess_kurtosis=kurtosis,
    )


def _sample_statistics(
    sample,
) -> SimulationStatistics:

    sample = np.asarray(
        sample,
        dtype=float,
    )

    mean = float(
        np.mean(sample)
    )

    variance = float(
        np.var(
            sample,
            ddof=1,
        )
    )

    standard_deviation = float(
        np.std(
            sample,
            ddof=1,
        )
    )

    # Constant samples have undefined standardized
    # third and fourth moments.
    if np.all(
        sample == sample[0]
    ):
        skewness = None
        kurtosis = None

    else:
        skewness = _optional_finite(
            sample_skew(
                sample,
                bias=False,
            )
        )

        kurtosis = _optional_finite(
            sample_kurtosis(
                sample,
                fisher=True,
                bias=False,
            )
        )

    return SimulationStatistics(
        mean=_optional_finite(
            mean
        ),
        variance=_optional_finite(
            variance
        ),
        standard_deviation=(
            _optional_finite(
                standard_deviation
            )
        ),
        skewness=skewness,
        excess_kurtosis=kurtosis,
    )


# ================================================================
# Simulation engine
# ================================================================


def simulate_distribution(
    distribution_key: str,
    raw_parameters: Mapping[str, Any],
    *,
    sample_size: Any = 1000,
    seed: Any = None,
) -> SimulationResult:

    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    parsed_sample_size = (
        _parse_sample_size(
            sample_size
        )
    )

    parsed_seed = _parse_seed(
        seed
    )

    if parsed_seed is None:
        effective_seed = secrets.randbelow(
            4_294_967_296
        )

    else:
        effective_seed = parsed_seed

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    random_generator = (
        np.random.default_rng(
            effective_seed
        )
    )

    try:
        sample = distribution.rvs(
            size=parsed_sample_size,
            random_state=random_generator,
        )

    except Exception as exc:
        raise SimulationInputError(
            non_field_errors=[
                (
                    "The random sample could not "
                    "be generated for the selected "
                    "distribution and parameters."
                )
            ]
        ) from exc

    sample = np.asarray(
        sample
    ).reshape(-1)

    if sample.size != parsed_sample_size:
        raise SimulationInputError(
            non_field_errors=[
                (
                    "The generated sample has an "
                    "unexpected size."
                )
            ]
        )

    numeric_sample = np.asarray(
        sample,
        dtype=float,
    )

    if not np.all(
        np.isfinite(
            numeric_sample
        )
    ):
        raise SimulationInputError(
            non_field_errors=[
                (
                    "The generated sample contains "
                    "non-finite values. Try another "
                    "seed or parameter configuration."
                )
            ]
        )

    if spec.category == "discrete":
        sample = numeric_sample.astype(
            int
        )

    else:
        sample = numeric_sample

    return SimulationResult(
        distribution_key=(
            distribution_key
        ),
        distribution_label=(
            spec.label
        ),
        category=(
            spec.category
        ),
        parameters=dict(
            parameters
        ),
        sample_size=(
            parsed_sample_size
        ),
        seed=effective_seed,
        sample=sample,
        theoretical=(
            _theoretical_statistics(
                distribution
            )
        ),
        simulated=(
            _sample_statistics(
                sample
            )
        ),
    )


# ================================================================
# Continuous display range
# ================================================================


def _continuous_display_range(
    distribution,
    sample,
) -> tuple[float, float]:

    sample = np.asarray(
        sample,
        dtype=float,
    )

    sample_low = float(
        np.quantile(
            sample,
            0.005,
        )
    )

    sample_high = float(
        np.quantile(
            sample,
            0.995,
        )
    )

    theoretical_low = (
        _optional_finite(
            distribution.ppf(
                0.005
            )
        )
    )

    theoretical_high = (
        _optional_finite(
            distribution.ppf(
                0.995
            )
        )
    )

    lower_candidates = [
        sample_low
    ]

    upper_candidates = [
        sample_high
    ]

    if theoretical_low is not None:
        lower_candidates.append(
            theoretical_low
        )

    if theoretical_high is not None:
        upper_candidates.append(
            theoretical_high
        )

    lower = min(
        lower_candidates
    )

    upper = max(
        upper_candidates
    )

    if lower == upper:
        padding = (
            abs(lower) * 0.1
            or 1.0
        )

        lower -= padding
        upper += padding

    return (
        float(lower),
        float(upper),
    )


# ================================================================
# Continuous figures
# ================================================================


def _continuous_distribution_figure(
    result,
    distribution,
):
    sample = np.asarray(
        result.sample,
        dtype=float,
    )

    lower, upper = (
        _continuous_display_range(
            distribution,
            sample,
        )
    )

    display_sample = sample[
        (
            sample >= lower
        )
        & (
            sample <= upper
        )
    ]

    if display_sample.size < 2:
        display_sample = sample

    try:
        edges = (
            np.histogram_bin_edges(
                display_sample,
                bins="fd",
            )
        )

    except Exception:
        edges = np.linspace(
            lower,
            upper,
            31,
        )

    number_of_bins = (
        len(edges) - 1
    )

    if number_of_bins < 10:
        edges = np.linspace(
            lower,
            upper,
            11,
        )

    elif number_of_bins > 80:
        edges = np.linspace(
            lower,
            upper,
            81,
        )

    counts, edges = np.histogram(
        sample,
        bins=edges,
    )

    widths = np.diff(
        edges
    )

    centers = (
        edges[:-1]
        + widths / 2.0
    )

    density = (
        counts
        / (
            result.sample_size
            * widths
        )
    )

    x = np.linspace(
        lower,
        upper,
        700,
    )

    theoretical_density = (
        np.asarray(
            distribution.pdf(x),
            dtype=float,
        )
    )

    theoretical_density[
        ~np.isfinite(
            theoretical_density
        )
    ] = np.nan

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=centers,
            y=density,
            width=widths,
            name="Simulated density",
            opacity=0.65,
            hovertemplate=(
                "x ≈ %{x:.6g}"
                "<br>Density = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=theoretical_density,
            mode="lines",
            name="Theoretical PDF",
            hovertemplate=(
                "x = %{x:.6g}"
                "<br>PDF = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": (
                f"{result.distribution_label}"
                " — simulated distribution"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="X",
        yaxis_title="Density",
        barmode="overlay",
    )

    return figure


def _continuous_cdf_figure(
    result,
    distribution,
):
    sample = np.asarray(
        result.sample,
        dtype=float,
    )

    sorted_sample = np.sort(
        sample
    )

    if (
        sorted_sample.size
        > MAX_ECDF_POINTS
    ):
        indexes = np.linspace(
            0,
            sorted_sample.size - 1,
            MAX_ECDF_POINTS,
            dtype=int,
        )

        ecdf_x = sorted_sample[
            indexes
        ]

        ecdf_y = (
            indexes + 1
        ) / sorted_sample.size

    else:
        ecdf_x = sorted_sample

        ecdf_y = (
            np.arange(
                1,
                sorted_sample.size + 1,
            )
            / sorted_sample.size
        )

    lower, upper = (
        _continuous_display_range(
            distribution,
            sample,
        )
    )

    theoretical_x = np.linspace(
        lower,
        upper,
        700,
    )

    theoretical_y = (
        distribution.cdf(
            theoretical_x
        )
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=ecdf_x,
            y=ecdf_y,
            mode="lines",
            name="Empirical CDF",
            hovertemplate=(
                "x = %{x:.6g}"
                "<br>ECDF = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=theoretical_x,
            y=theoretical_y,
            mode="lines",
            name="Theoretical CDF",
            hovertemplate=(
                "x = %{x:.6g}"
                "<br>CDF = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": (
                f"{result.distribution_label}"
                " — empirical vs theoretical CDF"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="X",
        yaxis_title="Cumulative probability",
    )

    figure.update_yaxes(
        range=[0, 1]
    )

    return figure


def _continuous_qq_figure(
    result,
    distribution,
):
    number_of_points = min(
        result.sample_size,
        MAX_QQ_POINTS,
    )

    probabilities = np.linspace(
        0.001,
        0.999,
        number_of_points,
    )

    theoretical = np.asarray(
        distribution.ppf(
            probabilities
        ),
        dtype=float,
    )

    observed = np.asarray(
        np.quantile(
            result.sample,
            probabilities,
        ),
        dtype=float,
    )

    valid = (
        np.isfinite(theoretical)
        & np.isfinite(observed)
    )

    theoretical = theoretical[
        valid
    ]

    observed = observed[
        valid
    ]

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=theoretical,
            y=observed,
            mode="markers",
            name="Sample quantiles",
            marker={
                "size": 6,
                "opacity": 0.65,
            },
            hovertemplate=(
                "Theoretical = %{x:.6g}"
                "<br>Observed = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    if theoretical.size:
        reference_low = min(
            float(
                np.min(
                    theoretical
                )
            ),
            float(
                np.min(
                    observed
                )
            ),
        )

        reference_high = max(
            float(
                np.max(
                    theoretical
                )
            ),
            float(
                np.max(
                    observed
                )
            ),
        )

        figure.add_trace(
            go.Scatter(
                x=[
                    reference_low,
                    reference_high,
                ],
                y=[
                    reference_low,
                    reference_high,
                ],
                mode="lines",
                name="Reference line",
            )
        )

    figure.update_layout(
        title={
            "text": (
                f"{result.distribution_label}"
                " — Q-Q plot"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=(
            "Theoretical quantiles"
        ),
        yaxis_title=(
            "Simulated quantiles"
        ),
    )

    return figure


# ================================================================
# Discrete support
# ================================================================


def _discrete_display_support(
    result,
    distribution,
):
    sample = np.asarray(
        result.sample,
        dtype=int,
    )

    sample_low = int(
        np.min(sample)
    )

    sample_high = int(
        np.max(sample)
    )

    theoretical_low = (
        _optional_finite(
            distribution.ppf(
                0.001
            )
        )
    )

    theoretical_high = (
        _optional_finite(
            distribution.ppf(
                0.999
            )
        )
    )

    low = sample_low
    high = sample_high

    if theoretical_low is not None:
        low = min(
            low,
            int(
                math.floor(
                    theoretical_low
                )
            ),
        )

    if theoretical_high is not None:
        high = max(
            high,
            int(
                math.ceil(
                    theoretical_high
                )
            ),
        )

    support_low, support_high = (
        distribution.support()
    )

    support_low = _optional_finite(
        support_low
    )

    support_high = _optional_finite(
        support_high
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

    if (
        high - low + 1
        > MAX_DISCRETE_POINTS
    ):
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
            round(
                median
            )
        )

        half = (
            MAX_DISCRETE_POINTS
            // 2
        )

        low = (
            center - half
        )

        high = (
            low
            + MAX_DISCRETE_POINTS
            - 1
        )

        if support_low is not None:
            if low < support_low:
                shift = (
                    int(
                        math.ceil(
                            support_low
                        )
                    )
                    - low
                )

                low += shift
                high += shift

        if support_high is not None:
            if high > support_high:
                shift = (
                    high
                    - int(
                        math.floor(
                            support_high
                        )
                    )
                )

                low -= shift
                high -= shift

    if high < low:
        high = low

    return np.arange(
        low,
        high + 1,
        dtype=int,
    )


def _observed_probabilities(
    sample,
    x,
):
    values, counts = np.unique(
        sample,
        return_counts=True,
    )

    probability_map = {
        int(value):
            count / len(sample)
        for value, count
        in zip(
            values,
            counts,
        )
    }

    return np.asarray(
        [
            probability_map.get(
                int(value),
                0.0,
            )
            for value in x
        ],
        dtype=float,
    )


# ================================================================
# Discrete figures
# ================================================================


def _discrete_distribution_figure(
    result,
    distribution,
):
    x = _discrete_display_support(
        result,
        distribution,
    )

    observed_probability = (
        _observed_probabilities(
            result.sample,
            x,
        )
    )

    expected_probability = (
        np.asarray(
            distribution.pmf(x),
            dtype=float,
        )
    )

    observed_count = (
        observed_probability
        * result.sample_size
    )

    expected_count = (
        expected_probability
        * result.sample_size
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=x,
            y=observed_count,
            name="Observed frequency",
            opacity=0.7,
            hovertemplate=(
                "X = %{x}"
                "<br>Observed = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=expected_count,
            mode="lines+markers",
            name="Expected frequency",
            hovertemplate=(
                "X = %{x}"
                "<br>Expected = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.update_layout(
        title={
            "text": (
                f"{result.distribution_label}"
                " — observed vs expected frequencies"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="X",
        yaxis_title="Frequency",
    )

    if len(x) <= 30:
        figure.update_xaxes(
            dtick=1
        )

    return figure


def _discrete_cdf_figure(
    result,
    distribution,
):
    x = _discrete_display_support(
        result,
        distribution,
    )

    sorted_sample = np.sort(
        np.asarray(
            result.sample,
            dtype=int,
        )
    )

    empirical = (
        np.searchsorted(
            sorted_sample,
            x,
            side="right",
        )
        / result.sample_size
    )

    theoretical = np.asarray(
        distribution.cdf(x),
        dtype=float,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=x,
            y=empirical,
            mode="lines+markers",
            name="Empirical CDF",
        )
    )

    figure.add_trace(
        go.Scatter(
            x=x,
            y=theoretical,
            mode="lines+markers",
            name="Theoretical CDF",
        )
    )

    figure.update_layout(
        title={
            "text": (
                f"{result.distribution_label}"
                " — empirical vs theoretical CDF"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="X",
        yaxis_title=(
            "Cumulative probability"
        ),
    )

    figure.update_yaxes(
        range=[0, 1]
    )

    if len(x) <= 30:
        figure.update_xaxes(
            dtick=1
        )

    return figure


def _discrete_probability_comparison_figure(
    result,
    distribution,
):
    x = _discrete_display_support(
        result,
        distribution,
    )

    observed = (
        _observed_probabilities(
            result.sample,
            x,
        )
    )

    expected = np.asarray(
        distribution.pmf(x),
        dtype=float,
    )

    maximum = max(
        float(
            np.max(
                observed
            )
        ),
        float(
            np.max(
                expected
            )
        ),
        1e-12,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=expected,
            y=observed,
            mode="markers",
            name="Observed vs expected",
            customdata=x,
            hovertemplate=(
                "X = %{customdata}"
                "<br>Expected = %{x:.6g}"
                "<br>Observed = %{y:.6g}"
                "<extra></extra>"
            ),
        )
    )

    figure.add_trace(
        go.Scatter(
            x=[0, maximum],
            y=[0, maximum],
            mode="lines",
            name="Perfect agreement",
        )
    )

    figure.update_layout(
        title={
            "text": (
                f"{result.distribution_label}"
                " — observed vs expected probabilities"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=(
            "Expected probability"
        ),
        yaxis_title=(
            "Observed probability"
        ),
    )

    return figure


# ================================================================
# Shared layout
# ================================================================


def _apply_simulation_layout(
    figure,
):
    figure.update_layout(
        template="plotly_white",
        hovermode="closest",
        height=500,
        margin={
            "l": 60,
            "r": 30,
            "t": 75,
            "b": 60,
        },
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
    )

    return figure


# ================================================================
# Public plotting API
# ================================================================


def build_simulation_figures(
    result: SimulationResult,
) -> dict[str, go.Figure]:

    distribution = create_distribution(
        result.distribution_key,
        result.parameters,
    )

    if result.category == "continuous":

        figures = {
            "distribution":
                _continuous_distribution_figure(
                    result,
                    distribution,
                ),

            "cdf":
                _continuous_cdf_figure(
                    result,
                    distribution,
                ),

            "qq":
                _continuous_qq_figure(
                    result,
                    distribution,
                ),
        }

    elif result.category == "discrete":

        figures = {
            "distribution":
                _discrete_distribution_figure(
                    result,
                    distribution,
                ),

            "cdf":
                _discrete_cdf_figure(
                    result,
                    distribution,
                ),

            "probability_comparison":
                (
                    _discrete_probability_comparison_figure(
                        result,
                        distribution,
                    )
                ),
        }

    else:
        raise SimulationInputError(
            non_field_errors=[
                (
                    "Unsupported distribution "
                    "category."
                )
            ]
        )

    return {
        key:
            _apply_simulation_layout(
                figure
            )
        for key, figure
        in figures.items()
    }


# ================================================================
# CSV export
# ================================================================


def simulation_to_csv(
    result: SimulationResult,
) -> str:

    buffer = io.StringIO()

    writer = csv.writer(
        buffer,
        lineterminator="\n",
    )

    writer.writerow(
        [
            "observation",
            "value",
        ]
    )

    for index, value in enumerate(
        result.sample,
        start=1,
    ):

        if result.category == "discrete":
            serialized_value = int(
                value
            )

        else:
            serialized_value = format(
                float(value),
                ".17g",
            )

        writer.writerow(
            [
                index,
                serialized_value,
            ]
        )

    return buffer.getvalue()


def simulation_figures_html(
    result: SimulationResult,
) -> dict[str, str]:

    figures = build_simulation_figures(
        result
    )

    html = {}

    for index, (
        key,
        figure,
    ) in enumerate(
        figures.items()
    ):
        safe_key = key.replace(
            "_",
            "-",
        )

        html[key] = figure.to_html(
            full_html=False,
            include_plotlyjs=(
                "cdn"
                if index == 0
                else False
            ),
            config=PLOT_CONFIG,
            div_id=(
                "probability-simulation-"
                f"{safe_key}-chart"
            ),
        )

    return html