import math
import secrets
import warnings

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


# ================================================================
# Limits
# ================================================================

MIN_REPETITIONS = 50
MAX_REPETITIONS = 20_000

MIN_SAMPLE_SIZE = 1
MAX_SAMPLE_SIZE = 10_000

MIN_CLT_SAMPLE_SIZES = 2
MAX_CLT_SAMPLE_SIZES = 6

MIN_LLN_SAMPLE_SIZE = 2
MAX_LLN_SAMPLE_SIZE = 100_000

MIN_LLN_PATHS = 1
MAX_LLN_PATHS = 20

MAX_LLN_OBSERVATIONS = 2_000_000

MAX_OBSERVATIONS_PER_CHUNK = 1_000_000
MAX_PLOT_POINTS = 2_000


SAMPLING_STATISTICS = {
    "mean": "Sample mean",
    "median": "Sample median",
    "variance": "Sample variance",
    "standard_deviation":
        "Sample standard deviation",
}


# ================================================================
# Results
# ================================================================


@dataclass(frozen=True)
class SamplingDistributionResult:
    distribution_key: str
    distribution_label: str
    category: str

    parameters: dict[str, int | float]

    statistic: str
    statistic_label: str

    sample_size: int
    repetitions: int
    seed: int

    values: np.ndarray

    source_mean: float | None
    source_variance: float | None

    theoretical_reference: float | None

    theoretical_reference_label: str | None

    normal_approximation_mean: float | None

    normal_approximation_sd: float | None

    empirical_mean: float
    empirical_standard_deviation: float


@dataclass(frozen=True)
class CLTResult:
    distribution_key: str
    distribution_label: str
    category: str

    parameters: dict[str, int | float]

    sample_sizes: tuple[int, ...]
    repetitions: int
    seed: int

    means_by_size: dict[int, np.ndarray]

    source_mean: float | None
    source_variance: float | None

    classical_clt_available: bool


@dataclass(frozen=True)
class LLNResult:
    distribution_key: str
    distribution_label: str
    category: str

    parameters: dict[str, int | float]

    max_sample_size: int
    paths: int
    seed: int

    theoretical_mean: float

    running_means: np.ndarray
    final_means: np.ndarray


# ================================================================
# Errors
# ================================================================


class SamplingInputError(ValueError):

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
            or "Invalid sampling input."
        )


# ================================================================
# Parsing
# ================================================================


def _parse_integer(
    raw_value: Any,
    *,
    field_name: str,
    label: str,
) -> int:

    if raw_value is None:
        raise SamplingInputError(
            field_errors={
                field_name:
                    f"{label} is required."
            }
        )

    if isinstance(
        raw_value,
        str,
    ):
        raw_value = raw_value.strip()

        if raw_value == "":
            raise SamplingInputError(
                field_errors={
                    field_name:
                        f"{label} is required."
                }
            )

    if isinstance(
        raw_value,
        bool,
    ):
        raise SamplingInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    try:
        value = float(
            raw_value
        )

    except (
        TypeError,
        ValueError,
    ):
        raise SamplingInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    if (
        not math.isfinite(value)
        or not value.is_integer()
    ):
        raise SamplingInputError(
            field_errors={
                field_name:
                    f"{label} must be an integer."
            }
        )

    return int(value)


def _parse_sample_size(
    raw_value,
) -> int:

    value = _parse_integer(
        raw_value,
        field_name="sample_size",
        label="Sample size",
    )

    if value < MIN_SAMPLE_SIZE:
        raise SamplingInputError(
            field_errors={
                "sample_size":
                    (
                        "Sample size must be "
                        f"at least {MIN_SAMPLE_SIZE}."
                    )
            }
        )

    if value > MAX_SAMPLE_SIZE:
        raise SamplingInputError(
            field_errors={
                "sample_size":
                    (
                        "Sample size cannot exceed "
                        f"{MAX_SAMPLE_SIZE:,}."
                    )
            }
        )

    return value


def _parse_repetitions(
    raw_value,
) -> int:

    value = _parse_integer(
        raw_value,
        field_name="repetitions",
        label="Number of repetitions",
    )

    if value < MIN_REPETITIONS:
        raise SamplingInputError(
            field_errors={
                "repetitions":
                    (
                        "Number of repetitions must "
                        f"be at least "
                        f"{MIN_REPETITIONS}."
                    )
            }
        )

    if value > MAX_REPETITIONS:
        raise SamplingInputError(
            field_errors={
                "repetitions":
                    (
                        "Number of repetitions "
                        "cannot exceed "
                        f"{MAX_REPETITIONS:,}."
                    )
            }
        )

    return value


def _parse_seed(
    raw_value,
) -> int:

    if (
        raw_value is None
        or (
            isinstance(raw_value, str)
            and raw_value.strip() == ""
        )
    ):
        return secrets.randbelow(
            4_294_967_296
        )

    value = _parse_integer(
        raw_value,
        field_name="seed",
        label="Random seed",
    )

    if value < 0:
        raise SamplingInputError(
            field_errors={
                "seed":
                    (
                        "Random seed must be "
                        "greater than or equal to 0."
                    )
            }
        )

    if value > 4_294_967_295:
        raise SamplingInputError(
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


def _parse_statistic(
    statistic,
) -> str:

    if statistic not in (
        SAMPLING_STATISTICS
    ):
        raise SamplingInputError(
            field_errors={
                "statistic":
                    (
                        "The selected sampling "
                        "statistic is not available."
                    )
            }
        )

    return statistic


# ================================================================
# Moments
# ================================================================


def _optional_finite(
    value,
) -> float | None:

    try:
        value = float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None

    if not math.isfinite(value):
        return None

    return value


def _source_moments(
    distribution,
) -> tuple[
    float | None,
    float | None,
]:

    with warnings.catch_warnings():

        warnings.simplefilter(
            "ignore",
            RuntimeWarning,
        )

        try:
            mean, variance = (
                distribution.stats(
                    moments="mv"
                )
            )

        except Exception:
            mean = (
                distribution.mean()
            )

            variance = (
                distribution.var()
            )

    return (
        _optional_finite(mean),
        _optional_finite(variance),
    )


# ================================================================
# Random generation
# ================================================================


def _draw_matrix(
    distribution,
    *,
    rows: int,
    columns: int,
    random_generator,
):
    try:
        raw = distribution.rvs(
            size=(
                rows,
                columns,
            ),
            random_state=random_generator,
        )

    except Exception as exc:
        raise SamplingInputError(
            non_field_errors=[
                (
                    "Random samples could not be "
                    "generated for the selected "
                    "distribution."
                )
            ]
        ) from exc

    values = np.asarray(
        raw,
        dtype=float,
    )

    expected_size = (
        rows * columns
    )

    if values.size != expected_size:
        raise SamplingInputError(
            non_field_errors=[
                (
                    "The generated random sample "
                    "has an unexpected size."
                )
            ]
        )

    values = values.reshape(
        rows,
        columns,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise SamplingInputError(
            non_field_errors=[
                (
                    "Generated samples contain "
                    "non-finite values."
                )
            ]
        )

    return values


def _calculate_statistic(
    samples,
    statistic,
):
    with np.errstate(
        all="ignore"
    ):

        if statistic == "mean":
            values = np.mean(
                samples,
                axis=1,
            )

        elif statistic == "median":
            values = np.median(
                samples,
                axis=1,
            )

        elif statistic == "variance":
            values = np.var(
                samples,
                axis=1,
                ddof=1,
            )

        elif statistic == (
            "standard_deviation"
        ):
            values = np.std(
                samples,
                axis=1,
                ddof=1,
            )

        else:
            raise SamplingInputError(
                field_errors={
                    "statistic":
                        (
                            "Unsupported sampling "
                            "statistic."
                        )
                }
            )

    values = np.asarray(
        values,
        dtype=float,
    )

    if not np.all(
        np.isfinite(values)
    ):
        raise SamplingInputError(
            non_field_errors=[
                (
                    "The selected statistic "
                    "produced non-finite values."
                )
            ]
        )

    return values


def _generate_statistic_values(
    distribution,
    *,
    statistic,
    sample_size,
    repetitions,
    random_generator,
):
    maximum_chunk_rows = max(
        1,
        MAX_OBSERVATIONS_PER_CHUNK
        // sample_size,
    )

    chunks = []

    remaining = repetitions

    while remaining > 0:

        rows = min(
            remaining,
            maximum_chunk_rows,
        )

        samples = _draw_matrix(
            distribution,
            rows=rows,
            columns=sample_size,
            random_generator=(
                random_generator
            ),
        )

        values = (
            _calculate_statistic(
                samples,
                statistic,
            )
        )

        chunks.append(
            values
        )

        remaining -= rows

    return np.concatenate(
        chunks
    )


# ================================================================
# Sampling distribution
# ================================================================


def simulate_sampling_distribution(
    distribution_key: str,
    raw_parameters: Mapping[
        str,
        Any,
    ],
    *,
    statistic: str = "mean",
    sample_size: Any = 30,
    repetitions: Any = 5000,
    seed: Any = None,
) -> SamplingDistributionResult:

    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    parsed_statistic = (
        _parse_statistic(
            statistic
        )
    )

    parsed_sample_size = (
        _parse_sample_size(
            sample_size
        )
    )

    parsed_repetitions = (
        _parse_repetitions(
            repetitions
        )
    )

    parsed_seed = _parse_seed(
        seed
    )

    if (
        parsed_statistic
        in {
            "variance",
            "standard_deviation",
        }
        and parsed_sample_size < 2
    ):
        raise SamplingInputError(
            field_errors={
                "sample_size":
                    (
                        "Sample size must be at "
                        "least 2 for sample variance "
                        "or standard deviation."
                    )
            }
        )

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    random_generator = (
        np.random.default_rng(
            parsed_seed
        )
    )

    values = (
        _generate_statistic_values(
            distribution,
            statistic=(
                parsed_statistic
            ),
            sample_size=(
                parsed_sample_size
            ),
            repetitions=(
                parsed_repetitions
            ),
            random_generator=(
                random_generator
            ),
        )
    )

    source_mean, source_variance = (
        _source_moments(
            distribution
        )
    )

    theoretical_reference = None
    theoretical_reference_label = None

    normal_approximation_mean = None
    normal_approximation_sd = None

    if parsed_statistic == "mean":

        theoretical_reference = (
            source_mean
        )

        theoretical_reference_label = (
            "Population mean"
        )

        if (
            source_mean is not None
            and source_variance is not None
            and source_variance >= 0
        ):
            normal_approximation_mean = (
                source_mean
            )

            normal_approximation_sd = (
                math.sqrt(
                    source_variance
                    / parsed_sample_size
                )
            )

    elif parsed_statistic == "variance":

        theoretical_reference = (
            source_variance
        )

        theoretical_reference_label = (
            "Population variance"
        )

    empirical_mean = float(
        np.mean(values)
    )

    empirical_standard_deviation = (
        float(
            np.std(
                values,
                ddof=1,
            )
        )
    )

    return SamplingDistributionResult(
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
        statistic=(
            parsed_statistic
        ),
        statistic_label=(
            SAMPLING_STATISTICS[
                parsed_statistic
            ]
        ),
        sample_size=(
            parsed_sample_size
        ),
        repetitions=(
            parsed_repetitions
        ),
        seed=parsed_seed,
        values=values,
        source_mean=(
            source_mean
        ),
        source_variance=(
            source_variance
        ),
        theoretical_reference=(
            theoretical_reference
        ),
        theoretical_reference_label=(
            theoretical_reference_label
        ),
        normal_approximation_mean=(
            normal_approximation_mean
        ),
        normal_approximation_sd=(
            normal_approximation_sd
        ),
        empirical_mean=(
            empirical_mean
        ),
        empirical_standard_deviation=(
            empirical_standard_deviation
        ),
    )


# ================================================================
# Sampling distribution plot
# ================================================================


def build_sampling_distribution_figure(
    result: SamplingDistributionResult,
):
    figure = go.Figure()

    figure.add_trace(
        go.Histogram(
            x=result.values,
            histnorm="probability density",
            nbinsx=50,
            name="Empirical sampling distribution",
            opacity=0.7,
        )
    )

    approximation_sd = (
        result.normal_approximation_sd
    )

    approximation_mean = (
        result.normal_approximation_mean
    )

    if (
        approximation_mean is not None
        and approximation_sd is not None
        and approximation_sd > 0
    ):
        low = float(
            np.quantile(
                result.values,
                0.0025,
            )
        )

        high = float(
            np.quantile(
                result.values,
                0.9975,
            )
        )

        low = min(
            low,
            approximation_mean
            - 4 * approximation_sd,
        )

        high = max(
            high,
            approximation_mean
            + 4 * approximation_sd,
        )

        x = np.linspace(
            low,
            high,
            700,
        )

        z = (
            x - approximation_mean
        ) / approximation_sd

        density = (
            np.exp(
                -0.5 * z ** 2
            )
            / (
                approximation_sd
                * math.sqrt(
                    2 * math.pi
                )
            )
        )

        figure.add_trace(
            go.Scatter(
                x=x,
                y=density,
                mode="lines",
                name="Normal approximation",
                line={
                    "dash": "dash",
                },
            )
        )

    if (
        result.theoretical_reference
        is not None
    ):
        figure.add_vline(
            x=(
                result
                .theoretical_reference
            ),
            line_dash="dot",
            annotation_text=(
                result
                .theoretical_reference_label
            ),
            annotation_position="top",
        )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                f"{result.statistic_label} — "
                f"{result.distribution_label}"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title=(
            result.statistic_label
        ),
        yaxis_title="Density",
        hovermode="closest",
        height=520,
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

    return figure


# ================================================================
# Central Limit Theorem
# ================================================================


def _parse_clt_sample_sizes(
    raw_sizes: Sequence[Any],
) -> tuple[int, ...]:

    if (
        isinstance(
            raw_sizes,
            (str, bytes),
        )
        or not isinstance(
            raw_sizes,
            Sequence,
        )
    ):
        raise SamplingInputError(
            field_errors={
                "sample_sizes":
                    (
                        "Sample sizes must be "
                        "provided as a sequence."
                    )
            }
        )

    parsed = []

    for raw_value in raw_sizes:

        value = _parse_integer(
            raw_value,
            field_name="sample_sizes",
            label="Sample size",
        )

        if (
            value < MIN_SAMPLE_SIZE
            or value > MAX_SAMPLE_SIZE
        ):
            raise SamplingInputError(
                field_errors={
                    "sample_sizes":
                        (
                            "Each sample size must "
                            f"be between "
                            f"{MIN_SAMPLE_SIZE} and "
                            f"{MAX_SAMPLE_SIZE:,}."
                        )
                }
            )

        if value not in parsed:
            parsed.append(
                value
            )

    if (
        len(parsed)
        < MIN_CLT_SAMPLE_SIZES
    ):
        raise SamplingInputError(
            field_errors={
                "sample_sizes":
                    (
                        "Select at least "
                        f"{MIN_CLT_SAMPLE_SIZES} "
                        "different sample sizes."
                    )
            }
        )

    if (
        len(parsed)
        > MAX_CLT_SAMPLE_SIZES
    ):
        raise SamplingInputError(
            field_errors={
                "sample_sizes":
                    (
                        "No more than "
                        f"{MAX_CLT_SAMPLE_SIZES} "
                        "sample sizes can be "
                        "compared."
                    )
            }
        )

    return tuple(
        parsed
    )


def simulate_clt(
    distribution_key: str,
    raw_parameters: Mapping[
        str,
        Any,
    ],
    *,
    sample_sizes: Sequence[Any] = (
        1,
        2,
        5,
        10,
        30,
    ),
    repetitions: Any = 5000,
    seed: Any = None,
) -> CLTResult:

    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    parsed_sizes = (
        _parse_clt_sample_sizes(
            sample_sizes
        )
    )

    parsed_repetitions = (
        _parse_repetitions(
            repetitions
        )
    )

    parsed_seed = _parse_seed(
        seed
    )

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    source_mean, source_variance = (
        _source_moments(
            distribution
        )
    )

    classical_clt_available = (
        source_mean is not None
        and source_variance is not None
        and source_variance >= 0
    )

    random_generator = (
        np.random.default_rng(
            parsed_seed
        )
    )

    means_by_size = {}

    for sample_size in parsed_sizes:

        means_by_size[
            sample_size
        ] = (
            _generate_statistic_values(
                distribution,
                statistic="mean",
                sample_size=sample_size,
                repetitions=(
                    parsed_repetitions
                ),
                random_generator=(
                    random_generator
                ),
            )
        )

    return CLTResult(
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
        sample_sizes=(
            parsed_sizes
        ),
        repetitions=(
            parsed_repetitions
        ),
        seed=parsed_seed,
        means_by_size=(
            means_by_size
        ),
        source_mean=(
            source_mean
        ),
        source_variance=(
            source_variance
        ),
        classical_clt_available=(
            classical_clt_available
        ),
    )


def build_clt_figure(
    result: CLTResult,
):
    figure = go.Figure()

    for sample_size in (
        result.sample_sizes
    ):
        values = (
            result.means_by_size[
                sample_size
            ]
        )

        low = float(
            np.quantile(
                values,
                0.005,
            )
        )

        high = float(
            np.quantile(
                values,
                0.995,
            )
        )

        if low == high:
            padding = (
                abs(low) * 0.1
                or 1.0
            )

            low -= padding
            high += padding

        edges = np.linspace(
            low,
            high,
            61,
        )

        counts, edges = (
            np.histogram(
                values,
                bins=edges,
                density=True,
            )
        )

        centers = (
            edges[:-1]
            + np.diff(edges) / 2
        )

        figure.add_trace(
            go.Scatter(
                x=centers,
                y=counts,
                mode="lines",
                name=(
                    f"n = {sample_size} "
                    "empirical"
                ),
            )
        )

        if (
            result.classical_clt_available
            and result.source_variance
            is not None
            and result.source_variance > 0
        ):
            sd = math.sqrt(
                result.source_variance
                / sample_size
            )

            x = np.linspace(
                low,
                high,
                400,
            )

            z = (
                x - result.source_mean
            ) / sd

            density = (
                np.exp(
                    -0.5 * z ** 2
                )
                / (
                    sd
                    * math.sqrt(
                        2 * math.pi
                    )
                )
            )

            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=density,
                    mode="lines",
                    name=(
                        f"n = {sample_size} "
                        "Normal approximation"
                    ),
                    line={
                        "dash": "dash",
                    },
                )
            )

    if result.source_mean is not None:
        figure.add_vline(
            x=result.source_mean,
            line_dash="dot",
            annotation_text=(
                "Population mean"
            ),
            annotation_position="top",
        )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "Central Limit Theorem — "
                f"{result.distribution_label}"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Sample mean",
        yaxis_title="Density",
        hovermode="closest",
        height=560,
        margin={
            "l": 60,
            "r": 30,
            "t": 75,
            "b": 60,
        },
    )

    return figure


# ================================================================
# Law of Large Numbers
# ================================================================


def _parse_lln_paths(
    raw_value,
) -> int:

    value = _parse_integer(
        raw_value,
        field_name="paths",
        label="Number of paths",
    )

    if value < MIN_LLN_PATHS:
        raise SamplingInputError(
            field_errors={
                "paths":
                    (
                        "Number of paths must be "
                        f"at least "
                        f"{MIN_LLN_PATHS}."
                    )
            }
        )

    if value > MAX_LLN_PATHS:
        raise SamplingInputError(
            field_errors={
                "paths":
                    (
                        "Number of paths cannot "
                        f"exceed "
                        f"{MAX_LLN_PATHS}."
                    )
            }
        )

    return value


def _parse_lln_sample_size(
    raw_value,
) -> int:

    value = _parse_integer(
        raw_value,
        field_name="max_sample_size",
        label="Maximum sample size",
    )

    if value < MIN_LLN_SAMPLE_SIZE:
        raise SamplingInputError(
            field_errors={
                "max_sample_size":
                    (
                        "Maximum sample size must "
                        f"be at least "
                        f"{MIN_LLN_SAMPLE_SIZE}."
                    )
            }
        )

    if value > MAX_LLN_SAMPLE_SIZE:
        raise SamplingInputError(
            field_errors={
                "max_sample_size":
                    (
                        "Maximum sample size "
                        "cannot exceed "
                        f"{MAX_LLN_SAMPLE_SIZE:,}."
                    )
            }
        )

    return value


def simulate_lln(
    distribution_key: str,
    raw_parameters: Mapping[
        str,
        Any,
    ],
    *,
    max_sample_size: Any = 5000,
    paths: Any = 5,
    seed: Any = None,
) -> LLNResult:

    spec = get_distribution_spec(
        distribution_key
    )

    parameters = (
        require_valid_distribution_parameters(
            distribution_key,
            raw_parameters,
        )
    )

    parsed_max_sample_size = (
        _parse_lln_sample_size(
            max_sample_size
        )
    )

    parsed_paths = (
        _parse_lln_paths(
            paths
        )
    )

    if (
        parsed_max_sample_size
        * parsed_paths
        > MAX_LLN_OBSERVATIONS
    ):
        raise SamplingInputError(
            non_field_errors=[
                (
                    "The requested LLN simulation "
                    "is too large. Reduce the "
                    "maximum sample size or number "
                    "of paths."
                )
            ]
        )

    parsed_seed = _parse_seed(
        seed
    )

    distribution = create_distribution(
        distribution_key,
        parameters,
    )

    source_mean, _ = (
        _source_moments(
            distribution
        )
    )

    if source_mean is None:
        raise SamplingInputError(
            non_field_errors=[
                (
                    "The classical Law of Large "
                    "Numbers visualization requires "
                    "a finite theoretical mean. "
                    f"{spec.label} does not provide "
                    "one for the selected parameters."
                )
            ]
        )

    random_generator = (
        np.random.default_rng(
            parsed_seed
        )
    )

    samples = _draw_matrix(
        distribution,
        rows=parsed_paths,
        columns=(
            parsed_max_sample_size
        ),
        random_generator=(
            random_generator
        ),
    )

    cumulative_sum = np.cumsum(
        samples,
        axis=1,
        dtype=float,
    )

    denominators = np.arange(
        1,
        parsed_max_sample_size + 1,
        dtype=float,
    )

    running_means = (
        cumulative_sum
        / denominators
    )

    if not np.all(
        np.isfinite(
            running_means
        )
    ):
        raise SamplingInputError(
            non_field_errors=[
                (
                    "The running means contain "
                    "non-finite values."
                )
            ]
        )

    final_means = (
        running_means[:, -1]
    )

    return LLNResult(
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
        max_sample_size=(
            parsed_max_sample_size
        ),
        paths=parsed_paths,
        seed=parsed_seed,
        theoretical_mean=(
            source_mean
        ),
        running_means=(
            running_means
        ),
        final_means=(
            final_means
        ),
    )


def build_lln_figure(
    result: LLNResult,
):
    figure = go.Figure()

    if (
        result.max_sample_size
        <= MAX_PLOT_POINTS
    ):
        indexes = np.arange(
            result.max_sample_size
        )

    else:
        indexes = np.unique(
            np.linspace(
                0,
                result.max_sample_size - 1,
                MAX_PLOT_POINTS,
                dtype=int,
            )
        )

    x = indexes + 1

    for path_index in range(
        result.paths
    ):
        figure.add_trace(
            go.Scatter(
                x=x,
                y=(
                    result.running_means[
                        path_index,
                        indexes,
                    ]
                ),
                mode="lines",
                name=(
                    f"Path "
                    f"{path_index + 1}"
                ),
            )
        )

    figure.add_hline(
        y=result.theoretical_mean,
        line_dash="dash",
        annotation_text=(
            "Theoretical mean"
        ),
        annotation_position="top right",
    )

    figure.update_layout(
        template="plotly_white",
        title={
            "text": (
                "Law of Large Numbers — "
                f"{result.distribution_label}"
            ),
            "x": 0.5,
            "xanchor": "center",
        },
        xaxis_title="Sample size (n)",
        yaxis_title="Running sample mean",
        hovermode="closest",
        height=540,
        margin={
            "l": 60,
            "r": 30,
            "t": 75,
            "b": 60,
        },
    )

    return figure


def _sampling_figure_html(
    figure,
    *,
    div_id,
) -> str:

    return figure.to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
        div_id=div_id,
    )


def sampling_distribution_figure_html(
    result: SamplingDistributionResult,
) -> str:

    return _sampling_figure_html(
        build_sampling_distribution_figure(
            result
        ),
        div_id=(
            "probability-sampling-"
            "distribution-chart"
        ),
    )


def clt_figure_html(
    result: CLTResult,
) -> str:

    return _sampling_figure_html(
        build_clt_figure(
            result
        ),
        div_id=(
            "probability-sampling-"
            "clt-chart"
        ),
    )


def lln_figure_html(
    result: LLNResult,
) -> str:

    return _sampling_figure_html(
        build_lln_figure(
            result
        ),
        div_id=(
            "probability-sampling-"
            "lln-chart"
        ),
    )