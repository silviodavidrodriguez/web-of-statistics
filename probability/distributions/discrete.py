from scipy import stats

from .metadata import (
    DistributionSpec,
    ParameterSpec,
)


def _probability(
    name="p",
    label="Probability",
    symbol="p",
    default=0.5,
):
    return ParameterSpec(
        name=name,
        label=label,
        symbol=symbol,
        default=default,
        kind="float",
        min_value=0,
        max_value=1,
    )


def _positive_int(
    name,
    label,
    symbol,
    default,
):
    return ParameterSpec(
        name=name,
        label=label,
        symbol=symbol,
        default=default,
        kind="int",
        min_value=1,
    )


def _nonnegative_int(
    name,
    label,
    symbol,
    default,
):
    return ParameterSpec(
        name=name,
        label=label,
        symbol=symbol,
        default=default,
        kind="int",
        min_value=0,
    )


def _validate_discrete_uniform(parameters):
    if parameters["low"] > parameters["high"]:
        return [
            "The upper value must be greater than "
            "or equal to the lower value."
        ]

    return []


def _validate_hypergeometric(parameters):
    population = parameters["population"]
    successes = parameters["successes"]
    draws = parameters["draws"]

    errors = []

    if successes > population:
        errors.append(
            "The number of successes in the "
            "population cannot exceed the "
            "population size."
        )

    if draws > population:
        errors.append(
            "The number of draws cannot exceed "
            "the population size."
        )

    return errors


# ------------------------------------------------------------------
# Factories
# ------------------------------------------------------------------

def bernoulli_factory(p):
    return stats.bernoulli(
        p,
    )


def binomial_factory(n, p):
    return stats.binom(
        n=n,
        p=p,
    )


def poisson_factory(rate):
    return stats.poisson(
        mu=rate,
    )


def geometric_factory(p):
    # scipy geom:
    # number of trials required to obtain
    # the first success.
    # Support: 1, 2, 3, ...
    return stats.geom(
        p=p,
    )


def negative_binomial_factory(r, p):
    # Number of failures before the r-th success.
    return stats.nbinom(
        n=r,
        p=p,
    )


def hypergeometric_factory(
    population,
    successes,
    draws,
):
    return stats.hypergeom(
        M=population,
        n=successes,
        N=draws,
    )


def discrete_uniform_factory(low, high):
    # scipy randint upper limit is exclusive.
    return stats.randint(
        low=low,
        high=high + 1,
    )


def zipf_factory(shape):
    return stats.zipf(
        a=shape,
    )


DISCRETE_DISTRIBUTIONS = {

    "bernoulli": DistributionSpec(
        key="bernoulli",
        label="Bernoulli",
        category="discrete",
        variable_symbol="X",
        parameters=(
            _probability(),
        ),
        factory=bernoulli_factory,
        description=(
            "Bernoulli distribution for a single "
            "binary trial."
        ),
        parameterization="X ~ Bernoulli(p)",
        supports_hazard=False,
    ),

    "binomial": DistributionSpec(
        key="binomial",
        label="Binomial",
        category="discrete",
        variable_symbol="X",
        parameters=(
            _nonnegative_int(
                "n",
                "Number of trials",
                "n",
                10,
            ),
            _probability(),
        ),
        factory=binomial_factory,
        description=(
            "Number of successes in n independent "
            "Bernoulli trials."
        ),
        parameterization="X ~ Binomial(n, p)",
        supports_hazard=False,
    ),

    "poisson": DistributionSpec(
        key="poisson",
        label="Poisson",
        category="discrete",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                name="rate",
                label="Rate",
                symbol="λ",
                default=3.0,
                min_value=0,
            ),
        ),
        factory=poisson_factory,
        description=(
            "Poisson distribution for event counts."
        ),
        parameterization="X ~ Poisson(λ)",
        supports_hazard=False,
    ),

    "geometric": DistributionSpec(
        key="geometric",
        label="Geometric",
        category="discrete",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                name="p",
                label="Probability of success",
                symbol="p",
                default=0.5,
                min_value=0,
                max_value=1,
                min_inclusive=False,
            ),
        ),
        factory=geometric_factory,
        description=(
            "Number of trials required to obtain "
            "the first success."
        ),
        parameterization="X ~ Geometric(p)",
        supports_hazard=False,
    ),

    "negative_binomial": DistributionSpec(
        key="negative_binomial",
        label="Negative Binomial",
        category="discrete",
        variable_symbol="X",
        parameters=(
            _positive_int(
                "r",
                "Required successes",
                "r",
                5,
            ),
            ParameterSpec(
                name="p",
                label="Probability of success",
                symbol="p",
                default=0.5,
                min_value=0,
                max_value=1,
                min_inclusive=False,
            ),
        ),
        factory=negative_binomial_factory,
        description=(
            "Number of failures observed before "
            "the r-th success."
        ),
        parameterization=(
            "X ~ Negative Binomial(r, p)"
        ),
        supports_hazard=False,
    ),

    "hypergeometric": DistributionSpec(
        key="hypergeometric",
        label="Hypergeometric",
        category="discrete",
        variable_symbol="X",
        parameters=(
            _positive_int(
                "population",
                "Population size",
                "N",
                50,
            ),
            _nonnegative_int(
                "successes",
                "Successes in population",
                "K",
                10,
            ),
            _nonnegative_int(
                "draws",
                "Number of draws",
                "n",
                5,
            ),
        ),
        factory=hypergeometric_factory,
        cross_validator=_validate_hypergeometric,
        description=(
            "Number of successes obtained when "
            "sampling without replacement."
        ),
        parameterization=(
            "X ~ Hypergeometric(N, K, n)"
        ),
        supports_hazard=False,
    ),

    "discrete_uniform": DistributionSpec(
        key="discrete_uniform",
        label="Discrete Uniform",
        category="discrete",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                name="low",
                label="Lowest integer",
                symbol="a",
                default=1,
                kind="int",
            ),
            ParameterSpec(
                name="high",
                label="Highest integer",
                symbol="b",
                default=6,
                kind="int",
            ),
        ),
        factory=discrete_uniform_factory,
        cross_validator=_validate_discrete_uniform,
        description=(
            "Discrete uniform distribution over "
            "consecutive integers."
        ),
        parameterization="X ~ Discrete Uniform(a, b)",
        supports_hazard=False,
    ),

    "zipf": DistributionSpec(
        key="zipf",
        label="Zipf",
        category="discrete",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                name="shape",
                label="Shape",
                symbol="a",
                default=2.0,
                min_value=1,
                min_inclusive=False,
            ),
        ),
        factory=zipf_factory,
        description="Zipf distribution.",
        parameterization="X ~ Zipf(a)",
        supports_hazard=False,
    ),
}