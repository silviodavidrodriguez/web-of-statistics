from .continuous import (
    CONTINUOUS_DISTRIBUTIONS,
)
from .discrete import (
    DISCRETE_DISTRIBUTIONS,
)
from .metadata import DistributionSpec


DISTRIBUTIONS: dict[
    str,
    DistributionSpec,
] = {
    **CONTINUOUS_DISTRIBUTIONS,
    **DISCRETE_DISTRIBUTIONS,
}


def get_distribution_spec(
    distribution_key: str,
) -> DistributionSpec:

    try:
        return DISTRIBUTIONS[
            distribution_key
        ]

    except KeyError as exc:
        raise ValueError(
            f"Unknown probability distribution: "
            f"{distribution_key}"
        ) from exc


def create_distribution(
    distribution_key: str,
    parameters: dict,
):
    spec = get_distribution_spec(
        distribution_key
    )

    return spec.factory(
        **parameters
    )


def get_continuous_distributions():
    return tuple(
        spec
        for spec in DISTRIBUTIONS.values()
        if spec.category == "continuous"
    )


def get_discrete_distributions():
    return tuple(
        spec
        for spec in DISTRIBUTIONS.values()
        if spec.category == "discrete"
    )