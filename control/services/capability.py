"""
Process capability and performance indices.

Capability:
    Cp, Cpl, Cpu, Cpk use within-process sigma.

Performance:
    Pp, Ppl, Ppu, Ppk use the overall sample
    standard deviation.

The caller may provide within_sigma from an appropriate
stable control-chart estimate, such as R-bar / d2 or
s-bar / c4.
"""

import math

from dataclasses import dataclass
from statistics import mean, stdev
from typing import Sequence


Number = int | float


class CapabilityInputError(ValueError):
    """Invalid input supplied to process capability analysis."""


@dataclass(frozen=True)
class ProcessCapabilityResult:
    sample_size: int
    mean: float

    lsl: float | None
    usl: float | None

    within_sigma: float | None
    overall_sigma: float

    cp: float | None
    cpl: float | None
    cpu: float | None
    cpk: float | None

    pp: float | None
    ppl: float | None
    ppu: float | None
    ppk: float | None


def _parse_observations(
    values: Sequence[Number],
) -> tuple[float, ...]:

    if isinstance(values, (str, bytes)):
        raise CapabilityInputError(
            "Observations must be a sequence."
        )

    parsed = []

    for index, value in enumerate(values, start=1):
        try:
            numeric = float(value)

        except (TypeError, ValueError):
            raise CapabilityInputError(
                f"Observation {index} must be numeric."
            )

        if not math.isfinite(numeric):
            raise CapabilityInputError(
                f"Observation {index} must be finite."
            )

        parsed.append(numeric)

    if len(parsed) < 2:
        raise CapabilityInputError(
            "At least two observations are required."
        )

    return tuple(parsed)


def _parse_limit(
    value,
    *,
    label,
) -> float | None:

    if value is None:
        return None

    try:
        numeric = float(value)

    except (TypeError, ValueError):
        raise CapabilityInputError(
            f"{label} must be numeric."
        )

    if not math.isfinite(numeric):
        raise CapabilityInputError(
            f"{label} must be finite."
        )

    return numeric


def _parse_within_sigma(
    value,
) -> float | None:

    if value is None:
        return None

    try:
        numeric = float(value)

    except (TypeError, ValueError):
        raise CapabilityInputError(
            "Within-process sigma must be numeric."
        )

    if not math.isfinite(numeric):
        raise CapabilityInputError(
            "Within-process sigma must be finite."
        )

    if numeric <= 0:
        raise CapabilityInputError(
            "Within-process sigma must be greater than 0."
        )

    return numeric


def _calculate_indices(
    *,
    process_mean: float,
    sigma: float,
    lsl: float | None,
    usl: float | None,
):
    lower = None
    upper = None
    two_sided = None

    if lsl is not None:
        lower = (
            process_mean - lsl
        ) / (
            3.0 * sigma
        )

    if usl is not None:
        upper = (
            usl - process_mean
        ) / (
            3.0 * sigma
        )

    if (
        lsl is not None
        and usl is not None
    ):
        two_sided = (
            usl - lsl
        ) / (
            6.0 * sigma
        )

    side_indices = [
        value
        for value in (lower, upper)
        if value is not None
    ]

    minimum_side = (
        min(side_indices)
        if side_indices
        else None
    )

    return (
        two_sided,
        lower,
        upper,
        minimum_side,
    )


def calculate_process_capability(
    observations: Sequence[Number],
    *,
    lsl: Number | None = None,
    usl: Number | None = None,
    within_sigma: Number | None = None,
) -> ProcessCapabilityResult:

    values = _parse_observations(
        observations
    )

    lower_specification = _parse_limit(
        lsl,
        label="LSL",
    )

    upper_specification = _parse_limit(
        usl,
        label="USL",
    )

    if (
        lower_specification is None
        and upper_specification is None
    ):
        raise CapabilityInputError(
            "At least one specification limit is required."
        )

    if (
        lower_specification is not None
        and upper_specification is not None
        and lower_specification
        >= upper_specification
    ):
        raise CapabilityInputError(
            "LSL must be lower than USL."
        )

    parsed_within_sigma = (
        _parse_within_sigma(
            within_sigma
        )
    )

    process_mean = float(
        mean(values)
    )

    overall_sigma = float(
        stdev(values)
    )

    cp = None
    cpl = None
    cpu = None
    cpk = None

    if parsed_within_sigma is not None:
        (
            cp,
            cpl,
            cpu,
            cpk,
        ) = _calculate_indices(
            process_mean=process_mean,
            sigma=parsed_within_sigma,
            lsl=lower_specification,
            usl=upper_specification,
        )

    pp = None
    ppl = None
    ppu = None
    ppk = None

    if overall_sigma > 0:
        (
            pp,
            ppl,
            ppu,
            ppk,
        ) = _calculate_indices(
            process_mean=process_mean,
            sigma=overall_sigma,
            lsl=lower_specification,
            usl=upper_specification,
        )

    return ProcessCapabilityResult(
        sample_size=len(values),
        mean=process_mean,

        lsl=lower_specification,
        usl=upper_specification,

        within_sigma=parsed_within_sigma,
        overall_sigma=overall_sigma,

        cp=cp,
        cpl=cpl,
        cpu=cpu,
        cpk=cpk,

        pp=pp,
        ppl=ppl,
        ppu=ppu,
        ppk=ppk,
    )