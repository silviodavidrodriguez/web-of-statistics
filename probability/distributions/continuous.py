import math

from scipy import stats

from .metadata import (
    DistributionSpec,
    ParameterSpec,
)


def _positive_float(
    name,
    label,
    symbol,
    default,
    help_text="",
):
    return ParameterSpec(
        name=name,
        label=label,
        symbol=symbol,
        default=default,
        kind="float",
        min_value=0,
        min_inclusive=False,
        help_text=help_text,
    )


def _positive_int(
    name,
    label,
    symbol,
    default,
    help_text="",
):
    return ParameterSpec(
        name=name,
        label=label,
        symbol=symbol,
        default=default,
        kind="int",
        min_value=1,
        help_text=help_text,
    )


def _validate_uniform(parameters):
    if parameters["lower"] >= parameters["upper"]:
        return [
            "The upper bound must be greater than "
            "the lower bound."
        ]

    return []


def _validate_triangular(parameters):
    lower = parameters["lower"]
    mode = parameters["mode"]
    upper = parameters["upper"]

    errors = []

    if lower >= upper:
        errors.append(
            "The upper bound must be greater than "
            "the lower bound."
        )

    if not lower <= mode <= upper:
        errors.append(
            "The mode must lie between the lower "
            "and upper bounds."
        )

    return errors


# ------------------------------------------------------------------
# Factories
# ------------------------------------------------------------------

def standard_normal_factory():
    return stats.norm(
        loc=0,
        scale=1,
    )


def normal_factory(mean, sd):
    return stats.norm(
        loc=mean,
        scale=sd,
    )


def student_t_factory(df):
    return stats.t(
        df=df,
    )


def chi_squared_factory(df):
    return stats.chi2(
        df=df,
    )


def fisher_f_factory(dfn, dfd):
    return stats.f(
        dfn=dfn,
        dfd=dfd,
    )


def uniform_factory(lower, upper):
    return stats.uniform(
        loc=lower,
        scale=upper - lower,
    )


def exponential_factory(rate):
    return stats.expon(
        scale=1 / rate,
    )


def gamma_factory(shape, scale):
    return stats.gamma(
        a=shape,
        scale=scale,
    )


def beta_factory(alpha, beta):
    return stats.beta(
        a=alpha,
        b=beta,
    )


def weibull_factory(shape, scale):
    return stats.weibull_min(
        c=shape,
        scale=scale,
    )


def lognormal_factory(mu, sigma):
    # If ln(X) ~ N(mu, sigma^2):
    #
    # scipy:
    # s = sigma
    # scale = exp(mu)
    return stats.lognorm(
        s=sigma,
        scale=math.exp(mu),
    )


def cauchy_factory(location, scale):
    return stats.cauchy(
        loc=location,
        scale=scale,
    )


def laplace_factory(location, scale):
    return stats.laplace(
        loc=location,
        scale=scale,
    )


def logistic_factory(location, scale):
    return stats.logistic(
        loc=location,
        scale=scale,
    )


def pareto_factory(shape, minimum):
    return stats.pareto(
        b=shape,
        scale=minimum,
    )


def rayleigh_factory(scale):
    return stats.rayleigh(
        scale=scale,
    )


def triangular_factory(lower, mode, upper):
    relative_mode = (
        (mode - lower)
        / (upper - lower)
    )

    return stats.triang(
        c=relative_mode,
        loc=lower,
        scale=upper - lower,
    )


def gumbel_factory(location, scale):
    return stats.gumbel_r(
        loc=location,
        scale=scale,
    )


def gev_factory(shape, location, scale):
    # Common extreme-value notation uses xi.
    #
    # scipy.stats.genextreme uses the opposite
    # sign convention:
    #
    # c = -xi
    return stats.genextreme(
        c=-shape,
        loc=location,
        scale=scale,
    )


def noncentral_t_factory(df, nc):
    return stats.nct(
        df=df,
        nc=nc,
    )


def noncentral_chi_squared_factory(df, nc):
    return stats.ncx2(
        df=df,
        nc=nc,
    )


def noncentral_f_factory(dfn, dfd, nc):
    return stats.ncf(
        dfn=dfn,
        dfd=dfd,
        nc=nc,
    )


# ------------------------------------------------------------------
# Parameter definitions
# ------------------------------------------------------------------

MEAN = ParameterSpec(
    name="mean",
    label="Mean",
    symbol="μ",
    default=0.0,
)

SD = _positive_float(
    name="sd",
    label="Standard deviation",
    symbol="σ",
    default=1.0,
)

LOCATION = ParameterSpec(
    name="location",
    label="Location",
    symbol="μ",
    default=0.0,
)

SCALE = _positive_float(
    name="scale",
    label="Scale",
    symbol="s",
    default=1.0,
)

DF = _positive_float(
    name="df",
    label="Degrees of freedom",
    symbol="df",
    default=10.0,
)

DFN = _positive_float(
    name="dfn",
    label="Numerator degrees of freedom",
    symbol="df₁",
    default=5.0,
)

DFD = _positive_float(
    name="dfd",
    label="Denominator degrees of freedom",
    symbol="df₂",
    default=10.0,
)

NONCENTRALITY = ParameterSpec(
    name="nc",
    label="Noncentrality parameter",
    symbol="δ",
    default=1.0,
)


CONTINUOUS_DISTRIBUTIONS = {

    "standard_normal": DistributionSpec(
        key="standard_normal",
        label="Standard Normal",
        category="continuous",
        variable_symbol="Z",
        parameters=(),
        factory=standard_normal_factory,
        description=(
            "Standard normal distribution with "
            "mean 0 and standard deviation 1."
        ),
        parameterization="Z ~ N(0, 1)",
    ),

    "normal": DistributionSpec(
        key="normal",
        label="Normal",
        category="continuous",
        variable_symbol="X",
        parameters=(
            MEAN,
            SD,
        ),
        factory=normal_factory,
        description=(
            "Normal distribution defined by its "
            "mean and standard deviation."
        ),
        parameterization="X ~ N(μ, σ²)",
    ),

    "student_t": DistributionSpec(
        key="student_t",
        label="Student's t",
        category="continuous",
        variable_symbol="T",
        parameters=(DF,),
        factory=student_t_factory,
        description=(
            "Student's t distribution."
        ),
        parameterization="T ~ t(df)",
    ),

    "chi_squared": DistributionSpec(
        key="chi_squared",
        label="Chi-Squared",
        category="continuous",
        variable_symbol="X²",
        parameters=(DF,),
        factory=chi_squared_factory,
        description=(
            "Chi-squared distribution."
        ),
        parameterization="X² ~ χ²(df)",
    ),

    "fisher_f": DistributionSpec(
        key="fisher_f",
        label="F Fisher-Snedecor",
        category="continuous",
        variable_symbol="F",
        parameters=(
            DFN,
            DFD,
        ),
        factory=fisher_f_factory,
        description=(
            "F distribution defined by numerator "
            "and denominator degrees of freedom."
        ),
        parameterization="F ~ F(df₁, df₂)",
    ),

    "uniform": DistributionSpec(
        key="uniform",
        label="Uniform",
        category="continuous",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                name="lower",
                label="Lower bound",
                symbol="a",
                default=0.0,
            ),
            ParameterSpec(
                name="upper",
                label="Upper bound",
                symbol="b",
                default=1.0,
            ),
        ),
        factory=uniform_factory,
        cross_validator=_validate_uniform,
        description=(
            "Continuous uniform distribution."
        ),
        parameterization="X ~ U(a, b)",
    ),

    "exponential": DistributionSpec(
        key="exponential",
        label="Exponential",
        category="continuous",
        variable_symbol="X",
        parameters=(
            _positive_float(
                name="rate",
                label="Rate",
                symbol="λ",
                default=1.0,
            ),
        ),
        factory=exponential_factory,
        description=(
            "Exponential distribution using the "
            "rate parameter λ."
        ),
        parameterization="X ~ Exp(λ)",
    ),

    "gamma": DistributionSpec(
        key="gamma",
        label="Gamma",
        category="continuous",
        variable_symbol="X",
        parameters=(
            _positive_float(
                "shape",
                "Shape",
                "k",
                2.0,
            ),
            _positive_float(
                "scale",
                "Scale",
                "θ",
                1.0,
            ),
        ),
        factory=gamma_factory,
        description="Gamma distribution.",
        parameterization="X ~ Gamma(k, θ)",
    ),

    "beta": DistributionSpec(
        key="beta",
        label="Beta",
        category="continuous",
        variable_symbol="X",
        parameters=(
            _positive_float(
                "alpha",
                "Alpha",
                "α",
                2.0,
            ),
            _positive_float(
                "beta",
                "Beta",
                "β",
                2.0,
            ),
        ),
        factory=beta_factory,
        description="Beta distribution on [0, 1].",
        parameterization="X ~ Beta(α, β)",
    ),

    "weibull": DistributionSpec(
        key="weibull",
        label="Weibull",
        category="continuous",
        variable_symbol="X",
        parameters=(
            _positive_float(
                "shape",
                "Shape",
                "k",
                2.0,
            ),
            _positive_float(
                "scale",
                "Scale",
                "λ",
                1.0,
            ),
        ),
        factory=weibull_factory,
        description="Weibull distribution.",
        parameterization="X ~ Weibull(k, λ)",
    ),

    "lognormal": DistributionSpec(
        key="lognormal",
        label="Lognormal",
        category="continuous",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                name="mu",
                label="Log-scale mean",
                symbol="μ",
                default=0.0,
            ),
            _positive_float(
                name="sigma",
                label="Log-scale standard deviation",
                symbol="σ",
                default=1.0,
            ),
        ),
        factory=lognormal_factory,
        description=(
            "Lognormal distribution where ln(X) "
            "is normally distributed."
        ),
        parameterization=(
            "ln(X) ~ N(μ, σ²)"
        ),
    ),

    "cauchy": DistributionSpec(
        key="cauchy",
        label="Cauchy",
        category="continuous",
        variable_symbol="X",
        parameters=(
            LOCATION,
            SCALE,
        ),
        factory=cauchy_factory,
        description="Cauchy distribution.",
        parameterization="X ~ Cauchy(x₀, γ)",
    ),

    "laplace": DistributionSpec(
        key="laplace",
        label="Laplace",
        category="continuous",
        variable_symbol="X",
        parameters=(
            LOCATION,
            SCALE,
        ),
        factory=laplace_factory,
        description="Laplace distribution.",
        parameterization="X ~ Laplace(μ, b)",
    ),

    "logistic": DistributionSpec(
        key="logistic",
        label="Logistic",
        category="continuous",
        variable_symbol="X",
        parameters=(
            LOCATION,
            SCALE,
        ),
        factory=logistic_factory,
        description="Logistic distribution.",
        parameterization="X ~ Logistic(μ, s)",
    ),

    "pareto": DistributionSpec(
        key="pareto",
        label="Pareto",
        category="continuous",
        variable_symbol="X",
        parameters=(
            _positive_float(
                "shape",
                "Shape",
                "α",
                3.0,
            ),
            _positive_float(
                "minimum",
                "Minimum",
                "xₘ",
                1.0,
            ),
        ),
        factory=pareto_factory,
        description="Pareto Type I distribution.",
        parameterization="X ~ Pareto(α, xₘ)",
    ),

    "rayleigh": DistributionSpec(
        key="rayleigh",
        label="Rayleigh",
        category="continuous",
        variable_symbol="X",
        parameters=(
            _positive_float(
                "scale",
                "Scale",
                "σ",
                1.0,
            ),
        ),
        factory=rayleigh_factory,
        description="Rayleigh distribution.",
        parameterization="X ~ Rayleigh(σ)",
    ),

    "triangular": DistributionSpec(
        key="triangular",
        label="Triangular",
        category="continuous",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                "lower",
                "Lower bound",
                "a",
                0.0,
            ),
            ParameterSpec(
                "mode",
                "Mode",
                "c",
                0.5,
            ),
            ParameterSpec(
                "upper",
                "Upper bound",
                "b",
                1.0,
            ),
        ),
        factory=triangular_factory,
        cross_validator=_validate_triangular,
        description="Triangular distribution.",
        parameterization="X ~ Triangular(a, c, b)",
    ),

    "gumbel": DistributionSpec(
        key="gumbel",
        label="Gumbel",
        category="continuous",
        variable_symbol="X",
        parameters=(
            LOCATION,
            SCALE,
        ),
        factory=gumbel_factory,
        description=(
            "Gumbel distribution for maxima."
        ),
        parameterization="X ~ Gumbel(μ, β)",
    ),

    "gev": DistributionSpec(
        key="gev",
        label="Generalized Extreme Value",
        category="continuous",
        variable_symbol="X",
        parameters=(
            ParameterSpec(
                "shape",
                "Shape",
                "ξ",
                0.0,
            ),
            LOCATION,
            SCALE,
        ),
        factory=gev_factory,
        description=(
            "Generalized Extreme Value distribution."
        ),
        parameterization="X ~ GEV(ξ, μ, σ)",
    ),

    "noncentral_t": DistributionSpec(
        key="noncentral_t",
        label="Noncentral Student's t",
        category="continuous",
        variable_symbol="T",
        parameters=(
            DF,
            NONCENTRALITY,
        ),
        factory=noncentral_t_factory,
        description=(
            "Noncentral Student's t distribution."
        ),
        parameterization="T ~ nct(df, δ)",
    ),

    "noncentral_chi_squared": DistributionSpec(
        key="noncentral_chi_squared",
        label="Noncentral Chi-Squared",
        category="continuous",
        variable_symbol="X²",
        parameters=(
            DF,
            ParameterSpec(
                name="nc",
                label="Noncentrality parameter",
                symbol="λ",
                default=1.0,
                min_value=0.0,
            ),
        ),
        factory=noncentral_chi_squared_factory,
        description=(
            "Noncentral chi-squared distribution."
        ),
        parameterization="X² ~ ncχ²(df, λ)",
    ),

    "noncentral_f": DistributionSpec(
        key="noncentral_f",
        label="Noncentral F",
        category="continuous",
        variable_symbol="F",
        parameters=(
            DFN,
            DFD,
            ParameterSpec(
                name="nc",
                label="Noncentrality parameter",
                symbol="λ",
                default=1.0,
                min_value=0.0,
            ),
        ),
        factory=noncentral_f_factory,
        description="Noncentral F distribution.",
        parameterization=(
            "F ~ ncF(df₁, df₂, λ)"
        ),
    ),
}