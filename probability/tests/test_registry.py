from django.test import SimpleTestCase

from probability.distributions import (
    DISTRIBUTIONS,
    create_distribution,
    get_continuous_distributions,
    get_discrete_distributions,
)

from probability.services import (
    get_default_parameters,
    validate_distribution_parameters,
)


class DistributionRegistryTests(
    SimpleTestCase
):

    def test_registry_contains_30_distributions(
        self,
    ):
        self.assertEqual(
            len(DISTRIBUTIONS),
            30,
        )

    def test_registry_category_counts(
        self,
    ):
        self.assertEqual(
            len(
                get_continuous_distributions()
            ),
            22,
        )

        self.assertEqual(
            len(
                get_discrete_distributions()
            ),
            8,
        )

    def test_all_default_parameters_are_valid(
        self,
    ):
        for key in DISTRIBUTIONS:

            with self.subTest(
                distribution=key
            ):
                parameters = (
                    get_default_parameters(
                        key
                    )
                )

                result = (
                    validate_distribution_parameters(
                        key,
                        parameters,
                    )
                )

                self.assertTrue(
                    result.is_valid,
                    msg=(
                        f"{key}: "
                        f"{result.field_errors} "
                        f"{result.non_field_errors}"
                    ),
                )

    def test_all_default_distributions_can_be_created(
        self,
    ):
        for key, spec in (
            DISTRIBUTIONS.items()
        ):

            with self.subTest(
                distribution=key
            ):
                parameters = (
                    get_default_parameters(
                        key
                    )
                )

                distribution = (
                    create_distribution(
                        key,
                        parameters,
                    )
                )

                self.assertTrue(
                    hasattr(
                        distribution,
                        "cdf",
                    )
                )

                self.assertTrue(
                    hasattr(
                        distribution,
                        "ppf",
                    )
                )

                if (
                    spec.category
                    == "continuous"
                ):
                    self.assertTrue(
                        hasattr(
                            distribution,
                            "pdf",
                        )
                    )

                else:
                    self.assertTrue(
                        hasattr(
                            distribution,
                            "pmf",
                        )
                    )

    def test_exponential_uses_rate_parameterization(
        self,
    ):
        distribution = create_distribution(
            "exponential",
            {
                "rate": 2.0,
            },
        )

        self.assertAlmostEqual(
            distribution.mean(),
            0.5,
            places=12,
        )

    def test_lognormal_uses_log_scale_parameters(
        self,
    ):
        distribution = create_distribution(
            "lognormal",
            {
                "mu": 0.0,
                "sigma": 1.0,
            },
        )

        self.assertAlmostEqual(
            distribution.median(),
            1.0,
            places=12,
        )