from django.test import SimpleTestCase

from probability.services import (
    DistributionValidationError,
    require_valid_distribution_parameters,
    validate_distribution_parameters,
)


class DistributionValidatorTests(
    SimpleTestCase
):

    def test_valid_normal_parameters(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "normal",
                {
                    "mean": "5",
                    "sd": "2",
                },
            )
        )

        self.assertTrue(
            result.is_valid
        )

        self.assertEqual(
            result.values,
            {
                "mean": 5.0,
                "sd": 2.0,
            },
        )

    def test_integer_parameter_is_converted_to_int(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "binomial",
                {
                    "n": "10",
                    "p": "0.5",
                },
            )
        )

        self.assertTrue(
            result.is_valid
        )

        self.assertIsInstance(
            result.values["n"],
            int,
        )

    def test_negative_standard_deviation_is_rejected(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "normal",
                {
                    "mean": "0",
                    "sd": "-1",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertIn(
            "sd",
            result.field_errors,
        )

    def test_probability_above_one_is_rejected(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "binomial",
                {
                    "n": "10",
                    "p": "1.2",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertIn(
            "p",
            result.field_errors,
        )

    def test_non_integer_count_is_rejected(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "binomial",
                {
                    "n": "10.5",
                    "p": "0.5",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertIn(
            "n",
            result.field_errors,
        )

    def test_nan_is_rejected(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "normal",
                {
                    "mean": "nan",
                    "sd": "1",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertIn(
            "finite",
            result.field_errors[
                "mean"
            ],
        )

    def test_infinity_is_rejected(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "normal",
                {
                    "mean": "0",
                    "sd": "inf",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertIn(
            "finite",
            result.field_errors[
                "sd"
            ],
        )

    def test_uniform_bounds_are_validated(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "uniform",
                {
                    "lower": "10",
                    "upper": "5",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertTrue(
            result.non_field_errors
        )

    def test_triangular_mode_is_validated(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "triangular",
                {
                    "lower": "0",
                    "mode": "8",
                    "upper": "5",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertTrue(
            result.non_field_errors
        )

    def test_hypergeometric_population_is_validated(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "hypergeometric",
                {
                    "population": "20",
                    "successes": "25",
                    "draws": "5",
                },
            )
        )

        self.assertFalse(
            result.is_valid
        )

    def test_unknown_distribution_is_rejected(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "not_a_distribution",
                {},
            )
        )

        self.assertFalse(
            result.is_valid
        )

        self.assertTrue(
            result.non_field_errors
        )

    def test_require_valid_parameters_raises_exception(
        self,
    ):
        with self.assertRaises(
            DistributionValidationError
        ):
            require_valid_distribution_parameters(
                "normal",
                {
                    "mean": "0",
                    "sd": "-1",
                },
            )

    def test_poisson_zero_rate_is_valid(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "poisson",
                {
                    "rate": "0",
                },
            )
        )

        self.assertTrue(
            result.is_valid
        )

    def test_binomial_zero_trials_is_valid(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "binomial",
                {
                    "n": "0",
                    "p": "0.5",
                },
            )
        )

        self.assertTrue(
            result.is_valid
        )

    def test_hypergeometric_zero_draws_is_valid(
        self,
    ):
        result = (
            validate_distribution_parameters(
                "hypergeometric",
                {
                    "population": "20",
                    "successes": "5",
                    "draws": "0",
                },
            )
        )

        self.assertTrue(
            result.is_valid
        )