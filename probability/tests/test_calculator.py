import math

from django.test import SimpleTestCase

from probability.services import (
    CalculatorInputError,
    calculate,
)


class ContinuousCalculatorTests(
    SimpleTestCase
):

    def test_standard_normal_left_probability(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "left",
            {
                "x": "1.96",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.9750021048517795,
            places=12,
        )

    def test_standard_normal_right_probability(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "right",
            {
                "x": "1.96",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.024997895148220435,
            places=12,
        )

    def test_standard_normal_between(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "between",
            {
                "a": "-1.96",
                "b": "1.96",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.950004209703559,
            places=12,
        )

    def test_standard_normal_density_at_zero(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "density",
            {
                "x": "0",
            },
        )

        expected = (
            1.0
            / math.sqrt(
                2.0 * math.pi
            )
        )

        self.assertAlmostEqual(
            result.value,
            expected,
            places=12,
        )

    def test_standard_normal_left_quantile(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "left_quantile",
            {
                "p": "0.975",
            },
        )

        self.assertAlmostEqual(
            result.value,
            1.959963984540054,
            places=12,
        )

    def test_standard_normal_right_quantile(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "right_quantile",
            {
                "p": "0.025",
            },
        )

        self.assertAlmostEqual(
            result.value,
            1.959963984540054,
            places=12,
        )

    def test_standard_normal_central_interval(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "central_interval",
            {
                "p": "0.95",
            },
        )

        lower, upper = result.value

        self.assertAlmostEqual(
            lower,
            -1.959963984540054,
            places=12,
        )

        self.assertAlmostEqual(
            upper,
            1.959963984540054,
            places=12,
        )

    def test_student_t_is_symmetric_at_zero(
        self,
    ):
        result = calculate(
            "student_t",
            {
                "df": "10",
            },
            "left",
            {
                "x": "0",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.5,
            places=12,
        )

    def test_chi_squared_at_zero(
        self,
    ):
        result = calculate(
            "chi_squared",
            {
                "df": "5",
            },
            "left",
            {
                "x": "0",
            },
        )

        self.assertEqual(
            result.value,
            0.0,
        )

    def test_exponential_known_probability(
        self,
    ):
        result = calculate(
            "exponential",
            {
                "rate": "1",
            },
            "left",
            {
                "x": "1",
            },
        )

        expected = (
            1.0
            - math.exp(-1.0)
        )

        self.assertAlmostEqual(
            result.value,
            expected,
            places=12,
        )

    def test_reversed_interval_is_rejected(
        self,
    ):
        with self.assertRaises(
            CalculatorInputError
        ):
            calculate(
                "standard_normal",
                {},
                "between",
                {
                    "a": "2",
                    "b": "-2",
                },
            )

    def test_invalid_quantile_probability_is_rejected(
        self,
    ):
        with self.assertRaises(
            CalculatorInputError
        ):
            calculate(
                "standard_normal",
                {},
                "left_quantile",
                {
                    "p": "1.5",
                },
            )


class DiscreteCalculatorTests(
    SimpleTestCase
):

    PARAMETERS = {
        "n": "10",
        "p": "0.5",
    }

    def test_binomial_mass(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "mass",
            {
                "x": "5",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.24609375,
            places=12,
        )

    def test_binomial_less(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "less",
            {
                "x": "5",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.376953125,
            places=12,
        )

    def test_binomial_less_equal(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "less_equal",
            {
                "x": "5",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.623046875,
            places=12,
        )

    def test_binomial_greater(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "greater",
            {
                "x": "5",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.376953125,
            places=12,
        )

    def test_binomial_greater_equal(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "greater_equal",
            {
                "x": "5",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.623046875,
            places=12,
        )

    def test_discrete_strict_and_inclusive_relationship(
        self,
    ):
        greater = calculate(
            "binomial",
            self.PARAMETERS,
            "greater",
            {
                "x": "5",
            },
        )

        equal = calculate(
            "binomial",
            self.PARAMETERS,
            "mass",
            {
                "x": "5",
            },
        )

        greater_equal = calculate(
            "binomial",
            self.PARAMETERS,
            "greater_equal",
            {
                "x": "5",
            },
        )

        self.assertAlmostEqual(
            greater.value
            + equal.value,
            greater_equal.value,
            places=12,
        )

    def test_binomial_between(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "between",
            {
                "a": "3",
                "b": "7",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.890625,
            places=12,
        )

    def test_binomial_outside(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "outside",
            {
                "a": "3",
                "b": "7",
            },
        )

        self.assertAlmostEqual(
            result.value,
            0.34375,
            places=12,
        )

    def test_binomial_quantile_is_integer(
        self,
    ):
        result = calculate(
            "binomial",
            self.PARAMETERS,
            "quantile",
            {
                "p": "0.5",
            },
        )

        self.assertEqual(
            result.value,
            5,
        )

        self.assertIsInstance(
            result.value,
            int,
        )

    def test_poisson_probability_at_zero(
        self,
    ):
        result = calculate(
            "poisson",
            {
                "rate": "3",
            },
            "mass",
            {
                "x": "0",
            },
        )

        self.assertAlmostEqual(
            result.value,
            math.exp(-3),
            places=12,
        )

    def test_poisson_zero_rate_is_degenerate_at_zero(
        self,
    ):
        result = calculate(
            "poisson",
            {
                "rate": "0",
            },
            "mass",
            {
                "x": "0",
            },
        )

        self.assertEqual(
            result.value,
            1.0,
        )

    def test_binomial_zero_trials_is_degenerate_at_zero(
        self,
    ):
        result = calculate(
            "binomial",
            {
                "n": "0",
                "p": "0.5",
            },
            "mass",
            {
                "x": "0",
            },
        )

        self.assertEqual(
            result.value,
            1.0,
        )

    def test_non_integer_discrete_value_is_rejected(
        self,
    ):
        with self.assertRaises(
            CalculatorInputError
        ):
            calculate(
                "binomial",
                self.PARAMETERS,
                "mass",
                {
                    "x": "4.5",
                },
            )