from django.test import SimpleTestCase

from probability.services import (
    calculate,
    format_calculation,
    format_number,
)


class CalculationFormattingTests(
    SimpleTestCase
):

    def test_standard_normal_left_expression(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "standard_normal",
                {},
                "left",
                {
                    "x": "1.96",
                },
            )
        )

        self.assertEqual(
            formatted.expression,
            "P(Z ≤ 1.96) = 0.975002",
        )

        self.assertEqual(
            formatted.result_display,
            "0.975002",
        )

        self.assertEqual(
            formatted.complement_display,
            "0.024998",
        )

    def test_standard_normal_right_expression(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "standard_normal",
                {},
                "right",
                {
                    "x": "1.96",
                },
            )
        )

        self.assertEqual(
            formatted.expression,
            "P(Z ≥ 1.96) = 0.024998",
        )

    def test_continuous_interval_expression(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "standard_normal",
                {},
                "between",
                {
                    "a": "-1.96",
                    "b": "1.96",
                },
            )
        )

        self.assertEqual(
            formatted.expression,
            (
                "P(-1.96 ≤ Z ≤ 1.96) "
                "= 0.950004"
            ),
        )

    def test_density_is_not_described_as_probability(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "standard_normal",
                {},
                "density",
                {
                    "x": "0",
                },
            )
        )

        self.assertEqual(
            formatted.result_label,
            "Density",
        )

        self.assertIn(
            "not itself a probability",
            formatted.interpretation,
        )

    def test_central_interval_formatting(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "standard_normal",
                {},
                "central_interval",
                {
                    "p": "0.95",
                },
            )
        )

        self.assertEqual(
            formatted.result_display,
            (
                "[-1.959964, "
                "1.959964]"
            ),
        )

        self.assertIn(
            "2.50%",
            formatted.interpretation,
        )

    def test_binomial_mass_expression(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "binomial",
                {
                    "n": "10",
                    "p": "0.5",
                },
                "mass",
                {
                    "x": "5",
                },
            )
        )

        self.assertEqual(
            formatted.expression,
            "P(X = 5) = 0.246094",
        )

    def test_binomial_greater_equal_expression(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "binomial",
                {
                    "n": "10",
                    "p": "0.5",
                },
                "greater_equal",
                {
                    "x": "5",
                },
            )
        )

        self.assertEqual(
            formatted.expression,
            "P(X ≥ 5) = 0.623047",
        )

        self.assertIn(
            "5 or more",
            formatted.interpretation,
        )

    def test_discrete_quantile_formatting(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "binomial",
                {
                    "n": "10",
                    "p": "0.5",
                },
                "quantile",
                {
                    "p": "0.5",
                },
            )
        )

        self.assertEqual(
            formatted.result_display,
            "5",
        )

        self.assertIn(
            "smallest integer",
            formatted.interpretation,
        )

    def test_parameter_summary(
        self,
    ):
        formatted = format_calculation(
            calculate(
                "normal",
                {
                    "mean": "100",
                    "sd": "15",
                },
                "left",
                {
                    "x": "115",
                },
            )
        )

        self.assertEqual(
            formatted.parameter_summary,
            (
                "Mean (μ) = 100",
                (
                    "Standard deviation "
                    "(σ) = 15"
                ),
            ),
        )

    def test_small_numbers_use_scientific_notation(
        self,
    ):
        self.assertEqual(
            format_number(
                0.000000123456
            ),
            "1.23456e-07",
        )