import math

from django.test import SimpleTestCase

from probability.services import (
    ComparisonCurve,
    ExplorerError,
    build_comparison_figure,
    build_explorer_figure,
    get_distribution_properties,
)


class ExplorerPropertiesTests(
    SimpleTestCase
):

    def test_standard_normal_properties(
        self,
    ):
        properties = (
            get_distribution_properties(
                "standard_normal",
                {},
            )
        )

        self.assertEqual(
            properties.mean,
            0.0,
        )

        self.assertEqual(
            properties.variance,
            1.0,
        )

        self.assertEqual(
            properties.standard_deviation,
            1.0,
        )

        self.assertEqual(
            properties.skewness,
            0.0,
        )

        self.assertEqual(
            properties.excess_kurtosis,
            0.0,
        )

    def test_normal_median_matches_mean(
        self,
    ):
        properties = (
            get_distribution_properties(
                "normal",
                {
                    "mean": "100",
                    "sd": "15",
                },
            )
        )

        self.assertAlmostEqual(
            properties.median,
            100.0,
            places=12,
        )

    def test_cauchy_undefined_moments(
        self,
    ):
        properties = (
            get_distribution_properties(
                "cauchy",
                {
                    "location": "0",
                    "scale": "1",
                },
            )
        )

        self.assertIsNone(
            properties.mean
        )

        self.assertIsNone(
            properties.variance
        )

        self.assertIsNone(
            properties.standard_deviation
        )

    def test_standard_normal_quantile(
        self,
    ):
        properties = (
            get_distribution_properties(
                "standard_normal",
                {},
            )
        )

        self.assertAlmostEqual(
            properties.quantiles[
                0.975
            ],
            1.959963984540054,
            places=12,
        )

    def test_invalid_quantile_probability_rejected(
        self,
    ):
        with self.assertRaises(
            ExplorerError
        ):
            get_distribution_properties(
                "standard_normal",
                {},
                quantile_probabilities=(
                    0.5,
                    1.0,
                ),
            )


class ExplorerPlotTests(
    SimpleTestCase
):

    def test_continuous_pdf(
        self,
    ):
        figure = build_explorer_figure(
            "standard_normal",
            {},
            view="pdf",
        )

        self.assertEqual(
            len(figure.data),
            1,
        )

        self.assertEqual(
            figure.data[0].type,
            "scatter",
        )

    def test_continuous_cdf(
        self,
    ):
        figure = build_explorer_figure(
            "standard_normal",
            {},
            view="cdf",
        )

        self.assertEqual(
            len(figure.data),
            1,
        )

        y = list(
            figure.data[0].y
        )

        self.assertLess(
            y[0],
            y[-1],
        )

    def test_exponential_hazard_is_constant(
        self,
    ):
        figure = build_explorer_figure(
            "exponential",
            {
                "rate": "2",
            },
            view="hazard",
        )

        y = [
            value
            for value
            in figure.data[0].y
            if (
                value is not None
                and math.isfinite(
                    float(value)
                )
            )
        ]

        self.assertTrue(y)

        for value in y[::100]:
            self.assertAlmostEqual(
                float(value),
                2.0,
                places=8,
            )

    def test_binomial_pmf(
        self,
    ):
        figure = build_explorer_figure(
            "binomial",
            {
                "n": "10",
                "p": "0.5",
            },
            view="pmf",
        )

        self.assertEqual(
            figure.data[0].type,
            "bar",
        )

        self.assertEqual(
            list(
                figure.data[0].x
            ),
            list(range(11)),
        )

    def test_invalid_discrete_pdf_rejected(
        self,
    ):
        with self.assertRaises(
            ExplorerError
        ):
            build_explorer_figure(
                "binomial",
                {
                    "n": "10",
                    "p": "0.5",
                },
                view="pdf",
            )


class ExplorerComparisonTests(
    SimpleTestCase
):

    def test_compare_student_t_configurations(
        self,
    ):
        figure = (
            build_comparison_figure(
                [
                    ComparisonCurve(
                        "student_t",
                        {
                            "df": 2,
                        },
                        "t, df = 2",
                    ),
                    ComparisonCurve(
                        "student_t",
                        {
                            "df": 5,
                        },
                        "t, df = 5",
                    ),
                    ComparisonCurve(
                        "student_t",
                        {
                            "df": 30,
                        },
                        "t, df = 30",
                    ),
                    ComparisonCurve(
                        "standard_normal",
                        {},
                        "Standard Normal",
                    ),
                ],
                view="pdf",
            )
        )

        self.assertEqual(
            len(figure.data),
            4,
        )

    def test_mixed_categories_rejected(
        self,
    ):
        with self.assertRaises(
            ExplorerError
        ):
            build_comparison_figure(
                [
                    ComparisonCurve(
                        "standard_normal",
                        {},
                        "Normal",
                    ),
                    ComparisonCurve(
                        "binomial",
                        {
                            "n": 10,
                            "p": 0.5,
                        },
                        "Binomial",
                    ),
                ]
            )