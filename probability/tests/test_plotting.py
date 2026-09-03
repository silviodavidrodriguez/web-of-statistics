from django.test import SimpleTestCase

from probability.distributions import (
    DISTRIBUTIONS,
    create_distribution,
)

from probability.services import (
    build_calculation_figure,
    calculate,
    get_default_parameters,
)


class ContinuousPlottingTests(
    SimpleTestCase
):

    def test_left_probability_has_shaded_region(
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

        figure = (
            build_calculation_figure(
                result
            )
        )

        self.assertEqual(
            len(figure.data),
            2,
        )

        self.assertEqual(
            figure.layout.title.text,
            "P(Z ≤ 1.96) = 0.975002",
        )

    def test_between_has_two_boundaries(
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

        figure = (
            build_calculation_figure(
                result
            )
        )

        self.assertEqual(
            len(figure.layout.shapes),
            2,
        )

    def test_outside_has_two_regions(
        self,
    ):
        result = calculate(
            "standard_normal",
            {},
            "outside",
            {
                "a": "-1.96",
                "b": "1.96",
            },
        )

        figure = (
            build_calculation_figure(
                result
            )
        )

        self.assertEqual(
            len(figure.data),
            3,
        )

    def test_density_has_selected_marker(
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

        figure = (
            build_calculation_figure(
                result
            )
        )

        self.assertEqual(
            len(figure.data),
            2,
        )


class DiscretePlottingTests(
    SimpleTestCase
):

    def test_binomial_support_is_plotted(
        self,
    ):
        result = calculate(
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

        figure = (
            build_calculation_figure(
                result
            )
        )

        self.assertEqual(
            list(figure.data[0].x),
            list(range(11)),
        )

    def test_binomial_greater_equal_highlights_six_values(
        self,
    ):
        result = calculate(
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

        figure = (
            build_calculation_figure(
                result
            )
        )

        colors = list(
            figure.data[0].marker.color
        )

        selected_count = sum(
            color == "#2563eb"
            for color in colors
        )

        self.assertEqual(
            selected_count,
            6,
        )

    def test_degenerate_poisson_can_be_plotted(
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

        figure = (
            build_calculation_figure(
                result
            )
        )

        self.assertIn(
            0,
            list(
                figure.data[0].x
            ),
        )

        zero_index = list(
            figure.data[0].x
        ).index(0)

        self.assertEqual(
            figure.data[0].y[
                zero_index
            ],
            1.0,
        )


class DistributionPlotCoverageTests(
    SimpleTestCase
):

    def test_every_default_distribution_can_be_plotted(
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

                median = (
                    distribution.ppf(
                        0.5
                    )
                )

                if (
                    spec.category
                    == "continuous"
                ):
                    result = calculate(
                        key,
                        parameters,
                        "left",
                        {
                            "x": str(
                                float(median)
                            ),
                        },
                    )

                else:
                    result = calculate(
                        key,
                        parameters,
                        "mass",
                        {
                            "x": str(
                                int(median)
                            ),
                        },
                    )

                figure = (
                    build_calculation_figure(
                        result
                    )
                )

                self.assertGreater(
                    len(
                        figure.data
                    ),
                    0,
                )