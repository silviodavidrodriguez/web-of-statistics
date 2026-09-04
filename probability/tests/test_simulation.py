import csv
import io

import numpy as np

from django.test import SimpleTestCase

from probability.distributions import (
    DISTRIBUTIONS,
)

from probability.services import (
    SimulationInputError,
    build_simulation_figures,
    get_default_parameters,
    simulate_distribution,
    simulation_to_csv,
)


class SimulationInputTests(
    SimpleTestCase
):

    def test_sample_size_below_minimum_is_rejected(
        self,
    ):
        with self.assertRaises(
            SimulationInputError
        ):
            simulate_distribution(
                "standard_normal",
                {},
                sample_size=1,
                seed=1,
            )

    def test_non_integer_sample_size_is_rejected(
        self,
    ):
        with self.assertRaises(
            SimulationInputError
        ):
            simulate_distribution(
                "standard_normal",
                {},
                sample_size="10.5",
                seed=1,
            )

    def test_negative_seed_is_rejected(
        self,
    ):
        with self.assertRaises(
            SimulationInputError
        ):
            simulate_distribution(
                "standard_normal",
                {},
                sample_size=100,
                seed=-1,
            )


class SimulationGenerationTests(
    SimpleTestCase
):

    def test_same_seed_is_reproducible(
        self,
    ):
        first = simulate_distribution(
            "standard_normal",
            {},
            sample_size=1000,
            seed=12345,
        )

        second = simulate_distribution(
            "standard_normal",
            {},
            sample_size=1000,
            seed=12345,
        )

        self.assertTrue(
            np.array_equal(
                first.sample,
                second.sample,
            )
        )

    def test_different_seeds_change_sample(
        self,
    ):
        first = simulate_distribution(
            "standard_normal",
            {},
            sample_size=1000,
            seed=1,
        )

        second = simulate_distribution(
            "standard_normal",
            {},
            sample_size=1000,
            seed=2,
        )

        self.assertFalse(
            np.array_equal(
                first.sample,
                second.sample,
            )
        )

    def test_standard_normal_simulation_statistics(
        self,
    ):
        result = simulate_distribution(
            "standard_normal",
            {},
            sample_size=5000,
            seed=123,
        )

        self.assertEqual(
            result.sample_size,
            5000,
        )

        self.assertEqual(
            result.theoretical.mean,
            0.0,
        )

        self.assertEqual(
            result.theoretical.variance,
            1.0,
        )

        self.assertAlmostEqual(
            result.simulated.mean,
            0.0,
            delta=0.08,
        )

        self.assertAlmostEqual(
            result.simulated.variance,
            1.0,
            delta=0.12,
        )

    def test_cauchy_theoretical_moments_are_undefined(
        self,
    ):
        result = simulate_distribution(
            "cauchy",
            {
                "location": 0,
                "scale": 1,
            },
            sample_size=1000,
            seed=123,
        )

        self.assertIsNone(
            result.theoretical.mean
        )

        self.assertIsNone(
            result.theoretical.variance
        )

        self.assertIsNone(
            result.theoretical.standard_deviation
        )

    def test_degenerate_poisson_is_all_zero(
        self,
    ):
        result = simulate_distribution(
            "poisson",
            {
                "rate": 0,
            },
            sample_size=100,
            seed=123,
        )

        self.assertTrue(
            np.all(
                result.sample == 0
            )
        )

        self.assertEqual(
            result.simulated.mean,
            0.0,
        )

    def test_every_default_distribution_can_be_simulated(
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
                    simulate_distribution(
                        key,
                        parameters,
                        sample_size=250,
                        seed=123,
                    )
                )

                self.assertEqual(
                    len(result.sample),
                    250,
                )

                self.assertTrue(
                    np.all(
                        np.isfinite(
                            result.sample
                        )
                    )
                )

    def test_blank_seed_generates_effective_seed(
        self,
    ):
        result = simulate_distribution(
            "standard_normal",
            {},
            sample_size=100,
            seed=None,
        )

        self.assertIsInstance(
            result.seed,
            int,
        )

        self.assertGreaterEqual(
            result.seed,
            0,
        )

        self.assertLessEqual(
            result.seed,
            4_294_967_295,
        )


    def test_generated_seed_reproduces_sample(
        self,
    ):
        first = simulate_distribution(
            "standard_normal",
            {},
            sample_size=100,
            seed=None,
        )

        second = simulate_distribution(
            "standard_normal",
            {},
            sample_size=100,
            seed=first.seed,
        )

        self.assertTrue(
            np.array_equal(
                first.sample,
                second.sample,
            )
        )


class SimulationPlotTests(
    SimpleTestCase
):

    def test_continuous_simulation_has_three_figures(
        self,
    ):
        result = simulate_distribution(
            "normal",
            {
                "mean": 0,
                "sd": 1,
            },
            sample_size=1000,
            seed=123,
        )

        figures = (
            build_simulation_figures(
                result
            )
        )

        self.assertEqual(
            set(figures),
            {
                "distribution",
                "cdf",
                "qq",
            },
        )

        self.assertEqual(
            len(
                figures[
                    "distribution"
                ].data
            ),
            2,
        )

    def test_discrete_simulation_has_three_figures(
        self,
    ):
        result = simulate_distribution(
            "binomial",
            {
                "n": 10,
                "p": 0.5,
            },
            sample_size=1000,
            seed=123,
        )

        figures = (
            build_simulation_figures(
                result
            )
        )

        self.assertEqual(
            set(figures),
            {
                "distribution",
                "cdf",
                "probability_comparison",
            },
        )

        self.assertEqual(
            len(
                figures[
                    "distribution"
                ].data
            ),
            2,
        )

    def test_continuous_qq_contains_finite_points(
        self,
    ):
        result = simulate_distribution(
            "standard_normal",
            {},
            sample_size=1000,
            seed=123,
        )

        figure = (
            build_simulation_figures(
                result
            )["qq"]
        )

        x = np.asarray(
            figure.data[0].x,
            dtype=float,
        )

        y = np.asarray(
            figure.data[0].y,
            dtype=float,
        )

        self.assertTrue(
            np.all(
                np.isfinite(x)
            )
        )

        self.assertTrue(
            np.all(
                np.isfinite(y)
            )
        )


class SimulationExportTests(
    SimpleTestCase
):

    def test_csv_contains_all_observations(
        self,
    ):
        result = simulate_distribution(
            "poisson",
            {
                "rate": 3,
            },
            sample_size=25,
            seed=123,
        )

        content = simulation_to_csv(
            result
        )

        rows = list(
            csv.reader(
                io.StringIO(
                    content
                )
            )
        )

        self.assertEqual(
            rows[0],
            [
                "observation",
                "value",
            ],
        )

        self.assertEqual(
            len(rows),
            26,
        )