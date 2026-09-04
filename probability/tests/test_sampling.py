import numpy as np

from django.test import SimpleTestCase

from probability.distributions import (
    DISTRIBUTIONS,
)

from probability.services import (
    SamplingInputError,
    build_clt_figure,
    build_lln_figure,
    build_sampling_distribution_figure,
    get_default_parameters,
    simulate_clt,
    simulate_lln,
    simulate_sampling_distribution,
)


class SamplingInputTests(
    SimpleTestCase
):

    def test_sample_size_zero_is_rejected(
        self,
    ):
        with self.assertRaises(
            SamplingInputError
        ):
            simulate_sampling_distribution(
                "standard_normal",
                {},
                sample_size=0,
                repetitions=100,
                seed=1,
            )

    def test_variance_requires_two_observations(
        self,
    ):
        with self.assertRaises(
            SamplingInputError
        ):
            simulate_sampling_distribution(
                "standard_normal",
                {},
                statistic="variance",
                sample_size=1,
                repetitions=100,
                seed=1,
            )

    def test_too_few_repetitions_are_rejected(
        self,
    ):
        with self.assertRaises(
            SamplingInputError
        ):
            simulate_sampling_distribution(
                "standard_normal",
                {},
                sample_size=10,
                repetitions=49,
                seed=1,
            )

    def test_negative_seed_is_rejected(
        self,
    ):
        with self.assertRaises(
            SamplingInputError
        ):
            simulate_sampling_distribution(
                "standard_normal",
                {},
                sample_size=10,
                repetitions=100,
                seed=-1,
            )


class SamplingDistributionTests(
    SimpleTestCase
):

    def test_sampling_distribution_is_reproducible(
        self,
    ):
        first = (
            simulate_sampling_distribution(
                "normal",
                {
                    "mean": 0,
                    "sd": 1,
                },
                statistic="mean",
                sample_size=10,
                repetitions=500,
                seed=123,
            )
        )

        second = (
            simulate_sampling_distribution(
                "normal",
                {
                    "mean": 0,
                    "sd": 1,
                },
                statistic="mean",
                sample_size=10,
                repetitions=500,
                seed=123,
            )
        )

        self.assertTrue(
            np.array_equal(
                first.values,
                second.values,
            )
        )

    def test_standard_normal_sample_mean(
        self,
    ):
        result = (
            simulate_sampling_distribution(
                "standard_normal",
                {},
                statistic="mean",
                sample_size=20,
                repetitions=5000,
                seed=123,
            )
        )

        self.assertAlmostEqual(
            result.empirical_mean,
            0.0,
            delta=0.03,
        )

        self.assertAlmostEqual(
            result.empirical_standard_deviation,
            1 / np.sqrt(20),
            delta=0.02,
        )

        self.assertEqual(
            result.theoretical_reference,
            0.0,
        )

    def test_sample_variance_reference(
        self,
    ):
        result = (
            simulate_sampling_distribution(
                "standard_normal",
                {},
                statistic="variance",
                sample_size=20,
                repetitions=4000,
                seed=123,
            )
        )

        self.assertEqual(
            result.theoretical_reference,
            1.0,
        )

        self.assertAlmostEqual(
            result.empirical_mean,
            1.0,
            delta=0.05,
        )

    def test_every_default_distribution_supports_sample_mean(
        self,
    ):
        for key in DISTRIBUTIONS:

            with self.subTest(
                distribution=key
            ):
                result = (
                    simulate_sampling_distribution(
                        key,
                        get_default_parameters(
                            key
                        ),
                        statistic="mean",
                        sample_size=3,
                        repetitions=100,
                        seed=123,
                    )
                )

                self.assertEqual(
                    len(result.values),
                    100,
                )

                self.assertTrue(
                    np.all(
                        np.isfinite(
                            result.values
                        )
                    )
                )


class CLTTests(
    SimpleTestCase
):

    def test_clt_multiple_sample_sizes(
        self,
    ):
        result = simulate_clt(
            "standard_normal",
            {},
            sample_sizes=(
                1,
                5,
                30,
            ),
            repetitions=500,
            seed=123,
        )

        self.assertEqual(
            result.sample_sizes,
            (
                1,
                5,
                30,
            ),
        )

        self.assertEqual(
            set(
                result.means_by_size
            ),
            {
                1,
                5,
                30,
            },
        )

    def test_clt_is_reproducible(
        self,
    ):
        first = simulate_clt(
            "standard_normal",
            {},
            sample_sizes=(
                2,
                10,
            ),
            repetitions=500,
            seed=123,
        )

        second = simulate_clt(
            "standard_normal",
            {},
            sample_sizes=(
                2,
                10,
            ),
            repetitions=500,
            seed=123,
        )

        for sample_size in (
            first.sample_sizes
        ):
            self.assertTrue(
                np.array_equal(
                    first.means_by_size[
                        sample_size
                    ],
                    second.means_by_size[
                        sample_size
                    ],
                )
            )

    def test_classical_clt_available_for_normal(
        self,
    ):
        result = simulate_clt(
            "normal",
            {
                "mean": 100,
                "sd": 15,
            },
            sample_sizes=(
                5,
                30,
            ),
            repetitions=500,
            seed=123,
        )

        self.assertTrue(
            result.classical_clt_available
        )

        self.assertEqual(
            result.source_mean,
            100.0,
        )

        self.assertEqual(
            result.source_variance,
            225.0,
        )

    def test_classical_clt_not_available_for_cauchy(
        self,
    ):
        result = simulate_clt(
            "cauchy",
            {
                "location": 0,
                "scale": 1,
            },
            sample_sizes=(
                5,
                30,
            ),
            repetitions=500,
            seed=123,
        )

        self.assertFalse(
            result.classical_clt_available
        )

        self.assertIsNone(
            result.source_mean
        )

        self.assertIsNone(
            result.source_variance
        )

    def test_clt_figure_contains_curves(
        self,
    ):
        result = simulate_clt(
            "exponential",
            {
                "rate": 1,
            },
            sample_sizes=(
                1,
                5,
                30,
            ),
            repetitions=500,
            seed=123,
        )

        figure = build_clt_figure(
            result
        )

        self.assertGreaterEqual(
            len(figure.data),
            3,
        )


class LLNTests(
    SimpleTestCase
):

    def test_lln_standard_normal(
        self,
    ):
        result = simulate_lln(
            "standard_normal",
            {},
            max_sample_size=5000,
            paths=5,
            seed=123,
        )

        self.assertEqual(
            result.running_means.shape,
            (
                5,
                5000,
            ),
        )

        self.assertEqual(
            result.theoretical_mean,
            0.0,
        )

        self.assertEqual(
            len(result.final_means),
            5,
        )

    def test_lln_is_reproducible(
        self,
    ):
        first = simulate_lln(
            "poisson",
            {
                "rate": 3,
            },
            max_sample_size=1000,
            paths=3,
            seed=123,
        )

        second = simulate_lln(
            "poisson",
            {
                "rate": 3,
            },
            max_sample_size=1000,
            paths=3,
            seed=123,
        )

        self.assertTrue(
            np.array_equal(
                first.running_means,
                second.running_means,
            )
        )

    def test_lln_rejects_undefined_mean(
        self,
    ):
        with self.assertRaises(
            SamplingInputError
        ):
            simulate_lln(
                "cauchy",
                {
                    "location": 0,
                    "scale": 1,
                },
                max_sample_size=1000,
                paths=3,
                seed=123,
            )

    def test_lln_figure_contains_paths(
        self,
    ):
        result = simulate_lln(
            "exponential",
            {
                "rate": 2,
            },
            max_sample_size=1000,
            paths=4,
            seed=123,
        )

        figure = build_lln_figure(
            result
        )

        self.assertEqual(
            len(figure.data),
            4,
        )