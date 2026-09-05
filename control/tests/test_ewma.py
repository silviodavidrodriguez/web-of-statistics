import math

from django.test import SimpleTestCase

from control.services import (
    EWMAInputError,
    calculate_ewma,
)


class EWMATests(SimpleTestCase):

    def test_first_subgroup_is_used(self):
        result = calculate_ewma(
            [
                [12, 12],
                [10, 10],
            ],
            target_mean=10,
            lambda_value=0.5,
            process_sigma=1,
        )

        self.assertAlmostEqual(
            result.ewma_values[0],
            11.0,
        )

    def test_known_ewma_recurrence(self):
        result = calculate_ewma(
            [
                [10, 10],
                [12, 12],
                [14, 14],
            ],
            target_mean=10,
            lambda_value=0.5,
            process_sigma=1,
        )

        self.assertEqual(
            result.subgroup_means,
            (
                10.0,
                12.0,
                14.0,
            ),
        )

        self.assertEqual(
            result.ewma_values,
            (
                10.0,
                11.0,
                12.5,
            ),
        )

    def test_dynamic_control_limits(self):
        result = calculate_ewma(
            [
                [10, 10],
                [10, 10],
                [10, 10],
            ],
            target_mean=10,
            lambda_value=0.2,
            process_sigma=2,
            control_limit_width=3,
        )

        sigma_xbar = (
            2
            / math.sqrt(2)
        )

        expected_se_1 = (
            sigma_xbar
            * math.sqrt(
                (
                    0.2
                    / 1.8
                )
                * (
                    1
                    - 0.8 ** 2
                )
            )
        )

        self.assertAlmostEqual(
            result
            .ewma_standard_errors[0],
            expected_se_1,
        )

        self.assertAlmostEqual(
            result
            .upper_control_limits[0],
            (
                10
                + 3
                * expected_se_1
            ),
        )

        self.assertAlmostEqual(
            result
            .lower_control_limits[0],
            (
                10
                - 3
                * expected_se_1
            ),
        )

    def test_limits_approach_steady_state(self):
        result = calculate_ewma(
            [[10, 10]] * 100,
            target_mean=10,
            lambda_value=0.2,
            process_sigma=2,
        )

        expected_se = (
            (
                2
                / math.sqrt(2)
            )
            * math.sqrt(
                0.2
                / 1.8
            )
        )

        self.assertAlmostEqual(
            result
            .ewma_standard_errors[-1],
            expected_se,
            places=8,
        )

    def test_signal_is_detected(self):
        result = calculate_ewma(
            [
                [10, 10],
                [10, 10],
                [15, 15],
                [15, 15],
            ],
            target_mean=10,
            lambda_value=0.5,
            process_sigma=1,
        )

        self.assertTrue(
            result.signal_indices
        )

        self.assertIn(
            3,
            result.signal_indices,
        )

    def test_target_defaults_to_global_mean(self):
        result = calculate_ewma(
            [
                [9, 11],
                [11, 13],
            ],
            lambda_value=0.2,
            process_sigma=1,
        )

        self.assertAlmostEqual(
            result.target_mean,
            11.0,
        )

    def test_pooled_within_sigma_is_used(self):
        result = calculate_ewma(
            [
                [9, 11],
                [11, 13],
                [13, 15],
            ],
            target_mean=12,
            lambda_value=0.2,
        )

        # Each subgroup has sample variance 2,
        # therefore pooled within sigma = sqrt(2).

        self.assertAlmostEqual(
            result.process_sigma,
            math.sqrt(2),
        )

        self.assertEqual(
            result.sigma_source,
            "pooled_within_subgroup",
        )

    def test_provided_sigma_is_preserved(self):
        result = calculate_ewma(
            [
                [9, 11],
                [11, 13],
            ],
            target_mean=11,
            lambda_value=0.2,
            process_sigma=2.5,
        )

        self.assertEqual(
            result.process_sigma,
            2.5,
        )

        self.assertEqual(
            result.sigma_source,
            "provided",
        )

    def test_lambda_one_reduces_to_subgroup_means(self):
        result = calculate_ewma(
            [
                [10, 10],
                [12, 12],
                [9, 9],
            ],
            target_mean=10,
            lambda_value=1,
            process_sigma=1,
        )

        self.assertEqual(
            result.ewma_values,
            result.subgroup_means,
        )

    def test_lambda_must_be_positive(self):
        with self.assertRaises(
            EWMAInputError
        ):
            calculate_ewma(
                [
                    [1, 2],
                    [2, 3],
                ],
                lambda_value=0,
                process_sigma=1,
            )

    def test_lambda_cannot_exceed_one(self):
        with self.assertRaises(
            EWMAInputError
        ):
            calculate_ewma(
                [
                    [1, 2],
                    [2, 3],
                ],
                lambda_value=1.1,
                process_sigma=1,
            )

    def test_sigma_must_be_positive(self):
        with self.assertRaises(
            EWMAInputError
        ):
            calculate_ewma(
                [
                    [1, 2],
                    [2, 3],
                ],
                lambda_value=0.2,
                process_sigma=0,
            )

    def test_unequal_subgroup_sizes_are_rejected(self):
        with self.assertRaises(
            EWMAInputError
        ):
            calculate_ewma(
                [
                    [1, 2],
                    [2, 3, 4],
                ],
                lambda_value=0.2,
                process_sigma=1,
            )