from django.test import SimpleTestCase

from control.services import (
    CUSUMInputError,
    calculate_cusum,
)


class CUSUMTests(SimpleTestCase):

    def test_positive_shift_accumulates(self):
        result = calculate_cusum(
            [
                [10],
                [10.5],
                [11],
                [11.5],
            ],
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        self.assertEqual(
            result.positive_cusum,
            (
                0.0,
                0.25,
                1.0,
                2.25,
            ),
        )

        self.assertEqual(
            result.negative_cusum,
            (
                -0.0,
                -0.0,
                -0.0,
                -0.0,
            ),
        )

        self.assertEqual(
            result.positive_signal_indices,
            (3, 4),
        )

        self.assertEqual(
            result.negative_signal_indices,
            (),
        )

    def test_negative_shift_accumulates(self):
        result = calculate_cusum(
            [
                [10],
                [9.5],
                [9],
                [8.5],
            ],
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        self.assertEqual(
            result.negative_cusum,
            (
                -0.0,
                -0.25,
                -1.0,
                -2.25,
            ),
        )

        self.assertEqual(
            result.positive_signal_indices,
            (),
        )

        self.assertEqual(
            result.negative_signal_indices,
            (3, 4),
        )

    def test_first_subgroup_is_included(self):
        result = calculate_cusum(
            [
                [12],
                [10],
            ],
            target_mean=10,
            reference_value=0.5,
            decision_interval=10,
        )

        self.assertAlmostEqual(
            result.positive_cusum[0],
            1.5,
        )

    def test_subgroup_means_are_used(self):
        result = calculate_cusum(
            [
                [9, 11],
                [11, 13],
            ],
            target_mean=10,
            reference_value=0.5,
            decision_interval=10,
        )

        self.assertEqual(
            result.subgroup_means,
            (
                10.0,
                12.0,
            ),
        )

        self.assertEqual(
            result.positive_cusum,
            (
                0.0,
                1.5,
            ),
        )

    def test_target_defaults_to_global_mean(self):
        result = calculate_cusum(
            [
                [9, 11],
                [11, 13],
            ],
            reference_value=0.5,
            decision_interval=5,
        )

        self.assertAlmostEqual(
            result.target_mean,
            11.0,
        )

    def test_reference_value_cannot_be_negative(self):
        with self.assertRaises(
            CUSUMInputError
        ):
            calculate_cusum(
                [[1], [2]],
                target_mean=1,
                reference_value=-0.5,
                decision_interval=5,
            )

    def test_decision_interval_must_be_positive(self):
        with self.assertRaises(
            CUSUMInputError
        ):
            calculate_cusum(
                [[1], [2]],
                target_mean=1,
                reference_value=0.5,
                decision_interval=0,
            )

    def test_subgroups_must_have_equal_size(self):
        with self.assertRaises(
            CUSUMInputError
        ):
            calculate_cusum(
                [
                    [1, 2],
                    [3],
                ],
                target_mean=2,
                reference_value=0.5,
                decision_interval=5,
            )