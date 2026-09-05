from django.test import SimpleTestCase

from control.services import (
    CUSUMInputError,
    calculate_cusum,
    calculate_vmask_cusum,
)


class VMaskCUSUMTests(SimpleTestCase):

    def test_cumulative_sum_includes_origin(self):
        result = calculate_vmask_cusum(
            [
                [10],
                [10.5],
                [11],
            ],
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        self.assertEqual(
            result.cumulative_sums,
            (
                0.0,
                0.0,
                0.5,
                1.5,
            ),
        )

    def test_lead_distance_is_h_over_k(self):
        result = calculate_vmask_cusum(
            [
                [10],
                [11],
            ],
            target_mean=10,
            reference_value=0.25,
            decision_interval=1.0,
        )

        self.assertAlmostEqual(
            result.lead_distance,
            4.0,
        )

    def test_positive_shift_matches_tabular_cusum(self):
        subgroups = [
            [10],
            [10.5],
            [11],
            [11.5],
        ]

        tabular = calculate_cusum(
            subgroups,
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        vmask = calculate_vmask_cusum(
            subgroups,
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        self.assertEqual(
            vmask.positive_signal_indices,
            tabular.positive_signal_indices,
        )

        self.assertEqual(
            vmask.positive_signal_indices,
            (3, 4),
        )

    def test_negative_shift_matches_tabular_cusum(self):
        subgroups = [
            [10],
            [9.5],
            [9],
            [8.5],
        ]

        tabular = calculate_cusum(
            subgroups,
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        vmask = calculate_vmask_cusum(
            subgroups,
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        self.assertEqual(
            vmask.negative_signal_indices,
            tabular.negative_signal_indices,
        )

        self.assertEqual(
            vmask.negative_signal_indices,
            (3, 4),
        )

    def test_final_mask_geometry(self):
        result = calculate_vmask_cusum(
            [
                [10],
                [10.5],
                [11],
            ],
            target_mean=10,
            reference_value=0.25,
            decision_interval=0.9,
        )

        # Final cumulative point:
        #
        # index = 3
        # S3 = 1.5
        #
        # d = 0.9 / 0.25 = 3.6
        #
        # vertex = (6.6, 1.5)

        self.assertAlmostEqual(
            result.final_vertex_x,
            6.6,
        )

        self.assertAlmostEqual(
            result.final_vertex_y,
            1.5,
        )

        # At the current point x=3 the
        # V-mask is H above and below S3.

        self.assertAlmostEqual(
            result.final_upper_boundary[-1],
            2.4,
        )

        self.assertAlmostEqual(
            result.final_lower_boundary[-1],
            0.6,
        )

    def test_stable_sequence_has_no_signal(self):
        result = calculate_vmask_cusum(
            [
                [10],
                [10.1],
                [9.9],
                [10],
            ],
            target_mean=10,
            reference_value=0.25,
            decision_interval=1.0,
        )

        self.assertEqual(
            result.positive_signal_indices,
            (),
        )

        self.assertEqual(
            result.negative_signal_indices,
            (),
        )

    def test_reference_value_must_be_positive(self):
        with self.assertRaises(
            CUSUMInputError
        ):
            calculate_vmask_cusum(
                [
                    [10],
                    [11],
                ],
                target_mean=10,
                reference_value=0,
                decision_interval=5,
            )

    def test_decision_interval_must_be_positive(self):
        with self.assertRaises(
            CUSUMInputError
        ):
            calculate_vmask_cusum(
                [
                    [10],
                    [11],
                ],
                target_mean=10,
                reference_value=0.5,
                decision_interval=0,
            )