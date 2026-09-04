from django.test import SimpleTestCase

from control.services import (
    GREEN,
    RED_LOWER,
    RED_UPPER,
    YELLOW_LOWER,
    YELLOW_UPPER,
    PrecontrolInputError,
    calculate_precontrol,
)


class PrecontrolTests(SimpleTestCase):

    def test_zone_boundaries(self):
        result = calculate_precontrol(
            [
                7.9,
                8.0,
                8.5,
                9.0,
                10.0,
                11.0,
                11.5,
                12.0,
                12.1,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        zones = tuple(
            point.zone
            for point in result.points
        )

        self.assertEqual(
            zones,
            (
                RED_LOWER,
                YELLOW_LOWER,
                YELLOW_LOWER,
                GREEN,
                GREEN,
                GREEN,
                YELLOW_UPPER,
                YELLOW_UPPER,
                RED_UPPER,
            ),
        )

    def test_limits_are_calculated_correctly(self):
        result = calculate_precontrol(
            [10],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.lower_spec_limit,
            8.0,
        )

        self.assertEqual(
            result.green_lower_limit,
            9.0,
        )

        self.assertEqual(
            result.green_upper_limit,
            11.0,
        )

        self.assertEqual(
            result.upper_spec_limit,
            12.0,
        )

    def test_five_consecutive_green_qualifies_setup(self):
        result = calculate_precontrol(
            [
                10.0,
                10.2,
                9.8,
                10.1,
                10.0,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.decision.status,
            "qualified",
        )

        self.assertEqual(
            result.decision.decision_index,
            5,
        )

        self.assertEqual(
            result.decision.reason,
            "five_consecutive_green",
        )

    def test_yellow_resets_green_count(self):
        result = calculate_precontrol(
            [
                10.0,
                10.0,
                11.5,
                10.0,
                10.0,
                10.0,
                10.0,
                10.0,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.decision.status,
            "qualified",
        )

        self.assertEqual(
            result.decision.decision_index,
            8,
        )

    def test_two_yellows_same_side_reject_setup(self):
        result = calculate_precontrol(
            [
                11.4,
                11.6,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.decision.status,
            "rejected",
        )

        self.assertEqual(
            result.decision.reason,
            "two_yellow_same_side",
        )

        self.assertEqual(
            result.decision.decision_index,
            2,
        )

    def test_two_yellows_opposite_sides_detect_variation(self):
        result = calculate_precontrol(
            [
                11.5,
                8.5,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.decision.status,
            "rejected",
        )

        self.assertEqual(
            result.decision.reason,
            "two_yellow_opposite_sides",
        )

        self.assertEqual(
            result.decision.decision_index,
            2,
        )

    def test_red_observation_rejects_setup(self):
        result = calculate_precontrol(
            [
                10,
                12.1,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.decision.status,
            "rejected",
        )

        self.assertEqual(
            result.decision.reason,
            "red_observation",
        )

        self.assertEqual(
            result.decision.decision_index,
            2,
        )

    def test_incomplete_sequence_is_pending(self):
        result = calculate_precontrol(
            [
                10,
                10.1,
                9.9,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.decision.status,
            "pending",
        )

        self.assertIsNone(
            result.decision.decision_index
        )

    def test_zone_counts_are_reported(self):
        result = calculate_precontrol(
            [
                10,
                8.5,
                11.5,
                7.5,
                12.5,
            ],
            nominal_value=10,
            tolerance_value=2,
        )

        self.assertEqual(
            result.green_count,
            1,
        )

        self.assertEqual(
            result.yellow_lower_count,
            1,
        )

        self.assertEqual(
            result.yellow_upper_count,
            1,
        )

        self.assertEqual(
            result.red_lower_count,
            1,
        )

        self.assertEqual(
            result.red_upper_count,
            1,
        )

    def test_tolerance_must_be_positive(self):
        with self.assertRaises(
            PrecontrolInputError
        ):
            calculate_precontrol(
                [10, 11],
                nominal_value=10,
                tolerance_value=0,
            )

    def test_non_finite_observation_is_rejected(self):
        with self.assertRaises(
            PrecontrolInputError
        ):
            calculate_precontrol(
                [
                    10,
                    float("nan"),
                ],
                nominal_value=10,
                tolerance_value=2,
            )