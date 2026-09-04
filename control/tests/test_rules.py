from django.test import SimpleTestCase

from control.services import (
    ControlRuleInputError,
    detect_nelson_rules,
    detect_nelson_rules_for_values,
)


def rules_found(signals):
    return {
        signal.rule
        for signal in signals
    }


class NelsonRuleTests(SimpleTestCase):

    def test_rule_1_point_beyond_three_sigma(self):
        signals = detect_nelson_rules(
            [0.1, -0.2, 3.2]
        )

        self.assertIn(
            1,
            rules_found(signals),
        )

        signal = next(
            item
            for item in signals
            if item.rule == 1
        )

        self.assertEqual(
            signal.point_indices,
            (3,),
        )

    def test_rule_2_nine_same_side(self):
        signals = detect_nelson_rules(
            [0.5] * 9
        )

        self.assertIn(
            2,
            rules_found(signals),
        )

    def test_rule_3_six_point_trend(self):
        signals = detect_nelson_rules(
            [-2, -1, 0, 1, 2, 2.5]
        )

        self.assertIn(
            3,
            rules_found(signals),
        )

    def test_rule_4_fourteen_alternating(self):
        values = [
            1 if index % 2 == 0 else -1
            for index in range(14)
        ]

        signals = detect_nelson_rules(
            values
        )

        self.assertIn(
            4,
            rules_found(signals),
        )

    def test_rule_5_two_of_three_beyond_two_sigma(self):
        signals = detect_nelson_rules(
            [2.2, 2.4, 0.1]
        )

        self.assertIn(
            5,
            rules_found(signals),
        )

    def test_rule_6_four_of_five_beyond_one_sigma(self):
        signals = detect_nelson_rules(
            [1.2, 1.3, 1.4, 0.2, 1.1]
        )

        self.assertIn(
            6,
            rules_found(signals),
        )

    def test_rule_7_fifteen_within_one_sigma(self):
        signals = detect_nelson_rules(
            [0.5] * 15
        )

        self.assertIn(
            7,
            rules_found(signals),
        )

    def test_rule_8_eight_outside_one_sigma(self):
        signals = detect_nelson_rules(
            [
                1.2,
                -1.3,
                1.4,
                -1.5,
                1.6,
                -1.7,
                1.8,
                -1.9,
            ]
        )

        self.assertIn(
            8,
            rules_found(signals),
        )

    def test_short_stable_sequence_has_no_signal(self):
        signals = detect_nelson_rules(
            [0.2, -0.3, 0.5, -0.2, 0.1]
        )

        self.assertEqual(
            signals,
            (),
        )

    def test_value_wrapper_standardizes_values(self):
        signals = detect_nelson_rules_for_values(
            [100, 101, 104],
            centerline=100,
            sigma=1,
        )

        self.assertIn(
            1,
            rules_found(signals),
        )

    def test_non_finite_z_score_is_rejected(self):
        with self.assertRaises(
            ControlRuleInputError
        ):
            detect_nelson_rules(
                [0, float("nan")]
            )

    def test_zero_sigma_is_rejected(self):
        with self.assertRaises(
            ControlRuleInputError
        ):
            detect_nelson_rules_for_values(
                [1, 1],
                centerline=1,
                sigma=0,
            )