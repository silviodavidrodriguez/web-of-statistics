import math

from django.test import (
    SimpleTestCase,
)

from control.services import (
    VariableChartInputError,
    calculate_individuals_mr,
    calculate_median_r,
    calculate_xbar_r,
    calculate_xbar_s,
)


SUBGROUPS_N5 = [
    [10, 11, 9, 10, 10],
    [12, 11, 10, 11, 11],
    [9, 10, 10, 9, 10],
    [11, 12, 11, 10, 11],
]


class XBarRTests(
    SimpleTestCase
):

    def test_known_xbar_r_values(
        self,
    ):
        result = calculate_xbar_r(
            SUBGROUPS_N5
        )

        self.assertEqual(
            result.subgroup_size,
            5,
        )

        self.assertEqual(
            result.subgroup_means,
            (
                10.0,
                11.0,
                9.6,
                11.0,
            ),
        )

        self.assertEqual(
            result.subgroup_ranges,
            (
                2.0,
                2.0,
                1.0,
                2.0,
            ),
        )

        self.assertAlmostEqual(
            result.x_centerline,
            10.4,
        )

        self.assertAlmostEqual(
            result.range_centerline,
            1.75,
        )

        self.assertAlmostEqual(
            result.x_upper_control_limit,
            11.40975,
        )

        self.assertAlmostEqual(
            result.x_lower_control_limit,
            9.39025,
        )

        self.assertAlmostEqual(
            result.range_upper_control_limit,
            3.6995,
        )

        self.assertAlmostEqual(
            result.range_lower_control_limit,
            0.0,
        )

        self.assertAlmostEqual(
            result.estimated_sigma,
            1.75 / 2.326,
        )

    def test_n8_uses_corrected_d4(
        self,
    ):
        subgroups = [
            [
                1, 2, 3, 4,
                5, 6, 7, 8,
            ],
            [
                2, 3, 4, 5,
                6, 7, 8, 9,
            ],
        ]

        result = calculate_xbar_r(
            subgroups
        )

        self.assertEqual(
            result.D4,
            1.864,
        )


class XBarSTests(
    SimpleTestCase
):

    def test_known_xbar_s_values(
        self,
    ):
        result = calculate_xbar_s(
            SUBGROUPS_N5
        )

        expected_standard_deviations = tuple(
            [
                math.sqrt(0.5),
                math.sqrt(0.5),
                math.sqrt(0.3),
                math.sqrt(0.5),
            ]
        )

        for actual, expected in zip(
            result
            .subgroup_standard_deviations,
            expected_standard_deviations,
        ):
            self.assertAlmostEqual(
                actual,
                expected,
            )

        expected_s_bar = (
            sum(
                expected_standard_deviations
            )
            / 4
        )

        self.assertAlmostEqual(
            result.x_centerline,
            10.4,
        )

        self.assertAlmostEqual(
            result.s_centerline,
            expected_s_bar,
        )

        self.assertAlmostEqual(
            result.x_upper_control_limit,
            (
                10.4
                + 1.427
                * expected_s_bar
            ),
        )

        self.assertAlmostEqual(
            result.x_lower_control_limit,
            (
                10.4
                - 1.427
                * expected_s_bar
            ),
        )

        self.assertAlmostEqual(
            result.s_upper_control_limit,
            (
                2.089
                * expected_s_bar
            ),
        )

        self.assertAlmostEqual(
            result.s_lower_control_limit,
            0.0,
        )

        self.assertAlmostEqual(
            result.estimated_sigma,
            expected_s_bar / 0.9400,
        )


class MedianRTests(
    SimpleTestCase
):

    def test_known_median_r_values(
        self,
    ):
        result = calculate_median_r(
            SUBGROUPS_N5
        )

        self.assertEqual(
            result.subgroup_medians,
            (
                10.0,
                11.0,
                10.0,
                11.0,
            ),
        )

        self.assertAlmostEqual(
            result.median_centerline,
            10.5,
        )

        self.assertAlmostEqual(
            result.range_centerline,
            1.75,
        )

        self.assertAlmostEqual(
            result
            .median_upper_control_limit,
            (
                10.5
                + 0.691 * 1.75
            ),
        )

        self.assertAlmostEqual(
            result
            .median_lower_control_limit,
            (
                10.5
                - 0.691 * 1.75
            ),
        )

    def test_n10_uses_corrected_a2_tilde(
        self,
    ):
        subgroups = [
            list(
                range(
                    1,
                    11,
                )
            ),
            list(
                range(
                    2,
                    12,
                )
            ),
        ]

        result = calculate_median_r(
            subgroups
        )

        self.assertEqual(
            result.A2_tilde,
            0.362,
        )


class IndividualsMRTests(
    SimpleTestCase
):

    def test_known_individuals_mr_values(
        self,
    ):
        result = (
            calculate_individuals_mr(
                [
                    10,
                    11,
                    9,
                    10,
                    12,
                ],
                moving_range_length=2,
            )
        )

        self.assertEqual(
            result.moving_ranges,
            (
                1.0,
                2.0,
                1.0,
                2.0,
            ),
        )

        self.assertAlmostEqual(
            result.individuals_centerline,
            10.4,
        )

        self.assertAlmostEqual(
            result.moving_range_centerline,
            1.5,
        )

        self.assertAlmostEqual(
            result
            .individuals_upper_control_limit,
            10.4 + 2.660 * 1.5,
        )

        self.assertAlmostEqual(
            result
            .individuals_lower_control_limit,
            10.4 - 2.660 * 1.5,
        )

        self.assertAlmostEqual(
            result
            .moving_range_upper_control_limit,
            3.267 * 1.5,
        )

        self.assertAlmostEqual(
            result
            .moving_range_lower_control_limit,
            0.0,
        )

        self.assertAlmostEqual(
            result.estimated_sigma,
            1.5 / 1.128,
        )

    def test_moving_range_length_three(
        self,
    ):
        result = (
            calculate_individuals_mr(
                [
                    10,
                    12,
                    11,
                    15,
                    14,
                ],
                moving_range_length=3,
            )
        )

        self.assertEqual(
            result.moving_ranges,
            (
                2.0,
                4.0,
                4.0,
            ),
        )

        self.assertEqual(
            result.E2,
            1.772,
        )


class VariableChartValidationTests(
    SimpleTestCase
):

    def test_subgroups_must_have_equal_size(
        self,
    ):
        with self.assertRaises(
            VariableChartInputError
        ):
            calculate_xbar_r(
                [
                    [1, 2, 3],
                    [1, 2],
                ]
            )

    def test_unsupported_subgroup_size_is_rejected(
        self,
    ):
        with self.assertRaises(
            VariableChartInputError
        ):
            calculate_xbar_r(
                [
                    [
                        1, 2, 3,
                        4, 5, 6,
                        7, 8, 9,
                        10, 11,
                    ],
                    [
                        2, 3, 4,
                        5, 6, 7,
                        8, 9, 10,
                        11, 12,
                    ],
                ]
            )

    def test_non_finite_value_is_rejected(
        self,
    ):
        with self.assertRaises(
            VariableChartInputError
        ):
            calculate_xbar_r(
                [
                    [
                        1,
                        2,
                        3,
                        4,
                        float("nan"),
                    ],
                    [
                        1,
                        2,
                        3,
                        4,
                        5,
                    ],
                ]
            )