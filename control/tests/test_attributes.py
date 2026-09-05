import math

from django.test import (
    SimpleTestCase,
)

from control.services import (
    AttributeChartInputError,
    calculate_c_chart,
    calculate_np_chart,
    calculate_p_chart,
    calculate_u_chart,
)


class PChartTests(
    SimpleTestCase
):

    def test_variable_sample_sizes(
        self,
    ):
        result = calculate_p_chart(
            [
                100,
                100,
                200,
            ],
            [
                5,
                10,
                20,
            ],
        )

        self.assertAlmostEqual(
            result.centerline,
            0.0875,
        )

        self.assertEqual(
            result.proportions,
            (
                0.05,
                0.10,
                0.10,
            ),
        )

        self.assertAlmostEqual(
            result.upper_control_limits[
                0
            ],
            0.1722699092,
            places=9,
        )

        self.assertAlmostEqual(
            result.lower_control_limits[
                0
            ],
            0.0027300908,
            places=9,
        )

        self.assertAlmostEqual(
            result.upper_control_limits[
                2
            ],
            0.1474413776,
            places=9,
        )

    def test_defectives_cannot_exceed_sample_size(
        self,
    ):
        with self.assertRaises(
            AttributeChartInputError
        ):
            calculate_p_chart(
                [100],
                [101],
            )


class NPChartTests(
    SimpleTestCase
):

    def test_known_limits(
        self,
    ):
        result = calculate_np_chart(
            [
                100,
                100,
                100,
                100,
            ],
            [
                5,
                10,
                15,
                10,
            ],
        )

        self.assertAlmostEqual(
            result.p_bar,
            0.1,
        )

        self.assertAlmostEqual(
            result.centerline,
            10.0,
        )

        self.assertAlmostEqual(
            result.upper_control_limit,
            19.0,
        )

        self.assertAlmostEqual(
            result.lower_control_limit,
            1.0,
        )

    def test_requires_constant_sample_size(
        self,
    ):
        with self.assertRaises(
            AttributeChartInputError
        ):
            calculate_np_chart(
                [
                    100,
                    120,
                ],
                [
                    5,
                    6,
                ],
            )


class CChartTests(
    SimpleTestCase
):

    def test_known_limits(
        self,
    ):
        result = calculate_c_chart(
            [
                2,
                4,
                3,
                5,
                6,
            ]
        )

        self.assertAlmostEqual(
            result.centerline,
            4.0,
        )

        self.assertAlmostEqual(
            result.upper_control_limit,
            10.0,
        )

        self.assertAlmostEqual(
            result.lower_control_limit,
            -2.0,
        )


class UChartTests(
    SimpleTestCase
):

    def test_variable_sample_sizes(
        self,
    ):
        result = calculate_u_chart(
            [
                100,
                200,
                100,
            ],
            [
                4,
                10,
                6,
            ],
        )

        self.assertAlmostEqual(
            result.centerline,
            0.05,
        )

        self.assertEqual(
            result.rates,
            (
                0.04,
                0.05,
                0.06,
            ),
        )

        self.assertAlmostEqual(
            result.upper_control_limits[
                0
            ],
            0.1170820393,
            places=9,
        )

        self.assertAlmostEqual(
            result.upper_control_limits[
                1
            ],
            0.0974341649,
            places=9,
        )

    def test_uses_pooled_average(
        self,
    ):
        result = calculate_u_chart(
            [
                100,
                200,
            ],
            [
                5,
                20,
            ],
        )

        self.assertAlmostEqual(
            result.centerline,
            25 / 300,
        )

        self.assertNotAlmostEqual(
            result.centerline,
            (
                0.05
                + 0.10
            ) / 2,
        )


class AttributeValidationTests(
    SimpleTestCase
):

    def test_negative_counts_are_rejected(
        self,
    ):
        with self.assertRaises(
            AttributeChartInputError
        ):
            calculate_c_chart(
                [
                    2,
                    -1,
                    4,
                ]
            )

    def test_counts_must_be_integer(
        self,
    ):
        with self.assertRaises(
            AttributeChartInputError
        ):
            calculate_c_chart(
                [
                    2,
                    3.5,
                    4,
                ]
            )