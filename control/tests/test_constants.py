from django.test import SimpleTestCase

from control.services import (
    INDIVIDUAL_MR_CONSTANTS,
    MEDIAN_R_CONSTANTS,
    XBAR_R_CONSTANTS,
    XBAR_S_CONSTANTS,
)


class ControlChartConstantsTests(
    SimpleTestCase
):

    def test_xbar_r_constants_match_source(
        self,
    ):
        expected = {
            2: {
                "A2": 1.880,
                "d2": 1.128,
                "D3": 0.000,
                "D4": 3.267,
            },
            3: {
                "A2": 1.023,
                "d2": 1.693,
                "D3": 0.000,
                "D4": 2.574,
            },
            4: {
                "A2": 0.729,
                "d2": 2.059,
                "D3": 0.000,
                "D4": 2.282,
            },
            5: {
                "A2": 0.577,
                "d2": 2.326,
                "D3": 0.000,
                "D4": 2.114,
            },
            6: {
                "A2": 0.483,
                "d2": 2.534,
                "D3": 0.000,
                "D4": 2.004,
            },
            7: {
                "A2": 0.419,
                "d2": 2.704,
                "D3": 0.076,
                "D4": 1.924,
            },
            8: {
                "A2": 0.373,
                "d2": 2.847,
                "D3": 0.136,
                "D4": 1.864,
            },
            9: {
                "A2": 0.337,
                "d2": 2.970,
                "D3": 0.184,
                "D4": 1.816,
            },
            10: {
                "A2": 0.308,
                "d2": 3.078,
                "D3": 0.223,
                "D4": 1.777,
            },
            15: {
                "A2": 0.223,
                "d2": 3.472,
                "D3": 0.347,
                "D4": 1.653,
            },
            25: {
                "A2": 0.153,
                "d2": 3.931,
                "D3": 0.459,
                "D4": 1.541,
            },
        }

        self.assertEqual(
            XBAR_R_CONSTANTS,
            expected,
        )

    def test_xbar_s_constants_match_source(
        self,
    ):
        expected = {
            2: {
                "A3": 2.659,
                "c4": 0.7979,
                "B3": 0.000,
                "B4": 3.267,
            },
            3: {
                "A3": 1.954,
                "c4": 0.8862,
                "B3": 0.000,
                "B4": 2.568,
            },
            4: {
                "A3": 1.628,
                "c4": 0.9213,
                "B3": 0.000,
                "B4": 2.266,
            },
            5: {
                "A3": 1.427,
                "c4": 0.9400,
                "B3": 0.000,
                "B4": 2.089,
            },
            6: {
                "A3": 1.287,
                "c4": 0.9515,
                "B3": 0.030,
                "B4": 1.970,
            },
            7: {
                "A3": 1.182,
                "c4": 0.9594,
                "B3": 0.118,
                "B4": 1.882,
            },
            8: {
                "A3": 1.099,
                "c4": 0.9650,
                "B3": 0.185,
                "B4": 1.815,
            },
            9: {
                "A3": 1.032,
                "c4": 0.9693,
                "B3": 0.239,
                "B4": 1.761,
            },
            10: {
                "A3": 0.975,
                "c4": 0.9727,
                "B3": 0.284,
                "B4": 1.716,
            },
            15: {
                "A3": 0.789,
                "c4": 0.9823,
                "B3": 0.428,
                "B4": 1.572,
            },
            25: {
                "A3": 0.606,
                "c4": 0.9896,
                "B3": 0.565,
                "B4": 1.435,
            },
        }

        self.assertEqual(
            XBAR_S_CONSTANTS,
            expected,
        )

    def test_median_r_constants_match_source(
        self,
    ):
        expected_a2_tilde = {
            2: 1.880,
            3: 1.187,
            4: 0.796,
            5: 0.691,
            6: 0.548,
            7: 0.508,
            8: 0.433,
            9: 0.412,
            10: 0.362,
        }

        expected_d4 = {
            2: 3.267,
            3: 2.574,
            4: 2.282,
            5: 2.114,
            6: 2.004,
            7: 1.924,
            8: 1.864,
            9: 1.816,
            10: 1.777,
        }

        for subgroup_size in (
            expected_a2_tilde
        ):
            with self.subTest(
                subgroup_size=(
                    subgroup_size
                )
            ):
                constants = (
                    MEDIAN_R_CONSTANTS[
                        subgroup_size
                    ]
                )

                self.assertEqual(
                    constants[
                        "A2_tilde"
                    ],
                    expected_a2_tilde[
                        subgroup_size
                    ],
                )

                self.assertEqual(
                    constants["D4"],
                    expected_d4[
                        subgroup_size
                    ],
                )

    def test_individual_mr_constants_match_source(
        self,
    ):
        expected_e2 = {
            2: 2.660,
            3: 1.772,
            4: 1.457,
            5: 1.290,
            6: 1.184,
            7: 1.109,
            8: 1.054,
            9: 1.010,
            10: 0.975,
        }

        for range_length, value in (
            expected_e2.items()
        ):
            with self.subTest(
                range_length=(
                    range_length
                )
            ):
                self.assertEqual(
                    INDIVIDUAL_MR_CONSTANTS[
                        range_length
                    ]["E2"],
                    value,
                )

    def test_xbar_r_regression_values_for_n8_n9(
        self,
    ):
        self.assertEqual(
            XBAR_R_CONSTANTS[8]["D4"],
            1.864,
        )

        self.assertEqual(
            XBAR_R_CONSTANTS[9]["D4"],
            1.816,
        )

    def test_median_regression_value_for_n10(
        self,
    ):
        self.assertEqual(
            MEDIAN_R_CONSTANTS[
                10
            ]["A2_tilde"],
            0.362,
        )