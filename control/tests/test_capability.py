from django.test import SimpleTestCase

from control.services import (
    CapabilityInputError,
    calculate_process_capability,
)


class ProcessCapabilityTests(SimpleTestCase):

    def test_centered_process_capability(self):
        result = calculate_process_capability(
            [
                9.8,
                10.0,
                10.1,
                10.2,
                9.9,
                10.0,
                10.1,
                9.9,
                10.2,
                9.8,
            ],
            lsl=9,
            usl=11,
            within_sigma=0.2,
        )

        self.assertAlmostEqual(
            result.mean,
            10.0,
        )

        self.assertAlmostEqual(
            result.cp,
            1.6666666667,
        )

        self.assertAlmostEqual(
            result.cpl,
            1.6666666667,
        )

        self.assertAlmostEqual(
            result.cpu,
            1.6666666667,
        )

        self.assertAlmostEqual(
            result.cpk,
            1.6666666667,
        )

    def test_performance_uses_overall_sample_sigma(self):
        result = calculate_process_capability(
            [
                9.8,
                10.0,
                10.1,
                10.2,
                9.9,
                10.0,
                10.1,
                9.9,
                10.2,
                9.8,
            ],
            lsl=9,
            usl=11,
            within_sigma=0.2,
        )

        self.assertAlmostEqual(
            result.overall_sigma,
            0.1490711985,
        )

        self.assertAlmostEqual(
            result.pp,
            2.2360679775,
        )

        self.assertAlmostEqual(
            result.ppk,
            2.2360679775,
        )

    def test_off_center_process_reduces_cpk(self):
        result = calculate_process_capability(
            [
                10.2,
                10.4,
                10.6,
            ],
            lsl=9,
            usl=11,
            within_sigma=0.2,
        )

        self.assertAlmostEqual(
            result.cp,
            1.6666666667,
        )

        self.assertAlmostEqual(
            result.cpl,
            2.3333333333,
        )

        self.assertAlmostEqual(
            result.cpu,
            1.0,
        )

        self.assertAlmostEqual(
            result.cpk,
            1.0,
        )

    def test_lower_specification_only(self):
        result = calculate_process_capability(
            [
                9.8,
                10.0,
                10.2,
            ],
            lsl=9,
            within_sigma=0.25,
        )

        self.assertIsNone(
            result.cp
        )

        self.assertAlmostEqual(
            result.cpl,
            1.3333333333,
        )

        self.assertIsNone(
            result.cpu
        )

        self.assertAlmostEqual(
            result.cpk,
            1.3333333333,
        )

    def test_upper_specification_only(self):
        result = calculate_process_capability(
            [
                9.8,
                10.0,
                10.2,
            ],
            usl=11,
            within_sigma=0.25,
        )

        self.assertIsNone(
            result.cp
        )

        self.assertIsNone(
            result.cpl
        )

        self.assertAlmostEqual(
            result.cpu,
            1.3333333333,
        )

        self.assertAlmostEqual(
            result.cpk,
            1.3333333333,
        )

    def test_capability_can_be_omitted(self):
        result = calculate_process_capability(
            [
                9.8,
                10.0,
                10.2,
            ],
            lsl=9,
            usl=11,
        )

        self.assertIsNone(
            result.cp
        )

        self.assertIsNone(
            result.cpk
        )

        self.assertAlmostEqual(
            result.pp,
            1.6666666667,
        )

        self.assertAlmostEqual(
            result.ppk,
            1.6666666667,
        )

    def test_zero_overall_variation_has_no_performance_index(self):
        result = calculate_process_capability(
            [
                10,
                10,
                10,
            ],
            lsl=9,
            usl=11,
            within_sigma=0.2,
        )

        self.assertEqual(
            result.overall_sigma,
            0.0,
        )

        self.assertIsNone(
            result.pp
        )

        self.assertIsNone(
            result.ppk
        )

        self.assertAlmostEqual(
            result.cp,
            1.6666666667,
        )

    def test_at_least_one_specification_is_required(self):
        with self.assertRaises(
            CapabilityInputError
        ):
            calculate_process_capability(
                [
                    9,
                    10,
                    11,
                ]
            )

    def test_lsl_must_be_lower_than_usl(self):
        with self.assertRaises(
            CapabilityInputError
        ):
            calculate_process_capability(
                [
                    9,
                    10,
                    11,
                ],
                lsl=11,
                usl=9,
            )

    def test_within_sigma_must_be_positive(self):
        with self.assertRaises(
            CapabilityInputError
        ):
            calculate_process_capability(
                [
                    9,
                    10,
                    11,
                ],
                lsl=8,
                usl=12,
                within_sigma=0,
            )

    def test_non_finite_observation_is_rejected(self):
        with self.assertRaises(
            CapabilityInputError
        ):
            calculate_process_capability(
                [
                    9,
                    float("nan"),
                    11,
                ],
                lsl=8,
                usl=12,
            )