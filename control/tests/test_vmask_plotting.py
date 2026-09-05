from types import SimpleNamespace

from django.test import SimpleTestCase

from control.services.plotting import _vmask_state, vmask_cusum_html


class VMaskPlottingTests(SimpleTestCase):
    def setUp(self):
        self.result = SimpleNamespace(
            cumulative_sums=(0.0, 0.0, 0.5, 1.5, 3.0),
            reference_value=0.25,
            lead_distance=3.6,
        )

    def test_historical_mask_identifies_violation_at_observation_three(self):
        state = _vmask_state(self.result, 3)

        self.assertEqual(state["current_index"], 3)
        self.assertEqual(state["violating_indices"], [1])
        self.assertEqual(state["direction"], "upward")
        self.assertEqual(state["detection_x"], [3])
        self.assertAlmostEqual(state["vertex_x"], 6.6)
        self.assertAlmostEqual(state["vertex_y"], 1.5)

    def test_final_mask_distinguishes_violations_from_detection_point(self):
        state = _vmask_state(self.result, 4)

        self.assertEqual(state["violating_indices"], [0, 1, 2, 3])
        self.assertEqual(state["direction"], "upward")
        self.assertEqual(state["detection_x"], [4])
        self.assertEqual(state["detection_y"], [3.0])

    def test_chart_contains_interactive_position_selector(self):
        html = vmask_cusum_html(self.result)

        self.assertIn("Observation 1", html)
        self.assertIn("Observation 3", html)
        self.assertIn("Observation 4", html)
        self.assertIn("V-mask violation", html)
        self.assertIn("Detection point", html)
        self.assertIn("Upward shift detected", html)
