from django.test import SimpleTestCase, override_settings
from django.urls import reverse


@override_settings(SECURE_SSL_REDIRECT=False)
class ControlViewTests(SimpleTestCase):
    def test_control_page_loads(self):
        response = self.client.get(reverse("control"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Statistical Process Control")
        self.assertContains(response, "X̄-R Chart")

    def test_xbar_r_post_renders_plotly_results(self):
        response = self.client.post(
            reverse("control") + "?tab=variables&tool=xbar_r",
            {
                "action": "calculate",
                "data": (
                    "10\t11\t9\t10\t10\n"
                    "12\t11\t10\t11\t11\n"
                    "9\t10\t10\t9\t10\n"
                    "11\t12\t11\t10\t11"
                ),
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Estimated σ")
        self.assertContains(response, "X̄ Chart")
        self.assertContains(response, "plotly-graph-div")

    def test_attribute_p_chart_accepts_variable_sample_sizes(self):
        response = self.client.post(
            reverse("control") + "?tab=attributes&tool=p_chart",
            {
                "action": "calculate",
                "data": "100\t5\n100\t10\n200\t20",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Total defectives")
        self.assertContains(response, "p Chart")

    def test_vmask_cusum_renders(self):
        response = self.client.post(
            reverse("control") + "?tab=advanced&tool=cusum",
            {
                "action": "calculate",
                "data": "10\n10.5\n11\n11.5",
                "cusum_method": "vmask",
                "target_mean": "10",
                "reference_value": "0.25",
                "decision_interval": "0.9",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Lead distance d")
        self.assertContains(response, "CUSUM V-mask")

    def test_ewma_renders_dynamic_limits(self):
        response = self.client.post(
            reverse("control") + "?tab=advanced&tool=ewma",
            {
                "action": "calculate",
                "data": "9\t11\n11\t13\n13\t15",
                "target_mean": "12",
                "lambda_value": "0.2",
                "process_sigma": "1.41421356237",
                "control_limit_width": "3",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Process σ")
        self.assertContains(response, "EWMA Chart")

    def test_precontrol_reports_qualified_setup(self):
        response = self.client.post(
            reverse("control") + "?tab=precontrol&tool=precontrol",
            {
                "action": "calculate",
                "data": "10\n10.2\n9.8\n10.1\n10",
                "nominal_value": "10",
                "tolerance_value": "2",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Qualified")
        self.assertContains(response, "Five consecutive observations")

    def test_capability_calculates_cp_cpk(self):
        response = self.client.post(
            reverse("control") + "?tab=capability&tool=capability",
            {
                "action": "calculate",
                "data": "9.8\n10.0\n10.1\n10.2\n9.9\n10.0\n10.1\n9.9\n10.2\n9.8",
                "lsl": "9",
                "usl": "11",
                "sigma_method": "provided",
                "within_sigma": "0.2",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Cpk")
        self.assertContains(response, "1.666667")
        self.assertContains(response, "Capability summary")

    def test_invalid_dataset_returns_clear_error(self):
        response = self.client.post(
            reverse("control") + "?tab=variables&tool=xbar_r",
            {
                "action": "calculate",
                "data": "1\t2\t3\n4\tbad\t6",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Non-numeric value detected")
