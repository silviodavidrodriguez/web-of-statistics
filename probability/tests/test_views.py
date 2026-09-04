from django.test import SimpleTestCase
from django.urls import reverse


class ProbabilityViewTests(
    SimpleTestCase
):

    def test_probability_page_loads(
        self,
    ):
        response = self.client.get(
            reverse("probability")
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "Probability Distributions &amp; Simulation",
        )

        self.assertContains(
            response,
            "Distribution Explorer",
        )

        self.assertContains(
            response,
            "Simulation",
        )

        self.assertContains(
            response,
            "Sampling",
        )

    def test_legacy_table_interface_is_removed(
        self,
    ):
        response = self.client.get(
            reverse("probability")
        )

        self.assertNotContains(
            response,
            "Standard Normal Table",
        )

        self.assertNotContains(
            response,
            "Check the box to enter a probability",
        )

        self.assertNotContains(
            response,
            "Check the box to use two tails",
        )

    def test_standard_normal_calculation(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=functions"
            ),
            {
                "distribution":
                    "standard_normal",
                "operation":
                    "left",
                "input_x":
                    "1.96",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "P(Z ≤ 1.96) = 0.975002",
        )

        self.assertContains(
            response,
            "probability-calculation-chart",
        )

    def test_binomial_calculation(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=functions"
            ),
            {
                "distribution":
                    "binomial",
                "operation":
                    "greater_equal",
                "param_n":
                    "10",
                "param_p":
                    "0.5",
                "input_x":
                    "5",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "P(X ≥ 5) = 0.623047",
        )

    def test_parameter_error_is_preserved(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=functions"
            ),
            {
                "distribution":
                    "normal",
                "operation":
                    "left",
                "param_mean":
                    "0",
                "param_sd":
                    "-1",
                "input_x":
                    "1",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            (
                "Standard deviation must be "
                "greater than 0."
            ),
        )

    def test_explorer_tab_loads(
        self,
    ):
        response = self.client.get(
            (
                reverse("probability")
                + "?tab=explorer"
            )
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "Distribution Explorer",
        )

        self.assertContains(
            response,
            "Theoretical properties",
        )

        self.assertContains(
            response,
            "probability-explorer-chart",
        )

    def test_explorer_normal_properties(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_distribution":
                    "normal",
                "explorer_view":
                    "pdf",
                "explorer_param_mean":
                    "100",
                "explorer_param_sd":
                    "15",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "Normal",
        )

        self.assertContains(
            response,
            "Standard deviation",
        )

        self.assertContains(
            response,
            "100",
        )


    def test_explorer_cauchy_shows_undefined_moments(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_distribution":
                    "cauchy",
                "explorer_view":
                    "pdf",
                "explorer_param_location":
                    "0",
                "explorer_param_scale":
                    "1",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "Undefined",
        )


    def test_explorer_parameter_error(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_distribution":
                    "normal",
                "explorer_view":
                    "pdf",
                "explorer_param_mean":
                    "0",
                "explorer_param_sd":
                    "-1",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            (
                "Standard deviation must be "
                "greater than 0."
            ),
        )

    def test_explorer_comparison_loads(
        self,
    ):
        response = self.client.get(
            (
                reverse("probability")
                + "?tab=explorer"
                + "&mode=compare"
            )
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "Compare distributions",
        )

        self.assertContains(
            response,
            (
                "probability-explorer-"
                "comparison-chart"
            ),
        )


    def test_explorer_student_t_comparison(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_mode":
                    "compare",

                "comparison_category":
                    "continuous",

                "comparison_view":
                    "pdf",

                "comparison_count":
                    "4",

                "compare_0_distribution":
                    "student_t",
                "compare_0_label":
                    "t2",
                "compare_0_param_df":
                    "2",

                "compare_1_distribution":
                    "student_t",
                "compare_1_label":
                    "t5",
                "compare_1_param_df":
                    "5",

                "compare_2_distribution":
                    "student_t",
                "compare_2_label":
                    "t30",
                "compare_2_param_df":
                    "30",

                "compare_3_distribution":
                    "standard_normal",
                "compare_3_label":
                    "Normal",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "t2",
        )

        self.assertContains(
            response,
            "t5",
        )

        self.assertContains(
            response,
            "t30",
        )

        self.assertContains(
            response,
            (
                "probability-explorer-"
                "comparison-chart"
            ),
        )


    def test_explorer_discrete_comparison(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_mode":
                    "compare",

                "comparison_category":
                    "discrete",

                "comparison_view":
                    "pmf",

                "comparison_count":
                    "2",

                "compare_0_distribution":
                    "binomial",
                "compare_0_label":
                    "Binomial",
                "compare_0_param_n":
                    "10",
                "compare_0_param_p":
                    "0.5",

                "compare_1_distribution":
                    "poisson",
                "compare_1_label":
                    "Poisson",
                "compare_1_param_rate":
                    "5",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            "Binomial",
        )

        self.assertContains(
            response,
            "Poisson",
        )

        self.assertContains(
            response,
            (
                "probability-explorer-"
                "comparison-chart"
            ),
        )


    def test_explorer_mixed_comparison_rejected(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_mode":
                    "compare",

                "comparison_category":
                    "continuous",

                "comparison_view":
                    "pdf",

                "comparison_count":
                    "2",

                "compare_0_distribution":
                    "standard_normal",
                "compare_0_label":
                    "Normal",

                "compare_1_distribution":
                    "binomial",
                "compare_1_label":
                    "Binomial",
                "compare_1_param_n":
                    "10",
                "compare_1_param_p":
                    "0.5",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            (
                "Binomial is not a "
                "continuous distribution"
            ),
        )

    def test_explorer_comparison_parameter_error(
        self,
    ):
        response = self.client.post(
            (
                reverse("probability")
                + "?tab=explorer"
            ),
            {
                "explorer_mode":
                    "compare",

                "comparison_category":
                    "continuous",

                "comparison_view":
                    "pdf",

                "comparison_count":
                    "2",

                "compare_0_distribution":
                    "normal",
                "compare_0_label":
                    "Invalid Normal",
                "compare_0_param_mean":
                    "0",
                "compare_0_param_sd":
                    "-1",

                "compare_1_distribution":
                    "standard_normal",
                "compare_1_label":
                    "Standard Normal",
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertContains(
            response,
            (
                "Standard deviation must be "
                "greater than 0."
            ),
        )