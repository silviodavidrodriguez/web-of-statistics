from django import forms
from django.utils.translation import gettext_lazy as _

class StudySetupForm(forms.Form):
    evaluations_data = forms.CharField(
        label=_("Sensory evaluations"),
        required=True,
        widget=forms.Textarea(
            attrs={
                "class": "form-control dataset-textarea",
                "rows": 10,
                "placeholder": (
                    "Consumer\tSample\tLiking\tColor_JAR\tCATA\n"
                    "1\t120\t7\t3\tCreamy;Smooth"
                ),
                "spellcheck": "false",
            }
        ),
    )

    consumers_data = forms.CharField(
        label=_("Consumer information"),
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "form-control dataset-textarea",
                "rows": 7,
                "placeholder": (
                    "Consumer\tAge\tGender\tEducation\n"
                    "1\t32\tFemale\tUniversity"
                ),
                "spellcheck": "false",
            }
        ),
    )

    general_data = forms.CharField(
        label=_("General study questions"),
        required=False,
        widget=forms.Textarea(
            attrs={
                "class": "form-control dataset-textarea",
                "rows": 7,
                "placeholder": (
                    "Consumer\tFunctional_Information\tPurchase_With_Information\n"
                    "1\tVery influential\t5"
                ),
                "spellcheck": "false",
            }
        ),
    )

    evaluations_has_header = forms.BooleanField(
        label=_("First row contains column names"),
        required=False,
        initial=True,
    )

    consumers_has_header = forms.BooleanField(
        label=_("First row contains column names"),
        required=False,
        initial=True,
    )

    general_has_header = forms.BooleanField(
        label=_("First row contains column names"),
        required=False,
        initial=True,
    )