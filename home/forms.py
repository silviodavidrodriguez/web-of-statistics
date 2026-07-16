from django import forms
from .models import ForumReply, ForumTopic

class ForumTopicForm(forms.ModelForm):
    # Campo trampa contra bots. Debe quedar vacío.
    website = forms.CharField(
        required=False,
        widget=forms.HiddenInput,
    )

    class Meta:
        model = ForumTopic
        fields = [
            "title",
            "author_name",
            "author_email",
            "message",
        ]
        widgets = {
            "title": forms.TextInput(
                attrs={
                    "class": "forum-input",
                    "placeholder": "Topic title",
                    "maxlength": 200,
                }
            ),
            "author_name": forms.TextInput(
                attrs={
                    "class": "forum-input",
                    "placeholder": "Your name",
                }
            ),
            "author_email": forms.EmailInput(
                attrs={
                    "class": "forum-input",
                    "placeholder": "Your email",
                }
            ),
            "message": forms.Textarea(
                attrs={
                    "class": "forum-textarea",
                    "placeholder": "Write your message",
                    "rows": 6,
                }
            ),
        }

    def clean_website(self):
        value = self.cleaned_data.get("website")

        if value:
            raise forms.ValidationError("Invalid submission.")

        return value

class ForumReplyForm(forms.ModelForm):
    parent_id = forms.IntegerField(
        required=False,
        widget=forms.HiddenInput,
    )

    website = forms.CharField(
        required=False,
        widget=forms.HiddenInput,
    )

    class Meta:
        model = ForumReply
        fields = [
            "author_name",
            "author_email",
            "message",
        ]
        widgets = {
            "author_name": forms.TextInput(
                attrs={
                    "class": "forum-input",
                    "placeholder": "Your name",
                }
            ),
            "author_email": forms.EmailInput(
                attrs={
                    "class": "forum-input",
                    "placeholder": "Your email",
                }
            ),
            "message": forms.Textarea(
                attrs={
                    "class": "forum-textarea",
                    "placeholder": "Write your reply",
                    "rows": 5,
                }
            ),
        }

    def clean_website(self):
        value = self.cleaned_data.get("website")

        if value:
            raise forms.ValidationError("Invalid submission.")

        return value