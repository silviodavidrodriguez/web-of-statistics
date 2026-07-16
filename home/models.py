from django.db import models
from django.urls import reverse

class Publication(models.Model):
    title = models.CharField("Title", max_length=500)
    authors = models.CharField("Authors", max_length=500)
    journal = models.CharField("Journal", max_length=250)
    volume = models.CharField("Volume", max_length=50, blank=True)
    article_number = models.CharField(
        "Article number or pages",
        max_length=100,
        blank=True,
    )
    year = models.PositiveIntegerField("Year")
    doi = models.URLField("DOI URL", blank=True)
    description = models.TextField("Description", blank=True)
    published = models.BooleanField("Visible on website", default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-year", "-created_at"]
        verbose_name = "Scientific publication"
        verbose_name_plural = "Scientific publications"

    def __str__(self):
        return f"{self.title} ({self.year})"

class ForumTopic(models.Model):
    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending review"),
        (STATUS_APPROVED, "Approved"),
        (STATUS_REJECTED, "Rejected"),
    ]

    title = models.CharField("Title", max_length=200)
    author_name = models.CharField("Name", max_length=100)
    author_email = models.EmailField("Email")
    message = models.TextField("Message")

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Forum topic"
        verbose_name_plural = "Forum topics"

    def __str__(self):
        return self.title

    def get_absolute_url(self):
        return reverse("forum_topic_detail", kwargs={"pk": self.pk})

class ForumReply(models.Model):
    STATUS_PENDING = "pending"
    STATUS_APPROVED = "approved"
    STATUS_REJECTED = "rejected"

    STATUS_CHOICES = [
        (STATUS_PENDING, "Pending review"),
        (STATUS_APPROVED, "Approved"),
        (STATUS_REJECTED, "Rejected"),
    ]

    topic = models.ForeignKey(
        ForumTopic,
        on_delete=models.CASCADE,
        related_name="replies",
    )

    # Permite responderle a otra respuesta.
    parent = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        related_name="child_replies",
        null=True,
        blank=True,
    )

    author_name = models.CharField("Name", max_length=100)
    author_email = models.EmailField("Email")
    message = models.TextField("Reply")

    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING,
    )

    created_at = models.DateTimeField(auto_now_add=True)
    reviewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["created_at"]
        verbose_name = "Forum reply"
        verbose_name_plural = "Forum replies"

    def __str__(self):
        return f"Reply by {self.author_name} in {self.topic.title}"