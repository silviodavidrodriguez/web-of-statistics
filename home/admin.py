from django.contrib import admin
from django.utils import timezone
from .models import ForumReply, ForumTopic, Publication

@admin.action(description="Approve selected items")
def approve_items(modeladmin, request, queryset):
    queryset.update(
        status="approved",
        reviewed_at=timezone.now(),
    )

@admin.action(description="Reject selected items")
def reject_items(modeladmin, request, queryset):
    queryset.update(
        status="rejected",
        reviewed_at=timezone.now(),
    )

@admin.register(Publication)
class PublicationAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "journal",
        "year",
        "published",
    )
    list_filter = ("published", "year", "journal")
    search_fields = ("title", "authors", "journal")
    list_editable = ("published",)

@admin.register(ForumTopic)
class ForumTopicAdmin(admin.ModelAdmin):
    list_display = (
        "title",
        "author_name",
        "status",
        "created_at",
        "reviewed_at",
    )
    list_filter = ("status", "created_at")
    search_fields = (
        "title",
        "message",
        "author_name",
        "author_email",
    )
    readonly_fields = ("created_at", "reviewed_at")
    actions = (approve_items, reject_items)

@admin.register(ForumReply)
class ForumReplyAdmin(admin.ModelAdmin):
    list_display = (
        "topic",
        "author_name",
        "status",
        "created_at",
        "reviewed_at",
    )
    list_filter = ("status", "created_at")
    search_fields = (
        "message",
        "author_name",
        "author_email",
        "topic__title",
    )
    readonly_fields = ("created_at", "reviewed_at")
    actions = (approve_items, reject_items)