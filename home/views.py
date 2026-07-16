from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from .forms import ForumReplyForm, ForumTopicForm
from .models import ForumReply, ForumTopic, Publication

def index(request):
    context = {
        "segment": "home",
    }

    return render(request, "home/index.html", context)

def publication_list(request):
    publications = Publication.objects.filter(published=True)

    context = {
        "segment": "publications",
        "publications": publications,
    }

    return render(
        request,
        "home/publication_list.html",
        context,
    )

def forum_list(request):
    topics = ForumTopic.objects.filter(
        status=ForumTopic.STATUS_APPROVED,
    )

    form = ForumTopicForm()

    context = {
        "segment": "forum",
        "topics": topics,
        "form": form,
    }

    return render(
        request,
        "home/forum_list.html",
        context,
    )

def forum_topic_create(request):
    if request.method != "POST":
        return redirect("forum")

    form = ForumTopicForm(request.POST)

    if form.is_valid():
        topic = form.save(commit=False)
        topic.status = ForumTopic.STATUS_PENDING
        topic.save()

        messages.success(
            request,
            (
                "Your topic was submitted successfully. "
                "It will be published after administrator review."
            ),
        )

        return redirect("forum")

    topics = ForumTopic.objects.filter(
        status=ForumTopic.STATUS_APPROVED,
    )

    context = {
        "segment": "forum",
        "topics": topics,
        "form": form,
    }

    return render(
        request,
        "home/forum_list.html",
        context,
        status=400,
    )

def forum_topic_detail(request, pk):
    topic = get_object_or_404(
        ForumTopic,
        pk=pk,
        status=ForumTopic.STATUS_APPROVED,
    )

    replies = topic.replies.filter(
        status=ForumReply.STATUS_APPROVED,
    ).select_related("parent")

    context = {
        "segment": "forum",
        "topic": topic,
        "replies": replies,
        "reply_form": ForumReplyForm(),
    }

    return render(
        request,
        "home/forum_topic_detail.html",
        context,
    )

def forum_reply_create(request, pk):
    topic = get_object_or_404(
        ForumTopic,
        pk=pk,
        status=ForumTopic.STATUS_APPROVED,
    )

    if request.method != "POST":
        return redirect(
            "forum_topic_detail",
            pk=topic.pk,
        )

    form = ForumReplyForm(request.POST)

    if form.is_valid():
        reply = form.save(commit=False)
        reply.topic = topic
        reply.status = ForumReply.STATUS_PENDING

        parent_id = form.cleaned_data.get("parent_id")

        if parent_id:
            reply.parent = get_object_or_404(
                ForumReply,
                pk=parent_id,
                topic=topic,
                status=ForumReply.STATUS_APPROVED,
            )

        reply.save()

        messages.success(
            request,
            (
                "Your reply was submitted successfully. "
                "It will be visible after administrator review."
            ),
        )

        return redirect(
            "forum_topic_detail",
            pk=topic.pk,
        )

    replies = topic.replies.filter(
        status=ForumReply.STATUS_APPROVED,
    ).select_related("parent")

    context = {
        "segment": "forum",
        "topic": topic,
        "replies": replies,
        "reply_form": form,
    }

    return render(
        request,
        "home/forum_topic_detail.html",
        context,
        status=400,
    )