from django.urls import path
from home import views

urlpatterns = [
    path(
        "",
        views.index,
        name="home",
    ),
    path(
        "publications/",
        views.publication_list,
        name="publications",
    ),
    path(
        "forum/",
        views.forum_list,
        name="forum",
    ),
    path(
        "forum/new/",
        views.forum_topic_create,
        name="forum_topic_create",
    ),
    path(
        "forum/topic/<int:pk>/",
        views.forum_topic_detail,
        name="forum_topic_detail",
    ),
    path(
        "forum/topic/<int:pk>/reply/",
        views.forum_reply_create,
        name="forum_reply_create",
    ),
]