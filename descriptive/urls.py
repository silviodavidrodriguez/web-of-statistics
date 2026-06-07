from django.urls import path
from descriptive import views

urlpatterns = [
    path('descriptive/', views.descriptive, name="descriptive"),
]