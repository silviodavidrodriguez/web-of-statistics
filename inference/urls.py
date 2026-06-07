from django.urls import path
from inference import views

urlpatterns = [
    path('inference/', views.inference, name="inference"),
]