from django.urls import path
from classification import views

urlpatterns = [
    path('classification/', views.classification, name="classification"),
]