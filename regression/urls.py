from django.urls import path
from regression import views

urlpatterns = [
    path('regression/', views.regression, name="regression"),
]