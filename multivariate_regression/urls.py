from django.urls import path
from multivariate_regression import views

urlpatterns = [
    path('multivariate_regression/', views.multivariate_regression, name="multivariate_regression"),
]