from django.urls import path
from multivariate import views

urlpatterns = [
    path('multivariate/', views.multivariate, name="multivariate"),
]