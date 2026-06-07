from django.urls import path
from probability import views

urlpatterns = [
    path('probability/', views.probability, name="probability"),
]