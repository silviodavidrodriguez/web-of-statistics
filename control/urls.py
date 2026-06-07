from django.urls import path
from control import views

urlpatterns = [
    path('control/', views.control, name="control"),
]