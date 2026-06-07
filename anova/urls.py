from django.urls import path
from anova import views

urlpatterns = [
    path('anova/', views.anova, name="anova"),
]