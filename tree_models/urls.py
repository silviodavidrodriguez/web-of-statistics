from django.urls import path
from tree_models import views

urlpatterns = [
    path('tree_models/', views.tree_models, name='tree_models'),
]