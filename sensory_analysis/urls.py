from django.urls import path
from sensory_analysis import views

urlpatterns = [
    path('sensory_analysis/', views.sensory_analysis, name="sensory_analysis"),
    path("sensory_analysis/variables/",views.variable_mapping,name="sensory_variable_mapping",),
    path("sensory_analysis/configuration/",views.study_configuration,name="sensory_study_configuration",),
    path("sensory_analysis/validation/",views.study_validation,name="sensory_study_validation",),
]