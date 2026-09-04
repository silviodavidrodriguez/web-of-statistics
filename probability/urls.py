from django.urls import path
from probability import views


urlpatterns = [
    path(
        "probability/",
        views.probability,
        name="probability",
    ),
    path(
        "probability/simulation/export/",
        views.probability_simulation_export,
        name="probability_simulation_export",
    ),
]