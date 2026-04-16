from django.urls import path
from .views import detect_body

urlpatterns = [
    path("detect-body/", detect_body),
]