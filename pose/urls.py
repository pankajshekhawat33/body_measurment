from django.urls import path
from .views import detect_body, test_model_status

urlpatterns = [
    path("detect-body/", detect_body),
    path("test-model-status/", test_model_status),
]