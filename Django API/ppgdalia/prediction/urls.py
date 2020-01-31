from django.urls import path
from . import views
from django.conf.urls import url, include
from rest_framework.routers import DefaultRouter

from prediction.views import EndpointViewSet
from prediction.views import MLAlgorithmViewSet
from prediction.views import MLAlgorithmStatusViewSet
from prediction.views import MLRequestViewSet
from prediction.views import Prediction

router = DefaultRouter(trailing_slash=False)
router.register(r"prediction", EndpointViewSet, basename="prediction")
router.register(r"mlalgorithms", MLAlgorithmViewSet, basename="mlalgorithms")
router.register(r"mlalgorithmstatuses", MLAlgorithmStatusViewSet, basename="mlalgorithmstatuses")
router.register(r"mlrequests", MLRequestViewSet, basename="mlrequests")

urlpatterns = [
    url(r"^api/v1/", include(router.urls)),
	url(
        r"^api/v1/(?P<endpoint_name>.+)/predict$", Prediction.as_view(), name="predict"
    ),
]