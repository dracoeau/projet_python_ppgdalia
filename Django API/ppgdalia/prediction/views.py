from django.shortcuts import render

# Create your views here.

from rest_framework import viewsets
from rest_framework import mixins

from prediction.models import Endpoint
from prediction.serializers import EndpointSerializer

from prediction.models import MLAlgorithm
from prediction.serializers import MLAlgorithmSerializer

from prediction.models import MLAlgorithmStatus
from prediction.serializers import MLAlgorithmStatusSerializer

from prediction.models import MLRequest
from prediction.serializers import MLRequestSerializer

from prediction.machine_learning.model_classifier.xgboost import XGBoostClassifier

import json
from numpy.random import rand
from rest_framework import views, status
from rest_framework.response import Response

class EndpointViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = EndpointSerializer
    queryset = Endpoint.objects.all()


class MLAlgorithmViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet
):
    serializer_class = MLAlgorithmSerializer
    queryset = MLAlgorithm.objects.all()


def deactivate_other_statuses(instance):
    old_statuses = MLAlgorithmStatus.objects.filter(parent_mlalgorithm = instance.parent_mlalgorithm,
                                                        created_at__lt=instance.created_at,
                                                        active=True)
    for i in range(len(old_statuses)):
        old_statuses[i].active = False
    MLAlgorithmStatus.objects.bulk_update(old_statuses, ["active"])

class MLAlgorithmStatusViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.CreateModelMixin
):
    serializer_class = MLAlgorithmStatusSerializer
    queryset = MLAlgorithmStatus.objects.all()
    def perform_create(self, serializer):
        try:
            with transaction.atomic():
                instance = serializer.save(active=True)
                # set active=False for other statuses
                deactivate_other_statuses(instance)



        except Exception as e:
            raise APIException(str(e))

class MLRequestViewSet(
    mixins.RetrieveModelMixin, mixins.ListModelMixin, viewsets.GenericViewSet,
    mixins.UpdateModelMixin
):
    serializer_class = MLRequestSerializer
    queryset = MLRequest.objects.all()

class Prediction(views.APIView):
	def post(self, request, endpoint_name, format=None):
		
		my_alg = XGBoostClassifier()
		prediction = my_alg.compute_prediction(request.data)

		return Response(prediction)