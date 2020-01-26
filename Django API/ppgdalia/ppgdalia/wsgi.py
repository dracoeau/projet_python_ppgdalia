"""
WSGI config for ppgdalia project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/3.0/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ppgdalia.settings')

application = get_wsgi_application()

import inspect
from prediction.ml.registry import MLRegistry
from prediction.ml.income_classifier.xgboost import XGBoostClassifier

try:
    registry = MLRegistry()
    rf = XGBoostClassifier()
    registry.add_algorithm(endpoint_name="income_classifier",
                            algorithm_object=rf,
                            algorithm_name="xgboost",
                            algorithm_status="production",
                            algorithm_version="0.0.1",
                            owner="Piotr",
                            algorithm_description="XGBoost model prediction",
                            algorithm_code=inspect.getsource(XGBoostClassifier))

except Exception as e:
    print("Exception while loading the algorithms to the registry,", str(e))