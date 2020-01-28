# predict the activity of a subject

from django.test import TestCase
from prediction.ml.income_classifier.xgboost import XGBoostClassifier
import numpy as np
import inspect
from prediction.ml.registry import MLRegistry

class MLTests(TestCase):
	def test_rf_algorithm(self):
		input_data = {
			"ACC_x_chest": 0.896337,
			"ACC_y_chest": -0.156904,
			"ACC_z_chest": -0.277930,
			"ACC_x_wrist": -0.746094,
			"ACC_y_wrist": 0.652344,
			"ACC_z_wrist": -0.261719,
			"TEMP_wrist": 30.910000,
			"Gender": 1.000000,
			"AGE": 34.000000,
			"SPORT": 6.000000,	
			"IMC": 23.547881,
			"label": 58.242769,
			"rpeaks": 1.000000
		}
		
		target_expected = 3
		
		my_alg = XGBoostClassifier()
		response = my_alg.compute_prediction(input_data)
		print(f"Prediction : activity = {np.argmax(response)} with an accuracy of {max(response)}")
		print(f"Expected prediction was : activity = {target_expected}")
		
def test_registry(self):
	registry = MLRegistry()
	self.assertEqual(len(registry.endpoints), 0)
	endpoint_name = "income_classifier"
	algorithm_object = XGBoostClassifier()
	algorithm_name = "xgboost"
	algorithm_status = "production"
	algorithm_version = "0.0.1"
	algorithm_owner = "Piotr"
	algorithm_description = "XGBoost model prediction"
	algorithm_code = inspect.getsource(XGBoostClassifier)
	# add to registry
	registry.add_algorithm(endpoint_name, algorithm_object, algorithm_name,
					algorithm_status, algorithm_version, algorithm_owner,
					algorithm_description, algorithm_code)
	# there should be one endpoint available
	self.assertEqual(len(registry.endpoints), 1)		