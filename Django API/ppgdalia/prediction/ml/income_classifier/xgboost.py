# predict the activity of a subject

import joblib
import pandas as pd
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score

class XGBoostClassifier:
	def __init__(self):
		self.model = joblib.load("prediction/ml/income_classifier/" + "xgboost.joblib")

	def predict(self, input_data):
		return self.model.predict_proba(input_data)

	def compute_prediction(self, input_data):
		input_data = pd.DataFrame(input_data, index=[0])
		prediction = self.predict(input_data)[0]  # only one sample
		return prediction


