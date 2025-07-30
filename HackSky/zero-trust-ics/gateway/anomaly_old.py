# gateway/anomaly.py

from sklearn.ensemble import IsolationForest
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(contamination=0.05)  # ~5% anomalies
        self.trained = False
        self.training_data = []

    def add_training_sample(self, feature_vector):
        self.training_data.append(feature_vector)
        if len(self.training_data) >= 50:  # train after 50 samples
            self.model.fit(self.training_data)
            self.trained = True

    def predict(self, feature_vector):
        if not self.trained:
            return 1  # treat all as normal until training is ready
        result = self.model.predict([feature_vector])[0]
        return result  # 1 = normal, -1 = anomaly
