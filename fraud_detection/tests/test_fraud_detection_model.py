import unittest
import numpy as np
from models.fraud_detection_model import FraudDetectionModel

class TestFraudDetectionModel(unittest.TestCase):
    def test_model_training(self):
        model = FraudDetectionModel()
        # Dummy data
        data = np.random.rand(100, 30)  # 100 samples, 30 features
        labels = np.random.randint(2, size=100)
        model.train(data, labels)
        # Test the model structure
        self.assertEqual(len(model.model.layers), 3)

if __name__ == '__main__':
    unittest.main()

