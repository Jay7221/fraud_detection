import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class FraudDetectionModel:
    def __init__(self):
        self.model = self.build_model()

    def build_model(self):
        model = Sequential([
            Dense(64, activation='relu', input_shape=(30,)),  # Assuming 30 features
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, data, labels):
        self.model.fit(data, labels, epochs=10, batch_size=32)

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model = tf.keras.models.load_model(path)

