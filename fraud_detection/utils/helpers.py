import numpy as np
from sklearn.preprocessing import StandardScaler

def process_data(data):
    scaler = StandardScaler()
    processed_data = scaler.fit_transform(data)
    return processed_data

