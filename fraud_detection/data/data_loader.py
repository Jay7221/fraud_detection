import pandas as pd

class DataLoader:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        data = pd.read_csv(self.data_path)
        labels = data.pop('label')  # Assuming 'label' is the column name for labels
        return data, labels

