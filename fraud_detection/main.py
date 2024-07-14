from config.settings import Config
from data.data_loader import DataLoader
from models.fraud_detection_model import FraudDetectionModel
from utils.helpers import process_data

def main():
    # Load configuration
    config = Config()

    # Load data
    data_loader = DataLoader(config.DATA_PATH)
    data, labels = data_loader.load_data()

    # Process data
    processed_data = process_data(data)

    # Train model
    model = FraudDetectionModel()
    model.train(processed_data, labels)

    # Save model
    model.save(config.MODEL_PATH)

if __name__ == "__main__":
    main()

