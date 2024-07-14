import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    DATA_PATH = os.getenv('DATA_PATH')
    MODEL_PATH = os.getenv('MODEL_PATH')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    EPOCHS = int(os.getenv('EPOCHS'))
