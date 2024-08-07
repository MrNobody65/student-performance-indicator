import sys

from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        pass

    def train(self):
        try:
            data_ingestion = DataIngestion()
            train_path, test_path = data_ingestion.initiate_data_ingestion()

            data_transformation = DataTransformation()
            train_arr, test_arr = data_transformation.initiate_data_transformation(train_path, test_path)

            model_trainer = ModelTrainer()
            model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    train_pipeline = TrainPipeline()
    train_pipeline.train()