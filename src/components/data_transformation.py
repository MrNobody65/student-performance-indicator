import sys
import os

from dataclasses import dataclass
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocess_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ["reading_score", "writing_score"]
            categorical_columns = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            logging.info("Create data transformer")
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_data_path, test_data_path):
        try:
            train_df = pd.read_csv(train_data_path)
            test_df = pd.read_csv(test_data_path)

            logging.info("Read train and test data")

            preprocessor = self.get_data_transformer()
            target_column_name = 'math_score'

            input_train = train_df.drop(columns=[target_column_name], axis=1)
            target_train = train_df[target_column_name]

            input_test = test_df.drop(columns=[target_column_name], axis=1)
            target_test = test_df[target_column_name]

            input_train_arr = preprocessor.fit_transform(input_train)
            input_test_arr = preprocessor.transform(input_test)

            train_arr = np.c_[input_train_arr, np.array(target_train)]
            test_arr = np.c_[input_test_arr, np.array(target_test)]

            logging.info("Apply transformation to data")

            save_object(
                file_path=self.data_transformation_config.preprocess_obj_file_path,
                obj=preprocessor
            )

            logging.info("Save preprocess object")

            return (
                train_arr,
                test_arr
            )
        except Exception as e:
            raise CustomException(e, sys)