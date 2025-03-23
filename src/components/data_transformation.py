import sys  # Importing sys module for system-specific functions
from dataclasses import dataclass  # Importing dataclass for defining configuration classes

import numpy as np  # Importing numpy for numerical operations
import pandas as pd  # Importing pandas for data manipulation
from sklearn.compose import ColumnTransformer  # Importing ColumnTransformer for feature transformation
from sklearn.impute import SimpleImputer  # Importing SimpleImputer for handling missing values
from sklearn.pipeline import Pipeline  # Importing Pipeline to build processing pipelines
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # Importing preprocessing techniques

from src.exception import CustomException  # Importing custom exception handling class
from src.logger import logging  # Importing logging module for logging messages
import os  # Importing os module for file operations

from src.utlis import save_object  # Importing save_object function to save preprocessor objects

@dataclass
class DataTransformationConfig:
    """
    Configuration class for defining file paths for saving preprocessing objects.
    """
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")  # Path to save the preprocessing object
    
class DataTransformation:
    """
    Class for handling data transformation including preprocessing.
    """
    def __init__(self):
        """
        Initializes DataTransformation with configuration paths.
        """
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        """
        Creates and returns a preprocessing object for transforming numerical and categorical features.
        
        Returns:
            ColumnTransformer: A preprocessor object with transformations for numerical and categorical features.
        """
        try:
            numerical_columns = ['writing_score', 'reading_score']  # Defining numerical columns
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            # Defining numerical pipeline
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Handling missing values
                    ("scaler", StandardScaler())  # Scaling numerical features
                ]
            )
            
            # Defining categorical pipeline
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy='most_frequent')),  # Handling missing values
                    ("one_hot_encoder", OneHotEncoder()),  # Encoding categorical variables
                    ("scaler", StandardScaler(with_mean=False))  # Scaling categorical features
                ]
            )
            
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combining numerical and categorical transformations
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            
            return preprocessor  # Returning the preprocessing object
        
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with error details
        
    
    def initiate_data_transformation(self, train_path, test_path):
        """
        Reads the dataset, applies transformations, and saves the preprocessing object.
        
        Args:
            train_path (str): Path to training dataset.
            test_path (str): Path to testing dataset.
        
        Returns:
            Tuple[np.array, np.array, str]: Transformed train and test arrays along with the preprocessor object path.
        """
        try:
            # Reading train and test datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")
            
            preprocessing_obj = self.get_data_transformer_object()  # Getting preprocessing object
            
            target_column_name = "math_score"  # Defining target variable
            numerical_columns = ['writing_score', 'reading_score']  # Defining numerical columns
            
            # Splitting input and target features for training and testing sets
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes.")
            
            # Applying transformations
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combining transformed input features with target values
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object.")

            # Saving the preprocessing object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
            
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with error details
