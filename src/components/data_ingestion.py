import os  # Importing os module for file and directory operations
import sys  # Importing sys module for system-specific functions
from src.exception import CustomException  # Importing custom exception handling class
from src.logger import logging  # Importing logging module for logging messages

import pandas as pd  # Importing pandas for data manipulation

from sklearn.model_selection import train_test_split  # Importing train-test split function from sklearn
from dataclasses import dataclass  # Importing dataclass for creating configuration classes

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.modal_trainer import ModelTrainerConfig
from src.components.modal_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    """
    Configuration class for defining data ingestion paths.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")  # Path to store training data
    test_data_path: str = os.path.join('artifacts', "test.csv")  # Path to store testing data
    raw_data_path: str = os.path.join('artifacts', "data.csv")  # Path to store raw data
    
class DataIngestion:
    """
    Class for handling data ingestion process.
    """
    def __init__(self):
        """
        Initializes DataIngestion with configuration paths.
        """
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        """
        Reads data, performs train-test split, and saves the files.
        
        Returns:
            Tuple[str, str]: Paths of train and test datasets.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Reading dataset
            df = pd.read_csv("data/stud.csv")  # Reading data from CSV file
            logging.info('Read the dataset as dataframe')
            
            # Creating directory for storing datasets
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Saving raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            logging.info('Train test split initiated')
            
            # Splitting dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Saving train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with error details
        

if __name__ == "__main__":
    """
    Runs the data ingestion process when script is executed.
    """
    obj = DataIngestion() # Creating an instance of DataIngestion class
    train_data,test_data = obj.initiate_data_ingestion()  # Initiating data ingestion process

    # Initiating data transformation process after data ingestion
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)
    
    modelTrainer = ModelTrainer()
    print(modelTrainer.initiate_model_trainer(train_arr,test_arr))