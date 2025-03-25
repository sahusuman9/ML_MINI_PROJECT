import sys  # Module for system-specific functions and exceptions
import pandas as pd  # Pandas for handling data in DataFrame format
from src.exception import CustomException  # Custom exception handling class
from src.utlis import load_object  # Utility function for loading saved models and preprocessors

class PredictPipeline:
    """
    Pipeline for making predictions using a trained model and a preprocessor.
    """
    def __init__(self):
        """
        Initializes the PredictPipeline class.
        """
        pass
    
    def predict(self, features):
        """
        Loads the trained model and preprocessor, transforms the input features, 
        and makes predictions.
        
        Args:
            features (pd.DataFrame): Input features for prediction.
        
        Returns:
            np.ndarray: Predictions made by the trained model.
        """
        try:
            # Paths to the trained model and preprocessor
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessor.pkl'
            
            # Loading trained model and preprocessor
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            
            # Transforming input data using the preprocessor
            data_scaled = preprocessor.transform(features)
            
            # Making predictions using the trained model
            preds = model.predict(data_scaled)
            
            return preds
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with detailed error information
    
class CustomData:
    """
    Class for capturing and structuring custom user input data.
    """
    def __init__(self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):
        """
        Initializes the CustomData class with input values.
        
        Args:
            gender (str): Gender of the student.
            race_ethnicity (str): Ethnic group of the student.
            parental_level_of_education (str): Education level of the student's parents.
            lunch (str): Type of lunch received by the student.
            test_preparation_course (str): Whether the student took a test preparation course.
            reading_score (int): Reading score of the student.
            writing_score (int): Writing score of the student.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        
    def get_data_as_data_frame(self):
        """
        Converts the user input data into a Pandas DataFrame for model inference.
        
        Returns:
            pd.DataFrame: Data structured in a tabular format.
        """
        try:
            # Creating a dictionary with input features
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)  # Returning the data as a Pandas DataFrame
        
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with error details
