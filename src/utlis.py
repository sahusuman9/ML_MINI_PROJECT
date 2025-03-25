import os  # Module for file and directory operations
import sys  # Module for system-specific functions and exceptions

import dill  # Module for object serialization and deserialization

import numpy as np  # NumPy for numerical computations
import pandas as pd  # Pandas for data handling

from src.exception import CustomException  # Custom exception handling class
from sklearn.metrics import r2_score  # R-squared metric for model evaluation
from sklearn.model_selection import GridSearchCV  # Hyperparameter tuning using Grid Search

def save_object(file_path, obj):
    """
    Saves a Python object to a file using dill.
    
    Args:
        file_path (str): Path to save the object.
        obj (object): Python object to be saved.
    """
    try:
        dir_path = os.path.dirname(file_path)
        
        # Create directory if it does not exist
        os.makedirs(dir_path, exist_ok=True)
        
        # Save object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e, sys)  # Raising custom exception with error details
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Trains and evaluates multiple machine learning models using GridSearchCV for hyperparameter tuning.
    
    Args:
        X_train (np.ndarray): Training feature set.
        y_train (np.ndarray): Training target variable.
        X_test (np.ndarray): Testing feature set.
        y_test (np.ndarray): Testing target variable.
        models (dict): Dictionary of models to train and evaluate.
        param (dict): Dictionary of hyperparameter grids for each model.
    
    Returns:
        dict: Dictionary containing R-squared scores of the evaluated models on the test set.
    """
    try:
        report = {}  # Dictionary to store model performance results

        for i in range(len(list(models))):
            model = list(models.values())[i]  # Get model instance
            para = param[list(models.keys())[i]]  # Get corresponding hyperparameters
            
            # Performing Grid Search for best hyperparameters
            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)
            
            # Setting best parameters to the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)  # Training model with best parameters
            
            # Making predictions on training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculating R-squared scores for train and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            # Storing test score in the report dictionary
            report[list(models.keys())[i]] = test_model_score
        
        return report  # Returning evaluation results

    except Exception as e:
        raise CustomException(e, sys)  # Raising custom exception with error details
    

def load_object(file_path):
    """
    Loads a serialized object from a file using dill.
    
    Args:
        file_path (str): Path to the saved object file.
    
    Returns:
        object: Loaded Python object.
    """
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)  # Loading object using dill
    
    except Exception as e:
        raise CustomException(e, sys)  # Raising custom exception with error details
