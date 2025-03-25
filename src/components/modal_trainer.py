import os  # Module for file and directory operations
import sys  # Module for system-specific functions and exceptions
from dataclasses import dataclass  # Dataclass module for defining configuration classes

# Importing machine learning models
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score  # Metric for model evaluation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Importing custom modules
from src.exception import CustomException  # Custom exception handling class
from src.logger import logging  # Logging module for tracking process execution
from src.utlis import save_object, evaluate_models  # Utility functions for saving models and evaluating performance

@dataclass
class ModelTrainerConfig:
    """
    Configuration class for defining the file path where the trained model will be saved.
    """
    trained_model_file_path = os.path.join('artifacts', "model.pkl")  # Path to save trained model

class ModelTrainer:
    """
    Class responsible for training multiple machine learning models and selecting the best one.
    """
    def __init__(self):
        """
        Initializes the ModelTrainer class with configuration settings.
        """
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains multiple regression models and selects the best one based on performance metrics.
        
        Args:
            train_array (numpy.ndarray): Training dataset with features and target variable.
            test_array (numpy.ndarray): Testing dataset with features and target variable.
        
        Returns:
            float: R-squared score of the best-selected model.
        """
        try:
            logging.info("Splitting train and test input data")
            
            # Splitting features and target variable for training and testing sets
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features of training data
                train_array[:, -1],   # Target variable of training data
                test_array[:, :-1],   # Features of testing data
                test_array[:, -1]     # Target variable of testing data
            )
            
            # Defining multiple regression models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            
            
            #Model Hyperparameter tuning
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }
            
            # Evaluating models and obtaining performance report
            model_report: dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param = params)
            
            # Finding the best model based on performance score
            best_model_score = max(sorted(model_report.values()))  # Highest R-squared score
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]  # Model name
            best_model = models[best_model_name]  # Selecting the best model
            
            # Ensuring the best model meets the minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("Best model found based on training and testing dataset performance")
            
            # Saving the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Making predictions using the best model
            predicted = best_model.predict(X_test)
            
            # Calculating R-squared score for model evaluation
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)  # Raising custom exception with detailed error information
