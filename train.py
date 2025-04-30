import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Add this!

import json
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.modal_trainer import ModelTrainer
from src.exception import CustomException
import logging

def main():
    logging.info("Training pipeline started.")

    try:
        # Step 1: Ingest data
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Step 2: Transform data
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Step 3: Train model
        model_trainer = ModelTrainer()
        r2_score, best_model = model_trainer.initiate_model_trainer(train_arr, test_arr)

        # Step 4: Save metrics
        metrics = {
            "r2_score": r2_score,
            "best_model": model_trainer.best_model_name
        }

        os.makedirs("artifacts", exist_ok=True)
        with open("artifacts/metrics.json", "w") as f:
            json.dump(metrics, f)

        logging.info("Training completed. Metrics saved to artifacts/metrics.json")

    except Exception as e:
        logging.error("An error occurred in the training pipeline.")
        raise CustomException(e, sys)

if __name__ == "__main__":
    main()
