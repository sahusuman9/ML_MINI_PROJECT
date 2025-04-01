# Student Math Score Prediction

## Overview
This project predicts students' math scores using machine learning. It follows a structured ML pipeline, including exploratory data analysis (EDA), data ingestion, transformation, model training, and evaluation.

## Project Structure
```
├── data/                          # Raw dataset (stud.csv)
├── notebooks/                     # Jupyter Notebooks for EDA and model training
│   ├── eda.ipynb                  # Exploratory Data Analysis (EDA)
│   ├── model_training.ipynb       # Model training and evaluation
├── src/                           # Source code
│   ├── components/                # ML pipeline components
│   │   ├── data_ingestion.py      # Handles data loading and splitting
│   │   ├── data_transformation.py # Preprocessing and feature engineering
│   │   ├── model_trainer.py       # Model training and evaluation
│   ├── pipeline/                  # Pipeline execution
│   │   ├── prediction_pipeline.py # Handles model inference
│   ├── logger.py                  # Logging utility
│   ├── exception.py               # Custom exception handling
│   ├── utils.py                   # Utility functions (saving objects, evaluation, etc.)
├── app.py                         # Flask API for model inference
├── artifacts/                      # Stores trained models and transformed data
├── README.md                       # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/student-math-score-prediction.git
   ```
2. Navigate to the project directory:
   ```bash
   cd student-math-score-prediction
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
### Run the project pipeline:
```bash
python src/components/data_ingestion.py
```

### Steps in the ML Pipeline:
1. **Exploratory Data Analysis (EDA)**:
   - Conducted in `notebooks/eda.ipynb` to understand data distributions, correlations, and feature importance.
   - Helps in identifying missing values and outliers.
2. **Data Ingestion**:
   - Reads `stud.csv` and splits it into training and test sets.
   - Saves data in the `artifacts` folder.
3. **Data Transformation**:
   - Handles missing values and encodes categorical features.
   - Scales numerical features.
   - Saves the transformation pipeline (`preprocessor.pkl`).
4. **Model Training**:
   - Implemented in `notebooks/model_training.ipynb`.
   - Trains multiple regression models (Random Forest, Decision Tree, Gradient Boosting, etc.).
   - Selects the best model based on R² score.
   - Saves the best model (`model.pkl`).
5. **Model Evaluation**:
   - Evaluates model performance on test data.
6. **Prediction Pipeline**:
   - `prediction_pipeline.py` loads the trained model and preprocessor.
   - Takes user input (reading and writing scores) and returns the predicted math score.

## Flask API for Model Deployment
The project includes a Flask-based API (`app.py`) to serve predictions:

- **Run the Flask app:**
  ```bash
  python app.py
  ```
- **API Routes:**
  - `GET /`: Health check endpoint.
  - `POST /predict`: Accepts student data in JSON format (reading and writing scores) and returns the predicted math score.

## Model Output
- The model predicts a student's **math score** based on their **reading and writing scores**.
- The API accepts input in the following JSON format:
  ```json
  {
      "reading_score": 78,
      "writing_score": 85
  }
  ```
- The API returns a JSON response with the predicted math score:
  ```json
  {
      "predicted_math_score": 80.5666666
  }
  ```

## Model Performance
- The model achieving the highest **R² score** is selected.
- If no model achieves a score >0.6, an exception is raised.
- The best performance model used is Linear Regression.
