from flask import Flask, request, render_template  # Importing Flask modules for web application
import numpy as np  # NumPy for numerical operations
import pandas as pd  # Pandas for data handling

from sklearn.preprocessing import StandardScaler  # StandardScaler for data normalization
from src.pipeline.predict_pipeline import CustomData, PredictPipeline  # Importing custom prediction pipeline

# Initializing Flask application
application = Flask(__name__)
app = application  # Creating an application instance

## Route for the home page
@app.route('/')
def index():
    """
    Renders the index.html template for the homepage.
    """
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handles user input for predictions. 
    Displays the form for GET requests and processes input data for POST requests.
    """
    if request.method == 'GET':
        return render_template('home.html')  # Render form page for input
    else:
        # Capturing user input from form fields
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),  # Correcting form input mapping
            writing_score=float(request.form.get('reading_score'))  # Correcting form input mapping
        )
        
        # Converting user input data into a Pandas DataFrame
        pred_df = data.get_data_as_data_frame()
        print(pred_df)  # Debugging statement to check input data
        
        # Initializing prediction pipeline
        predict_pipeline = PredictPipeline()
        
        # Making predictions using the trained model
        results = predict_pipeline.predict(pred_df)
        
        # Rendering the result on the home.html page
        return render_template("home.html", results=results[0])
    
if __name__ == "__main__":
    """
    Runs the Flask web application.
    """
    app.run(host="0.0.0.0", debug=True)  # Running Flask app on all available network interfaces
