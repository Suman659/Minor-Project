from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load trained models
flight_model = pickle.load(open('flight_model.pkl', 'rb'))
activity_model = pickle.load(open('activity_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_flight', methods=['POST'])
def predict_flight():
    # Collect user input from form (index.html)
    data = request.form
    # Example: extract the user inputs from the form
    dep_time = data['departure_time']
    arr_time = data['arrival_time']
    # More fields would be extracted similarly...

    # Prepare the data for prediction (adjust according to your model)
    input_data = np.array([[dep_time, arr_time]])  # Modify as per model
    prediction = flight_model.predict(input_data)

    # Send prediction to frontend
    return jsonify({'fare': prediction[0]})

@app.route('/predict_activity', methods=['POST'])
def predict_activity():
    # Collect activity input data from form (activity.html)
    data = request.form
    activity_input = data['activity_input']  # Example field

    # Prepare the data for prediction (adjust according to your model)
    input_data = np.array([[activity_input]])  # Modify as per model
    prediction = activity_model.predict(input_data)

    # Send the prediction to frontend
    return jsonify({'activity_prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
