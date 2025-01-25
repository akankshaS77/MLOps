import json
import joblib
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*does not have valid feature names.*')

# Correct path to the model inside the 'models' folder
model_path = 'models/rf_model'

# Load the pre-trained Random Forest model
rf_model = joblib.load(model_path)

def predict_from_json(input_json):
    """
    This function takes JSON input, parses it, and predicts using the trained Random Forest model.
    """
    # Parse the JSON data
    data = json.loads(input_json)

    # Extract input features (ensure it's in the correct shape for the model)
    features = np.array(data['input']).reshape(1, -1)  # Reshaping to match model input shape

    # Make prediction using the Random Forest model
    prediction = rf_model.predict(features)
    
    return prediction[0]

# Example input JSON
input_json = '''
{
  "input": [
    0.172848, 0.365408, 0.574790, 0.253927, 1.000000, 0.666667, 0.080751, 0.434783,
    0.161963, 0.542373, 0.658346, 0.101629, 0.183971, 1.000000, 0.277778, 0.073048, 0.42073
  ]
}
'''

# Use the function to make a prediction
result = predict_from_json(input_json)

# Print the prediction result
print(f"Prediction Result********: {result}")

# Optionally, assert the result (example checking if the result is 0)
def test_prediction():
    # Your prediction logic here
    assert result == 0  # Correct result from your model
