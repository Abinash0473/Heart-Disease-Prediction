import pickle
import numpy as np

# Load the trained model
def load_model():
    with open('model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Function to predict heart disease risk
def predict_heart_disease(input_data, scaler):
    model = load_model()
    input_data = np.array(input_data).reshape(1, -1)
    input_data = scaler.transform(input_data)
    prediction = model.predict(input_data)
    return prediction[0]
