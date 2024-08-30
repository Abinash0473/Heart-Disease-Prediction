import pickle

# Load the scaler
def load_scaler():
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return scaler
