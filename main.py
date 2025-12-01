from flask import Flask,jsonify, request
import pandas as pd
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

loaded_model = joblib.load("models/model_diabetes.pkl")

loaded_scaler = joblib.load("models/standard_scaler.pkl")

columns = [
        'Pregnancies', 
        'Glucose', 
        'BloodPressure', 
        'SkinThickness', 
        'Insulin', 
        'BMI', 
        'DiabetesPedigreeFunction', 
        'Age'
    ]

@app.route('/')
def index():
    return jsonify({
        "meta" : {
            "status" : "success",
            "message" : "Welcome to Diabetes"
        },
        "data" : None
    })

@app.route ('/api/predict', methods = {"POST"})
def predict():
    data = request.get_json()

    X_input = pd.DataFrame([data], columns=columns)

    X_input_scaled = loaded_scaler.transform (X_input)

    prediction = loaded_model.predict(X_input_scaled)

    return jsonify({
        "meta": {
            "status" : "Success",
            "message" : "Prediction"
        },
        "data" : prediction.tolist()[0]
    })

if __name__ == '__main__': 
    app.run(debug = True)