import pandas as pd
import tensorflow as tf
import numpy as np
import io

import joblib
from flask_cors import CORS
from flask import Flask, jsonify, request

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input

app = Flask(__name__)
CORS(app)

load_model = joblib.load("models/model_diabetes.pkl")
load_model_rps = tf.keras.models.load_model("models/RPS.keras")

load_scaler = joblib.load("models/standard_scaler.pkl")

columns = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

rps_class_names = ["Paper", "Rock", "Scissors"]


@app.route("/")
def index():
    return jsonify(
        {"meta": {"status": "success", "message": "Welcome to Api"}, "data": None}
    )


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    x_input = pd.DataFrame([data], columns=columns)

    x_input_scaled = load_scaler.transform(x_input)

    prediction = load_model.predict(x_input_scaled)

    return jsonify(
        {
            "meta": {"status": "success", "message": "Prediction"},
            "data": prediction.tolist()[0],
        }
    )


@app.route("/api/predict-rps", methods=["POST"])
def predict_rps():
    if "file" not in request.files:
        return jsonify(
            {
                "meta": {"status": 400, "message": "No file part in the request"},
            }
        )

    file = request.files["file"]
    img_bytes = io.BytesIO(file.read())

    img = image.load_img(img_bytes, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)
    img_batch = np.expand_dims(img_array, axis=0)

    predictions = load_model_rps.predict(img_batch)
    predicted_class_index = np.argmax(predictions)

    return jsonify(
        {
            "meta": {"status": 200, "message": "Prediction Successful"},
            "data": {
                "prediction": rps_class_names[predicted_class_index],
                "probability": f"{np.max(predictions) * 100:.2f}%",
            },
        }
    )


if __name__ == "__main__":
    app.run(debug=True)
