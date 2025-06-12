from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.normpath(os.path.join(BASE_DIR, "../model"))

# Load model, scaler, dan fitur
with open(os.path.join(MODEL_DIR, "knn_best_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler_selected.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "features_selected.pkl"), "rb") as f:
    selected_features = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "CBR KNN Diabetes Diagnosis API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Validasi input: pastikan semua fitur tersedia
        if not all(feature in data for feature in selected_features):
            return jsonify({
                "success": False,
                "message": f"Input JSON harus memiliki semua fitur: {selected_features}"
            }), 400

        # Ambil data dalam urutan fitur yang sesuai
        input_data = np.array([[data[feature] for feature in selected_features]])

        # Scaling
        input_scaled = scaler.transform(input_data)

        # Prediksi
        pred = model.predict(input_scaled)[0]
        result = "Diabetes" if pred == 1 else "Tidak Diabetes"

        return jsonify({
            "success": True,
            "prediction": int(pred),
            "result": result
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
