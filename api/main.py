from flask import Flask, request, jsonify
import pickle
import numpy as np
import os
from serverless_wsgi import handle_request  

app = Flask(__name__)

# Load model langsung dari folder yang sama (untuk Vercel)
try:
    with open("knn_best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open("scaler_selected.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    with open("features_selected.pkl", "rb") as f:
        selected_features = pickle.load(f)
except Exception as e:
    print(f"Error loading model files: {str(e)}")
    raise

@app.route("/")
def home():
    return "CBR KNN Diabetes Diagnosis API"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not all(feature in data for feature in selected_features):
            return jsonify({
                "success": False,
                "message": f"Input JSON harus memiliki semua fitur: {selected_features}"
            }), 400

        input_data = np.array([[data[feature] for feature in selected_features]])
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]

        return jsonify({
            "success": True,
            "prediction": int(pred),
            "result": "Diabetes" if pred == 1 else "Tidak Diabetes"
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "note": "Pastikan semua file model ada di direktori yang benar"
        }), 500

# Wrapper untuk Vercel
def vercel_handler(request):
    return handle_request(app, request)