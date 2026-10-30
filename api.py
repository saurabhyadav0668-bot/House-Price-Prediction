"""
Flask REST API for House Price Prediction
Provides enterprise-grade microservice endpoints for model serving:
- GET  /health          : Service health check
- GET  /metrics         : Benchmark evaluation metrics of all trained models
- POST /predict         : Real-time inference on a single property
- POST /predict_batch   : High-throughput batch inference on a list of properties
"""

import os
import json
from flask import Flask, request, jsonify
from src.predict import HousePricePredictor

app = Flask(__name__)

# Initialize predictor singleton
predictor = HousePricePredictor()

# Load model comparison
base_dir = os.path.dirname(os.path.abspath(__file__))
comp_file = os.path.join(base_dir, "models", "model_comparison.json")
model_metrics = []
if os.path.exists(comp_file):
    with open(comp_file, "r") as f:
        model_metrics = json.load(f)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "House Price Prediction REST API",
        "version": "1.0.0",
        "endpoints": {
            "GET /health": "Microservice health status",
            "GET /metrics": "Evaluation comparison metrics across models",
            "POST /predict": "Predict valuation for a single property",
            "POST /predict_batch": "Predict valuations for multiple properties"
        }
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": predictor.model is not None,
        "model_architecture": predictor.model_name
    }), 200

@app.route("/metrics", methods=["GET"])
def metrics():
    return jsonify({
        "winning_model": predictor.model_name,
        "models_benchmarked": model_metrics
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON payload provided"}), 400
            
        result = predictor.predict(data)
        return jsonify({
            "status": "success",
            "valuation": result
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    try:
        payload = request.get_json(force=True)
        if not payload or not isinstance(payload, list):
            return jsonify({"error": "Payload must be a JSON array of property objects"}), 400
            
        results = [predictor.predict(item) for item in payload]
        return jsonify({
            "status": "success",
            "count": len(results),
            "valuations": results
        }), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
