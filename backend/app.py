from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import joblib
import os
import json
from pathlib import Path

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

MODEL_PATH = Path(__file__).parent.parent / "models"

# Load model artifacts
model = None
scaler = None
feature_names = None
label_encoder = None

def load_model():
    global model, scaler, feature_names, label_encoder
    try:
        model = joblib.load(MODEL_PATH / "model.pkl")
        scaler = joblib.load(MODEL_PATH / "scaler.pkl")
        with open(MODEL_PATH / "feature_names.json") as f:
            feature_names = json.load(f)
        label_encoder = joblib.load(MODEL_PATH / "label_encoder.pkl")
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"⚠️  Model not found. Run train.py first. Error: {e}")

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        features = [
            float(data.get("age", 18)),
            float(data.get("studyTime", 2)),
            float(data.get("failures", 0)),
            float(data.get("absences", 0)),
            float(data.get("g1", 10)),
            float(data.get("g2", 10)),
            float(data.get("travelTime", 1)),
            float(data.get("freeTime", 3)),
            float(data.get("goOut", 3)),
            float(data.get("health", 3)),
            int(data.get("internet", 1)),
            int(data.get("paid", 0)),
            int(data.get("activities", 0)),
            int(data.get("nursery", 1)),
            int(data.get("higher", 1)),
            int(data.get("romantic", 0)),
            int(data.get("famrel", 4)),
            int(data.get("schoolsup", 0)),
            int(data.get("famsup", 1)),
        ]

        X = np.array(features).reshape(1, -1)
        X_scaled = scaler.transform(X)
        prediction = model.predict(X_scaled)[0]
        probabilities = model.predict_proba(X_scaled)[0]

        grade_label = label_encoder.inverse_transform([prediction])[0]
        classes = label_encoder.inverse_transform(model.classes_)

        prob_dict = {str(cls): round(float(prob) * 100, 1) 
                     for cls, prob in zip(classes, probabilities)}

        # Compute performance score 0-100
        grade_map = {"A": 90, "B": 75, "C": 60, "D": 45, "F": 20}
        score = grade_map.get(grade_label, 50)

        # Risk level
        if grade_label in ["A", "B"]:
            risk = "Low"
        elif grade_label == "C":
            risk = "Medium"
        else:
            risk = "High"

        # Recommendations
        recommendations = generate_recommendations(data, grade_label)

        return jsonify({
            "grade": grade_label,
            "score": score,
            "risk": risk,
            "probabilities": prob_dict,
            "recommendations": recommendations,
            "confidence": round(float(max(probabilities)) * 100, 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def generate_recommendations(data, grade):
    recs = []
    study = float(data.get("studyTime", 2))
    absences = float(data.get("absences", 0))
    failures = float(data.get("failures", 0))
    internet = int(data.get("internet", 1))
    paid = int(data.get("paid", 0))
    health = float(data.get("health", 3))
    goOut = float(data.get("goOut", 3))

    if study < 2:
        recs.append({"icon": "📚", "text": "Increase weekly study time to at least 5 hours for better outcomes."})
    if absences > 10:
        recs.append({"icon": "📅", "text": "High absenteeism detected. Regular attendance strongly correlates with better grades."})
    if failures > 0:
        recs.append({"icon": "🎯", "text": "Past course failures noted. Consider tutoring or extra support sessions."})
    if not internet:
        recs.append({"icon": "🌐", "text": "Access to the internet for study resources can significantly improve performance."})
    if not paid:
        recs.append({"icon": "💡", "text": "Enrolling in paid supplementary classes may boost understanding."})
    if health < 3:
        recs.append({"icon": "❤️", "text": "Poor health impacts academic focus. Prioritize rest and wellbeing."})
    if goOut > 4:
        recs.append({"icon": "⚖️", "text": "High social activity may reduce study time. Balance is key."})
    if grade in ["A", "B"]:
        recs.append({"icon": "🏆", "text": "Great performance! Consider mentoring peers or joining academic clubs."})

    return recs[:4] if recs else [{"icon": "✅", "text": "Keep up the consistent effort and study habits!"}]


@app.route("/api/stats", methods=["GET"])
def stats():
    """Return model statistics"""
    stats_path = MODEL_PATH / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            return jsonify(json.load(f))
    return jsonify({"accuracy": "N/A", "samples": "N/A"})


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
