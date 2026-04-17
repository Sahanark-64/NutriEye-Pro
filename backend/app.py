"""
app.py - NutriEye Flask Backend
Endpoints:
  POST /predict  - accepts image file, returns food + nutrition
  POST /webcam   - accepts base64 image from webcam, returns food + nutrition
  POST /bmi      - calculates BMI from weight/height
  GET  /foods    - returns list of all foods in the nutrition database
"""

import os
import base64
import tempfile

import cv2
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

from model import predict_food

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)

# Load nutrition database once at startup
CSV_PATH = os.path.join(os.path.dirname(__file__), "nutrition.csv")
df = pd.read_csv(CSV_PATH)
df["food"] = df["food"].str.lower().str.strip()


def get_nutrition(food_label: str) -> dict:
    """Look up nutrition data for a food label."""
    food_label = food_label.lower().strip().replace(" ", "_")
    row = df[df["food"] == food_label]
    if not row.empty:
        return row.iloc[0].to_dict()
    # Try partial match
    partial = df[df["food"].str.contains(food_label, na=False)]
    if not partial.empty:
        return partial.iloc[0].to_dict()
    return {
        "food": food_label,
        "calories": "N/A",
        "protein": "N/A",
        "carbs": "N/A",
        "fat": "N/A",
        "vitamins": "N/A"
    }


@app.route("/predict", methods=["POST"])
def predict():
    """Accept an uploaded image file and return nutrition info."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to a temp file
    suffix = os.path.splitext(file.filename)[-1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        result = predict_food(tmp_path)
        nutrition = get_nutrition(result["food"])
        nutrition["confidence"] = result["confidence"]
        return jsonify(nutrition)
    finally:
        os.unlink(tmp_path)


@app.route("/webcam", methods=["POST"])
def webcam():
    """Accept a base64-encoded image from the webcam and return nutrition info."""
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image data provided"}), 400

    # Decode base64 → numpy array → save as temp file
    img_data = data["image"]
    if "," in img_data:
        img_data = img_data.split(",")[1]  # strip data:image/jpeg;base64,

    img_bytes = base64.b64decode(img_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, frame)
        tmp_path = tmp.name

    try:
        result = predict_food(tmp_path)
        nutrition = get_nutrition(result["food"])
        nutrition["confidence"] = result["confidence"]
        return jsonify(nutrition)
    finally:
        os.unlink(tmp_path)


@app.route("/bmi", methods=["POST"])
def bmi():
    """Calculate BMI and return category."""
    data = request.get_json()
    try:
        weight = float(data["weight"])
        height = float(data["height"]) / 100  # cm → m
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Provide weight (kg) and height (cm)"}), 400

    bmi_value = round(weight / (height ** 2), 1)

    if bmi_value < 18.5:
        category = "Underweight"
        color = "#3498db"
    elif bmi_value < 25:
        category = "Normal weight"
        color = "#2ecc71"
    elif bmi_value < 30:
        category = "Overweight"
        color = "#f39c12"
    else:
        category = "Obese"
        color = "#e74c3c"

    return jsonify({"bmi": bmi_value, "category": category, "color": color})


@app.route("/")
def index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/foods", methods=["GET"])
def foods():
    """Return all food names in the nutrition database."""
    return jsonify({"foods": df["food"].tolist()})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
