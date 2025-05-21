from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load model and vectorizer (make sure these .pkl files are in the same folder)
clf_persuasion = joblib.load("clf_persuasion.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET"])
def home():
    return "Flask API is running locally!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")
        features = vectorizer.transform([text])
        prediction = clf_persuasion.predict(features)[0]
        return jsonify({"prediction": prediction})
    except Exception as e:
        app.logger.error(f"Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
