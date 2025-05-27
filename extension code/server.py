from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sentence_transformers import SentenceTransformer, util
import joblib  # for loading models
import re

import json

# Load your LIWC dictionary
with open("LIWKDictionary.json") as f:
    liwc_dict = json.load(f)

# Word lists
booster_words = [
   "absolute", "absolutely", "accura", "all", "altogether", "always", "apparent",
    "assur", "blatant", "certain", "clear", "clearly", "commit", "commitment", "commits", "committed", "committing",
    "complete", "completed", "completely", "completes", "confidence", "confident", "confidently",
    "correct", "defined", "definite", "definitely", "definitive", "directly", "distinct", "entire",
    "especially", "essential", "ever", "every", "everybod", "everyday", "everyone", "everything", "everytime", "everywhere",
    "evident", "exact", "explicit", "extremely", "fact", "facts", "factual", "forever", "frankly",
    "fundamental", "fundamentalis", "fundamentally", "fundamentals", "guarant", "implicit", "indeed", "inevitab",
    "infallib", "invariab", "irrefu", "must", "namely", "necessari", "necessary", "never",
    "nothing", "nowhere", "obvious", "obviously", "particularly", "perfect", "perfected", "perfecting", "perfection",
    "perfectly", "perfects", "positive", "positively", "positives", "positivi", "precis", "promise", "proof",
    "prove", "proving", "pure", "purely", "pureness", "purest", "purity", "specific", "specifically", "specifics",
    "sure", "total", "totally", "true", "truest", "truly", "truth", "unambigu", "undeniab", "undoubt",
    "unquestion", "visibly", "wholly"
]

hedging_words = [
    "might", "could", "may", "can", "would",
    "possibly", "perhaps", "apparently", "seemingly", "likely",
    "conceivably", "presumably", "reportedly",
    "suggest", "indicate", "appear", "seem", "imply", "speculate", "propose", "estimate",
    "possible", "probable", "potential", "uncertain", "likely"
]

superlatives = [
    "best", "greatest", "most important", "ultimate", "most significant", "highest", "largest", "leading"
]

# Combine LIWC-based categories with custom lists
hedging = (
    liwc_dict.get("54\nTentat", []) +
    liwc_dict.get("12\nAuxverb", []) +
    liwc_dict.get("15\nNegate", [])
)
assertive = (
    liwc_dict.get("55\nCertain", []) +
    liwc_dict.get("82\nAchieve", []) +
    liwc_dict.get("83\nPower", []) +
    liwc_dict.get("84\nReward", []) +
    liwc_dict.get("85\nRisk", []) +
    liwc_dict.get("50\nCogProc", [])
)
pathos = (
    liwc_dict.get("30\nAffect", []) +
    liwc_dict.get("31\nPosemo", []) +
    liwc_dict.get("80\nDrives", [])
)

hedging += hedging_words
assertive += booster_words
pathos += superlatives

# Create sets (lowercased)
hedging_set = set(word.lower() for word in hedging)
assertive_set = set(word.lower() for word in assertive)
pathos_set = set(word.lower() for word in pathos)

# Your final sets
final_hedging = hedging_set
final_assertive = assertive_set
final_pathos = pathos_set


app = Flask(__name__)
CORS(app)

# Load spaCy model once
nlp = spacy.load("en_core_web_sm")

# Load SentenceTransformer and NICHD facts once
model = SentenceTransformer('all-MiniLM-L6-v2')
with open("nichd_filtered_facts.txt", "r", encoding="utf-8") as f:
    nichd_facts = [line.strip() for line in f]
nichd_embeddings = model.encode(nichd_facts, convert_to_tensor=True)

# Load your TF-IDF vectorizer and trained RF classifier
vectorizer = joblib.load('vectorizer.pkl')
clf_persuasion = joblib.load('clf_persuasion.pkl')
clf_accuracy = joblib.load('clf_accuracy.pkl')

def count_liwc_words(doc):
    hedging = sum(1 for token in doc if token.text.lower() in final_hedging)
    assertive = sum(1 for token in doc if token.text.lower() in final_assertive)
    pathos = sum(1 for token in doc if token.text.lower() in final_pathos)
    return pd.Series({
        "hedging_count": hedging,
        "assertive_count": assertive,
        "pathos_count": pathos,
        "word_count": len(doc)
    })

def compute_ethos_logos(doc):
    # Return a pandas Series of ethos and logos scores
    return pd.Series({'ethos_score': 0.03, 'logos_score': 0.04})

def extract_spacy_features(text):
    # Return pandas Series of additional spaCy features
    return pd.Series({'some_spacy_feature': 0.5})

def compute_confidence_score(features):
    # Return a float confidence score based on your logic
    return 0.75

def compute_persuasion_score(features):
    # Return a float persuasion score based on your logic
    return 0.65

def assign_confidence_label_combined(features):
    # Return string label like 'High Confidence', 'Low Confidence'
    return "High Confidence"

def assign_persuasion_label(score):
    # Return string label based on score thresholds
    if score > 0.7:
        return "High"
    elif score < 0.4:
        return "Low"
    else:
        return "Medium"

def get_combined_label(response, persuasion_label, ethos_rate, logos_rate):
    resp_embedding = model.encode(response, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resp_embedding, nichd_embeddings)
    max_score = cosine_scores.max().item()

    if max_score < 0.4:
        accuracy_label = "Possible Misinformation"
        if ethos_rate > 0.02 or logos_rate > 0.02:
            accuracy_label += " (but rhetorically persuasive)"
    elif max_score < 0.6:
        accuracy_label = "Neutral / Unclear"
    else:
        accuracy_label = "Likely aligned with NICHD"

    if persuasion_label == 'High':
        persuasion_label_desc = "Highly Persuasive"
    elif persuasion_label == 'Low':
        persuasion_label_desc = "Low Persuasion"
    else:
        persuasion_label_desc = "Moderately Persuasive"

    return f"{persuasion_label_desc} / {accuracy_label}"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    # spaCy doc
    doc = nlp(text)

    # Extract features
    liwc_feats = count_liwc_words(doc)
    rhetorical_feats = compute_ethos_logos(doc)
    spacy_feats = extract_spacy_features(text)

    # Combine features into one vector/Series
    combined_feats = pd.concat([liwc_feats, rhetorical_feats, spacy_feats])
    combined_feats['pathos_rate'] = combined_feats['pathos_count'] / max(combined_feats['word_count'], 1)

    # Number of expected total features by the model
    expected_total_features = 1011

    # TF-IDF vector
    tfidf_vector = vectorizer.transform([text])

    # Number of features in TF-IDF
    num_tfidf_features = tfidf_vector.shape[1]

    # How many dense features we need
    num_dense_needed = expected_total_features - num_tfidf_features

    # Current dense feature vector
    dense_feats_array = combined_feats.values.reshape(1, -1)

    # If not enough, pad with zeros
    if dense_feats_array.shape[1] < num_dense_needed:
        pad_width = num_dense_needed - dense_feats_array.shape[1]
        dense_feats_array = np.hstack([dense_feats_array, np.zeros((1, pad_width))])
    elif dense_feats_array.shape[1] > num_dense_needed:
        # Truncate if too many
        dense_feats_array = dense_feats_array[:, :num_dense_needed]

    # Combine features
    X = hstack([tfidf_vector, dense_feats_array])

    # Predict label and probabilities with RF model
    pred_label = clf_persuasion.predict(X)[0]
    pred_proba = clf_persuasion.predict_proba(X).max()

    # Compute your confidence and persuasion scores
    confidence_score = compute_confidence_score(combined_feats)
    persuasion_score = compute_persuasion_score(combined_feats)

    # Assign custom labels
    confidence_label = assign_confidence_label_combined(combined_feats)
    persuasion_label = assign_persuasion_label(persuasion_score)

    # Get combined semantic + rhetorical label
    combined_label = get_combined_label(text, persuasion_label, combined_feats['ethos_score'], combined_feats['logos_score'])

    # Return JSON response
    return jsonify({
        "prediction_label": pred_label,
        "confidence_score": confidence_score,
        "confidence_label": confidence_label,
        "persuasion_score": persuasion_score,
        "persuasion_label": persuasion_label,
        "combined_label": combined_label
    })

if __name__ == "__main__":
    app.run(debug=True)
