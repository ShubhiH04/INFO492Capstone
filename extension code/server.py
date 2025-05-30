from flask import Flask, request, jsonify
from flask_cors import CORS
import spacy
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix
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
    "suggest", "indicate", "appear", "seem", "seems", "imply", "speculate", "propose", "estimate",
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

hedging_set = set(word.lower() for word in hedging)
assertive_set = set(word.lower() for word in assertive)
pathos_set = set(word.lower() for word in pathos)

custom_hedging_set = set(word.lower() for word in hedging_words)
custom_assertive_set = set(word.lower() for word in booster_words)
custom_pathos_set = set(word.lower() for word in superlatives)

# Remove overlaps so no duplicates between LIWC and custom sets
custom_hedging_set -= hedging_set
custom_assertive_set -= assertive_set
custom_pathos_set -= pathos_set

# Final combined sets to use in your feature extraction
final_hedging = hedging_set.union(custom_hedging_set)
final_assertive = assertive_set.union(custom_assertive_set)
final_pathos = pathos_set.union(custom_pathos_set)


# Your final sets
final_hedging = hedging_set
final_assertive = assertive_set
final_pathos = pathos_set

authority_list = [
    "cdc", "who", "mayo clinic", "johns hopkins", "obstetrician", "gynecologist",
    "dr", "doctor", "physician", "clinician", "acog", "health department", "nih"
]

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

def detect_authority(doc):
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PERSON'] and ent.text.lower() in authority_list:
            return 1
    return 0

def detect_citation(doc):
    text = doc.text.lower()
    return int(bool(re.search(r'https?://', text) or re.search(r'(according to|published in|a study by)', text)))

def detect_call_to_action(doc):
    patterns = ['you should', 'consider', 'make sure to', 'talk to your doctor', 'schedule a visit']
    return int(any(p in doc.text.lower() for p in patterns))

def detect_structured_sentences(doc):
    sents = list(doc.sents)
    if not sents:
        return 0
    complete = sum(1 for sent in sents if any(tok.dep_ == 'nsubj' for tok in sent) and any(tok.dep_ == 'ROOT' for tok in sent))
    return int(complete / len(sents) > 0.7)

def compute_ethos_logos(doc):
    ethos = detect_authority(doc) + detect_citation(doc)
    logos = detect_call_to_action(doc) + detect_structured_sentences(doc)
    return pd.Series({
        "ethos_score": ethos,
        "logos_score": logos
    })

def extract_spacy_features(text):
    doc = nlp(text)
    liwc = count_liwc_words(doc)
    rhetorical = compute_ethos_logos(doc)
    return pd.concat([liwc, rhetorical], axis=0)

def compute_confidence_score(row, weights=None):
    if row["word_count"] == 0:
        return 0  # Avoid divide-by-zero
    
    if weights is None:
        weights = {
            "assertive_rate": 0.3,
            "hedging_rate": -0.6
        }
    
    # Basic weighted confidence score
    score = (
        weights["assertive_rate"] * row["assertive_count"] / row["word_count"] +
        weights["hedging_rate"] * row["hedging_count"] / row["word_count"]
    )
    
    # Flag cases where both assertive and hedging are high (conflict)
    if row["assertive_count"] >= 3 and row["hedging_count"] >= 3:
        score = 0.4  # downgrade to medium confidence
    
    return max(0, min(1, round(score, 3)))

def compute_persuasion_score(row, weights=None):
    if weights is None:
        weights = {
            'confidence': 0.4,
            'ethos': 0.2,
            'logos': 0.2,
            'pathos': 0.2
        }

    pathos_rate = row["pathos_count"] / row["word_count"] if row["word_count"] else 0

    persuasion_score = (
        weights['confidence'] * row['confidence_score'] +
        weights['ethos'] * (row['ethos_score'] / 2) +
        weights['logos'] * (row['logos_score'] / 2) +
        weights['pathos'] * pathos_rate
    )

    return max(0, min(1, round(persuasion_score, 3)))

def assign_confidence_label_combined(score):
    # Debug print for tracking scores
    print(f"Confidence score: {score}")
    if score > 0.7:
        return "High Confidence"
    elif score < 0.4:
        return "Low Confidence"
    else:
        return "Medium Confidence"

def assign_persuasion_label(score):
    print(f"Persuasion score: {score}")
    if score > 0.6:
        return "High"
    elif score < 0.4:
        return "Low"
    else:
        return "Medium"

def get_combined_label(response, persuasion_label, ethos_rate, logos_rate):
    resp_embedding = model.encode(response, convert_to_tensor=True)
    cosine_scores = util.cos_sim(resp_embedding, nichd_embeddings)
    max_score = cosine_scores.max().item()
    print(f"Max cosine similarity: {max_score}")

    if max_score < 0.4:
        accuracy_label = "Possible Misinformation"
        # You can adjust this threshold if needed
        if ethos_rate > 0.05 or logos_rate > 0.05:
            accuracy_label += " (but rhetorically persuasive)"
    elif max_score < 0.6:
        accuracy_label = "Unclear"
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
    print("POST /predict called")
    try:
        data = request.get_json(force=True)
        text = data.get("text", "")

        doc = nlp(text)

        # Extract features once via extract_spacy_features (includes liwc + ethos/logos)
        combined_feats = extract_spacy_features(text)

        # Add pathos_rate
        combined_feats['pathos_rate'] = combined_feats['pathos_count'] / max(combined_feats['word_count'], 1)

        # Prepare TF-IDF features
        tfidf_vector = vectorizer.transform([text])
        num_tfidf_features = tfidf_vector.shape[1]

        # Dense features array
        dense_feats_array = combined_feats.values.reshape(1, -1)
        expected_total_features = clf_persuasion.n_features_in_
        num_dense_needed = expected_total_features - num_tfidf_features

        # Pad or truncate dense features
        if dense_feats_array.shape[1] < num_dense_needed:
            pad_width = num_dense_needed - dense_feats_array.shape[1]
            dense_feats_array = np.hstack([dense_feats_array, np.zeros((1, pad_width))])
        elif dense_feats_array.shape[1] > num_dense_needed:
            dense_feats_array = dense_feats_array[:, :num_dense_needed]

        # Final combined feature matrix
        X = hstack([tfidf_vector, dense_feats_array])

        # Model prediction
        pred_label = clf_persuasion.predict(X)[0]
        pred_proba = clf_persuasion.predict_proba(X).max()

        # Compute scores and labels
        confidence_score = compute_confidence_score(combined_feats)
        combined_feats["confidence_score"] = confidence_score
        
        persuasion_score = compute_persuasion_score(combined_feats)
        confidence_label = assign_confidence_label_combined(confidence_score)
        persuasion_label = assign_persuasion_label(persuasion_score)

        combined_label = get_combined_label(
            response=text,
            persuasion_label=persuasion_label,
            ethos_rate=combined_feats['ethos_score'],
            logos_rate=combined_feats['logos_score']
        )

        print("Assertive count:", combined_feats["assertive_count"])
        print("Hedging count:", combined_feats["hedging_count"])
        print("Word count:", combined_feats["word_count"])
        print("Confidence score:", confidence_score)
        print("Max cosine similarity to NICHD facts:", max_score)

        return jsonify({
            "prediction_label": pred_label,
            "prediction_probability": round(pred_proba, 3),
            "confidence_score": confidence_score,
            "confidence_label": confidence_label,
            "persuasion_score": persuasion_score,
            "persuasion_label": persuasion_label,
            "combined_label": combined_label,
            "assertive_rate": round(combined_feats["assertive_count"] / combined_feats["word_count"], 3),
            "hedging_rate": round(combined_feats["hedging_count"] / combined_feats["word_count"], 3)
        })

    except Exception as e:
        import traceback
        print(f"Error in /predict: {e}")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


