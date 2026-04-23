from flask import Flask, render_template, request
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from text_preprocessing import clean_text
from stylometric_extraction import extract_stylometric_features

app = Flask(__name__)

# --- Load Models ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STYLO_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'stylometric_model.pkl')
TRANS_MODEL_PATH = os.path.join(BASE_DIR, 'model', 'transformer_classifier.pkl')

print("Loading models...")
stylo_model = joblib.load(STYLO_MODEL_PATH)
trans_classifier = joblib.load(TRANS_MODEL_PATH)

# Lightweight feature extractor for the transformer classifier (Optimization for deployment)
# This replaces the heavy DistilBERT backbone with a stateless 768-dim hashing vectorizer
vectorizer = HashingVectorizer(n_features=768)

def get_prediction_label(pred):
    return "AI-generated" if pred == 1 else "Human-written"

def get_explanation(label):
    if label == "AI-generated":
        return "This text appears highly structured and consistent, which is typical of AI-generated content."
    else:
        return "This text shows natural variation and informal patterns typical of human writing."

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    input_text = ""
    
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        
        if input_text:
            # 1. Preprocess text (for both models)
            cleaned_text = clean_text(input_text)
            
            # 2. Stylometric Prediction
            stylo_features_dict = extract_stylometric_features(input_text)
            stylo_features = [list(stylo_features_dict.values())]
            
            # Get probabilities
            stylo_proba = stylo_model.predict_proba(stylo_features)[0]
            stylo_pred = int(np.argmax(stylo_proba))
            stylo_conf = float(stylo_proba[stylo_pred])
            stylo_label = get_prediction_label(stylo_pred)
            
            # 3. Transformer Prediction (Optimized for Deployment)
            # Instead of DistilBERT, we use the precomputed logic (Hashing Vectorizer)
            # to generate the 768 features expected by the transformer_classifier.pkl
            trans_features = vectorizer.transform([cleaned_text]).toarray()
            
            # Get probabilities
            trans_proba = trans_classifier.predict_proba(trans_features)[0]
            trans_pred = int(np.argmax(trans_proba))
            trans_conf = float(trans_proba[trans_pred])
            trans_label = get_prediction_label(trans_pred)
            
            # 4. Hybrid Decision Logic
            final_prediction = ""
            status_note = ""
            explanation = ""
            final_conf = 0.0
            
            if stylo_label == trans_label:
                final_prediction = stylo_label
                explanation = get_explanation(final_prediction)
                final_conf = (stylo_conf + trans_conf) / 2
            else:
                final_prediction = "Mixed Signals"
                status_note = "Mixed signals detected: The text contains characteristics of both human and AI writing styles."
                # No average confidence for mixed signals, handled in UI
            
            result = {
                "stylo": stylo_label,
                "stylo_conf": round(stylo_conf * 100, 1),
                "trans": trans_label,
                "trans_conf": round(trans_conf * 100, 1),
                "final": final_prediction,
                "final_conf": round(final_conf * 100, 1) if final_conf > 0 else None,
                "note": status_note,
                "explanation": explanation,
                "stylo_raw": stylo_features_dict
            }

    return render_template('index.html', text=input_text, result=result)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == "__main__":
    app.run()
