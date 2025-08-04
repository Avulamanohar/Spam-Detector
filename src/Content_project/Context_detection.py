# predict_message.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import joblib
import re
from Content_project.preprocess import preprocess_text

BASE_DIR= os.path.dirname(__file__)




def predict_message(model, vectorizer, message):
    processed = preprocess_text(message)
    vectorized = vectorizer.transform([processed])
    prediction = model.predict(vectorized)[0]
    return " Ham" if prediction == 1 else "Spam"

# Load model and vectorizer
model = joblib.load(os.path.join(BASE_DIR,'context_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR,'context_vectorizer.pkl'))

# At the end of Context_detection.py

def classify_context(text):
    preprocessed = preprocess_text(text)
    vect = vectorizer.transform([preprocessed])
    return model.predict(vect)[0]

print(classify_context("hi jknsdnfklnsdklf  emanna na you this  good."))