# src/predict.py

import pickle
from src.config import MODEL_PATH, VECTORIZER_PATH
from src.preprocessing import preprocess_text

# Load the artifacts once when the module is loaded
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Error: Model or vectorizer not found. Please run train.py first.")
    model = None
    vectorizer = None

def predict_spam(message: str) -> str:
    """
    Predicts if a new message is spam or ham.
    
    Args:
        message (str): The input message string.
        
    Returns:
        str: 'Spam' or 'Ham'.
    """
    if not model or not vectorizer:
        return "Error: Model not loaded."
        
    # 1. Preprocess the new message
    cleaned_message = preprocess_text(message)
    
    # 2. Vectorize the message
    message_tfidf = vectorizer.transform([cleaned_message])
    
    # 3. Predict
    prediction = model.predict(message_tfidf)
    
    # 4. Return the result
    return "Spam" if prediction[0] == 1 else "Ham"