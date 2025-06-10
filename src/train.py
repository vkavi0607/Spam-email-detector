# src/train.py

import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Import from our custom modules
from config import DATA_PATH, MODEL_PATH, VECTORIZER_PATH, TFIDF_MAX_FEATURES
from preprocessing import preprocess_text

def train_model():
    """Trains the spam detection model and saves the artifacts."""
    
    print("Starting model training process...")

    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, sep='\t', header=None, names=['label', 'message'])
    
    # 2. Preprocess Data
    print("Preprocessing text data...")
    df['cleaned_message'] = df['message'].apply(preprocess_text)
    
    # 3. Feature Engineering (TF-IDF)
    print("Creating TF-IDF features...")
    tfidf_vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    
    X = tfidf_vectorizer.fit_transform(df['cleaned_message'])
    y = df['label'].map({'ham': 0, 'spam': 1})
    
    # 4. Train Model
    print("Training the Multinomial Naive Bayes model...")
    model = MultinomialNB()
    model.fit(X, y)
    
    # 5. Evaluate (on the whole dataset for simplicity here)
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"\nTraining complete. Model accuracy on training data: {accuracy:.4f}")
    print("\nClassification Report on Training Data:")
    print(classification_report(y, y_pred, target_names=['Ham', 'Spam']))
    
    # 6. Save Artifacts
    print(f"Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"Saving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
        
    print("\nTraining process finished successfully!")

if __name__ == '__main__':
    train_model()