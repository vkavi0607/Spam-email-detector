# src/preprocessing.py

import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure NLTK data is downloaded
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Cleans and preprocesses a single text message.
    """
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    
    # 3. Tokenization and remove stopwords
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    
    # 4. Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # 5. Join back into a string
    return ' '.join(tokens)