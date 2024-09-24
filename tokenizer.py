# tokenizer.py
import nltk
from nltk.tokenize import word_tokenize

# Ensure required NLTK resources are downloaded
nltk.download('punkt', quiet=True)

def tokenize_text(text):
    tokens = word_tokenize(text)
    return [word for word in tokens if word.isalpha()]
