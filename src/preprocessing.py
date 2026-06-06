# ----------------------------------------------------- 
# Resume Job Category Classifier - Text Preprocessing 
# -----------------------------------------------------

# Import Required Libraries

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download Required NLTK Resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# load dataset
df = pd.read_csv('data/final_resume_data.csv')

# Initialize Stopwords and Lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Text Preprocessing Function
def preprocess_text(text):

    # Convert non-string values to string
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = text.strip()
    text = re.sub(r"http\S+|www\S+", "", text)  # Remove URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text)   # Remove special characters
    text = re.sub(r"\d+", "", text)     # Remove numbers

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords
    words = [
        word for word in words
        if word not in stop_words
    ]

    # Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
    ]
    
    # Join tokens back to string
    return ' '.join(words)

# Apply Preprocessing on Text Column
df["cleaned_text"] = df["combined_text"].apply(preprocess_text)

# Display processed text 
print("\nProcessed Text:")
print(df["cleaned_text"].head())

# Keep the required columns only
df = df[["cleaned_text", "job_position"]]

# save preprocessed dataset 
df.to_csv("preprocessed_resume_data.csv", index=False)
print("\nPreprocessed dataset saved to 'preprocessed_resume_data.csv' successfully!")