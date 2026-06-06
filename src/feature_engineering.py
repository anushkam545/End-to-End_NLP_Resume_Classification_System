# ----------------------------------------------------- 
# Resume Job Category Classifier - Feature Engineering 
# -----------------------------------------------------

# Import Required Libraries
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def prepare_features():
    # load dataset
    df = pd.read_csv('data/preprocessed_resume_data.csv')

    # Encode Target Labels
    label_encoder = LabelEncoder()
    df["encoded_job_position"] = label_encoder.fit_transform(df["job_position"])

    # Split Data into Features and Target
    X = df["cleaned_text"]              # Features (text data)
    y = df["encoded_job_position"]       # Target (encoded job positions)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y, 
                                                        test_size=0.2,
                                                        random_state=42)

    # Text Vectorization using TF-IDF
    tfidf = TfidfVectorizer(max_features=5000)  # Limit to top 5000 features
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    return ( X_train_tfidf,
             X_test_tfidf,
             y_train,
             y_test,
             tfidf,
             label_encoder )

