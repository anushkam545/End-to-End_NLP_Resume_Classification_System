# ---------------------------------------------------------------- 
# Resume Job Category Classifier - Machine Learning Model Training 
# -----------------------------------------------------=----------

# Import Required Libraries
import pandas as pd
import pickle                # for saving the trained models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from feature_engineering import prepare_features

# getTF-IDF features and labels
X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf, label_encoder = prepare_features()

# Trainng model functions for Logistic Regression and Support Vector Machine (SVM) models

def train_models(model):
    # fit the model
    model.fit(X_train_tfidf, y_train)

    # predict categories
    y_pred = model.predict(X_test_tfidf)

    # evaluate the model
    print(f"{model.__class__.__name__} Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Train Logistic Regression Model
# -------------------------------

# initialize the model
lr_model = LogisticRegression(max_iter=1000)
train_models(lr_model)

# Train Support Vector Machine (SVM) Model
# ---------------------------------------

# initialize the model
svm_model = LinearSVC(max_iter=1000)
train_models(svm_model)

# compare the two models and save the best one 
print("Comparing Models...")
print("\nLogistic Regression Accuracy:")
print(accuracy_score(y_test, lr_model.predict(X_test_tfidf)))
print("\nSupport Vector Machine Accuracy:")
print(accuracy_score(y_test, svm_model.predict(X_test_tfidf)))

# Save the best model (Support Vector Machine performs better)
with open('ml_model.pkl', 'wb') as f:
    pickle.dump(svm_model, f)
print("Best model (Support Vector Machine) saved as 'ml_model.pkl'")

# save TF-IDF vectorizer for future use in prediction
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)
print("TF-IDF vectorizer saved as 'tfidf_vectorizer.pkl'")