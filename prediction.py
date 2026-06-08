# --------------------------------------------
# Resume Job Category Classifier - prediction
# --------------------------------------------

# import required libraries
# prediction on custom input
import pickle
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
from transformers import AutoModelForSequenceClassification

MODEL_PATH = "models/bert_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model.to(device)
model.eval()


def predict_job_category(text):

    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_label = prediction.cpu().item()
    category = label_encoder.inverse_transform(
        [predicted_label]
    )[0]

    return category, confidence.cpu().item()

# test prediction
if __name__ == "__main__":
    sample_text = "Experienced Data Scientist skilled in Python, Machine Learning, Deep Learning, SQL, Data Analysis and NLP."
    category, confidence = predict_job_category(sample_text)
    print(f"Predicted Category: {category}, Confidence: {confidence:.4f}")  