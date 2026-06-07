# -------------------------------------------------------------
# Resume Job Category Classifier - Deep Learning Model Training 
# -----------------------------------------------------=-------

# Import Required Libraries

# pip install transformers

import pandas as pd
import pickle                # for saving the trained models
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
from nltk.tokenize import word_tokenize
from transformers import (DistilBertTokenizer,
                          DistilBertForSequenceClassification)
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('preprocessed_resume_data.csv')
df.head()

# Encode Target Labels
label_encoder = LabelEncoder()
df["encoded_job_position"] = label_encoder.fit_transform(df["job_position"])

# save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Label encoder saved successfully!")

# features and labels
X = df["cleaned_text"]              # Features (text data)
y = df["encoded_job_position"]       # Target (encoded job positions)
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42 )

# ----------
# LSTM Model
# ----------

# Split text into words
train_tokens = X_train.apply( lambda x: x.split() )
test_tokens = X_test.apply( lambda x: x.split() )

# Build Vocabulary
vocab = {}
for tokens in train_tokens:
    for word in tokens:
        if word not in vocab:
            vocab[word] = len(vocab) + 1  # Start indexing from 1

print("Vocabulary Size:")
print(len(vocab))

# convert tokens into integer sequences
def text_to_sequence(tokens):
    return [vocab.get(word, 0) for word in tokens]  # Use 0 for unknown words
train_sequences = train_tokens.apply(text_to_sequence)
test_sequences = test_tokens.apply(text_to_sequence)

# create padded sequences
max_length= 100  # maximum sequence length
def pad_sequence(sequence):
    if len(sequence) < max_length:
        return sequence + [0] * (max_length - len(sequence))  # Pad with 0s
    else:
        return sequence[:max_length]  # Truncate if longer than max_length
padded_train_sequences = train_sequences.apply(pad_sequence)
padded_test_sequences = test_sequences.apply(pad_sequence)

# Combine padded sequences to match the original dataframe length
all_padded_sequences = pd.concat([padded_train_sequences, padded_test_sequences]).sort_index()

# Convert to PyTorch Tensors using the full aligned dataset
X = torch.tensor(all_padded_sequences.tolist(), dtype=torch.long)
y = torch.tensor(df["encoded_job_position"].values, dtype=torch.long)

# Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# initialize LSTM model
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        hidden = hidden.squeeze(0)
        predictions = self.fc(hidden)
        return predictions

# Initialize Model, Loss Function, and Optimizer
vocab_size = len(vocab) + 1  # +1 for padding token
embedding_dim = 128
hidden_dim = 128
output_dim = len(label_encoder.classes_)  # Number of job categories

model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

# evaluation mode
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        predictions = model(X_batch)
        _, predicted_labels = torch.max(predictions, 1)
        total += y_batch.size(0)
        correct += (predicted_labels == y_batch).sum().item()

# accuracy calculation
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# ----------
# BERT Model
# ----------

# initialize BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# create BERT dataset
class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Use the tokenizer call method directly
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# DistilBERT model initialization
bert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=output_dim)

# fine-tuning BERT model
optimizer = AdamW(bert_model.parameters(), lr=2e-5)

# 1. Prepare BERT DataLoaders
train_texts, test_texts, train_labels, test_labels = train_test_split(df['cleaned_text'].tolist(), df['encoded_job_position'].tolist(), test_size=0.2, random_state=42)

train_bert_dataset = BERTDataset(train_texts, train_labels, tokenizer)
test_bert_dataset = BERTDataset(test_texts, test_labels, tokenizer)

bert_train_loader = DataLoader(train_bert_dataset, batch_size=16, shuffle=True)
bert_test_loader = DataLoader(test_bert_dataset, batch_size=16)

# 2. BERT training loop
num_epochs = 5
bert_model.to('cuda' if torch.cuda.is_available() else 'cpu')

for epoch in range(num_epochs):
    bert_model.train()
    total_loss = 0
    for batch in bert_train_loader:
        optimizer.zero_grad()

        # Move batch to device
        input_ids = batch['input_ids'].to(bert_model.device)
        attention_mask = batch['attention_mask'].to(bert_model.device)
        labels = batch['labels'].to(bert_model.device)

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(bert_train_loader):.4f}")

# BERT evaluation
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)
bert_model.eval()

correct, total = 0, 0
with torch.no_grad():
    for batch in bert_test_loader:
        # Move batch to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted_labels = torch.max(outputs.logits, 1)

        total += labels.size(0)
        correct += (predicted_labels == labels).sum().item()

# accuracy calculation
accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")

# Save the fine-tuned BERT model and tokenizer
bert_model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")