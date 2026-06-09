# 📄 End-to-End NLP Resume Classification System

An end-to-end NLP pipeline that automatically classifies resumes into job categories using TF-IDF, classical Machine Learning, PyTorch, and Hugging Face Transformers (DistilBERT).

---

## 🚀 Features

- Upload or paste resume text for instant job category prediction
- Fine-tuned **DistilBERT** model for high-accuracy classification
- Classical ML baseline with **TF-IDF + Scikit-learn**
- **Streamlit** web UI for interactive predictions
- **FastAPI** REST endpoint for programmatic access
- Confidence score displayed alongside predicted category

---

## 🗂️ Project Structure

```
End-to-End_NLP_Resume_Classification_System/
├── data/               # Raw and processed datasets
├── models/             # Saved model weights and artifacts
├── notebook/           # EDA and training notebooks
├── src/                # Core source code (preprocessing, training, evaluation)
├── app.py              # Streamlit web application
├── api.py              # FastAPI REST API
├── prediction.py       # Inference logic
├── requirements.txt    # Python dependencies
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.x |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, WordCloud |
| Classical ML | Scikit-learn, SciPy |
| NLP | NLTK, TF-IDF |
| Deep Learning | PyTorch, TorchVision, TorchAudio |
| Transformers | Hugging Face Transformers, DistilBERT |
| Web UI | Streamlit |
| REST API | FastAPI, Uvicorn |
| Cloud | boto3 (AWS, optional) |

---

## ⚙️ Installation

```bash
git clone https://github.com/anushkam545/End-to-End_NLP_Resume_Classification_System.git
cd End-to-End_NLP_Resume_Classification_System
pip install -r requirements.txt
```

---

## 🖥️ Usage

### Streamlit App

```bash
streamlit run app.py
```

Open `http://localhost:8501` → Upload or paste resume text → Click **Predict Category**.

### FastAPI

```bash
uvicorn api:app --reload
```

API available at `http://localhost:8000`.

---

## 🔍 How It Works

1. **Preprocessing** — Resume text is cleaned and tokenized using NLTK.
2. **Feature Extraction** — TF-IDF vectors (classical) or DistilBERT tokenizer (deep learning).
3. **Model Inference** — Fine-tuned DistilBERT predicts the job category with a confidence score.
4. **Output** — Predicted category + confidence displayed in UI or returned via API.

---

## 📦 Dependencies

Key packages (see `requirements.txt` for pinned versions):

- `torch`, `torchvision`, `torchaudio`
- `transformers`, `tokenizers`, `safetensors`
- `scikit-learn`, `nltk`
- `streamlit`
- `fastapi`, `uvicorn`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

---

 

 
