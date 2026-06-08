 # --------------------------------------------
# Resume Job Category Classifier - FastAPI
# --------------------------------------------

# import required libraries

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

from prediction import predict_job_category

# Create FastAPI App
 
app = FastAPI(
    title="Resume Job Category Classifier API",
    description="Predict job category from resume text using DistilBERT",
    version="1.0"
)

# Request Schema
 
class ResumeRequest(BaseModel):
    resume_text: str

# Home Endpoint

@app.get("/")
def home():

    return {
        "message": "Resume Classification API is running successfully"
    }

# Prediction Endpoint

@app.post("/predict")
def predict(request: ResumeRequest):

    category, confidence = predict_job_category(
        request.resume_text
    )

    return {
        "predicted_category": category,
        "confidence_score": round(confidence, 4)
    }
 
# Run API Server

if __name__ == "__main__":

    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

