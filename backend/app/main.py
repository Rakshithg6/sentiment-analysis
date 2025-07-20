from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline

# Load Hugging Face sentiment analysis pipeline (default: distilbert-base-uncased-finetuned-sst-2-english)
sentiment_pipeline = pipeline("sentiment-analysis")

app = FastAPI(title="Electronix AI Sentiment API")

# Set up CORS
origins = [
    "*",  # Allow all origins for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        result = sentiment_pipeline(req.text)
        label = result[0]["label"].lower()
        score = float(result[0]["score"])
        return PredictResponse(label=label, score=score)
    except Exception as e:
        print(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

