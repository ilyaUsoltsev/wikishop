from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pickle
from train import preprocess_text

# Initialize FastAPI app
app = FastAPI(title="Toxic Comment Classifier", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Load model (with lazy loading)
model = None
vectorizer = None


def load_model():
    """Load the latest trained model."""
    global model, vectorizer

    try:
        model = pickle.load(open("models/latest_model.pkl", "rb"))
        vectorizer = pickle.load(open("models/latest_vectorizer.pkl", "rb"))
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please train a model first.")
        raise HTTPException(status_code=500, detail="Model not available")


# Pydantic models for request/response
class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    is_toxic: bool
    confidence: float
    message: str


class FeedbackRequest(BaseModel):
    comment_id: int
    is_correct: bool


# API Endpoints
@app.get("/")
async def serve_frontend():
    """Serve the main frontend page."""
    return FileResponse("frontend/index.html")


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_text(request: ClassifyRequest):
    """Classify a text comment as toxic or not toxic."""

    # Load model if not already loaded
    if model is None or vectorizer is None:
        load_model()

    # Preprocess text
    cleaned_text = preprocess_text(request.text)

    if not cleaned_text.strip():
        raise HTTPException(status_code=400, detail="Empty or invalid text")

    # Vectorize and predict
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    confidence_scores = model.predict_proba(text_vec)[0]
    confidence = float(max(confidence_scores))

    is_toxic = bool(prediction)

    # Create response message
    if is_toxic:
        message = f"⚠️ This comment appears to be toxic (confidence: {confidence:.1%})"
    else:
        message = (
            f"✅ This comment appears to be non-toxic (confidence: {confidence:.1%})"
        )

    return ClassifyResponse(
        is_toxic=is_toxic,
        confidence=confidence,
        message=message,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
