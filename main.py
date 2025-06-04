from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib

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
threshold = None


def load_model():
    """Load the latest trained model."""
    global model, threshold

    try:
        model_data = joblib.load("models/latest_model.pkl")
        model = model_data["model"]
        threshold = model_data["threshold"]
        print("Model loaded successfully!")
    except FileNotFoundError:
        print("Model files not found. Please train a model first.")
        raise HTTPException(status_code=500, detail="Model not available")


# Pydantic models for request/response
class ClassifyRequest(BaseModel):
    text: str


class ClassifyResponse(BaseModel):
    is_toxic: bool
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
    if model is None or threshold is None:
        load_model()

    # Preprocess text
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Empty or invalid text")

    # if not english - throw error
    if not text.isascii():
        raise HTTPException(
            status_code=400, detail="Text must be in English (ASCII characters only)"
        )

    # Vectorize and predict
    proba = model.predict_proba([text])[:, 1]

    is_toxic = proba[0] > threshold
    print(f"Predicted probability: {proba[0]:.4f}, Threshold: {threshold:.4f}")

    # Create response message
    if is_toxic:
        message = f"⚠️ This comment appears to be toxic"
    else:
        message = f"✅ This comment appears to be non-toxic"

    return ClassifyResponse(
        is_toxic=is_toxic,
        message=message,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
