from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from audio_utils import decode_base64_audio, extract_features
from model import predict

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "CoderStorm"   # change later if you want

app = FastAPI(
    title="AI Generated Voice Detection API",
    version="1.0"
)

# -----------------------------
# Request Schema
# -----------------------------
from pydantic import BaseModel, Field

class AudioRequest(BaseModel):
    language: str
    audio_format: str = Field(..., alias="audioFormat")
    audio_base64: str = Field(..., alias="audioBase64")

    class Config:
        allow_population_by_field_name = True

# -----------------------------
# Health Check (optional but safe)
# -----------------------------
@app.get("/")
def root():
    return {"status": "API is running"}

# -----------------------------
# Main Prediction Endpoint
# ----------------------------- 
@app.get("/")
def root():
    return {"status": "API is running"}
@app.post("/predict")
def detect_voice(
    request: AudioRequest,
    x_api_key: str = Header(None)
):
    # 1️⃣ API Key check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # 2️⃣ Decode audio
    audio, sr = decode_base64_audio(request.audio_base64)

    if audio is None:
        # Fallback (never crash)
        return {
            "classification": "HUMAN",
            "confidence_score": 0.5
        }

    # 3️⃣ Feature extraction
    features = extract_features(audio, sr)

    if features is None:
        return {
            "classification": "HUMAN",
            "confidence_score": 0.5
        }

    # 4️⃣ ML prediction
    label, confidence = predict(features)

    # 5️⃣ Final JSON response (STRICT)
    return {
        "classification": label,
        "confidence_score": confidence
    }
