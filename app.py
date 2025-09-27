# app.py - Full Hair Analyzer + Haircut API
import io
import base64
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from gradio_client import Client, handle_file

# -------------------------------
# Config
# -------------------------------
MONGO_URI = os.getenv("MONGODB_URL")  # MongoDB URL
DB_NAME = "hair_analyzer_db"
BEARER_TOKEN = os.getenv("API_KEY")   # Set your API key

if not MONGO_URI or not BEARER_TOKEN:
    raise ValueError("MONGODB_URL and API_KEY must be set in environment variables")

# -------------------------------
# MongoDB setup
# -------------------------------
mongo_client = MongoClient(MONGO_URI)
db = mongo_client[DB_NAME]
fs = gridfs.GridFS(db)

# -------------------------------
# FastAPI setup
# -------------------------------
app = FastAPI(title="Hair Analyzer API")

# Hugging Face Spaces
HF_ANALYZE_SPACE = "LogicGoInfotechSpaces/hairanalyzer"
HF_HAIRCUT_SPACE = "LogicGoInfotechSpaces/haircutidentifier"

analyze_client = Client(HF_ANALYZE_SPACE)
haircut_client = Client(HF_HAIRCUT_SPACE)

# -------------------------------
# Auth
# -------------------------------
security = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# -------------------------------
# Upload Endpoint
# -------------------------------
@app.post("/source-image")
def upload_image(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    try:
        file_bytes = file.file.read()
        # Store in GridFS
        file_id = fs.put(file_bytes, filename=file.filename, contentType=file.content_type)
        return {"id": str(file_id), "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# -------------------------------
# Analyze Endpoint
# -------------------------------
@app.get("/analyze/{image_id}")
def analyze_image(image_id: str, auth: bool = Depends(check_auth)):
    try:
        # Fetch image from GridFS
        grid_out = fs.get(ObjectId(image_id))
        img_bytes = grid_out.read()

        # Save to temp file for HF Gradio clients
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp") as tmp:
            tmp.write(img_bytes)
            tmp.flush()

            # Analyze Hair Type, Face Shape, Gender
            analyze_result = analyze_client.predict(
                img=handle_file(tmp.name),
                api_name="/predict"
            )

            # Analyze Haircut
            haircut_result = haircut_client.predict(
                img=handle_file(tmp.name),
                api_name="/predict"
            )

        # Combine results
        response = {
            "hair_type": analyze_result[0],
            "face_shape": analyze_result[1],
            "gender": analyze_result[2],
            "haircut": haircut_result
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# -------------------------------
# Health endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
