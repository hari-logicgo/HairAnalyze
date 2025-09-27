import io
import base64
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from bson import ObjectId
import gridfs
from gradio_client import Client, handle_file
import os

# -------------------------------
# Config
# -------------------------------
MONGO_URI = os.getenv("MONGODB_URL") 
DB_NAME = "hair_analyzer_db"
BEARER_TOKEN = os.getenv("API_KEY")

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

# Hugging Face Space
HF_SPACE = "LogicGoInfotechSpaces/hairanalyzer"
HF_TOKEN = None  # set your HF token if private
client = Client(HF_SPACE, hf_token=HF_TOKEN)

# -------------------------------
# Auth
# -------------------------------
security = HTTPBearer()

def check_auth(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != BEARER_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")

# -------------------------------
# Upload endpoint
# -------------------------------
@app.post("/upload")
def upload_image(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    try:
        file_bytes = file.file.read()
        file_base64 = base64.b64encode(file_bytes).decode("utf-8")

        # Store in MongoDB
        file_id = fs.put(file_bytes, filename=file.filename, contentType=file.content_type, base64=file_base64)

        return {"id": str(file_id), "filename": file.filename}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# -------------------------------
# Analyze endpoint
# -------------------------------
@app.get("/analyze/{image_id}")
def analyze_image(image_id: str, auth: bool = Depends(check_auth)):
    try:
        # Fetch image from GridFS
        grid_out = fs.get(ObjectId(image_id))
        img_bytes = grid_out.read()

        # Save to temp file in /tmp for Gradio client
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp") as tmp:
            tmp.write(img_bytes)
            tmp.flush()

            # Call HF Gradio Space
            result = client.predict(
                img=handle_file(tmp.name),
                api_name="/predict"
            )

        return JSONResponse(content={
            "hair_type": result[0],
            "face_shape": result[1],
            "gender": result[2]
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# -------------------------------
# Health endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
