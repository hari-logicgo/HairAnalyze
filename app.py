# app.py - Hair Analyzer + Haircut + Hairstyle Swap
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
app = FastAPI(title="Hair Analyzer + Hairstyle Swap API")

# Hugging Face Spaces
HF_ANALYZE_SPACE = "LogicGoInfotechSpaces/hairanalyzer"
HF_HAIRCUT_SPACE = "LogicGoInfotechSpaces/haircutidentifier"
HF_HAIR_SWAP_SPACE = "AIRI-Institute/HairFastGAN"

analyze_client = Client(HF_ANALYZE_SPACE)
haircut_client = Client(HF_HAIRCUT_SPACE)
hair_swap_client = Client(HF_HAIR_SWAP_SPACE)

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
@app.post("/upload")
def upload_image(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    try:
        file_bytes = file.file.read()
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
        grid_out = fs.get(ObjectId(image_id))
        img_bytes = grid_out.read()

        with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp") as tmp:
            tmp.write(img_bytes)
            tmp.flush()

            analyze_result = analyze_client.predict(
                img=handle_file(tmp.name),
                api_name="/predict"
            )
            haircut_result = haircut_client.predict(
                img=handle_file(tmp.name),
                api_name="/predict"
            )

        return JSONResponse(content={
            "hair_type": analyze_result[0],
            "face_shape": analyze_result[1],
            "gender": analyze_result[2],
            "haircut": haircut_result
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# -------------------------------
# Upload Reference Hairstyle Image
# -------------------------------
@app.post("/reference-image")
def upload_reference_image(file: UploadFile = File(...), auth: bool = Depends(check_auth)):
    try:
        file_bytes = file.file.read()
        file_id = fs.put(file_bytes, filename=file.filename, contentType=file.content_type)
        return {"id": str(file_id), "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reference upload failed: {str(e)}")

# -------------------------------
# Hairstyle Swap Endpoint
# -------------------------------
@app.get("/swap/{source_id}/{ref_id}")
def swap_hairstyle(source_id: str, ref_id: str, auth: bool = Depends(check_auth)):
    try:
        # Fetch source and reference images from GridFS
        src_bytes = fs.get(ObjectId(source_id)).read()
        ref_bytes = fs.get(ObjectId(ref_id)).read()

        # Save temp files for HF
        with tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp") as src_tmp, \
             tempfile.NamedTemporaryFile(suffix=".jpg", dir="/tmp") as ref_tmp:

            src_tmp.write(src_bytes)
            src_tmp.flush()
            ref_tmp.write(ref_bytes)
            ref_tmp.flush()

            # Preprocess source and reference
            src_preprocessed = hair_swap_client.predict(
                img=handle_file(src_tmp.name),
                align=["Face", "Shape", "Color"],
                api_name="/resize_inner"
            )
            ref_preprocessed = hair_swap_client.predict(
                img=handle_file(ref_tmp.name),
                align=["Face", "Shape", "Color"],
                api_name="/resize_inner_1"
            )

            # Swap hair
            swap_result = hair_swap_client.predict(
                face=handle_file(src_preprocessed["value"] if isinstance(src_preprocessed, dict) else src_preprocessed),
                shape=handle_file(ref_preprocessed["value"] if isinstance(ref_preprocessed, dict) else ref_preprocessed),
                color=handle_file(ref_preprocessed["value"] if isinstance(ref_preprocessed, dict) else ref_preprocessed),
                blending="Article",
                poisson_iters=0,
                poisson_erosion=15,
                api_name="/swap_hair"
            )

            # Extract actual file path
            swap_file_path = swap_result[0]["value"] if isinstance(swap_result[0], dict) else swap_result[0]

            # Convert to base64
            with open(swap_file_path, "rb") as f:
                swap_base64 = base64.b64encode(f.read()).decode("utf-8")

        return JSONResponse(content={
            "result_image_base64": swap_base64,
            "message": swap_result[1]  # Optional text message from HF
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hair swap failed: {str(e)}")

# -------------------------------
# Health endpoint
# -------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}
