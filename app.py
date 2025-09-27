# app.py - FastAPI application for image storage in MongoDB with base64 encoding

import os
import base64
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from pymongo import MongoClient
from bson.objectid import ObjectId
from typing import Optional

# Initialize FastAPI app
app = FastAPI(
    title="Image API with MongoDB",
    description="API to store and retrieve images as base64 strings in MongoDB. Endpoints are protected with API key authentication."
)

# Security scheme for API key (HTTP Bearer)
security = HTTPBearer()

# Environment variables
MONGODB_URL = os.getenv("MONGODB_URL")  # e.g., mongodb://username:password@host:port/dbname
API_KEY = os.getenv("API_KEY")  # Set this in your environment for authentication

if not MONGODB_URL:
    raise ValueError("MONGODB_URL environment variable is not set.")
if not API_KEY:
    raise ValueError("API_KEY environment variable is not set.")

# MongoDB client
client = MongoClient(MONGODB_URL)
db = client.get_default_database()  # Or specify db name if not in URL
collection = db.images  # Collection to store images

# Pydantic model for image data
class ImageData(BaseModel):
    image_base64: str  # Base64 encoded image string
    filename: Optional[str] = None
    description: Optional[str] = None

class ImageResponse(BaseModel):
    id: str
    image_base64: str
    filename: Optional[str]
    description: Optional[str]

# Dependency for API key authentication
def authenticate(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return True

@app.get("/")
def root():
    return {"message": "Welcome to the Image API. Use /docs for Swagger UI."}

# Endpoint to upload image as base64
@app.post("/images/", response_model=ImageResponse, status_code=201)
def upload_image(image: ImageData, auth: bool = Depends(authenticate)):
    # Insert into MongoDB
    result = collection.insert_one({
        "image_base64": image.image_base64,
        "filename": image.filename,
        "description": image.description
    })
    return {
        "id": str(result.inserted_id),
        "image_base64": image.image_base64,
        "filename": image.filename,
        "description": image.description
    }

# Endpoint to retrieve image by ID
@app.get("/images/{image_id}", response_model=ImageResponse)
def get_image(image_id: str, auth: bool = Depends(authenticate)):
    try:
        obj_id = ObjectId(image_id)
    except:
        raise HTTPException(status_code=400, detail="Invalid image ID format")
    
    image_doc = collection.find_one({"_id": obj_id})
    if not image_doc:
        raise HTTPException(status_code=404, detail="Image not found")
    
    return {
        "id": str(image_doc["_id"]),
        "image_base64": image_doc["image_base64"],
        "filename": image_doc.get("filename"),
        "description": image_doc.get("description")
    }

# Run the app (for local testing; Render will use its own command)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)