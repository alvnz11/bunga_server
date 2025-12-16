"""
FastAPI Server untuk Klasifikasi Bunga
Integrasi dengan Flutter Mobile App
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import modules
from feature_extraction import extract_all_features
from model_loader import ModelLoader
from database import Database

# Initialize FastAPI
app = FastAPI(
    title="Flower Classification API",
    description="API Server untuk klasifikasi gambar bunga menggunakan SVM, KNN, dan Manhattan Distance",
    version="1.0.0"
)

# CORS Middleware untuk Flutter
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get base directory (folder where main.py is located)
BASE_DIR = Path(__file__).resolve().parent

# Folders
UPLOAD_FOLDER = BASE_DIR / "uploads"
MODELS_FOLDER = BASE_DIR / "models"
DATABASE_FOLDER = BASE_DIR / "database"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)
os.makedirs(DATABASE_FOLDER, exist_ok=True)

# Initialize Model Loader & Database
model_loader = ModelLoader(models_dir=str(MODELS_FOLDER))
db = Database(db_path=str(DATABASE_FOLDER / 'flower_classification.db'))

# Accuracy metrics (akan diload saat startup)
accuracy_metrics = {}


@app.on_event("startup")
async def startup_event():
    """Load models saat server startup"""
    print("\n" + "="*70)
    print("🚀 STARTING FLOWER CLASSIFICATION SERVER")
    print("="*70)
    
    # Load models
    success = model_loader.load_models()
    
    if not success:
        print("\n⚠ Warning: Some models failed to load")
        print("Please ensure all model files are in the 'models/' directory:")
        print("  - svm_model.pkl")
        print("  - knn_model.pkl")
        print("  - scaler.pkl")
        print("  - label_encoder.pkl")
        print("  - accuracy_metrics.json")
    
    # Load accuracy metrics
    acc_path = MODELS_FOLDER / 'accuracy_metrics.json'
    if acc_path.exists():
        with open(acc_path, 'r') as f:
            global accuracy_metrics
            accuracy_metrics = json.load(f)
        print(f"✓ Accuracy metrics loaded from {acc_path}")
    else:
        print(f"⚠ Accuracy metrics not found at {acc_path}")
    
    print("\n✓ Server is ready!")
    print("="*70 + "\n")


@app.get("/")
async def root():
    """Root endpoint - Health check"""
    return {
        "message": "Flower Classification API is running!",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "predict": "/predict",
            "history": "/history",
            "statistics": "/statistics",
            "accuracy": "/accuracy",
            "health": "/health"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    models_loaded = all([
        model_loader.svm_model is not None,
        model_loader.knn_model is not None,
        model_loader.scaler is not None,
        model_loader.label_encoder is not None
    ])
    
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "svm_loaded": model_loader.svm_model is not None,
        "knn_loaded": model_loader.knn_model is not None,
        "scaler_loaded": model_loader.scaler is not None,
        "label_encoder_loaded": model_loader.label_encoder is not None,
        "accuracy_metrics_loaded": len(accuracy_metrics) > 0,
        "database_connected": True,
        "timestamp": datetime.now().isoformat()
    }


@app.post("/predict")
async def predict_flower(file: UploadFile = File(...)):
    """
    Endpoint untuk prediksi klasifikasi bunga
    
    Args:
        file: File gambar bunga (JPEG/PNG)
    
    Returns:
        JSON dengan hasil prediksi dari 3 metode (SVM, KNN, Manhattan)
    """
    try:
        # Validate file extension
        allowed_extensions = ['.jpg', '.jpeg', '.png']
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = str(UPLOAD_FOLDER / filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n📸 Processing image: {filename}")
        
        # Load image
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to load image")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Extract features
        print("🔍 Extracting features...")
        features = extract_all_features(img_rgb)
        
        # Debug: print features
        print(f"DEBUG - Total features: {len(features)}")
        print(f"DEBUG - First 10 weighted features: {features[:10]}")
        print(f"DEBUG - Feature stats - mean: {np.mean(features):.6f}, std: {np.std(features):.6f}")
        
        # Predict with SVM
        print("🤖 Predicting with SVM...")
        prediction_svm, confidence_svm = model_loader.predict_svm(features)
        
        # Predict with KNN
        print("🤖 Predicting with KNN...")
        prediction_knn, confidence_knn = model_loader.predict_knn(features)
        
        # Save to database
        print("💾 Saving to database...")
        db_record = db.save_prediction(
            image_filename=filename,
            image_path=file_path,
            prediction_svm=prediction_svm,
            confidence_svm=confidence_svm,
            prediction_knn=prediction_knn,
            confidence_knn=confidence_knn
        )
        
        # Prepare response
        response = {
            "success": True,
            "id": db_record['id'],
            "filename": filename,
            "predictions": {
                "svm": {
                    "class": prediction_svm,
                    "confidence": round(confidence_svm, 2)
                },
                "knn": {
                    "class": prediction_knn,
                    "confidence": round(confidence_knn, 2)
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        print(f"✅ Prediction completed: SVM={prediction_svm} ({confidence_svm:.1f}%), KNN={prediction_knn} ({confidence_knn:.1f}%)")
        
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.get("/history")
async def get_history(limit: int = 50):
    """
    Get prediction history
    
    Args:
        limit: Maximum number of records to return (default: 50)
    """
    try:
        history = db.get_all_predictions(limit=limit)
        return {
            "success": True,
            "count": len(history),
            "data": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/history/{prediction_id}")
async def get_prediction_by_id(prediction_id: int):
    """Get single prediction by ID"""
    try:
        prediction = db.get_prediction_by_id(prediction_id)
        if not prediction:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {
            "success": True,
            "data": prediction
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/statistics")
async def get_statistics():
    """Get statistics dari prediction history"""
    try:
        stats = db.get_statistics()
        return {
            "success": True,
            "data": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.delete("/history/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """Delete prediction by ID"""
    try:
        success = db.delete_prediction(prediction_id)
        if not success:
            raise HTTPException(status_code=404, detail="Prediction not found")
        
        return {
            "success": True,
            "message": "Prediction deleted successfully"
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")


@app.get("/classes")
async def get_classes():
    """Get available flower classes"""
    try:
        classes = model_loader.get_class_names()
        return {
            "success": True,
            "classes": classes
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/accuracy")
async def get_accuracy():
    """Get model accuracy metrics"""
    try:
        if not accuracy_metrics:
            raise HTTPException(status_code=404, detail="Accuracy metrics not loaded")
        
        return {
            "success": True,
            "data": accuracy_metrics
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    )