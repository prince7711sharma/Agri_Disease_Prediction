from fastapi import APIRouter, File, UploadFile
from app.services.model_service import model_service
from app.services.image_service import preprocess_image, validate_image
from app.schemas.response import PredictResponse, HealthResponse
import os

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok ✅",
        "model_loaded": model_service.model is not None,
        "total_classes": len(model_service.class_names),
        "accuracy": "97.78%",
        "version": os.getenv("APP_VERSION", "1.0.0")
    }

@router.get("/classes")
def get_classes():
    return {
        "total": len(model_service.class_names),
        "classes": model_service.class_names
    }

@router.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):

    # Read file
    image_bytes = await file.read()

    # Validate
    max_mb = int(os.getenv("MAX_FILE_SIZE_MB", 10))
    error = validate_image(file.content_type, len(image_bytes), max_mb)
    if error:
        return {"success": False, "error": error}

    # Preprocess
    img_array = preprocess_image(image_bytes)

    # Predict
    result = model_service.predict(img_array)

    return {
        "success": True,
        "prediction": {
            "class_name": result["top_class"],
            "display_name": result["info"]["display_name"],
            "confidence": round(result["top_conf"] * 100, 2),
            "is_healthy": "healthy" in result["top_class"].lower(),
            "severity": result["info"]["severity"],
            "symptoms": result["info"]["symptoms"],
            "treatment": result["info"]["treatment"],
            "prevention": result["info"]["prevention"],
            "warning": result["warning"]
        },
        "top3": result["top3"]
    }