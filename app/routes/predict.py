from fastapi import APIRouter, File, UploadFile
from app.services.image_service import preprocess_image, validate_image
import os

router = APIRouter()

# 🔥 Lazy load
model_service = None

def get_model_service():
    global model_service
    if model_service is None:
        print("⏳ Initializing model...")
        from app.services.model_service import ModelService
        model_service = ModelService()
    return model_service


@router.get("/health")
def health():
    try:
        ms = get_model_service()
        return {
            "status": "ok",
            "model_loaded": ms.model is not None,
            "classes": len(ms.class_names)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        print("🚀 Request received")

        ms = get_model_service()

        image_bytes = await file.read()
        print("📦 File size:", len(image_bytes))

        error = validate_image(file.content_type, len(image_bytes), 10)
        if error:
            return {"error": error}

        img_array = preprocess_image(image_bytes)
        print("🧠 Image shape:", img_array.shape)

        result = ms.predict(img_array)
        print("✅ Prediction done")

        return {
            "success": True,
            "data": result
        }

    except Exception as e:
        import traceback
        print("❌ ERROR:", str(e))
        print(traceback.format_exc())
        return {"success": False, "error": str(e)}
