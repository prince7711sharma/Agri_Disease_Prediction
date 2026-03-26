# from fastapi import APIRouter, File, UploadFile
# from app.services.image_service import preprocess_image, validate_image
# from app.schemas.response import PredictResponse, HealthResponse
# import os

# router = APIRouter()

# # 🔥 Lazy load (IMPORTANT)
# model_service = None

# def get_model_service():
#     global model_service
#     if model_service is None:
#         print("⏳ Initializing model service...")
#         from app.services.model_service import ModelService
#         model_service = ModelService()
#     return model_service


# @router.get("/health", response_model=HealthResponse)
# def health():
#     ms = get_model_service()
#     return {
#         "status": "ok ✅",
#         "model_loaded": ms.model is not None,
#         "total_classes": len(ms.class_names),
#         "accuracy": "97.78%",
#         "version": os.getenv("APP_VERSION", "1.0.0")
#     }


# @router.get("/classes")
# def get_classes():
#     ms = get_model_service()
#     return {
#         "total": len(ms.class_names),
#         "classes": ms.class_names
#     }


# @router.post("/predict", response_model=PredictResponse)
# async def predict(file: UploadFile = File(...)):

#     ms = get_model_service()

#     # Read file
#     image_bytes = await file.read()

#     # Validate
#     max_mb = int(os.getenv("MAX_FILE_SIZE_MB", 10))
#     error = validate_image(file.content_type, len(image_bytes), max_mb)
#     if error:
#         return {"success": False, "error": error}

#     # Preprocess
#     img_array = preprocess_image(image_bytes)

#     # Predict
#     result = ms.predict(img_array)

#     return {
#         "success": True,
#         "prediction": {
#             "class_name": result["top_class"],
#             "display_name": result["info"]["display_name"],
#             "confidence": round(result["top_conf"] * 100, 2),
#             "is_healthy": "healthy" in result["top_class"].lower(),
#             "severity": result["info"]["severity"],
#             "symptoms": result["info"]["symptoms"],
#             "treatment": result["info"]["treatment"],
#             "prevention": result["info"]["prevention"],
#             "warning": result["warning"]
#         },
#         "top3": result["top3"]
#     }

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
