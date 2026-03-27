import os
import requests
import numpy as np
import json
from dotenv import load_dotenv

# ✅ Use TensorFlow Keras (IMPORTANT FIX)
from tensorflow import keras

load_dotenv()

# 🔥 Hugging Face Model URL
MODEL_URL = "https://huggingface.co/vksharma7711/plant-disease-model/resolve/main/plant_disease_model.keras"
MODEL_PATH = "model/plant_disease_model.keras"


def download_model():
    """Download model if missing or corrupted"""

    try:
        # ✅ Check existence + size
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            print("⬇️ Downloading model from Hugging Face...")

            os.makedirs("model", exist_ok=True)

            response = requests.get(MODEL_URL, stream=True, timeout=120)

            if response.status_code == 200:
                with open(MODEL_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)

                size = os.path.getsize(MODEL_PATH)
                print(f"✅ Model downloaded successfully ({size} bytes)")

                # ❗ Safety check
                if size < 1_000_000:
                    raise Exception("❌ Downloaded file is corrupted (too small)")
            else:
                raise Exception(f"❌ Failed to download model: {response.status_code}")

        else:
            print("✅ Model already exists and is valid")

    except Exception as e:
        print("❌ Model download failed:", str(e))
        raise


class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.disease_info = {}
        self._load()

    def _load(self):
        print("⏳ Initializing model service...")

        # ✅ Step 1: Download model
        download_model()

        # ✅ Debug info
        print("MODEL PATH:", MODEL_PATH)
        print("EXISTS:", os.path.exists(MODEL_PATH))
        print("SIZE:", os.path.getsize(MODEL_PATH))

        # ❗ Final safety check
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            raise Exception("❌ Model file missing or corrupted")

        # ✅ Step 2: Load model
        print("⏳ Loading model into memory...")
        self.model = keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        print("✅ Model loaded successfully")

        # ✅ Step 3: Load class names
        class_path = os.getenv("CLASS_NAMES_PATH", "model/class_names.json")
        with open(class_path, 'r') as f:
            self.class_names = json.load(f)

        print(f"✅ Classes loaded: {len(self.class_names)}")

        # ✅ Step 4: Load disease info
        info_path = os.getenv("DISEASE_INFO_PATH", "app/data/disease_info.json")
        with open(info_path, 'r') as f:
            self.disease_info = json.load(f)

        print("✅ Disease info loaded!")

    def get_disease_info(self, class_name: str) -> dict:
        """Return disease info with fallback"""

        if class_name in self.disease_info:
            return self.disease_info[class_name]

        if "healthy" in class_name.lower():
            return {
                "display_name": "Healthy Plant",
                "crop": class_name.split("___")[0],
                "severity": "none",
                "symptoms": "No disease symptoms detected.",
                "treatment": "No treatment needed.",
                "prevention": "Continue good agricultural practices."
            }

        crop = class_name.split("___")[0] if "___" in class_name else "Unknown"
        disease = class_name.split("___")[1].replace("_", " ") if "___" in class_name else class_name

        return {
            "display_name": f"{crop} - {disease}",
            "crop": crop,
            "severity": "unknown",
            "symptoms": "Symptoms not available.",
            "treatment": "Consult a local agronomist for treatment advice.",
            "prevention": "Practice crop rotation and good field hygiene."
        }

    def predict(self, img_array: np.ndarray, top_k: int = 3) -> dict:
        """Run prediction"""

        if self.model is None:
            raise Exception("❌ Model not loaded")

        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))

        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:top_k]

        top_idx = top_indices[0]
        top_class = self.class_names[top_idx]
        top_conf = float(predictions[top_idx])
        info = self.get_disease_info(top_class)

        warning = None
        if top_conf < threshold:
            warning = "⚠️ Low confidence. Please upload a clearer leaf image."

        return {
            "top_class": top_class,
            "top_conf": round(top_conf * 100, 2),
            "info": info,
            "warning": warning,
            "top3": [
                {
                    "class_name": self.class_names[i],
                    "confidence": round(float(predictions[i]) * 100, 2)
                }
                for i in top_indices
            ]
        }


# ✅ Singleton instance
model_service = ModelService()
