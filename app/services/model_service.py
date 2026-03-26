import tensorflow as tf
import numpy as np
import json
import os
from dotenv import load_dotenv

# 🔥 Fix logs + CPU issues
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()

class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.disease_info = {}
        self._load()

    def _load(self):
        print("⏳ Loading model...")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # ✅ Define paths properly
        model_path = os.path.join(BASE_DIR, "../../model/plant_disease_model.keras")
        class_path = os.path.join(BASE_DIR, "../../model/class_names.json")
        info_path = os.path.join(BASE_DIR, "../data/disease_info.json")

        print("📂 Model path:", model_path)
        print("📂 Exists:", os.path.exists(model_path))

        # ✅ Load model (NO safe_mode)
        try:
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            print("✅ Model loaded")

        except Exception as e:
            print("❌ Model load failed:", str(e))
            raise e

        # ✅ Load class names
        with open(class_path, 'r') as f:
            self.class_names = json.load(f)

        # ✅ Load disease info
        with open(info_path, 'r') as f:
            self.disease_info = json.load(f)

        print("✅ All resources loaded")

    # def _load(self):
    #     print("⏳ Loading model...")
    #
    #     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #
    #     # ✅ FIXED PATHS (Render safe)
    #     model_path = os.path.join(BASE_DIR, "../../model/plant_disease_model.keras")
    #     class_path = os.path.join(BASE_DIR, "../../model/class_names.json")
    #     info_path = os.path.join(BASE_DIR, "../data/disease_info.json")
    #
    #     print("📂 Model path:", model_path)
    #     print("📂 Exists:", os.path.exists(model_path))
    #
    #     try:
    #         self.model = tf.keras.models.load_model(
    #             model_path,
    #             compile=False,
    #             safe_mode=False
    #         )
    #         print("✅ Model loaded")
    #
    #     except Exception as e:
    #         print("❌ Model load failed:", str(e))
    #         raise e
    #
    #     # Load class names
    #     with open(class_path, 'r') as f:
    #         self.class_names = json.load(f)
    #
    #     # Load disease info
    #     with open(info_path, 'r') as f:
    #         self.disease_info = json.load(f)
    #
    #     print("✅ All resources loaded")

    def get_disease_info(self, class_name: str) -> dict:
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

        return {
            "display_name": class_name,
            "crop": "Unknown",
            "severity": "unknown",
            "symptoms": "Not available",
            "treatment": "Consult expert",
            "prevention": "Good farming practices"
        }

    def predict(self, img_array: np.ndarray):
        predictions = self.model.predict(img_array, verbose=0)[0]
        top_idx = int(np.argmax(predictions))

        return {
            "top_class": self.class_names[top_idx],
            "top_conf": float(predictions[top_idx]),
            "info": self.get_disease_info(self.class_names[top_idx]),
            "top3": [
                {
                    "class_name": self.class_names[i],
                    "confidence": float(predictions[i])
                }
                for i in np.argsort(predictions)[::-1][:3]
            ]

        }

