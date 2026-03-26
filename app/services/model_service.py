# import tensorflow as tf
# import numpy as np
# import json
# import os
# from dotenv import load_dotenv

# load_dotenv()

# class ModelService:
#     def __init__(self):
#         self.model = None
#         self.class_names = []
#         self.disease_info = {}
#         self._load()

#     def _load(self):
#         """Load model, class names and disease info on startup."""
#         print("⏳ Loading AgritechAI model...")

#         # Load Keras model
#         model_path = os.getenv("MODEL_PATH", "model/plant_disease_model.keras")
#         self.model = tf.keras.models.load_model(model_path)
#         print(f"✅ Model loaded from: {model_path}")

#         # Load class names
#         class_path = os.getenv("CLASS_NAMES_PATH", "model/class_names.json")
#         with open(class_path, 'r') as f:
#             self.class_names = json.load(f)
#         print(f"✅ Classes loaded: {len(self.class_names)}")

#         # Load disease info
#         info_path = os.getenv("DISEASE_INFO_PATH", "app/data/disease_info.json")
#         with open(info_path, 'r') as f:
#             self.disease_info = json.load(f)
#         print(f"✅ Disease info loaded!")

#     def get_disease_info(self, class_name: str) -> dict:
#         """Get disease info — exact match, healthy fallback, or generic."""
#         if class_name in self.disease_info:
#             return self.disease_info[class_name]
#         if "healthy" in class_name.lower():
#             return {
#                 "display_name": "Healthy Plant",
#                 "crop": class_name.split("___")[0],
#                 "severity": "none",
#                 "symptoms": "No disease symptoms detected.",
#                 "treatment": "No treatment needed.",
#                 "prevention": "Continue good agricultural practices."
#             }
#         crop = class_name.split("___")[0] if "___" in class_name else "Unknown"
#         disease = class_name.split("___")[1].replace("_", " ") if "___" in class_name else class_name
#         return {
#             "display_name": f"{crop} - {disease}",
#             "crop": crop,
#             "severity": "unknown",
#             "symptoms": "Symptoms not available.",
#             "treatment": "Consult a local agronomist for treatment advice.",
#             "prevention": "Practice crop rotation and good field hygiene."
#         }

#     def predict(self, img_array: np.ndarray, top_k: int = 3) -> dict:
#         """Run inference and return top-k predictions."""
#         threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))

#         predictions = self.model.predict(img_array, verbose=0)[0]
#         top_indices = np.argsort(predictions)[::-1][:top_k]

#         # Top prediction
#         top_idx = top_indices[0]
#         top_class = self.class_names[top_idx]
#         top_conf = float(predictions[top_idx])
#         info = self.get_disease_info(top_class)

#         # Warning if low confidence
#         warning = None
#         if top_conf < threshold:
#             warning = "⚠️ Low confidence. Please upload a clearer leaf image."

#         return {
#             "top_class": top_class,
#             "top_conf": top_conf,
#             "info": info,
#             "warning": warning,
#             "top3": [
#                 {
#                     "class_name": self.class_names[i],
#                     "confidence": round(float(predictions[i]) * 100, 2)
#                 }
#                 for i in top_indices
#             ]
#         }

# # Single instance (loaded once at startup)
# model_service = ModelService()
import tensorflow as tf
import numpy as np
import json
import os
from dotenv import load_dotenv

# 🔥 Fix for Render / CPU issues
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()

class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.disease_info = {}
        self._load()

    def _load(self):
        """Load model, class names and disease info on startup."""
        print("⏳ Loading AgritechAI model...")

        try:
            # ✅ Load Keras model safely
            model_path = os.getenv("MODEL_PATH", "model/plant_disease_model.keras")
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False,     # 🔥 FIX 1
                safe_mode=False    # 🔥 FIX 2 (for .keras)
            )
            print(f"✅ Model loaded from: {model_path}")

        except Exception as e:
            print("❌ Model loading failed!")
            print("Error:", str(e))
            raise e  # stop app if model fails

        try:
            # Load class names
            class_path = os.getenv("CLASS_NAMES_PATH", "model/class_names.json")
            with open(class_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"✅ Classes loaded: {len(self.class_names)}")

        except Exception as e:
            print("❌ Failed to load class names:", str(e))
            raise e

        try:
            # Load disease info
            info_path = os.getenv("DISEASE_INFO_PATH", "app/data/disease_info.json")
            with open(info_path, 'r') as f:
                self.disease_info = json.load(f)
            print(f"✅ Disease info loaded!")

        except Exception as e:
            print("❌ Failed to load disease info:", str(e))
            raise e

    def get_disease_info(self, class_name: str) -> dict:
        """Get disease info — exact match, healthy fallback, or generic."""
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
        """Run inference and return top-k predictions."""
        threshold = float(os.getenv("CONFIDENCE_THRESHOLD", 0.70))

        predictions = self.model.predict(img_array, verbose=0)[0]
        top_indices = np.argsort(predictions)[::-1][:top_k]

        # Top prediction
        top_idx = top_indices[0]
        top_class = self.class_names[top_idx]
        top_conf = float(predictions[top_idx])
        info = self.get_disease_info(top_class)

        # Warning if low confidence
        warning = None
        if top_conf < threshold:
            warning = "⚠️ Low confidence. Please upload a clearer leaf image."

        return {
            "top_class": top_class,
            "top_conf": top_conf,
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

# ✅ Load once at startup
model_service = ModelService()
