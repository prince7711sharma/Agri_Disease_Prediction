

import tensorflow as tf
import numpy as np
import json
import os

# 🔥 reduce logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class ModelService:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.disease_info = {}
        self._load()

    def _load(self):
        print("⏳ Loading model...")

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))

        # ✅ Correct paths (Render safe)
        model_path = os.path.join(BASE_DIR, "../../model/plant_disease_model.keras")
        class_path = os.path.join(BASE_DIR, "../../model/class_names.json")
        info_path = os.path.join(BASE_DIR, "../data/disease_info.json")

        print("📂 Model path:", model_path)
        print("📂 Exists:", os.path.exists(model_path))

        # ✅ Load model (NO safe_mode)
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        print("✅ Model loaded")

        # Load class names
        with open(class_path, "r") as f:
            self.class_names = json.load(f)

        # Load disease info
        with open(info_path, "r") as f:
            self.disease_info = json.load(f)

        print("✅ All files loaded")

    def predict(self, img_array: np.ndarray):
        preds = self.model.predict(img_array, verbose=0)[0]
        idx = int(np.argmax(preds))

        return {
            "top_class": self.class_names[idx],
            "top_conf": float(preds[idx]),
            "top3": [
                {
                    "class_name": self.class_names[i],
                    "confidence": float(preds[i])
                }
                for i in np.argsort(preds)[::-1][:3]
            ]
        }
