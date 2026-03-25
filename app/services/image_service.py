from PIL import Image
import numpy as np
import io

IMG_SIZE = 224

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """
    Load image from bytes, resize to 224x224,
    and return as float32 numpy array with batch dim.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # (1, 224, 224, 3)
    return img_array

def validate_image(content_type: str, file_size: int, max_mb: int = 10) -> str | None:
    """
    Validate image file type and size.
    Returns error message or None if valid.
    """
    allowed_types = ["image/jpeg", "image/png", "image/jpg", "image/webp"]
    if content_type not in allowed_types:
        return "Invalid file type. Please upload JPG, PNG or WEBP."
    if file_size > max_mb * 1024 * 1024:
        return f"File too large. Max size is {max_mb}MB."
    return None