from pydantic import BaseModel
from typing import Optional, List

class DiseaseInfo(BaseModel):
    display_name: str
    crop: str
    severity: str
    symptoms: str
    treatment: str
    prevention: str

class TopPrediction(BaseModel):
    class_name: str
    display_name: str
    confidence: float
    is_healthy: bool
    severity: str
    symptoms: str
    treatment: str
    prevention: str
    warning: Optional[str] = None

class Top3Item(BaseModel):
    class_name: str
    confidence: float

class PredictResponse(BaseModel):
    success: bool
    prediction: TopPrediction
    top3: List[Top3Item]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    total_classes: int
    accuracy: str
    version: str