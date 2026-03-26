from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.predict import router
from dotenv import load_dotenv
import os

load_dotenv()

app = FastAPI(
    title=os.getenv("APP_NAME", "AgritechAI"),
    description="🌿 Plant Disease Detection API — 97.78% Accuracy",
    version=os.getenv("APP_VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(router, prefix="/api/v1", tags=["Plant Disease Detection"])

@app.get("/")
def root():
    return {
        "name": "🌿 AgritechAI Plant Disease Detection API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "accuracy": "97.78%",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict"
    }