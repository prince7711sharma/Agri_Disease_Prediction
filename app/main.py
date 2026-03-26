import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# 🔥 Hide TF logs + CPU fix
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

load_dotenv()

app = FastAPI(
    title=os.getenv("APP_NAME", "AgritechAI"),
    description="🌿 Plant Disease Detection API — 97.78% Accuracy",
    version=os.getenv("APP_VERSION", "1.0.0"),
    docs_url="/docs",
    redoc_url="/redoc"
)

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Root route (works even if model fails)
@app.get("/")
def root():
    return {
        "name": "🌿 AgritechAI Plant Disease Detection API",
        "version": os.getenv("APP_VERSION", "1.0.0"),
        "status": "running",
        "docs": "/docs"
    }

# ✅ Import router AFTER app starts (prevents crash)
try:
    from app.routes.predict import router
    app.include_router(router, prefix="/api/v1", tags=["Plant Disease Detection"])
    print("✅ Routes loaded successfully")

except Exception as e:
    print("❌ Failed to load routes:", str(e))