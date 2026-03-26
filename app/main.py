
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# 🔥 Important
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = FastAPI(title="AgritechAI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "API Running 🚀"}

# Load routes safely
try:
    from app.routes.predict import router
    app.include_router(router, prefix="/api/v1")
    print("✅ Routes loaded")
except Exception as e:
    print("❌ Route load error:", str(e))
