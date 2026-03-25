import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,       # Auto-reload on code changes
        workers=1          # Use 1 worker (increase in production)
    )