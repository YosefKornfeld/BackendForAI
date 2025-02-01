from fastapi import FastAPI
from routes import qa
from config import settings

app = FastAPI(title="Halacha Q&A System with SurrealDB")
app.include_router(qa.router)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model": settings.embedding_model}
