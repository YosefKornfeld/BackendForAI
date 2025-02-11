# main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import qa  # your router module

app = FastAPI(title="Halacha Q&A System with SurrealDB")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or use ["*"] for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(qa.router)

@app.get("/health")
def health_check():
    return {"status": "healthy"}
