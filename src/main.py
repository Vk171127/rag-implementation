# src/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv

load_dotenv()

from modules.database.db_manager import db_manager
from modules.vector.vector_store import vector_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything BEFORE yield runs on startup
    await db_manager.initialize()
    vector_store.initialize()
    yield  # App is running here
    
    # Everything AFTER yield runs on shutdown
    await db_manager.close()

# Pass lifespan to FastAPI
app = FastAPI(title="TestGen API", version="1.0.0", lifespan=lifespan)

@app.get("/health")
async def health():
    return {"status": "ok", "message": "TestGen is running"}

@app.get("/test-vector")
async def test_vector():
    # Store two documents
    vector_store.add_documents(
        documents=[
            "The user must be able to login with email and password",
            "The system shall send a confirmation email after registration",
            "The app must generate monthly PDF reports"
        ],
        metadatas=[
            {"session_id": "test", "type": "requirement"},
            {"session_id": "test", "type": "requirement"},
            {"session_id": "test", "type": "requirement"},
        ],
        ids=["req_1", "req_2", "req_3"]
    )

    # Search for something semantically related
    results = vector_store.search("authentication and user access", n_results=2)
    return {"results": results}
