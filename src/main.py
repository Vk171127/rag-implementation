# src/main.py

from contextlib import asynccontextmanager
from fastapi import FastAPI
from dotenv import load_dotenv
import time

load_dotenv()

from modules.database.db_manager import db_manager
from modules.vector.vector_store import vector_store
from modules.cache.redis_manager import redis_manager
from agents.requirement_analyzer import analyze_requirements
from agents.test_case_generator import generate_test_cases



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Everything BEFORE yield runs on startup
    await db_manager.initialize()
    vector_store.initialize()
    await redis_manager.initialize()
    yield  # App is running here
    
    # Everything AFTER yield runs on shutdown
    await db_manager.close()
    await redis_manager.close()        

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

@app.get("/test-redis")
async def test_redis():
    # Save something
    await redis_manager.set("test_key", {"message": "hello from redis"}, ttl=60)

    # Read it back
    value = await redis_manager.get("test_key")
    return {"cached_value": value}

@app.get("/test-cache-speed")
async def test_cache_speed():
    
    # Simulate slow operation (like ChromaDB search)
    start = time.time()
    await redis_manager.set("benchmark_key", {"data": "some result"}, ttl=60)
    # Fake a slow DB call
    import asyncio
    await asyncio.sleep(0.5)  # pretend this took 500ms
    slow_time = time.time() - start

    # Now read from cache (fast)
    start = time.time()
    value = await redis_manager.get("benchmark_key")
    fast_time = time.time() - start

    return {
        "slow_path_ms": round(slow_time * 1000, 2),
        "cache_hit_ms": round(fast_time * 1000, 2),
        "speedup": f"{round(slow_time / max(fast_time, 0.0001))}x faster"
    }

@app.get("/test-agents")
async def test_agents():
    # Step 1: Analyze raw requirements
    raw = "users should login with email, also need to reset password, account should lock after wrong attempts"
    analyzed = await analyze_requirements(raw)

    if analyzed["status"] != "success":
        return {"error": analyzed["message"]}

    # Step 2: Generate test cases from structured requirements
    test_cases = await generate_test_cases(
        requirements=analyzed["requirements"],
        rag_context=["Login should use JWT tokens", "Lockout after 5 attempts is standard"]
    )

    return {
        "requirements": analyzed["requirements"],
        "test_cases": test_cases["test_cases"],
        "used_rag": test_cases["used_rag_context"]
    }
