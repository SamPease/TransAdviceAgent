import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import run_rag, ensure_vectorstore_files  # your existing function

logger = logging.getLogger(__name__)

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Try to download vectorstore files at startup to reduce repeated wake-up downloads.
    try:
        ensure_vectorstore_files()
    except Exception:
        # Don't block startup permanently for transient HF issues; log and continue.
        logger.exception("Failed to ensure vectorstore files at startup")
    yield


app = FastAPI(lifespan=lifespan)

# --- Secure CORS setup ---
origins = [
    "https://sampease.github.io",  # only allow your GitHub Pages domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],   # allow GET, POST, OPTIONS, etc.
    allow_headers=["*"],   # allow Content-Type, Authorization, etc.
)
# --- end CORS setup ---

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    try:
        result = await run_rag(query.question)

        return {
            "answer": result.get("final_answer", "No answer generated"),
            "sources": result.get("sources", [])
        }
    except Exception as e:
        # Return a real error status so clients can distinguish failures from
        # answers (previously this returned 200 with an "error" key, which the
        # frontend silently rendered as "No answer returned").
        logger.exception("Error handling /ask")
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Health check + root route ---
@app.get("/")
@app.head("/")   # ✅ handle Render's HEAD /
def read_root():
    return {"message": "TransAdviceAgent API is running."}
