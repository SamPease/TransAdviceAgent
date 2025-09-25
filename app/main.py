from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import run_rag, ensure_vectorstore_files  # your existing function

app = FastAPI()


@app.on_event("startup")
def startup_download_vectorstore():
    # Try to download vectorstore files at startup to reduce repeated wake-up downloads.
    try:
        ensure_vectorstore_files()
    except Exception:
        # Don't block startup permanently for transient HF issues; log and continue.
        import logging
        logging.getLogger(__name__).exception("Failed to ensure vectorstore files at startup")

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
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}

# --- Health check + root route ---
@app.get("/")
@app.head("/")   # ✅ handle Render's HEAD /
def read_root():
    return {"message": "TransAdviceAgent API is running."}
