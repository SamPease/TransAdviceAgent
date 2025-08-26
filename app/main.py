from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .rag_pipeline import run_rag  # your existing function

app = FastAPI()

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
        
        # Just extract URLs from sources
        urls = [source.get("url", "") for source in result.get("sources", [])]
        
        return {
            "answer": result.get("final_answer", "No answer generated"),
            "sources": urls
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
