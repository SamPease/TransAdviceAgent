from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .rag_pipeline import run_rag  # your existing function

app = FastAPI()

# --- CORS setup ---
origins = [
    "https://sampease.github.io/",  # replace with your GitHub Pages URL
    "http://localhost:8000",           # optional: for local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --- end CORS setup ---

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask(query: Query):
    try:
        result = await run_rag(query.question)
        answer = result.get("final_answer")
        sources = result.get("sources")
        return {"answer": answer, "sources": sources}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return {"error": str(e)}
