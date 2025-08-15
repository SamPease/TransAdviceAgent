from fastapi import FastAPI
from pydantic import BaseModel
from .rag_pipeline import run_rag  # your existing function

app = FastAPI()

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

