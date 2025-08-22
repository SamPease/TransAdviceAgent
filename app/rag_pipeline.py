import asyncio
from typing import TypedDict, List
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import psutil
import sqlite3
import json

# Load env variables
load_dotenv()

# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
DOC_BATCH_SIZE = 10          # how many docs to fetch from FAISS per summarization step
SUMMARY_BATCH_SIZE = 10      # how many summaries to reduce at once
DOCS_FETCH_LIMIT = 200       # how many docs to pull from FAISS in initial search
DOCS_KEEP_LIMIT = 100        # how many docs to keep after reranking

# --------------------------
# Prompts
# --------------------------
map_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an assistant reading documents to answer a question.
Question: {question}

Read the following documents carefully and produce a concise summary that captures all key information relevant to the question. 

Documents:
{context}

Summary:
"""
)

reduce_prompt = PromptTemplate(
    input_variables=["question", "summaries"],
    template="""
You are an assistant tasked with combining summaries into a single, coherent summary. 

Question: {question}

Batch summaries:
{summaries}

Combined summary:
"""
)

final_prompt = PromptTemplate(
    input_variables=["question", "final_summary"],
    template="""
You are an expert assistant. Using the following summary, answer the question as accurately and clearly as possible.

Question: {question}

Summary:
{final_summary}

Answer:
"""
)

# --------------------------
# Streaming callback (optional)
# --------------------------
from langchain.callbacks.base import BaseCallbackHandler
class StreamingPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

# --------------------------
# Global cached models
# --------------------------
_embedding_model = None
_vectorstore = None

# --------------------------
# SQLite docstore
# --------------------------
import sqlite3
import json
import threading
from typing import List
from langchain.schema import Document

class SQLiteDocstore:
    """Minimal thread-safe docstore for lazy-loading documents from SQLite."""
    def __init__(self, db_path="docs.sqlite"):
        # Allow connection to be used across threads
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.Lock()

    def get_by_ids(self, ids: List[str]) -> List[Document]:
        docs = []
        with self.lock:
            for doc_id in ids:
                row = self.conn.execute(
                    "SELECT content, metadata FROM docs WHERE id=?",
                    (doc_id,)
                ).fetchone()
                if row:
                    content, metadata_json = row
                    metadata = json.loads(metadata_json)
                    docs.append(Document(page_content=content, metadata=metadata))
        return docs

    def insert_documents(self, documents: List[Document]):
        """Optional helper to insert docs into SQLite"""
        with self.lock:
            for doc in documents:
                doc_id = doc.metadata.get("id") or os.urandom(8).hex()
                self.conn.execute(
                    "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
                    (doc_id, doc.page_content, json.dumps(doc.metadata))
                )
            self.conn.commit()

def get_vectorstore():
    global _embedding_model, _vectorstore
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    if _vectorstore is None:
        faiss_path = "./app/faiss_index"
        if os.path.exists(faiss_path + "/index.faiss"):
            _vectorstore = FAISS.load_local(
                faiss_path,
                _embedding_model,
                allow_dangerous_deserialization=True
            )
            # attach SQLite-backed docstore instead of loading index.pkl
            _vectorstore.docstore = SQLiteDocstore("./app/faiss_index/docs.sqlite")
        else:
            raise RuntimeError("FAISS index not found. Please build it first.")
    return _vectorstore

def get_llm(streaming=False):
    callbacks = [StreamingPrintHandler()] if streaming else []
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=0,
        streaming=streaming,
        callbacks=callbacks,
    )

# --------------------------
# Utilities
# --------------------------
def log_memory(tag=""):
    mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    print(f"[MEMORY] {tag}: {mem:.2f} MB")

async def summarize_batch(batch: List[Document], llm, question: str) -> str:
    batch_text = "\n\n".join(doc.page_content for doc in batch)
    prompt = map_prompt.format(question=question, context=batch_text)
    result = await llm.ainvoke(prompt)
    return result.content

# --------------------------
# State definition
# --------------------------
class ChatAgentState(TypedDict):
    query: str
    doc_ids: List[str]
    metadata_map: dict
    final_answer: str

# --------------------------
# LangGraph nodes
# --------------------------
async def retrieve_node(state):
    query = state["query"]
    vectorstore = get_vectorstore()

    log_memory("before retrieval")

    # Query embedding
    query_emb = _embedding_model.embed_query(query)

    # Initial FAISS search (IDs only)
    D, I = vectorstore.index.search(np.array([query_emb], dtype=np.float32), DOCS_FETCH_LIMIT)
    candidate_indices = I[0].tolist()
    candidate_doc_ids = [
        vectorstore.index_to_docstore_id[idx]
        for idx in candidate_indices if idx in vectorstore.index_to_docstore_id
    ]

    if not candidate_doc_ids:
        return {"doc_ids": [], "metadata_map": {}, "query": query}

    # ---- Memory-efficient MMR ----
    lambda_mult = 0.7
    k = min(DOCS_KEEP_LIMIT, len(candidate_doc_ids))

    selected_doc_ids = []
    selected_embs = []

    # Seed with best match
    best_idx = candidate_indices[0]
    selected_doc_ids.append(vectorstore.index_to_docstore_id[best_idx])
    selected_embs.append(vectorstore.index.reconstruct(best_idx))

    # Remove it from candidates
    remaining = [(idx, doc_id) for idx, doc_id in zip(candidate_indices, candidate_doc_ids)
                 if doc_id not in selected_doc_ids]

    while len(selected_doc_ids) < k and remaining:
        scores = []
        for idx, doc_id in remaining:
            emb = vectorstore.index.reconstruct(idx)
            sim_to_query = cosine_similarity([query_emb], [emb])[0][0]
            sim_to_selected = max(
                cosine_similarity([emb], selected_embs)[0]
            )
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            scores.append((mmr_score, idx, doc_id, emb))
        _, idx, doc_id, emb = max(scores, key=lambda x: x[0])
        selected_doc_ids.append(doc_id)
        selected_embs.append(emb)
        remaining = [(i, d) for (i, d) in remaining if d != doc_id]

    # Load only metadata
    metadata_map = {}
    for doc_id in selected_doc_ids:
        docs = vectorstore.docstore.get_by_ids([doc_id])
        if docs:
            metadata_map[doc_id] = docs[0].metadata

    log_memory("after retrieval")
    return {"doc_ids": selected_doc_ids, "metadata_map": metadata_map, "query": query}


async def summarize_node(state):
    doc_ids = state["doc_ids"]
    if not doc_ids:
        return {
            "final_answer": "No documents found for this query.",
            "query": state["query"],
            "used_doc_ids": [],
            "metadata_map": {}
        }

    vectorstore = get_vectorstore()
    llm = get_llm(streaming=False)

    level_summaries = []

    log_memory("before summarization")

    # --------------------------
    # Prefetch first batch asynchronously
    # --------------------------
    next_batch_start = 0
    current_task = None
    if next_batch_start < len(doc_ids):
        batch_ids = doc_ids[next_batch_start:next_batch_start + DOC_BATCH_SIZE]
        current_task = asyncio.to_thread(vectorstore.docstore.get_by_ids, batch_ids)
        next_batch_start += DOC_BATCH_SIZE

    batch_index = 1
    while current_task:
        # Wait for the current batch to load
        batch_docs = await current_task

        # Start prefetching next batch if available
        if next_batch_start < len(doc_ids):
            batch_ids = doc_ids[next_batch_start:next_batch_start + DOC_BATCH_SIZE]
            next_task = asyncio.to_thread(vectorstore.docstore.get_by_ids, batch_ids)
            next_batch_start += DOC_BATCH_SIZE
        else:
            next_task = None

        # Summarize current batch
        summary = await summarize_batch(batch_docs, llm, state["query"])
        level_summaries.append(summary)

        del batch_docs
        log_memory(f"after summarization batch {batch_index}")
        batch_index += 1

        # Move to next batch
        current_task = next_task

    # --------------------------
    # Iterative reduction of summaries
    # --------------------------
    while len(level_summaries) > 1:
        next_level = []
        for i in range(0, len(level_summaries), SUMMARY_BATCH_SIZE):
            batch = level_summaries[i:i + SUMMARY_BATCH_SIZE]
            prompt = reduce_prompt.format(
                question=state["query"],
                summaries="\n\n".join(batch)
            )
            result = await llm.ainvoke(prompt)
            next_level.append(result.content)
        level_summaries = next_level

    # --------------------------
    # Final answer generation
    # --------------------------
    final_prompt_text = final_prompt.format(
        question=state["query"],
        final_summary=level_summaries[0]
    )
    final_answer = await llm.ainvoke(final_prompt_text)

    log_memory("after final summarization")

    return {
        "final_answer": final_answer,
        "query": state["query"],
        "used_doc_ids": doc_ids,
        "metadata_map": state.get("metadata_map")
    }

async def output_node(state):
    return {
        "final_answer": state["final_answer"],
        "used_doc_ids": state.get("used_doc_ids"),
        "metadata_map": state.get("metadata_map")
    }

# --------------------------
# Workflow setup
# --------------------------
workflow = StateGraph(ChatAgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("output", output_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "summarize")
workflow.add_edge("summarize", "output")
workflow.add_edge("output", END)
workflow = workflow.compile()

# --------------------------
# Async wrapper for FastAPI
# --------------------------
async def run_rag(user_query: str) -> dict:
    result = await workflow.ainvoke({"query": user_query})
    result["sources"] = [
        result["metadata_map"][doc_id] for doc_id in result.get("used_doc_ids", [])
    ]
    return result

# --------------------------
# Terminal debugging
# --------------------------
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    while True:
        q = input("Enter question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer = asyncio.run(run_rag(q))
        print("\nFinal Answer:\n", answer["final_answer"], "\n")
        print("Sources metadata:\n", answer["sources"], "\n")
