import asyncio
import json
import os
import sqlite3
from typing import TypedDict, List

import numpy as np
import faiss
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END

# For debugging/profiling (commented out)
# import psutil
# from sklearn.metrics.pairwise import cosine_similarity

# Load env variables
load_dotenv()

# --------------------------
# Global model instance
# --------------------------
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                )

    return _embedding_model



# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
DOC_BATCH_SIZE = 10          # how many docs to fetch from FAISS per summarization step
SUMMARY_BATCH_SIZE = 10      # how many summaries to reduce at once
DOCS_FETCH_LIMIT = 200       # how many docs to pull from FAISS in initial search
DOCS_KEEP_LIMIT = 100        # how many docs to keep after reranking
VECTORSTORE_PATH = "./app/vectorstore"  # path to FAISS index directory

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
            # Use a single query with IN clause for better performance
            placeholders = ','.join('?' * len(ids))
            rows = self.conn.execute(
                f"SELECT content, metadata FROM docs WHERE id IN ({placeholders})",
                ids
            ).fetchall()
            
            # Create a mapping for faster lookup
            row_map = {json.loads(metadata)["id"]: (content, metadata) 
                      for content, metadata in rows}
            
            # Maintain order of requested ids
            for doc_id in ids:
                if doc_id in row_map:
                    content, metadata_json = row_map[doc_id]
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
    # Use memory-mapped IO for FAISS index
    index = faiss.read_index(VECTORSTORE_PATH + "/index.faiss", faiss.IO_FLAG_MMAP)

    with open(VECTORSTORE_PATH + "/id_map.json") as f:
        id_map = json.load(f)

    # Use the singleton embedding model
    return FAISS(
        embedding_function=get_embedding_model(),  # Use stable singleton
        index=index,
        docstore=SQLiteDocstore(VECTORSTORE_PATH + "/docs.sqlite"),
        index_to_docstore_id=id_map,
    )

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
# def log_memory(tag=""):
#     mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#     print(f"[MEMORY] {tag}: {mem:.2f} MB")

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

    # Debug: Check SQLite contents
    print("\n[DEBUG][retrieve] Checking SQLite database contents...")
    conn = sqlite3.connect(VECTORSTORE_PATH + "/docs.sqlite")
    
    # Get total counts
    doc_count = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    parent_count = conn.execute(
        "SELECT COUNT(DISTINCT json_extract(metadata, '$.parent_id')) FROM docs"
    ).fetchone()[0]
    print(f"[DEBUG][retrieve] Total chunks in SQLite: {doc_count}")
    print(f"[DEBUG][retrieve] Total unique parent posts: {parent_count}")
    
    # Sample some documents
    sample_docs = conn.execute("SELECT id, metadata FROM docs LIMIT 5").fetchall()
    print(f"\n[DEBUG][retrieve] Sample docs in SQLite:")
    for doc_id, metadata_json in sample_docs:
        metadata = json.loads(metadata_json)
        parent_id = metadata.get('parent_id', 'N/A')
        chunk_index = metadata.get('chunk_index', 'N/A')
        total_chunks = metadata.get('total_chunks', 'N/A')
        print(f"  - Chunk ID: {doc_id}")
        print(f"    Parent Post: {parent_id}")
        print(f"    Chunk {chunk_index} of {total_chunks}")
    conn.close()

    # Load id_map for document lookup
    with open(VECTORSTORE_PATH + "/id_map.json") as f:
        id_map = json.load(f)

    # Generate query embedding and search
    query_emb = get_embedding_model().embed_query(query)
    D, I = vectorstore.index.search(np.array([query_emb], dtype=np.float32), DOCS_FETCH_LIMIT)
    print(f"[DEBUG][retrieve] FAISS search complete. Top 3 distances: {D[0][:3]}")
    candidate_indices = I[0].tolist()
    print(f"[DEBUG][retrieve] Found {len(candidate_indices)} candidate indices")

    # First verify which documents exist in SQLite
    print("\n[DEBUG][retrieve] Verifying document existence in SQLite...")
    conn = sqlite3.connect(VECTORSTORE_PATH + "/docs.sqlite")
    existing_ids = set()
    for row in conn.execute("SELECT id FROM docs").fetchall():
        existing_ids.add(row[0])
    print(f"[DEBUG][retrieve] Found {len(existing_ids)} documents in SQLite")

    # Load id mapping explicitly to ensure we have the latest version
    with open(VECTORSTORE_PATH + "/id_map.json") as f:
        id_map = json.load(f)
    print(f"[DEBUG][retrieve] Loaded id_map with {len(id_map)} entries")

    # Map FAISS indices to document IDs
    candidate_doc_ids = []
    valid_id_map = {k: v for k, v in id_map.items() if v in existing_ids}
    print(f"[DEBUG][retrieve] {len(valid_id_map)} of {len(id_map)} mappings point to existing documents")
    
    for idx in candidate_indices:
        key = str(idx)  # id_map uses string keys
        if key in id_map:
            doc_id = id_map[key]
            candidate_doc_ids.append(doc_id)
            # print(f"[DEBUG][retrieve] Mapped index {idx} to doc_id {doc_id}")
        else:
            print(f"[WARN] No mapping found for FAISS index {idx}")
    print(f"[DEBUG][retrieve] Mapped to {len(candidate_doc_ids)} valid doc IDs")
    if len(candidate_doc_ids) != len(candidate_indices):
        print(f"[DEBUG][retrieve] WARNING: {len(candidate_indices) - len(candidate_doc_ids)} indices had no matching doc IDs")
        print(f"[DEBUG][retrieve] First few missing indices: {[idx for idx in candidate_indices[:10] if idx not in vectorstore.index_to_docstore_id]}")

    if not candidate_doc_ids:
        print("[DEBUG][retrieve] No valid document IDs found!")
        return {"doc_ids": [], "metadata_map": {}, "query": query}

    # ---- Memory-efficient MMR ----
    print("\n[DEBUG][retrieve] Starting MMR reranking...")
    lambda_mult = 0.7
    k = min(DOCS_KEEP_LIMIT, len(candidate_doc_ids))
    print(f"[DEBUG][retrieve] Will select top {k} documents from {len(candidate_doc_ids)} candidates")

    selected_indices = []
    selected_doc_ids = []
    selected_embs = []

    # Seed with best match
    best_idx = candidate_indices[0]
    best_doc_id = id_map[str(best_idx)]
    selected_indices.append(best_idx)
    selected_doc_ids.append(best_doc_id)
    selected_embs.append(vectorstore.index.reconstruct(best_idx))

    # Remove it from candidates
    remaining = [(idx, id_map[str(idx)]) for idx in candidate_indices[1:]
                if str(idx) in id_map]

    while len(selected_doc_ids) < k and remaining:
        scores = []
        for idx, doc_id in remaining:
            emb = vectorstore.index.reconstruct(idx)
            # Use numpy operations for similarity calculations
            sim_to_query = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            sim_to_selected = max(np.dot(emb, sel_emb) / (np.linalg.norm(emb) * np.linalg.norm(sel_emb))
                                for sel_emb in selected_embs)
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
            scores.append((mmr_score, idx, doc_id, emb))
        
        if not scores:
            break
            
        _, idx, doc_id, emb = max(scores, key=lambda x: x[0])
        selected_indices.append(idx)
        selected_doc_ids.append(doc_id)
        selected_embs.append(emb)
        # print(f"[DEBUG][retrieve] Selected: index={idx}, doc_id={doc_id}, mmr_score={_:.4f}")
        remaining = [(i, d) for (i, d) in remaining if d != doc_id]

    print(f"[DEBUG][retrieve] Final selection: {len(selected_doc_ids)} documents")

    # Load metadata from SQLite
    print("[DEBUG][retrieve] Loading metadata from SQLite...")
    metadata_map = {}
    for doc_id in selected_doc_ids:
        docs = vectorstore.docstore.get_by_ids([doc_id])
        if docs:
            metadata_map[doc_id] = docs[0].metadata
            # print(f"[DEBUG][retrieve] Loaded metadata for doc_id {doc_id}")
        else:
            print(f"[WARN] Could not find document {doc_id} in SQLite")

    # log_memory("after retrieval")
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

    return {
        "final_answer": final_answer.content,  # Extract the content from AIMessage
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
