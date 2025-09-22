import asyncio
import json
import os
import sqlite3
import logging
from typing import TypedDict, List

import numpy as np
import faiss
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langsmith import Client
from huggingface_hub import hf_hub_download
import shutil
import time
import random
from collections import deque
import threading


# For debugging/profiling (commented out)
# import psutil
# from sklearn.metrics.pairwise import cosine_similarity

# Load env variables
load_dotenv()

# Module logger
logger = logging.getLogger(__name__)
# If the root logger has no handlers configured, configure basic logging so
# running the script from CLI will show INFO-level messages by default.
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

# --------------------------
# Global model instance
# --------------------------
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    elif os.environ.get("LOCAL_EMBEDDING_MODEL", "").lower() in ("1", "true", "yes"):
        _embedding_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
    else:
        _embedding_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                )

    return _embedding_model


async def embed_query_with_retry(text: str):
    """Run the HF endpoint embedding in a thread and apply global rate/ retry limits.

    This ensures embedding requests share the same rate limiting and retry
    instrumentation as LLM calls. Returns the embedding vector (list/np.array).
    """
    # Wrap the blocking embed call in asyncio.to_thread so call_with_limits can
    # await it like other async providers. We also add debug logging so delays
    # caused by rate-limiting are visible.
    def sync_embed():
        model = get_embedding_model()
        return model.embed_query(text)

    # call_with_limits expects a zero-arg coroutine factory; provide a lambda
    # that returns the awaitable for the threaded call.
    try:
        resp = await call_with_limits(lambda: asyncio.to_thread(sync_embed))
    except Exception as exc:
        logger.exception("Embedding failed after retries: %s", exc)
        raise
    return resp



# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
DOC_BATCH_SIZE = 10          # how many docs to fetch from FAISS per summarization step
SUMMARY_BATCH_SIZE = 10      # how many summaries to reduce at once
DOCS_FETCH_LIMIT = 200       # how many docs to pull from FAISS in initial search
DOCS_KEEP_LIMIT = 100        # how many docs to keep after reranking
# Allow the deploy environment to override where the vectorstore is stored.
# Prefer a repo-relative path so build-time downloads are included in the image.
VECTORSTORE_PATH = os.environ.get("VECTORSTORE_PATH", "./app/vectorstore")  # path to FAISS index directory


def ensure_vectorstore_files(repo_id: str = "SamPease/TransAdviceAgent", files: list | None = None, repo_type: str = "dataset") -> None:
    """Ensure vectorstore files exist on disk by downloading from HuggingFace if missing.

    This is safe to call at build or startup. For private datasets set HF_TOKEN env var.
    """
    files = files or ["index.faiss", "docs.sqlite", "id_map.json"]
    global VECTORSTORE_PATH
    # Try to create the target directory; if the environment is read-only (e.g. trying
    # to write to a protected root path), fall back to /tmp so the process can continue.
    try:
        os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    except Exception as exc:
        logger.warning("Could not create VECTORSTORE_PATH '%s': %s — falling back to /tmp", VECTORSTORE_PATH, exc)
        fallback = os.environ.get("FALLBACK_VECTORSTORE_PATH", "/tmp/app_vectorstore")
        try:
            os.makedirs(fallback, exist_ok=True)
            VECTORSTORE_PATH = fallback
        except Exception:
            # Last resort: continue and let later operations raise readable errors
            logger.exception("Failed to create fallback vectorstore path %s", fallback)
    # Accept multiple common env var names for HF token so Render/.env values work
    token = (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
    )
    for fn in files:
        dest = os.path.join(VECTORSTORE_PATH, fn)
        # skip if file already present and non-empty
        try:
            if os.path.exists(dest) and os.path.getsize(dest) > 0:
                continue
        except Exception:
            pass

        # download to HF cache and copy into place
        try:
            local = hf_hub_download(repo_id=repo_id, filename=fn, repo_type=repo_type, token=token)
            if os.path.abspath(local) != os.path.abspath(dest):
                shutil.copy(local, dest)
        except Exception as exc:
            logger.warning("Could not download %s from %s: %s", fn, repo_id, exc)

# --------------------------
# Prompts
# --------------------------
client = Client()
enhance_prompt = client.pull_prompt("query_enhancement", include_model=True)
map_prompt = client.pull_prompt("map_prompt", include_model=True)
reduce_prompt = client.pull_prompt("reduce_prompt", include_model=True)
final_prompt = client.pull_prompt("final_prompt", include_model=True)

# --------------------------
# Streaming callback (optional)
# --------------------------
from langchain.callbacks.base import BaseCallbackHandler
class StreamingPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)


# --------------------------
# Global rate limit / concurrency controls for LLM calls
# --------------------------
# Tunable parameters — adjust to match provider limits (safe defaults below)
GLOBAL_CONCURRENCY = 10       # max concurrent LLM calls across the app
GLOBAL_RPS = 45              # targets (calls per minute); we'll convert to per-second window
RPS_PERIOD = 60.0            # seconds for the GLOBAL_RPS window
MAX_RETRIES = 4
BASE_BACKOFF = 0.5           # seconds
MAX_BACKOFF = 30.0           # seconds

# Runtime primitives
_semaphore = asyncio.Semaphore(GLOBAL_CONCURRENCY)

class SimpleRateLimiter:
    """Sliding-window rate limiter suitable for low-volume use.

    Keeps timestamps of recent calls and blocks until a slot is available.
    """
    def __init__(self, max_calls: int, period: float):
        self.max_calls = max_calls
        self.period = period
        self.timestamps = deque()
        self.lock = asyncio.Lock()

    async def acquire(self):
        async with self.lock:
            now = time.monotonic()
            # pop expired timestamps
            while self.timestamps and now - self.timestamps[0] >= self.period:
                self.timestamps.popleft()
            if len(self.timestamps) < self.max_calls:
                self.timestamps.append(now)
                return
            wait = self.period - (now - self.timestamps[0])
        # Log when we're being delayed by the global rate limiter so it's
        # obvious in logs that embedding/LLM calls are being throttled.
        try:
            logger.info("Rate limiter: waiting %.2fs for slot (max_calls=%d, period=%.1fs)", wait, self.max_calls, self.period)
        except Exception:
            pass
        await asyncio.sleep(wait)
        await self.acquire()

# Create a process-global limiter (uses GLOBAL_RPS over RPS_PERIOD seconds)
_rate_limiter = SimpleRateLimiter(max_calls=GLOBAL_RPS, period=RPS_PERIOD)


def _extract_retry_after_from(obj) -> float | None:
    # obj can be a response-like object or an exception with .response
    headers = {}
    try:
        if hasattr(obj, "headers") and obj.headers:
            headers = {k.lower(): v for k, v in obj.headers.items()}
        elif hasattr(obj, "response") and getattr(obj.response, "headers", None):
            headers = {k.lower(): v for k, v in obj.response.headers.items()}
    except Exception:
        return None

    if not headers:
        return None
    ra = headers.get("retry-after") or headers.get("Retry-After")
    if ra is None:
        return None
    try:
        return float(ra)
    except Exception:
        # sometimes it's an HTTP-date; ignore in that case
        return None


def _is_rate_limit_exception(exc: Exception) -> bool:
    # Best-effort detection for provider rate-limit errors
    msg = str(exc).lower()
    if hasattr(exc, "status_code") and getattr(exc, "status_code") == 429:
        return True
    if hasattr(exc, "response") and getattr(exc.response, "status", None) == 429:
        return True
    if "rate limit" in msg or "rate_limited" in msg or "429" in msg:
        return True
    return False


async def call_with_limits(call_coro_fn, *, max_retries: int = MAX_RETRIES):
    """Call an async LLM coroutine factory with global concurrency, rate
    limiting and retry/backoff on rate limits or transient errors.

    call_coro_fn must be a zero-argument callable returning an awaitable.
    """
    attempt = 0
    while True:
        # Wait for rate token and a free concurrency slot
        await _rate_limiter.acquire()
        async with _semaphore:
            try:
                resp = await call_coro_fn()
            except Exception as exc:
                # If it's clearly a rate-limit error, honor Retry-After when available
                if _is_rate_limit_exception(exc):
                    ra = _extract_retry_after_from(exc)
                    if ra:
                        try:
                            logger.info("Rate-limited: sleeping %.2fs (Retry-After) before retrying (attempt=%d)", ra, attempt + 1)
                        except Exception:
                            pass
                        await asyncio.sleep(ra)
                        attempt += 1
                        if attempt > max_retries:
                            raise
                        continue
                    # otherwise fall through to exponential backoff
                # For other transient errors, retry with backoff
                attempt += 1
                if attempt > max_retries:
                    raise
                backoff = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1))) * random.uniform(0.8, 1.2)
                try:
                    logger.info("Transient error: sleeping %.2fs before retrying (attempt=%d): %s", backoff, attempt, str(exc))
                except Exception:
                    pass
                await asyncio.sleep(backoff)
                continue

        # If the response itself encodes a rate limit (some clients return 429)
        status = getattr(resp, "status", None) or getattr(resp, "status_code", None)
        headers = getattr(resp, "headers", {}) or {}
        if status == 429 or (isinstance(status, int) and status >= 500 and "rate" in str(status).lower()):
            ra = _extract_retry_after_from(resp) or None
            if ra:
                try:
                    logger.info("Response indicates rate limit: sleeping %.2fs (Retry-After) before retrying (attempt=%d)", ra, attempt + 1)
                except Exception:
                    pass
                await asyncio.sleep(ra)
                attempt += 1
                if attempt > max_retries:
                    raise RuntimeError("rate-limited, max retries exceeded")
                continue
            attempt += 1
            if attempt > max_retries:
                raise RuntimeError("rate-limited, max retries exceeded")
            backoff = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** (attempt - 1))) * random.uniform(0.8, 1.2)
            try:
                logger.info("Response indicates transient server/rate issue: sleeping %.2fs before retrying (attempt=%d)", backoff, attempt)
            except Exception:
                pass
            await asyncio.sleep(backoff)
            continue

        return resp



# --------------------------
# SQLite docstore
# --------------------------
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

# def get_llm(streaming=False):
#     callbacks = [StreamingPrintHandler()] if streaming else []
#     return ChatAnthropic(
#         model=MODEL_NAME,
#         temperature=0,
#         streaming=streaming,
#         callbacks=callbacks,
#     )

# --------------------------
# Utilities
# --------------------------
# def log_memory(tag=""):
#     mem = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
#     print(f"[MEMORY] {tag}: {mem:.2f} MB")

async def summarize_batch(batch: List[Document], question: str) -> str:
    batch_text = "\n\n".join(doc.page_content for doc in batch)
    resp = await call_with_limits(lambda: map_prompt.ainvoke({"question": question, "context": batch_text}))
    return getattr(resp, "content", resp)

async def reduce_batch(batch: List[str], question: str) -> str:
    batch_text = "\n\n".join(batch)
    resp = await call_with_limits(lambda: reduce_prompt.ainvoke({"question": question, "summaries": batch_text}))
    return getattr(resp, "content", resp)


def _extract_url_from_metadata(metadata: dict) -> str | None:
    """Try common metadata keys for a URL and normalize reddit permalinks.

    Returns the first plausible URL string or None if none found.
    """
    if not metadata:
        return None

    # Common keys that might contain a URL
    for key in ("url", "source", "source_url", "permalink", "href", "link", "uri"):
        if key in metadata and metadata[key]:
            val = metadata[key]
            # Normalize reddit permalinks
            if key == "permalink":
                if val.startswith("/"):
                    return f"https://reddit.com{val}"
                if val.startswith("http"):
                    return val
                return f"https://reddit.com/{val}"

            # If it's already a full URL, return it
            if isinstance(val, str) and val.startswith("http"):
                return val
            # Otherwise return stringified form
            return str(val)

    # Fallback: if metadata has subreddit/id/permalink structure, try to build a reddit url
    subreddit = metadata.get("subreddit") or metadata.get("subreddit_name")
    post_id = metadata.get("id") or metadata.get("post_id")
    if subreddit and post_id:
        return f"https://reddit.com/r/{subreddit}/comments/{post_id}"

    return None


def _normalize_url_for_dedupe(url: str) -> str:
    """Normalize URL for duplicate detection: strip whitespace and trailing slash.

    This is intentionally lightweight to avoid changing meaningful URL case,
    but removes common incidental differences.
    """
    if not url:
        return ""
    return url.strip().rstrip('/')


def _dedupe_urls_preserve_order(urls: list) -> list:
    seen = set()
    out = []
    for u in urls:
        key = _normalize_url_for_dedupe(u)
        if not key:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


# --------------------------
# State definition
# --------------------------
class ChatAgentState(TypedDict, total=False):
    # Make state keys optional so nodes can add fields like 'sources' without
    # being stripped by the StateGraph schema. Include 'sources' here so the
    # compiled workflow will preserve it in the final result.
    query: str
    doc_ids: List[str]
    metadata_map: dict
    summary: Document
    final_answer: str
    sources: List[dict]

# --------------------------
# LangGraph nodes
# --------------------------
async def retrieve_node(state):
    query = state["query"]
    result = await call_with_limits(lambda: enhance_prompt.ainvoke({"question": query}))
    enhanced_query = result.content
    vectorstore = get_vectorstore()

    # Quick DB check
    conn = sqlite3.connect(VECTORSTORE_PATH + "/docs.sqlite")
    doc_count = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    logger.info("Found %d documents in database", doc_count)
    conn.close()

    # Load id_map for document lookup
    with open(VECTORSTORE_PATH + "/id_map.json") as f:
        id_map = json.load(f)

    # Generate query embedding and search. Use embed_query_with_retry so
    # embeddings go through the same rate limiting / retry logic and produce
    # informative logs when delayed.
    logger.info("Requesting embedding for enhanced query")
    start_emb = time.monotonic()
    query_emb = await embed_query_with_retry(enhanced_query)
    elapsed_emb = time.monotonic() - start_emb
    logger.info("Embedding completed in %.3fs", elapsed_emb)
    D, I = vectorstore.index.search(np.array([query_emb], dtype=np.float32), DOCS_FETCH_LIMIT)
    candidate_indices = I[0].tolist()
    logger.info("Initial search found %d relevant documents", len(candidate_indices))

    # Verify document existence
    conn = sqlite3.connect(VECTORSTORE_PATH + "/docs.sqlite")
    existing_ids = set(row[0] for row in conn.execute("SELECT id FROM docs").fetchall())

    # Map FAISS indices to document IDs
    candidate_doc_ids = []
    for idx in candidate_indices:
        key = str(idx)
        if key in id_map and id_map[key] in existing_ids:
            candidate_doc_ids.append(id_map[key])
    
    logger.info("Found %d valid documents to process", len(candidate_doc_ids))

    if not candidate_doc_ids:
        logger.debug("No valid document IDs found!")
        return {"doc_ids": [], "metadata_map": {}, "query": query}

    # ---- Memory-efficient MMR ----
    logger.debug("Starting MMR reranking...")
    lambda_mult = 0.7
    k = min(DOCS_KEEP_LIMIT, len(candidate_doc_ids))
    logger.debug("Will select top %d documents from %d candidates", k, len(candidate_doc_ids))

    selected_indices = []
    selected_doc_ids = []
    selected_embs = []
    selected_sims = []

    # Seed with best match
    best_idx = candidate_indices[0]
    best_doc_id = id_map[str(best_idx)]
    selected_indices.append(best_idx)
    selected_doc_ids.append(best_doc_id)
    best_emb = vectorstore.index.reconstruct(best_idx)
    selected_embs.append(best_emb)
    # compute and store sim_to_query for the seeded document
    try:
        sim_best = float(np.dot(query_emb, best_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(best_emb)))
    except Exception:
        sim_best = None
    selected_sims.append(sim_best)

    # Remove it from candidates
    remaining = [(idx, id_map[str(idx)]) for idx in candidate_indices[1:]
                if str(idx) in id_map]

    while len(selected_doc_ids) < k and remaining:
        scores = []
        for idx, doc_id in remaining:
            emb = vectorstore.index.reconstruct(idx)
            # Use numpy operations for similarity calculations
            try:
                sim_to_query = float(np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb)))
            except Exception:
                sim_to_query = None
            sim_to_selected = max(np.dot(emb, sel_emb) / (np.linalg.norm(emb) * np.linalg.norm(sel_emb))
                                for sel_emb in selected_embs)
            mmr_score = (lambda_mult * sim_to_query) - ((1 - lambda_mult) * sim_to_selected) if sim_to_query is not None else -999.0
            # store sim_to_query alongside candidate so we can reuse it when selected
            scores.append((mmr_score, idx, doc_id, emb, sim_to_query))
        
        if not scores:
            break

        _, idx, doc_id, emb, sim_selected = max(scores, key=lambda x: x[0])
        selected_indices.append(idx)
        selected_doc_ids.append(doc_id)
        selected_embs.append(emb)
        selected_sims.append(sim_selected)
        # print(f"[DEBUG][retrieve] Selected: index={idx}, doc_id={doc_id}, mmr_score={_:.4f}")
        remaining = [(i, d) for (i, d) in remaining if d != doc_id]

    logger.debug("Final selection: %d documents", len(selected_doc_ids))

    # Load metadata from SQLite
    logger.debug("Loading metadata from SQLite...")
    metadata_map = {}
    for doc_id in selected_doc_ids:
        docs = vectorstore.docstore.get_by_ids([doc_id])
        if docs:
            metadata_map[doc_id] = docs[0].metadata
        else:
            logger.warning("Could not find document %s in SQLite", doc_id)

    # --- Compute similarity to query and provide both MMR selection order and
    #     a pure-relevance sort. MMR selects diverse items; if you need
    #     strict relevance ordering we sort by sim_to_query here before
    #     returning results. ---
    try:
        # Build sim_list from stored sims collected during MMR selection to avoid
        # recomputing similarity a second time.
        sim_list = list(zip(selected_doc_ids, selected_sims))

        # Log top N in the MMR-chosen order (how they were selected)
        top_n = min(10, len(sim_list))
        logger.info("Top %d retrieved documents (MMR selection order):", top_n)
        top_ids_mmr = [doc_id for doc_id, _ in sim_list[:top_n]]
        top_docs_mmr = vectorstore.docstore.get_by_ids(top_ids_mmr)
        for (doc_id, sim), doc in zip(sim_list[:top_n], top_docs_mmr):
            url = None
            if doc and getattr(doc, "metadata", None):
                url = doc.metadata.get("url") or doc.metadata.get("permalink")
            logger.info("  %s  score=%s  url=%s", doc_id, f"{sim:.4f}" if sim is not None else "<err>", url)

        # Now produce a strict relevance-sorted ordering (descending sim)
        sorted_by_relevance = sorted([t for t in sim_list if t[1] is not None], key=lambda x: x[1], reverse=True)
        # If some sims are None, append them at the end in original order
        none_sims = [t for t in sim_list if t[1] is None]
        sorted_by_relevance.extend(none_sims)

        # Reorder selected_doc_ids and selected_embs to be relevance-sorted for return
        relevance_ordered_ids = [doc_id for doc_id, _ in sorted_by_relevance]
        id_to_emb = {doc_id: emb for doc_id, emb in zip(selected_doc_ids, selected_embs)}
        selected_doc_ids = relevance_ordered_ids
        selected_embs = [id_to_emb.get(did) for did in selected_doc_ids]

        # Attach relevance scores into metadata_map so downstream nodes can use them
        # Use 0.0 for None sims to keep a numeric field available.
        for doc_id, sim in sim_list:
            try:
                metadata_map.setdefault(doc_id, {})["relevance"] = float(sim) if sim is not None else 0.0
            except Exception:
                metadata_map.setdefault(doc_id, {})["relevance"] = 0.0

        # Log top N in the relevance-sorted order for easy comparison
        logger.info("Top %d retrieved documents (relevance-sorted):", top_n)
        top_ids_rel = [doc_id for doc_id, _ in sorted_by_relevance[:top_n]]
        top_docs_rel = vectorstore.docstore.get_by_ids(top_ids_rel)
        for (doc_id, sim), doc in zip(sorted_by_relevance[:top_n], top_docs_rel):
            url = None
            if doc and getattr(doc, "metadata", None):
                url = doc.metadata.get("url") or doc.metadata.get("permalink")
            logger.info("  %s  score=%s  url=%s", doc_id, f"{sim:.4f}" if sim is not None else "<err>", url)
    except Exception:
        logger.debug("Failed to compute/log relevance list", exc_info=True)

    # Return selected doc ids, metadata_map (only for selected docs), and the query.
    return {
        "doc_ids": selected_doc_ids,
        "metadata_map": metadata_map,
        "query": query,
    }


async def summarize_node(state):
    doc_ids = state["doc_ids"]
    if not doc_ids:
        print("[INFO] No relevant documents found for query")
        return {
            "final_answer": "No documents found for this query.",
            "query": state["query"],
            "used_doc_ids": [],
            "metadata_map": {}
        }

    logger.info("Starting summarization process for %d documents", len(doc_ids))
    logger.info("Using batch size of %d for initial summarization", DOC_BATCH_SIZE)
    
    vectorstore = get_vectorstore()
    # llm = get_llm(streaming=False)

    level_summaries = []

    # --------------------------
    # Prefetch first batch asynchronously
    # --------------------------
    # Note: removed eager prefetch via asyncio.to_thread() here to avoid creating
    # an un-awaited coroutine (which produced a RuntimeWarning). We load batches
    # via `load_batch` below which properly awaits asyncio.to_thread.

    # Create coroutines for document loading
    async def load_batch(batch_ids):
        return await asyncio.to_thread(vectorstore.docstore.get_by_ids, batch_ids)
    
    # Setup parallel document loading
    doc_batches = []
    for i in range(0, len(doc_ids), DOC_BATCH_SIZE):
        batch_ids = doc_ids[i:i + DOC_BATCH_SIZE]
        doc_batches.append(load_batch(batch_ids))
    
    # Wait for all document batches to load
    logger.info("Loading %d document batches in parallel", len(doc_batches))
    loaded_batches = await asyncio.gather(*doc_batches)

    # Summarize all batches in parallel
    logger.info("Starting parallel summarization of %d batches", len(loaded_batches))
    summarization_tasks = []
    for batch_index, batch_docs in enumerate(loaded_batches, 1):
        # Log batch size and a small sample of document IDs (if available)
        sample_ids = []
        try:
            for d in batch_docs[:3]:
                mid = None
                try:
                    mid = (getattr(d, "metadata", {}) or {}).get("id")
                except Exception:
                    mid = None
                sample_ids.append(mid or (d.page_content[:60] if getattr(d, "page_content", None) else "<no-id>"))
        except Exception:
            sample_ids = ["<err>"]
        logger.info("Queuing batch %d (%d documents) sample_ids=%s", batch_index, len(batch_docs), sample_ids)
        # wrap map prompt calls with global rate/concurrency limits
        # Bind batch_docs into the lambda's default argument to avoid the
        # common late-binding closure issue where all lambdas would capture
        # the same final value of `batch_docs`.
        summarization_tasks.append(
            call_with_limits(lambda bd=batch_docs: map_prompt.ainvoke({
                "question": state["query"],
                "context": "\n\n".join(doc.page_content for doc in bd)
            }))
        )
    
    # Wait for all summaries
    level_summaries = await asyncio.gather(*summarization_tasks)
    # Ensure each summary is a plain string. Some clients return message objects
    # (e.g., with a .content attribute); normalize those here so reductions can
    # safely join summaries.
    logger.info("Completed parallel summarization of %d batches", len(level_summaries))

    # --------------------------
    # Iterative reduction of summaries
    # --------------------------
    reduction_round = 1
    while len(level_summaries) > 1:
        level_summaries = [getattr(s, "content", s) for s in level_summaries]

        logger.info("Starting reduction round %d", reduction_round)
        logger.info("Combining %d summaries in batches of %d", len(level_summaries), SUMMARY_BATCH_SIZE)

        # Process reduction batches in parallel
        reduction_tasks = []
        for i in range(0, len(level_summaries), SUMMARY_BATCH_SIZE):
            batch = level_summaries[i:i + SUMMARY_BATCH_SIZE]
            # wrap reduce calls with rate/concurrency guard
            reduction_tasks.append(call_with_limits(lambda b=batch: reduce_prompt.ainvoke({"question": state["query"], "summaries": "\n\n".join(b)})))

        # Wait for all reductions to complete
        results = await asyncio.gather(*reduction_tasks)
        next_level = results

        logger.info("Reduction round %d complete. Summaries reduced from %d to %d", reduction_round, len(level_summaries), len(next_level))
        level_summaries = next_level
        reduction_round += 1



    # Propagate metadata_map and the final answer. Summarize node doesn't modify metadata.
    metadata_map = state.get("metadata_map") or {}
    return {
        "summary": level_summaries[0],
        "metadata_map": metadata_map,
    }

async def output_node(state):
     # --------------------------
    # Final answer generation
    # --------------------------
    logger.info("Generating final answer from consolidated summary")
    final_summary = state.get("summary").content

    final_resp = await call_with_limits(lambda: final_prompt.ainvoke({"question": state["query"], "final_summary": final_summary}))
    final_answer = getattr(final_resp, "content", final_resp)
    logger.info("Answer generation complete")


    metadata_map = state.get("metadata_map") or {}

    # Build a list of (doc_id, relevance, url) directly from metadata_map so
    # ordering is driven solely by relevance recorded at retrieval time.
    url_entries = []
    for did, meta in metadata_map.items():
        meta = meta or {}
        rel = meta.get("relevance")
        url = _extract_url_from_metadata(meta) or ""
        url_entries.append((did, float(rel) if rel is not None else 0.0, url))

    # Sort by relevance descending
    url_entries.sort(key=lambda x: x[1], reverse=True)

    # Extract urls and dedupe while preserving the relevance order
    ordered_urls = _dedupe_urls_preserve_order([u for _, _, u in url_entries])

    # Build sources and include relevance for downstream visibility
    # Sources remain compatible with main.py (source.get('url')) but now also
    # include 'relevance' if callers want it.
    # We map back relevance values for the final sources list.
    # Map each url -> relevance, keeping the first-seen relevance (highest
    # because url_entries was sorted by relevance). This avoids overwriting
    # a high relevance with a later lower-relevance duplicate.
    url_to_rel = {}
    # Also map url -> first-seen doc_id so we can lookup other metadata fields
    # such as a title (if present) for inclusion in the final sources list.
    url_to_docid = {}
    for _, r, u in url_entries:
        if u and u not in url_to_rel:
            url_to_rel[u] = r
            url_to_docid[u] = _

    sources = []
    for u in ordered_urls:
        entry = {"url": u, "relevance": url_to_rel.get(u, 0.0)}
        # Only include a 'title' key if the corresponding metadata contains one.
        try:
            doc_id = url_to_docid.get(u)
            if doc_id:
                meta = metadata_map.get(doc_id) or {}
                # Accept common title keys: 'title' or 'heading'
                title = meta.get("title") or meta.get("heading")
                if title:
                    entry["title"] = title
        except Exception:
            # Don't fail the whole pipeline if title extraction fails; skip title.
            logger.debug("Failed to extract title for url %s", u, exc_info=True)

        sources.append(entry)

    # Debug log the final source ordering for easy inspection during runs.
    try:
        logger.info("Output node final sources (top 10): %s", sources[:10])
    except Exception:
        logger.debug("Unable to log output sources", exc_info=True)

    return {
        "final_answer": final_answer,
        "sources": sources,
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
    # Minimal output: return exactly what the workflow produced.
    final_answer = result.get("final_answer", "")
    sources = result.get("sources")
    # Enforce contract: output_node must set `sources` (list of dicts with 'url')
    if sources is None:
        # Log diagnostic information to help debug why `sources` is missing.
        try:
            keys = list(result.keys())
        except Exception:
            keys = f"<uninspectable result type: {type(result)!r}>"

        # Try to extract useful summaries without risking large dumps.
        meta_len = None
        doc_ids_len = None
        try:
            meta_len = len(result.get("metadata_map") or {})
        except Exception:
            meta_len = "<err>"
        try:
            doc_ids_len = len(result.get("doc_ids") or [])
        except Exception:
            doc_ids_len = "<err>"

        logger.error("Pipeline contract violated: 'sources' missing from workflow result. result keys=%s, metadata_map_len=%s, doc_ids_len=%s", keys, meta_len, doc_ids_len)
        # Surface the keys in the raised error to make debugging easier for callers.
        raise RuntimeError(f"Pipeline contract violated: 'sources' missing from workflow result. result_keys={keys}, metadata_map_len={meta_len}, doc_ids_len={doc_ids_len}")

    return {"final_answer": final_answer, "sources": sources}

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
        print("Document URLs:\n", answer.get("sources"), "\n")
