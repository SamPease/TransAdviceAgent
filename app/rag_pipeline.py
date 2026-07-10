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
MODEL_NAME = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5")
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
# Pull only the prompt templates from LangSmith and bind the model here.
# Previously we pulled with include_model=True, which baked the model choice
# into the LangSmith prompt config — when that model was retired
# (claude-3-5-haiku-latest), every request failed with a 404. Binding the
# model in code keeps it under version control and easy to update.
llm = ChatAnthropic(model=MODEL_NAME, temperature=0, max_tokens=4096)

client = Client()
enhance_prompt = client.pull_prompt("query_enhancement") | llm
map_prompt = client.pull_prompt("map_prompt") | llm
reduce_prompt = client.pull_prompt("reduce_prompt") | llm
final_prompt = client.pull_prompt("final_prompt") | llm

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
                f"SELECT id, content, metadata FROM docs WHERE id IN ({placeholders})",
                ids
            ).fetchall()

            # Create a mapping for faster lookup by the DB id column
            # rows are tuples (id, content, metadata)
            row_map = {row_id: (content, metadata) for (row_id, content, metadata) in rows}

            # Maintain order of requested ids
            for doc_id in ids:
                if doc_id in row_map:
                    content, metadata_json = row_map[doc_id]
                    metadata = json.loads(metadata_json) if metadata_json else {}
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
    index.nprobe = 10  # number of clusters to search; tune for speed/accuracy

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
# Embedding helpers (module-level)
# --------------------------
def _parse_embedding(raw, doc_id: str):
    """Parse a stored `Embedding` column value (float32 blob, JSON array, or
    sequence) into a float32 numpy array. Returns None if missing/unparseable."""
    try:
        if isinstance(raw, (bytes, bytearray)):
            return np.frombuffer(raw, dtype=np.float32)
        if isinstance(raw, str):
            return np.array(json.loads(raw), dtype=np.float32)
        if raw is not None:
            return np.array(raw, dtype=np.float32)
    except Exception:
        logger.warning("Could not parse stored embedding for doc id %s", doc_id)
    return None


def _get_embeddings_from_sql(vstore, doc_ids: List[str]) -> dict:
    """Fetch embeddings for many doc ids from the SQLite `Embedding` column in
    a single query. Returns {doc_id: np.ndarray}; ids with missing or
    unparseable embeddings are omitted.

    A few hundred 384-dim float32 vectors is well under 1 MB, so holding them
    all for the duration of one request is cheap — much cheaper than the
    previous approach of re-reading embeddings from SQLite on every iteration
    of the MMR loop (~20k point queries per request).
    """
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    with vstore.docstore.lock:
        rows = vstore.docstore.conn.execute(
            f"SELECT id, Embedding FROM docs WHERE id IN ({placeholders})", doc_ids
        ).fetchall()
    out = {}
    for doc_id, raw in rows:
        emb = _parse_embedding(raw, doc_id)
        if emb is not None:
            out[doc_id] = emb
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

    # Map FAISS indices to document IDs, then verify existence in SQLite.
    # Only check the candidate ids — loading every id in the docstore into a
    # Python set (the previous approach) rescanned the whole table per request.
    mapped_ids = [id_map[str(idx)] for idx in candidate_indices if str(idx) in id_map]
    conn = sqlite3.connect(VECTORSTORE_PATH + "/docs.sqlite")
    try:
        placeholders = ",".join("?" * len(mapped_ids))
        existing_ids = set(
            row[0]
            for row in conn.execute(
                f"SELECT id FROM docs WHERE id IN ({placeholders})", mapped_ids
            )
        ) if mapped_ids else set()
    finally:
        conn.close()
    candidate_doc_ids = [doc_id for doc_id in mapped_ids if doc_id in existing_ids]


    logger.info("Found %d valid documents to process", len(candidate_doc_ids))

    if not candidate_doc_ids:
        logger.debug("No valid document IDs found!")
        return {"doc_ids": [], "metadata_map": {}, "query": query}

    # ---- MMR reranking (vectorized) ----
    # Fetch all candidate embeddings with one query and keep them as a single
    # (n, dim) float32 matrix (~0.3 MB for 200 candidates), instead of
    # re-reading embeddings from SQLite inside the selection loop.
    logger.debug("Starting MMR reranking...")
    lambda_mult = 0.7

    emb_map = _get_embeddings_from_sql(vectorstore, candidate_doc_ids)
    query_vec = np.asarray(query_emb, dtype=np.float32)
    dim = query_vec.shape[0]
    ids = [d for d in candidate_doc_ids if d in emb_map and emb_map[d].shape == (dim,)]
    dropped = len(candidate_doc_ids) - len(ids)
    if dropped:
        logger.warning("Dropped %d candidates with missing/malformed embeddings", dropped)
    if not ids:
        logger.warning("No candidate embeddings available for this query")
        return {"doc_ids": [], "metadata_map": {}, "query": query}

    k = min(DOCS_KEEP_LIMIT, len(ids))
    logger.debug("Will select top %d documents from %d candidates", k, len(ids))

    # Unit-normalize once so every cosine similarity below is a plain dot product.
    E = np.stack([emb_map[d] for d in ids])
    E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    q = query_vec / max(float(np.linalg.norm(query_vec)), 1e-12)
    sim_to_query = E @ q

    # Seed with the best FAISS match (ids preserve FAISS ranking order), then
    # greedily add the candidate with the highest MMR score. The similarity to
    # the selected set is updated incrementally, so the loop is O(k * n * dim).
    selected = [0]
    max_sim_to_selected = E @ E[0]
    while len(selected) < k:
        mmr = lambda_mult * sim_to_query - (1 - lambda_mult) * max_sim_to_selected
        mmr[selected] = -np.inf
        nxt = int(np.argmax(mmr))
        selected.append(nxt)
        max_sim_to_selected = np.maximum(max_sim_to_selected, E @ E[nxt])

    selected_doc_ids = [ids[i] for i in selected]
    selected_sims = [float(sim_to_query[i]) for i in selected]
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

    # --- Provide both the MMR selection order (logged) and a strict
    #     relevance-sorted ordering (returned). MMR selects diverse items;
    #     downstream consumers want doc ids sorted by similarity to the query. ---
    try:
        sim_list = list(zip(selected_doc_ids, selected_sims))

        # Log top N in the MMR-chosen order (how they were selected)
        top_n = min(10, len(sim_list))
        logger.info("Top %d retrieved documents (MMR selection order):", top_n)
        top_docs_mmr = vectorstore.docstore.get_by_ids([d for d, _ in sim_list[:top_n]])
        for (doc_id, sim), doc in zip(sim_list[:top_n], top_docs_mmr):
            url = None
            if doc and getattr(doc, "metadata", None):
                url = doc.metadata.get("url") or doc.metadata.get("permalink")
            logger.info("  %s  score=%.4f  url=%s", doc_id, sim, url)

        # Strict relevance-sorted ordering (descending sim) for the return value
        sorted_by_relevance = sorted(sim_list, key=lambda x: x[1], reverse=True)
        selected_doc_ids = [doc_id for doc_id, _ in sorted_by_relevance]

        # Attach relevance scores into metadata_map so downstream nodes can use them
        for doc_id, sim in sim_list:
            metadata_map.setdefault(doc_id, {})["relevance"] = float(sim)

        # Log top N in the relevance-sorted order for easy comparison
        logger.info("Top %d retrieved documents (relevance-sorted):", top_n)
        top_docs_rel = vectorstore.docstore.get_by_ids([d for d, _ in sorted_by_relevance[:top_n]])
        for (doc_id, sim), doc in zip(sorted_by_relevance[:top_n], top_docs_rel):
            url = None
            if doc and getattr(doc, "metadata", None):
                url = doc.metadata.get("url") or doc.metadata.get("permalink")
            logger.info("  %s  score=%.4f  url=%s", doc_id, sim, url)
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
    summary = state.get("summary")
    if summary is not None:
        final_summary = getattr(summary, "content", summary)
        final_resp = await call_with_limits(lambda: final_prompt.ainvoke({"question": state["query"], "final_summary": final_summary}))
        final_answer = getattr(final_resp, "content", final_resp)
        logger.info("Answer generation complete")
    else:
        # No summary (e.g. retrieval found no documents) — pass through any
        # answer set upstream instead of crashing on summary.content.
        final_answer = state.get("final_answer", "No documents found for this query.")
        logger.info("No summary in state; skipping final answer generation")


    metadata_map = state.get("metadata_map") or {}

    # Build a relevance-sorted list of sources (ordered by relevance desc).
    # Each source is a dict with the fields: 'title', 'url', and 'relevance'.
    # If a document has no URL, set both title and url to the string 'private'.
    sources: List[dict] = []

    # Build list of (doc_id, relevance) and sort by relevance desc. Missing
    # relevance values are treated as 0.0.
    sim_list = []
    for did, meta in metadata_map.items():
        meta = meta or {}
        rel = meta.get("relevance")
        try:
            sim_list.append((did, float(rel) if rel is not None else 0.0))
        except Exception:
            sim_list.append((did, 0.0))
    sim_list.sort(key=lambda x: x[1], reverse=True)

    # Build raw source entries (preserving relevance) in relevance order.
    raw_sources = []
    for doc_id, rel in sim_list:
        meta = metadata_map.get(doc_id) or {}
        relevance = float(rel) if rel is not None else 0.0
        url = _extract_url_from_metadata(meta)

        if not url:
            # Per requirement: if the url is missing, return 'private' for both
            # title and url.
            raw_sources.append({"title": "private", "url": "private", "relevance": relevance})
            continue

        # Title should come from metadata when available. Fall back to
        # a sensible value (url) if no explicit title exists.
        title = None
        try:
            title = meta.get("title") or meta.get("heading")
        except Exception:
            title = None
        if not title:
            title = url

        raw_sources.append({"source": meta.get("source"), "title": title, "url": url, "relevance": relevance})

    # Deduplicate entries while preserving relevance order. Use normalized URL
    # as the dedupe key; treat missing/empty normalized keys as 'private'.
    seen = set()
    for s in raw_sources:
        key = _normalize_url_for_dedupe(s.get("url", "")) or "private"
        if key in seen:
            continue
        seen.add(key)
        sources.append(s)

    # Debug log the final source ordering for easy inspection during runs.
    try:
        logger.info("Output node final sources (top 10): %s", sources[:10])
    except Exception:
        logger.debug("Unable to log output sources", exc_info=True)

    return {"final_answer": final_answer, "sources": sources}

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
