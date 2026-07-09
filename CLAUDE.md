# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A RAG-powered Q&A API about transgender care topics. FastAPI serves a single `/ask` endpoint that runs a LangGraph pipeline over a FAISS vector index + SQLite docstore built from Reddit posts, WhatsApp exports, Wikipedia articles, and curated documents. Answers are generated with Anthropic Claude and returned with relevance-sorted source attributions.

## Commands

```bash
pip install -r requirements.txt          # install deps (use .venv)

uvicorn app.main:app --reload            # run the API server
python app/rag_pipeline.py               # interactive CLI REPL for the pipeline (no server)

pytest                                   # run tests
pytest tests/test_rag_pipeline_output_contract.py::test_output_node_builds_sources_order_and_dedup  # single test

python scripts/build_vectorstore_all.py  # rebuild FAISS index + SQLite docstore from data/
python -m scripts.fetch_vectorstore --repo-id SamPease/TransAdviceAgent  # download vectorstore artifacts from HF
```

**Environment variables** (loaded from `.env` via python-dotenv):
- `ANTHROPIC_API_KEY` — required for generation
- `LANGSMITH_API_KEY` — required even to *import* `app.rag_pipeline` (see below)
- `HF_TOKEN` (or `HUGGINGFACEHUB_API_TOKEN` / `HUGGINGFACE_TOKEN`) — HF endpoint embeddings and private dataset download
- `LOCAL_EMBEDDING_MODEL=1` — use local sentence-transformers instead of the HF inference endpoint
- `VECTORSTORE_PATH` — override vectorstore dir (default `./app/vectorstore`)
- `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USER_AGENT` — only for `scripts/get_reddit.py`

**Import-time side effects / test caveat:** `app/rag_pipeline.py` pulls its four prompt templates (`query_enhancement`, `map_prompt`, `reduce_prompt`, `final_prompt`) from LangSmith at module import and pipes each into a `ChatAnthropic` constructed in code (`MODEL_NAME`, overridable via `ANTHROPIC_MODEL`). Importing the module — including running pytest — therefore requires network access and a valid `LANGSMITH_API_KEY`. Do not pull prompts with `include_model=True`: that bakes the model choice into LangSmith config, which is how the pipeline previously broke when its pinned model (`claude-3-5-haiku-latest`) was retired.

## Architecture

Almost all logic lives in `app/rag_pipeline.py`. `app/main.py` is a thin FastAPI wrapper (CORS locked to `https://sampease.github.io`; downloads vectorstore files on startup).

### LangGraph pipeline (retrieve → summarize → output)

State is `ChatAgentState` (TypedDict, `total=False`). Nodes communicate via `doc_ids`, `metadata_map` (doc_id → metadata incl. injected `relevance` score), `summary`, `final_answer`, `sources`.

1. **retrieve_node** — enhances the query with the `query_enhancement` prompt, embeds it, searches FAISS for `DOCS_FETCH_LIMIT` (200) candidates, then does a memory-efficient MMR rerank down to `DOCS_KEEP_LIMIT` (100). MMR pulls per-document embeddings from the SQLite `Embedding` column (`_get_embedding_from_sql`) rather than reconstructing from FAISS — the index is IVFPQ (lossy/compressed), so exact embeddings only exist in SQLite. Returns doc IDs relevance-sorted, with `relevance` stashed into each doc's metadata.
2. **summarize_node** — map/reduce summarization: docs in batches of `DOC_BATCH_SIZE` (10) through `map_prompt` in parallel, then iterative reduction rounds through `reduce_prompt` until one summary remains.
3. **output_node** — generates the final answer with `final_prompt` and builds `sources`: relevance-sorted dicts of `{title, url, relevance}`, deduped by normalized URL; docs without any URL become `{"title": "private", "url": "private"}`.

**Output contract:** `run_rag` raises `RuntimeError` if `sources` is missing from the workflow result — `output_node` must always set it, and `sources` must be declared in `ChatAgentState` or the compiled graph strips it. `tests/test_rag_pipeline_output_contract.py` covers source ordering/dedupe.

### Rate limiting

All LLM and embedding calls go through `call_with_limits()` — a process-global semaphore (`GLOBAL_CONCURRENCY=10`) + sliding-window rate limiter (45 calls/60s) with Retry-After-aware exponential backoff. When adding any new LLM/embedding call, wrap it in `call_with_limits(lambda: ...)` (bind loop variables as lambda defaults to avoid late-binding bugs — see existing nodes).

### Vectorstore artifacts

`app/vectorstore/` is a **git submodule** pointing at the HuggingFace dataset `SamPease/TransAdviceAgent` (files tracked with LFS): `index.faiss` (IVFPQ, mmap'd at load), `docs.sqlite` (columns: `id`, `content`, `metadata` JSON, `Embedding` float32 blob), and `id_map.json` (FAISS position → doc id). `ensure_vectorstore_files()` downloads missing files from HF at build/startup, falling back to `/tmp` if the target dir is read-only. Rebuilding is done by `scripts/build_vectorstore_all.py` (chunk_size=1000, overlap=100, all-MiniLM-L6-v2 embeddings, IVFPQ nlist=1024/m=64/nbits=8); other scripts in `scripts/` are one-off data collection (`get_reddit.py`, `fetch_wikipedia_from_csv.py`) and docstore repair/inspection tools.

`data/` is gitignored (raw source data lives only locally).

### Deployment

Deployed on Render: `render_build.sh` installs deps and pre-fetches vectorstore files. The frontend is a static page at sampease.github.io.
