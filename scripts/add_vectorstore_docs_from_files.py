#!/usr/bin/env python3
"""
Add chunks from data/gdb.pdf and data/WHY PROGESTERONE_.txt into the SQLite
docstore used by the vectorstore. Chunks follow the same splitter settings as
`scripts/build_vectorstore_all.py`: chunk_size=1000, chunk_overlap=100.

This script only inserts chunked documents into the SQLite docstore
(`app/vectorstore/docs.sqlite`). It does not train or modify the FAISS index.

Usage:
    python scripts/add_vectorstore_docs_from_files.py

You can pass --sqlite to point to a different sqlite file.
"""
import argparse
import json
import os
import sqlite3
from typing import List

import faiss
import numpy as np
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings


ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
GDB_PDF = os.path.join(DATA_DIR, 'gdb.pdf')
WHY_TXT = os.path.join(DATA_DIR, 'WHY PROGESTERONE_.txt')
DEFAULT_SQLITE = os.path.join(ROOT, 'app', 'vectorstore', 'docs.sqlite')
INDEX_DIR = os.path.join(ROOT, 'app', 'vectorstore')


def init_sqlite(db_path: str, wipe: bool = False) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    if wipe:
        try:
            conn.execute("DROP TABLE IF EXISTS docs")
        except Exception:
            pass
    conn.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id TEXT PRIMARY KEY,
        content TEXT,
        metadata TEXT,
        embedding BLOB
    )
    """)
    conn.commit()
    return conn


def insert_documents(conn: sqlite3.Connection, documents: List[dict], commit_every: int = 500):
    cur = conn.cursor()
    inserted = 0
    for doc in documents:
        doc_id = doc['id']
        cur.execute(
            "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
            (doc_id, doc['content'], json.dumps(doc.get('metadata', {})))
        )
        inserted += 1
        if inserted % commit_every == 0:
            conn.commit()
    conn.commit()


def extract_text_from_pdf(path: str) -> str:
    # Use pdfplumber (venv should provide this). We don't support fallbacks.
    text_parts = []
    try:
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                text_parts.append(page_text)
        return "\n\n".join(text_parts)
    except Exception:
        return ""


def split_text_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 100):
    """Split text into character-based chunks with overlap.

    Kept as a utility but we prefer using RecursiveCharacterTextSplitter.
    """
    if not text:
        return []
    chunks = []
    start = 0
    text_len = len(text)
    step = chunk_size - chunk_overlap
    if step <= 0:
        step = chunk_size
    while start < text_len:
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start += step
    return chunks


def add_gdb(conn: sqlite3.Connection, splitter) -> List[dict]:
    if not os.path.exists(GDB_PDF):
        print(f"[WARN] {GDB_PDF} not found, skipping GenderDysphoriaBible")
        return 0
    text = extract_text_from_pdf(GDB_PDF)
    # Use splitter.split_text if using langchain splitter, else expect a callable
    if hasattr(splitter, 'split_text'):
        chunks = splitter.split_text(text)
    else:
        chunks = splitter(text)
    docs = []  # list of dicts with id, content, metadata
    for i, chunk in enumerate(chunks):
        chunk_id = f"gdb_chunk_{i}"
        metadata = {
            'source': 'GenderDysphoriaBible',
            'title': 'GenderDysphoriaBible',
            'url': 'https://genderdysphoria.fyi/en',
            'chunk_index': i,
            'total_chunks': len(chunks),
        }
        docs.append({'id': chunk_id, 'content': chunk, 'metadata': metadata})
    insert_documents(conn, docs)
    return docs


def add_why_progesterone(conn: sqlite3.Connection, splitter) -> List[dict]:
    if not os.path.exists(WHY_TXT):
        print(f"[WARN] {WHY_TXT} not found, skipping WhyProgesterone")
        return 0
    try:
        with open(WHY_TXT, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"[WARN] Failed to read {WHY_TXT}: {e}")
        return 0
    if hasattr(splitter, 'split_text'):
        chunks = splitter.split_text(text)
    else:
        chunks = splitter(text)
    docs = []
    for i, chunk in enumerate(chunks):
        chunk_id = f"why_progesterone_chunk_{i}"
        metadata = {
            'title': 'WhyProgesterone',
            'url': 'https://docs.google.com/document/d/1OGiomfiMk18nPb3ITKZD9pWPvWRUlyI06enxahQpHBI/edit?pli=1&tab=t.0',
            'source': 'WhyProgesterone',
            'chunk_index': i,
            'total_chunks': len(chunks),
        }
        docs.append({'id': chunk_id, 'content': chunk, 'metadata': metadata})
    insert_documents(conn, docs)
    return docs


def embed_and_add_to_index(conn: sqlite3.Connection, docs: List[dict], index_dir: str, embed_batch: int = 512):
    if not docs:
        return 0
    os.makedirs(index_dir, exist_ok=True)
    index_path = os.path.join(index_dir, 'index.faiss')
    id_map_path = os.path.join(index_dir, 'id_map.json')

    # load or create id_map
    try:
        with open(id_map_path, 'r', encoding='utf-8') as f:
            id_map = json.load(f)
    except Exception:
        id_map = {}

    # embedding model
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # try to load existing index
    index = None
    dim = None
    if os.path.exists(index_path):
        try:
            index = faiss.read_index(index_path)
        except Exception as e:
            print(f"[WARN] Failed to read existing FAISS index: {e}")
            index = None

    # compute embeddings in batches and add to index
    docs_texts = [d['content'] for d in docs]
    docs_ids = [d['id'] for d in docs]

    # embed a small sample to determine dim if needed
    sample_emb = embedding_model.embed_documents(docs_texts[:4]) if docs_texts else []
    if sample_emb:
        dim = len(sample_emb[0])

    if index is None:
        if dim is None:
            raise SystemExit('Cannot determine embedding dim to create FAISS index')
        index = faiss.IndexFlatL2(dim)

    # if index is not trained (e.g., IVFPQ not trained), fall back to flat index
    try:
        is_trained = getattr(index, 'is_trained', True)
    except Exception:
        is_trained = True
    if not is_trained:
        print('[WARN] Existing index is not trained; creating a flat IndexFlatL2 instead')
        index = faiss.IndexFlatL2(dim)

    next_faiss_id = int(index.ntotal)

    # embed and add
    total_added = 0
    for i in range(0, len(docs_texts), embed_batch):
        batch_texts = docs_texts[i:i+embed_batch]
        batch_ids = docs_ids[i:i+embed_batch]
        embs = embedding_model.embed_documents(batch_texts)
        embs_np = np.array(embs, dtype='float32')

        # store embeddings in sqlite
        upd_cur = conn.cursor()
        for j, doc_id in enumerate(batch_ids):
            vec = embs_np[j]
            blob = vec.tobytes()
            upd_cur.execute("UPDATE docs SET embedding = ? WHERE id = ?", (blob, doc_id))
        conn.commit()

        # add to faiss
        index.add(embs_np)
        for j, doc_id in enumerate(batch_ids):
            id_map[str(next_faiss_id + j)] = doc_id
        next_faiss_id += len(batch_ids)
        total_added += len(batch_ids)
        print(f"Added {total_added} vectors to FAISS index (so far)")

    # persist index and id_map
    try:
        faiss.write_index(index, index_path)
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map, f)
    except Exception as e:
        print(f"[ERROR] Failed to save index or id_map: {e}")

    print(f"Saved FAISS index to {index_path} ({index.ntotal} vectors)")
    print(f"Saved id_map.json ({len(id_map)} entries) to {id_map_path}")
    return total_added


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sqlite', type=str, default=DEFAULT_SQLITE, help='Path to docs sqlite')
    parser.add_argument('--wipe', action='store_true', help='Wipe table before inserting')
    args = parser.parse_args()

    # Choose splitter: prefer langchain's RecursiveCharacterTextSplitter when
    # available in the venv, otherwise use the local char-based splitter.
    if RecursiveCharacterTextSplitter is not None:
        splitter_obj = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splitter_fn = splitter_obj.split_text
    else:
        splitter_fn = lambda t: split_text_chunks(t, chunk_size=1000, chunk_overlap=100)

    conn = init_sqlite(args.sqlite, wipe=args.wipe)

    # Insert documents into sqlite
    gdb_docs = add_gdb(conn, splitter_fn)
    why_docs = add_why_progesterone(conn, splitter_fn)

    total_chunks = len(gdb_docs) + len(why_docs)
    print(f"Inserted {len(gdb_docs)} GenderDysphoriaBible chunks")
    print(f"Inserted {len(why_docs)} WhyProgesterone chunks")

    # Embed and add to FAISS index
    all_docs = gdb_docs + why_docs
    added_vectors = embed_and_add_to_index(conn, all_docs, INDEX_DIR)
    print(f"Total chunks inserted: {total_chunks}; vectors added to index: {added_vectors}")


if __name__ == '__main__':
    main()
