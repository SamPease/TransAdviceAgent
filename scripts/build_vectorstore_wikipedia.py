#!/usr/bin/env python3
import os
import json
import sqlite3
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

ROOT = os.path.dirname(os.path.dirname(__file__))
WIKI_DATA_ROOT = os.path.join(ROOT, "data", "wikipedia")
INDEX_DIR = os.path.join(ROOT, "app", "vectorstore")
SQLITE_PATH = os.path.join(INDEX_DIR, "docs.sqlite")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

def init_sqlite(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
    CREATE TABLE IF NOT EXISTS docs (
        id TEXT PRIMARY KEY,
        content TEXT,
        metadata TEXT
    )
    """)
    conn.commit()
    return conn

def get_existing_ids(conn):
    rows = conn.execute("SELECT id FROM docs").fetchall()
    return set(r[0] for r in rows)

def safe_chunk_id(pageid, i):
    return f"{pageid}_chunk_{i}"

def iter_wikipedia_jsons(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fn.endswith('.json'):
                continue
            yield os.path.join(dirpath, fn)

def load_documents(existing_ids=None):
    existing_ids = existing_ids or set()
    documents = []
    skipped = 0

    for path in iter_wikipedia_jsons(WIKI_DATA_ROOT):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                page = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue

        pageid = page.get('pageid') or page.get('id') or os.path.splitext(os.path.basename(path))[0]
        title = page.get('title', '')
        url = page.get('url')
        text = page.get('text')
        categories = page.get('categories', [])

        if not text:
            skipped += 1
            print(f"[INFO] Skipping {path} (empty text)")
            continue

        content = title + '\n\n' + text
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = safe_chunk_id(pageid, i)
            if chunk_id in existing_ids:
                # Already indexed
                continue
            metadata = {
                'id': chunk_id,
                'parent_id': str(pageid),
                'title': title,
                'url': url,
                'pageid': pageid,
                'categories': categories,
                'chunk_index': i,
                'total_chunks': len(chunks)
            }
            documents.append(Document(page_content=chunk, metadata=metadata))

    print(f"[INFO] Prepared {len(documents)} new chunks (skipped {skipped} pages with no text)")
    return documents

def insert_documents(conn, documents):
    for doc in documents:
        doc_id = doc.metadata.get('id')
        conn.execute(
            "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
            (doc_id, doc.page_content, json.dumps(doc.metadata))
        )
    conn.commit()

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = init_sqlite(SQLITE_PATH)
    existing_ids = get_existing_ids(conn)
    print(f"[INFO] {len(existing_ids)} chunk ids already in SQLite docstore")

    new_docs = load_documents(existing_ids)
    if not new_docs:
        print("[INFO] No new documents to index. Exiting.")
        return

    # Insert metadata/content into SQLite docstore
    insert_documents(conn, new_docs)
    print(f"[INFO] Inserted {len(new_docs)} document chunks into SQLite")

    # Embed documents
    texts = [d.page_content for d in new_docs]
    print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
    embeddings = embedding_model.embed_documents(texts)
    embs = np.array(embeddings, dtype='float32')

    # Build or update FAISS index
    index_path = os.path.join(INDEX_DIR, 'index.faiss')
    id_map_path = os.path.join(INDEX_DIR, 'id_map.json')

    if os.path.exists(index_path):
        print('[INFO] Loading existing FAISS index...')
        old_index = faiss.read_index(index_path)
        with open(id_map_path, 'r', encoding='utf-8') as f:
            old_id_map = json.load(f)
        old_size = old_index.ntotal
        old_index.add(embs)

        # extend id map
        id_map = old_id_map.copy()
        for i, doc in enumerate(new_docs):
            id_map[str(old_size + i)] = doc.metadata['id']

        faiss.write_index(old_index, index_path)
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map, f)
        print(f"[INFO] Updated FAISS index; new size: {old_index.ntotal}")
    else:
        print('[INFO] Creating new FAISS index...')
        faiss_index = FAISS.from_documents(new_docs, embedding_model)
        # ensure id_map is strings
        id_map = {str(i): new_docs[i].metadata['id'] for i in range(len(new_docs))}
        faiss.write_index(faiss_index.index, index_path)
        with open(id_map_path, 'w', encoding='utf-8') as f:
            json.dump(id_map, f)
        print(f"[INFO] Created new FAISS index with {faiss_index.index.ntotal} vectors")

if __name__ == '__main__':
    main()
