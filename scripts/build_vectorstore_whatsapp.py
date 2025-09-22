#!/usr/bin/env python3
import os
import json
import sqlite3
import faiss
import numpy
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data", "WhatsApp")
INDEX_DIR = os.path.join(ROOT, "app", "vectorstore")
SQLITE_PATH = os.path.join(INDEX_DIR, "docs.sqlite")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

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

def load_txt_files(data_dir, existing_ids=None):
    documents = []
    existing_ids = existing_ids or set()

    for filename in os.listdir(data_dir):
        if filename.endswith(".txt"):
            path = os.path.join(data_dir, filename)
            file_id = os.path.splitext(filename)[0]
            if file_id in existing_ids:
                print(f"Skipping already indexed file {file_id}")
                continue

            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            chunks = text_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{file_id}_chunk_{i}"
                metadata = {"id": chunk_id, "source_file": filename, "parent_id": file_id, "chunk_index": i}
                documents.append(Document(page_content=chunk, metadata=metadata))
            print(f"Split {filename} into {len(chunks)} chunks")
    return documents

def insert_documents(conn, documents):
    doc_ids = []
    for doc in documents:
        doc_id = doc.metadata.get("id")
        doc_ids.append(doc_id)
        conn.execute(
            "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
            (doc_id, doc.page_content, json.dumps(doc.metadata))
        )
    conn.commit()
    return doc_ids

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = init_sqlite(SQLITE_PATH)
    existing_ids = get_existing_ids(conn)
    new_docs = load_txt_files(DATA_DIR, existing_ids)

    if not new_docs:
        print("No new WhatsApp docs to index.")
        return

    print(f"Embedding {len(new_docs)} new chunks...")
    embeddings = embedding_model.embed_documents([d.page_content for d in new_docs])

    # Build or update FAISS
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Updating existing FAISS index...")
        old_index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "id_map.json"), "r") as f:
            old_id_map = json.load(f)
        old_size = old_index.ntotal
        old_index.add(numpy.array(embeddings, dtype="float32"))
        id_map = old_id_map.copy()
        for i, doc in enumerate(new_docs):
            id_map[str(old_size + i)] = doc.metadata["id"]
        faiss_index = FAISS(embedding_function=embedding_model, index=old_index, docstore=None, index_to_docstore_id=id_map)
    else:
        print("Creating new FAISS index for WhatsApp docs...")
        faiss_index = FAISS.from_documents(new_docs, embedding_model)
        id_map = {}
        for i, doc in enumerate(new_docs):
            id_map[str(i)] = doc.metadata["id"]
        faiss_index.index_to_docstore_id = id_map

    faiss.write_index(faiss_index.index, os.path.join(INDEX_DIR, "index.faiss"))
    with open(os.path.join(INDEX_DIR, "id_map.json"), "w") as f:
        json.dump(faiss_index.index_to_docstore_id, f)

    print(f"Saved FAISS index to {INDEX_DIR}")

if __name__ == "__main__":
    main()
