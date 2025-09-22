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
DATA_DIR = os.path.join(ROOT, "data", "reddit")
INDEX_DIR = os.path.join(ROOT, "app", "vectorstore")
SQLITE_PATH = os.path.join(INDEX_DIR, "docs.sqlite")

# --------------------------
# Embeddings and text splitting
# --------------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# --------------------------
# SQLite docstore utilities
# --------------------------
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

def insert_documents(conn, documents):
    doc_ids = []  # Track assigned IDs
    
    for doc in documents:
        # Get the chunk ID that was assigned during document creation
        doc_id = doc.metadata.get("id")
        if not doc_id:
            print(f"[ERROR] Document missing ID in metadata: {doc.metadata}")
            continue
            
        doc_ids.append(doc_id)
        
        print(f"[DEBUG] Inserting document {doc_id}")
        conn.execute(
            "INSERT OR REPLACE INTO docs (id, content, metadata) VALUES (?, ?, ?)",
            (doc_id, doc.page_content, json.dumps(doc.metadata))
        )
        
        if len(doc_ids) % 100 == 0:
            print(f"[DEBUG] Inserted {len(doc_ids)} documents...")
            conn.commit()  # Periodic commits
            
    conn.commit()  # Final commit
    
    # Count unique parent posts
    unique_parents = len(set(doc.metadata.get("parent_id") for doc in documents))
    print(f"[DEBUG] Total documents inserted: {len(doc_ids)}")
    print(f"[DEBUG] Total unique parent posts: {unique_parents}")
    return doc_ids

def get_existing_ids(conn):
    rows = conn.execute("SELECT id FROM docs").fetchall()
    return set(r[0] for r in rows)

# --------------------------
# Load JSON and create document chunks
# --------------------------
def load_json_files(data_dir, existing_ids=None):
    documents = []
    existing_ids = existing_ids or set()

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                post = json.load(f)
                post_id = post["id"]
                if post_id in existing_ids:
                    print(f"Skipping already indexed post {post_id}")
                    continue

                text_parts = [post["title"], post.get("selftext", "")]
                for comment in post.get("comments", []):
                    text_parts.append(comment.get("body", ""))
                content = "\n".join(text_parts)
                metadata = {
                    "id": post_id,
                    "url": post["url"],
                    "author": post["author"],
                    "num_comments": post["num_comments"],
                    "score": post["score"],
                    "created_utc": post["created_utc"]
                }

                # Split into chunks
                chunks = text_splitter.split_text(content)
                for i, chunk in enumerate(chunks):
                    chunk_id = f"{post_id}_chunk_{i}"
                    chunk_metadata = metadata.copy()
                    chunk_metadata["id"] = chunk_id
                    chunk_metadata["parent_id"] = post_id
                    chunk_metadata["chunk_index"] = i
                    chunk_metadata["total_chunks"] = len(chunks)
                    documents.append(Document(page_content=chunk, metadata=chunk_metadata))
                print(f"[DEBUG] Created {len(chunks)} chunks for post {post_id}")
    return documents

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    conn = init_sqlite(SQLITE_PATH)

    existing_ids = get_existing_ids(conn)
    print(f"{len(existing_ids)} documents already in SQL docstore")

    new_docs = load_json_files(DATA_DIR, existing_ids)
    print(f"Found {len(new_docs)} new document chunks")

    if not new_docs:
        print("No new documents to add. Exiting.")
        return

    doc_ids = insert_documents(conn, new_docs)
    print(f"[DEBUG] Inserted {len(doc_ids)} documents into SQLite")
    print(f"[DEBUG] Sample doc IDs: {doc_ids[:5]}")

    # Build or update FAISS index
    if os.path.exists(os.path.join(INDEX_DIR, "index.faiss")):
        print("Loading existing index...")
        old_index = faiss.read_index(os.path.join(INDEX_DIR, "index.faiss"))
        with open(os.path.join(INDEX_DIR, "id_map.json"), "r") as f:
            old_id_map = json.load(f)
        old_size = old_index.ntotal
        print(f"Existing index size: {old_size}")
        print("Embedding new documents...")
        embeddings = embedding_model.embed_documents([d.page_content for d in new_docs])
        old_index.add(numpy.array(embeddings, dtype="float32"))
        print(f"Updated index size: {old_index.ntotal}")
        id_map = old_id_map.copy()
        for i, doc in enumerate(new_docs):
            doc_id = doc.metadata["id"]
            id_map[str(old_size + i)] = doc_id
            print(f"[DEBUG] Mapped FAISS index {old_size + i} -> doc_id {doc_id}")
        faiss_index = FAISS(
            embedding_function=embedding_model,
            index=old_index,
            docstore=None,
            index_to_docstore_id=id_map
        )
        print(f"Index updated with {len(new_docs)} new documents")
    else:
        print("Creating new index...")
        faiss_index = FAISS.from_documents(new_docs, embedding_model)
        corrected_map = {}
        for idx, _ in enumerate(new_docs):
            doc_id = new_docs[idx].metadata["id"]
            corrected_map[str(idx)] = doc_id
            print(f"[DEBUG] Mapped FAISS index {idx} -> doc_id {doc_id}")
        faiss_index.index_to_docstore_id = corrected_map
        print("Created new FAISS index")

    print("[DEBUG] Saving FAISS index...")
    faiss.write_index(faiss_index.index, os.path.join(INDEX_DIR, "index.faiss"))
    print(f"[DEBUG] FAISS index saved with {faiss_index.index.ntotal} vectors")

    print("[DEBUG] Saving id_map.json...")
    with open(os.path.join(INDEX_DIR, "id_map.json"), "w") as f:
        json.dump(faiss_index.index_to_docstore_id, f)
    print("[DEBUG] Sample id_map entries:")
    for idx, doc_id in list(faiss_index.index_to_docstore_id.items())[:5]:
        print(f"  {idx} -> {doc_id}")

    print(f"FAISS index + id_map.json saved to {INDEX_DIR}")

if __name__ == "__main__":
    main()
