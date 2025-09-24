#!/usr/bin/env python3
"""
Build a combined FAISS IndexIVFPQ index and SQLite docstore from reddit, WhatsApp,
and wikipedia data. This script will (by default) overwrite the existing
`app/vectorstore/docs.sqlite`, `index.faiss`, and `id_map.json`.

Defaults (per your request):
 - chunk_size=1000, chunk_overlap=100
 - embedding model: sentence-transformers/all-MiniLM-L6-v2
 - index type: IndexIVFPQ
 - nlist=1024, m=64, nbits=8
 - train size: 50000 random vectors (or fewer if DB smaller)

Usage:
    python scripts/build_vectorstore_all.py

You can override parameters with command-line flags.
"""
import argparse
import json
import os
import random
import sqlite3
import faiss
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_REDDIT = os.path.join(ROOT, "data", "reddit")
DATA_WHATSAPP = os.path.join(ROOT, "data", "WhatsApp")
DATA_WIKI = os.path.join(ROOT, "data", "wikipedia")
INDEX_DIR = os.path.join(ROOT, "app", "vectorstore")
SQLITE_PATH = os.path.join(INDEX_DIR, "docs.sqlite")


def init_sqlite(db_path, wipe=False):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    if wipe:
        # remove old table then recreate
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


def insert_documents(conn, documents, commit_every=500):
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


def reddit_iterator(data_dir, text_splitter):
    # yields dicts with id, content, metadata
    for filename in os.listdir(data_dir):
        if not filename.endswith('.json'):
            continue
        path = os.path.join(data_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                post = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        post_id = post.get('id')
        if not post_id:
            continue
        text_parts = [post.get('title', ''), post.get('selftext', '')]
        for c in post.get('comments', []):
            text_parts.append(c.get('body', ''))
        content = '\n'.join(text_parts)
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{post_id}_chunk_{i}"
            # Extract additional metadata fields from the original reddit JSON
            # Include the fields the user requested: id, title, author, score, url, num_comments, created_utc
            permalink = post.get('permalink')
            if isinstance(permalink, str) and permalink.startswith('/'):
                url = f"https://reddit.com{permalink}"
            else:
                url = post.get('url') or permalink
            num_comments = post.get('num_comments') if post.get('num_comments') is not None else post.get('comments_count')
            created_utc = post.get('created_utc') or post.get('created') or post.get('created_tuc')
            metadata = {
                'source': 'reddit',
                'parent_id': post_id,
                'id': post_id,
                'title': post.get('title'),
                'author': post.get('author'),
                'score': post.get('score'),
                'url': url,
                'num_comments': num_comments,
                'created_utc': created_utc,
                'chunk_index': i,
                'total_chunks': len(chunks),
            }
            yield {'id': chunk_id, 'content': chunk, 'metadata': metadata}


def whatsapp_iterator(data_dir, text_splitter):
    for filename in os.listdir(data_dir):
        if not filename.endswith('.txt'):
            continue
        file_id = os.path.splitext(filename)[0]
        path = os.path.join(data_dir, filename)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"[WARN] Failed to read {path}: {e}")
            continue
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_id}_chunk_{i}"
            metadata = {'source': 'whatsapp', 'source_file': filename, 'chunk_index': i, 'total_chunks': len(chunks)}
            yield {'id': chunk_id, 'content': chunk, 'metadata': metadata}


def iter_wikipedia_jsons(root):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.json'):
                yield os.path.join(dirpath, fn)


def wikipedia_iterator(root, text_splitter):
    for path in iter_wikipedia_jsons(root):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                page = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load {path}: {e}")
            continue
        pageid = page.get('pageid') or page.get('id') or os.path.splitext(os.path.basename(path))[0]
        title = page.get('title', '')
        text = page.get('text')
        if not text:
            continue
        content = title + '\n\n' + text
        chunks = text_splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{pageid}_chunk_{i}"
            # Prefer explicit URL fields if present, else fall back to curid style
            url = page.get('fullurl') or page.get('canonicalurl') or page.get('url')
            if not url and pageid is not None:
                url = f"https://en.wikipedia.org/?curid={pageid}"
            metadata = {
                'source': 'wikipedia',
                'pageid': pageid,
                'title': title,
                'url': url,
                'chunk_index': i,
                'total_chunks': len(chunks),
            }
            yield {'id': chunk_id, 'content': chunk, 'metadata': metadata}


def build_index(conn, embedding_model, index_dir, nlist, m, nbits, train_size, embed_batch=512):
    os.makedirs(index_dir, exist_ok=True)
    total = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    print(f"Total documents in SQLite docstore: {total}")
    if total == 0:
        raise SystemExit('No documents to index')

    # determine embedding dimension by embedding a small sample
    cur = conn.cursor()
    cur.execute("SELECT content FROM docs LIMIT 4")
    samples = [r[0] for r in cur.fetchall()]
    sample_embs = embedding_model.embed_documents(samples)
    dim = len(sample_embs[0])
    print(f"Detected embedding dimension: {dim}")

    # validate m
    if dim % m != 0:
        print(f"Warning: embedding dim {dim} is not divisible by m={m}. Adjusting m to divisor.")
        # choose largest divisor <= m
        for candidate in range(min(m, dim), 0, -1):
            if dim % candidate == 0:
                m = candidate
                print(f"Adjusted m to {m}")
                break

    # build quantizer + index
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits)

    # sample random documents for training
    train_size = min(train_size, total)
    print(f"Sampling {train_size} random vectors for training")
    cur.execute(f"SELECT content FROM docs ORDER BY RANDOM() LIMIT {train_size}")
    train_texts = [r[0] for r in cur.fetchall()]

    # embed train_texts in batches
    train_vecs = []
    for i in range(0, len(train_texts), embed_batch):
        batch_texts = train_texts[i:i+embed_batch]
        embs = embedding_model.embed_documents(batch_texts)
        train_vecs.extend(embs)
        print(f"Embedded training batch {i // embed_batch + 1}/{(len(train_texts) + embed_batch - 1) // embed_batch}")
    train_np = np.array(train_vecs, dtype='float32')

    print(f"Training IndexIVFPQ on {train_np.shape[0]} vectors...")
    index.train(train_np)

    # add all vectors in batches and build id_map
    id_map = {}
    next_faiss_id = 0
    cur2 = conn.cursor()
    offset = 0
    batch_size = embed_batch
    while True:
        cur2.execute("SELECT id, content FROM docs LIMIT ? OFFSET ?", (batch_size, offset))
        rows = cur2.fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        embs = []
        for i in range(0, len(texts), embed_batch):
            embs_batch = embedding_model.embed_documents(texts[i:i+embed_batch])
            embs.extend(embs_batch)
        embs_np = np.array(embs, dtype='float32')
        # store embeddings in sqlite as BLOBs (float32 bytes) so we can use them later
        upd_cur = conn.cursor()
        for i, doc_id in enumerate(ids):
            vec = embs_np[i]
            blob = vec.tobytes()
            upd_cur.execute("UPDATE docs SET embedding = ? WHERE id = ?", (blob, doc_id))
        conn.commit()
        index.add(embs_np)
        for i, doc_id in enumerate(ids):
            id_map[str(next_faiss_id + i)] = doc_id
        next_faiss_id += len(ids)
        offset += len(rows)
        print(f"Added {next_faiss_id}/{total} vectors to index")

    index_path = os.path.join(index_dir, 'index.faiss')
    id_map_path = os.path.join(index_dir, 'id_map.json')
    faiss.write_index(index, index_path)
    with open(id_map_path, 'w', encoding='utf-8') as f:
        json.dump(id_map, f)

    print(f"Saved IndexIVFPQ ({index.ntotal} vectors) to {index_path}")
    print(f"Saved id_map.json ({len(id_map)} entries) to {id_map_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlist', type=int, default=1024)
    parser.add_argument('--m', type=int, default=64)
    parser.add_argument('--nbits', type=int, default=8)
    parser.add_argument('--train-size', type=int, default=50000)
    parser.add_argument('--wipe', action='store_true', help='Wipe existing sqlite table before building')
    parser.add_argument('--embed-batch', type=int, default=512)
    args = parser.parse_args()

    # embedding model + splitter
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    # init sqlite (wipe if requested)
    conn = init_sqlite(SQLITE_PATH, wipe=True)

    # Clear table to rebuild fresh (user requested rebuild)
    conn.execute('DELETE FROM docs')
    conn.commit()

    # Insert documents from reddit, whatsapp, wikipedia
    print('Indexing reddit data...')
    batch = []
    batch_size = 200
    for doc in reddit_iterator(DATA_REDDIT, text_splitter):
        batch.append(doc)
        if len(batch) >= batch_size:
            insert_documents(conn, batch)
            batch = []
    if batch:
        insert_documents(conn, batch)

    print('Indexing WhatsApp data...')
    batch = []
    for doc in whatsapp_iterator(DATA_WHATSAPP, text_splitter):
        batch.append(doc)
        if len(batch) >= batch_size:
            insert_documents(conn, batch)
            batch = []
    if batch:
        insert_documents(conn, batch)

    print('Indexing Wikipedia data...')
    batch = []
    for doc in wikipedia_iterator(DATA_WIKI, text_splitter):
        batch.append(doc)
        if len(batch) >= batch_size:
            insert_documents(conn, batch)
            batch = []
    if batch:
        insert_documents(conn, batch)

    print('Finished inserting documents into SQLite')

    build_index(conn, embedding_model, INDEX_DIR, args.nlist, args.m, args.nbits, args.train_size, embed_batch=args.embed_batch)


if __name__ == '__main__':
    main()
