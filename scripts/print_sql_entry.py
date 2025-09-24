#!/usr/bin/env python3
"""
Print one entry from the SQLite `docs` table used by the project.

Usage:
  python scripts/print_sql_entry.py --db ./app/vectorstore/docs.sqlite --id <doc_id>
  # or to show the first row when --id is omitted
  python scripts/print_sql_entry.py --db ./app/vectorstore/docs.sqlite

The script prints id, a short preview of content, the parsed metadata (pretty JSON),
and if present, a short sample of the `Embedding` field inside metadata.

This is intentionally simple and prints helpful debug info for quick terminal inspection.
"""

import argparse
import json
import sqlite3
import sys
from typing import Any
import numpy as np


def print_pretty(obj: Any) -> None:
    try:
        print(json.dumps(obj, indent=2, ensure_ascii=False))
    except Exception:
        print(repr(obj))


def main():
    p = argparse.ArgumentParser(description="Print a docs.sqlite row and metadata sample")
    p.add_argument("--db", default="./app/vectorstore/docs.sqlite", help="Path to docs.sqlite")
    p.add_argument("--id", help="Document id to lookup. If omitted, the first row will be shown.")
    args = p.parse_args()

    try:
        conn = sqlite3.connect(args.db)
    except Exception as exc:
        print(f"ERROR: could not open database '{args.db}': {exc}")
        sys.exit(2)

    cur = conn.cursor()

    if args.id:
        cur.execute("SELECT id, content, metadata, Embedding FROM docs WHERE id=?", (args.id,))
    else:
        cur.execute("SELECT id, content, metadata, Embedding FROM docs LIMIT 1")

    row = cur.fetchone()
    if not row:
        if args.id:
            print(f"No row found for id={args.id}")
        else:
            print(f"No rows found in database {args.db}")
        sys.exit(1)

    doc_id, content, metadata_json, embedding_col = row
    print("--- ROW SUMMARY ---")
    print(f"id: {doc_id}")
    if content:
        preview = content[:500].replace('\n', ' ') if isinstance(content, str) else repr(content)
        print(f"content preview (first 500 chars):\n{preview}\n")
    else:
        print("content: <empty>\n")

    # Parse metadata JSON
    try:
        meta = json.loads(metadata_json) if isinstance(metadata_json, str) else metadata_json
    except Exception as exc:
        print(f"ERROR: failed to parse metadata JSON for id={doc_id}: {exc}")
        print("Raw metadata field:")
        print(metadata_json)
        sys.exit(3)

    print("--- METADATA (pretty) ---")
    print_pretty(meta)

    # The repository stores embeddings in their own column named 'Embedding'.
    if embedding_col is None:
        print("\nEmbedding: NOT FOUND in Embedding column")
        sys.exit(4)

    # Try to parse or show a sample from the embedding column. If it's a
    # binary blob, interpret as float32; if it's JSON text, parse it; else
    # assume it's a sequence.
    try:
        if isinstance(embedding_col, (bytes, bytearray)):
            print(f"Raw Embedding column byte length: {len(embedding_col)} bytes")
            try:
                emb_arr = np.frombuffer(embedding_col, dtype=np.float32)
                emb_list = emb_arr.tolist()
            except Exception as exc:
                print(f"ERROR: failed to interpret Embedding bytes as float32 for id={doc_id}: {exc}")
                print("Raw Embedding column value (repr, truncated):")
                print(repr(embedding_col)[:1000])
                sys.exit(6)
        elif isinstance(embedding_col, str):
            emb_list = json.loads(embedding_col)
        else:
            emb_list = embedding_col
    except Exception as exc:
        print(f"ERROR: failed to parse Embedding column for id={doc_id}: {exc}")
        print("Raw Embedding column value:")
        print(repr(embedding_col)[:1000])
        sys.exit(5)

    try:
        emb_len = len(emb_list)
    except Exception:
        emb_len = "<unknown>"
    print(f"\nEmbedding: length={emb_len}")
    try:
        sample = emb_list[:10]
        print(f"sample (first up to 10 elements): {sample}")
    except Exception:
        print("Unable to display embedding sample")

    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
