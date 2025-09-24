#!/usr/bin/env python3
"""Fix/augment metadata in the SQLite docstore for reddit and wikipedia documents.

The script attempts to avoid a full rebuild by matching chunk IDs / parent ids
to the original source JSON files under `data/reddit` and `data/wikipedia` and
copying desired metadata fields into each chunk's metadata JSON.

Usage:
  python3 scripts/fix_docstore_metadata.py [--db PATH] [--reddit DIR] [--wiki DIR] [--commit-batch N] [--dry-run]

By default it will only print a summary of proposed changes. Use --dry-run to
inspect without modifying the DB; omit it to apply updates.
"""
import argparse
import json
import os
import sqlite3
from typing import Dict, Optional, Any


def build_reddit_index(reddit_dir: str) -> Dict[str, str]:
    """Scan reddit JSON files and build mapping from post id -> filepath."""
    mapping = {}
    if not os.path.isdir(reddit_dir):
        return mapping
    for fn in os.listdir(reddit_dir):
        if not fn.endswith('.json'):
            continue
        path = os.path.join(reddit_dir, fn)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
        except Exception:
            continue
        pid = obj.get('id')
        if pid:
            mapping[str(pid)] = path
    return mapping


def build_wiki_index(wiki_root: str) -> Dict[str, str]:
    """Scan wikipedia JSON files (recursively) and map pageid -> filepath."""
    mapping = {}
    if not os.path.isdir(wiki_root):
        return mapping
    for dirpath, _, filenames in os.walk(wiki_root):
        for fn in filenames:
            if not fn.endswith('.json'):
                continue
            path = os.path.join(dirpath, fn)
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    page = json.load(f)
            except Exception:
                continue
            pid = page.get('pageid') or page.get('id')
            if pid is None:
                # fallback to filename (without .json)
                pid = os.path.splitext(fn)[0]
            mapping[str(pid)] = path
    return mapping


def normalize_reddit_url(post: dict) -> Optional[str]:
    # Try common URL fields
    for key in ('url', 'permalink', 'link', 'href'):
        v = post.get(key)
        if not v:
            continue
        if key == 'permalink':
            if isinstance(v, str) and v.startswith('/'):
                return f"https://reddit.com{v}"
            if isinstance(v, str) and v.startswith('http'):
                return v
            return f"https://reddit.com/{v}"
        if isinstance(v, str) and v.startswith('http'):
            return v
        return str(v)
    # fallback: try to build from subreddit + id
    sub = post.get('subreddit') or post.get('subreddit_name')
    pid = post.get('id')
    if sub and pid:
        return f"https://reddit.com/r/{sub}/comments/{pid}"
    return None


def extract_reddit_fields(post: dict) -> Dict[str, Any]:
    # The user requested: id, title, author, score, url, num_comments, created_utc
    # Normalize several common field names and store the timestamp under
    # the exact key 'created_utc' to match the user's request.
    created = post.get('created_utc') or post.get('created_utc') or post.get('created') or None
    num_comments = post.get('num_comments') if post.get('num_comments') is not None else post.get('comments_count')
    return {
        'id': post.get('id'),
        'title': post.get('title'),
        'author': post.get('author'),
        'score': post.get('score'),
        'url': normalize_reddit_url(post),
        'num_comments': num_comments,
        'created_utc': created,
    }


def extract_wiki_fields(page: dict) -> Dict[str, Any]:
    # The user requested: title, url, pageid
    pid = page.get('pageid') or page.get('id') or None
    # prefer explicit url-like fields if present
    url = page.get('fullurl') or page.get('canonicalurl') or page.get('url')
    if not url and pid is not None:
        # fallback to curid style
        url = f"https://en.wikipedia.org/?curid={pid}"
    return {
        'title': page.get('title'),
        'pageid': pid,
        'url': url,
    }


def load_docs_with_source(conn: sqlite3.Connection, source_keyword: str):
    cur = conn.cursor()
    # Use a LIKE filter to limit rows scanned; this avoids loading entire DB into memory
    q = "SELECT id, metadata FROM docs WHERE metadata LIKE ?"
    pattern = f"%\"source\": \"{source_keyword}\"%"
    cur.execute(q, (pattern,))
    for row in cur.fetchall():
        yield row[0], row[1]


def update_metadata_for_reddit(conn: sqlite3.Connection, reddit_dir: str, dry_run: bool, batch: int = 500):
    idx = build_reddit_index(reddit_dir)
    print(f'Indexed {len(idx)} reddit source files under {reddit_dir}')

    cur = conn.cursor()
    # collect unique parent_ids from metadata
    parent_map = {}
    for doc_id, meta_text in load_docs_with_source(conn, 'reddit'):
        try:
            meta = json.loads(meta_text) if meta_text else {}
        except Exception:
            meta = {}
        parent = meta.get('parent_id') or meta.get('post_id')
        if not parent:
            continue
        parent_map.setdefault(str(parent), []).append(doc_id)

    print(f'Found {len(parent_map)} reddit parent posts with chunks to update')

    updates = 0
    to_update = []
    for parent_id, chunk_ids in parent_map.items():
        src_path = idx.get(str(parent_id))
        if not src_path:
            # source JSON not found; skip
            continue
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                post = json.load(f)
        except Exception:
            continue
        extra = extract_reddit_fields(post)
        # Merge into each chunk's metadata
        for chunk_id in chunk_ids:
            cur.execute('SELECT metadata FROM docs WHERE id=?', (chunk_id,))
            row = cur.fetchone()
            if not row:
                continue
            try:
                meta = json.loads(row[0]) if row[0] else {}
            except Exception:
                meta = {}
            # only add fields if missing or different
            changed = False
            for k, v in extra.items():
                if v is None:
                    continue
                if meta.get(k) != v:
                    meta[k] = v
                    changed = True
            if changed:
                updates += 1
                to_update.append((json.dumps(meta, ensure_ascii=False), chunk_id))
        # commit in batches
        if len(to_update) >= batch:
            print(f'Applying {len(to_update)} reddit metadata updates...')
            if not dry_run:
                cur.executemany('UPDATE docs SET metadata=? WHERE id=?', to_update)
                conn.commit()
            to_update = []

    if to_update:
        print(f'Applying final {len(to_update)} reddit metadata updates...')
        if not dry_run:
            cur.executemany('UPDATE docs SET metadata=? WHERE id=?', to_update)
            conn.commit()

    print(f'Reddit metadata updates prepared: {updates} (dry_run={dry_run})')


def update_metadata_for_wikipedia(conn: sqlite3.Connection, wiki_dir: str, dry_run: bool, batch: int = 500):
    idx = build_wiki_index(wiki_dir)
    print(f'Indexed {len(idx)} wikipedia JSON files under {wiki_dir}')

    cur = conn.cursor()
    page_map = {}
    for doc_id, meta_text in load_docs_with_source(conn, 'wikipedia'):
        try:
            meta = json.loads(meta_text) if meta_text else {}
        except Exception:
            meta = {}
        pageid = meta.get('pageid') or meta.get('page_id') or None
        if pageid is None:
            # Perhaps the chunk id encodes the pageid as prefix
            if '_chunk_' in doc_id:
                pageid = doc_id.split('_chunk_')[0]
        if pageid is None:
            continue
        page_map.setdefault(str(pageid), []).append(doc_id)

    print(f'Found {len(page_map)} wikipedia pages with chunks to update')

    updates = 0
    to_update = []
    for pageid, chunk_ids in page_map.items():
        src_path = idx.get(str(pageid))
        if not src_path:
            # source JSON not found; skip
            continue
        try:
            with open(src_path, 'r', encoding='utf-8') as f:
                page = json.load(f)
        except Exception:
            continue
        extra = extract_wiki_fields(page)
        for chunk_id in chunk_ids:
            cur.execute('SELECT metadata FROM docs WHERE id=?', (chunk_id,))
            row = cur.fetchone()
            if not row:
                continue
            try:
                meta = json.loads(row[0]) if row[0] else {}
            except Exception:
                meta = {}
            changed = False
            for k, v in extra.items():
                if v is None:
                    continue
                if meta.get(k) != v:
                    meta[k] = v
                    changed = True
            if changed:
                updates += 1
                to_update.append((json.dumps(meta, ensure_ascii=False), chunk_id))
        if len(to_update) >= batch:
            print(f'Applying {len(to_update)} wikipedia metadata updates...')
            if not dry_run:
                cur.executemany('UPDATE docs SET metadata=? WHERE id=?', to_update)
                conn.commit()
            to_update = []

    if to_update:
        print(f'Applying final {len(to_update)} wikipedia metadata updates...')
        if not dry_run:
            cur.executemany('UPDATE docs SET metadata=? WHERE id=?', to_update)
            conn.commit()

    print(f'Wikipedia metadata updates prepared: {updates} (dry_run={dry_run})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', default='app/vectorstore/docs.sqlite', help='Path to docs.sqlite')
    parser.add_argument('--reddit', default='data/reddit', help='Directory with reddit JSON files')
    parser.add_argument('--wiki', default='data/wikipedia/lgbt', help='Root directory with wikipedia JSON files')
    parser.add_argument('--commit-batch', type=int, default=500, help='How many updates to apply per transaction')
    parser.add_argument('--dry-run', action='store_true', help='Do not modify DB, only report')
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print('DB not found:', args.db)
        raise SystemExit(1)

    conn = sqlite3.connect(args.db)

    try:
        print('Starting reddit metadata update...')
        update_metadata_for_reddit(conn, args.reddit, args.dry_run, batch=args.commit_batch)
        print('\nStarting wikipedia metadata update...')
        update_metadata_for_wikipedia(conn, args.wiki, args.dry_run, batch=args.commit_batch)
    finally:
        conn.close()


if __name__ == '__main__':
    main()
