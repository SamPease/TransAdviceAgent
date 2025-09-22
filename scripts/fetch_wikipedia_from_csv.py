#!/usr/bin/env python3
"""Download Wikipedia pages listed in a PetScan CSV (pageid,title).

Saves JSON files under `data/wikipedia/<outdir>/` and maintains a persistent index so
re-runs skip already-downloaded pages. Designed to be polite: batch requests, sleep between
requests, and exponential backoff on 429/5xx responses.

Usage (from repo root):
  .venv/bin/python3 scripts/fetch_wikipedia_from_csv.py data/wikipedia/petscan_lgbt.csv --outdir lgbt --batch 20 --sleep 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from typing import Dict, List

import requests

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TransAdviceAgent/1.0 (https://github.com/SamPease/TransAdviceAgent)"


def safe_filename(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "", title)


def load_index(index_path: str) -> Dict[str, Dict]:
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index_atomic(index: Dict, index_path: str):
    tmp = index_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    os.replace(tmp, index_path)


def read_csv_rows(csv_path: str) -> List[Dict]:
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # pet scan typically exports pageid and title; accept either column names
            if "pageid" in r and r.get("pageid"):
                rows.append({"pageid": int(r.get("pageid")), "title": r.get("title")})
            elif "page_id" in r and r.get("page_id"):
                rows.append({"pageid": int(r.get("page_id")), "title": r.get("title")})
    return rows


def fetch_by_pageids(pageids: List[int], props: str = "extracts|info|categories") -> Dict[int, Dict]:
    params = {
        "action": "query",
        "format": "json",
        "pageids": "|".join(str(p) for p in pageids),
        "prop": props,
        "inprop": "url",
        "explaintext": 1,
        "cllimit": "max",
    }
    headers = {"User-Agent": USER_AGENT}
    backoff = 1.0
    while True:
        r = requests.get(WIKI_API, params=params, headers=headers, timeout=60)
        if r.status_code in (429, 503, 500):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        out = {}
        for pid, p in pages.items():
            out[int(pid)] = p
        return out


def fetch_wikitext_by_pageid(pageid: int) -> str | None:
    """Fallback to get page wikitext via revisions when extracts are missing."""
    params = {
        "action": "query",
        "format": "json",
        "pageids": str(pageid),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
    }
    headers = {"User-Agent": USER_AGENT}
    backoff = 1.0
    while True:
        r = requests.get(WIKI_API, params=params, headers=headers, timeout=60)
        if r.status_code in (429, 503, 500):
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)
            continue
        r.raise_for_status()
        data = r.json()
        pages = data.get("query", {}).get("pages", {})
        p = pages.get(str(pageid)) or pages.get(pageid)
        if not p:
            return None
        revs = p.get("revisions")
        if not revs:
            return None
        # modern API stores content under slots->main->* or directly in '*'
        rv = revs[0]
        if isinstance(rv, dict):
            # try slots
            slots = rv.get("slots")
            if slots and "main" in slots and "*" in slots["main"]:
                return slots["main"]["*"]
            # fallback
            return rv.get("*") or rv.get("content")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch Wikipedia pages from a PetScan CSV of pageids")
    parser.add_argument("csv_path", help="path to PetScan CSV (must include pageid,title)")
    parser.add_argument("--outdir", default="lgbt", help="subfolder under data/wikipedia to store pages")
    parser.add_argument("--batch", type=int, default=20, help="pageids per batch (default 20)")
    parser.add_argument("--sleep", type=float, default=0.5, help="seconds to sleep between batches")
    parser.add_argument("--index", default="data/wikipedia/downloaded_index.json", help="path for persistent index JSON")
    args = parser.parse_args()

    rows = read_csv_rows(args.csv_path)
    if not rows:
        print("No rows found in CSV")
        return

    os.makedirs("data/wikipedia/" + args.outdir, exist_ok=True)
    index = load_index(args.index)

    # Build mapping from pageid to title from CSV (CSV order preserved for prioritized fetch)
    pageid_to_title = {r["pageid"]: r["title"] for r in rows}
    all_pageids = list(pageid_to_title.keys())

    to_fetch = [pid for pid in all_pageids if pageid_to_title[pid] not in index]
    print(f"Total pageids from CSV: {len(all_pageids)}, to fetch: {len(to_fetch)}")

    for i in range(0, len(to_fetch), args.batch):
        batch = to_fetch[i : i + args.batch]
        try:
            pages = fetch_by_pageids(batch)
        except Exception as e:
            print(f"Batch fetch failed (will retry next batch): {e}")
            time.sleep(5)
            continue

        for pid in batch:
            p = pages.get(pid)
            title = pageid_to_title.get(pid)
            if not p:
                print(f"No page returned for pageid {pid} (title '{title}')")
                continue
            fname = safe_filename(title) or str(pid)
            path = os.path.join("data/wikipedia/" + args.outdir, f"{fname}.json")
            if os.path.exists(path):
                # record in index and skip
                index[title] = {"path": os.path.relpath(path, "data/wikipedia"), "pageid": pid}
                continue
            out = {
                "title": p.get("title", title),
                "url": p.get("fullurl"),
                "text": p.get("extract"),
                "categories": [c.get("title") for c in p.get("categories", [])] if p.get("categories") else [],
                "pageid": pid,
            }
            try:
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(out, f, ensure_ascii=False, indent=2)
                index[out["title"]] = {"path": os.path.relpath(path, "data/wikipedia"), "pageid": pid}
                save_index_atomic(index, args.index)
            except Exception as e:
                print(f"Failed to save {title}: {e}")

        print(f"Batches done: {i + len(batch)} / {len(to_fetch)}; sleeping {args.sleep}s")
        time.sleep(args.sleep)

    print("All done")


if __name__ == "__main__":
    main()
