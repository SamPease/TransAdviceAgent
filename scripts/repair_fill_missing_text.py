#!/usr/bin/env python3
"""Repair JSON article files by filling missing 'text' fields.

Finds all JSON files under a base data directory, and for any file where the "text" field is null/empty,
attempts to fetch wikitext via pageid (preferred) or by title, then writes it back to the file and
updates the persistent index if present.

Run in the venv:
  .venv/bin/python3 scripts/repair_fill_missing_text.py --base data/wikipedia --log data/wikipedia/lgbt/repair.log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Optional

import requests

import requests


def fetch_wikitext_by_pageid(pageid: int) -> Optional[str]:
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
        # pages can have numeric keys as strings
        p = None
        if str(pageid) in pages:
            p = pages.get(str(pageid))
        else:
            # try numeric key
            for k, v in pages.items():
                if int(k) == pageid:
                    p = v
                    break
        if not p:
            return None
        revs = p.get("revisions")
        if not revs:
            return None
        rv = revs[0]
        if isinstance(rv, dict):
            slots = rv.get("slots")
            if slots and "main" in slots and "*" in slots["main"]:
                return slots["main"]["*"]
            return rv.get("*") or rv.get("content")
        return None

WIKI_API = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TransAdviceAgent/1.0 (https://github.com/SamPease/TransAdviceAgent)"


def safe_filename(title: str) -> str:
    return re.sub(r"[\\/:*?\"<>|]", "", title)


def fetch_wikitext_by_title(title: str) -> Optional[str]:
    # Try extracts first, then fallback to revisions for wikitext
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": 1,
        "inprop": "url",
    }
    headers = {"User-Agent": USER_AGENT}
    r = requests.get(WIKI_API, params=params, headers=headers, timeout=30)
    if r.status_code >= 500:
        return None
    r.raise_for_status()
    data = r.json()
    pages = data.get("query", {}).get("pages", {})
    for pid, p in pages.items():
        text = p.get("extract")
        if text:
            return text
        # fallback to revisions
        pageid = p.get("pageid")
        if pageid:
            return fetch_wikitext_by_pageid(pageid)
    return None


def repair_file(path: str, index: dict) -> bool:
    """Return True if file was updated."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    text = data.get("text")
    if text:
        return False

    pageid = data.get("pageid")
    title = data.get("title")
    fetched = None
    if pageid:
        fetched = fetch_wikitext_by_pageid(pageid)
    if not fetched and title:
        fetched = fetch_wikitext_by_title(title)
    if not fetched:
        return False

    data["text"] = fetched
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        # update index if title present
        if title is not None:
            index[title] = index.get(title, {})
            index[title]["repaired_at"] = int(time.time())
        return True
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="data/wikipedia", help="base directory to scan")
    parser.add_argument("--log", default=None, help="optional log file")
    parser.add_argument("--index", default=os.path.join("data", "wikipedia", "downloaded_index.json"), help="path to index JSON to update")
    args = parser.parse_args()

    index = {}
    if os.path.exists(args.index):
        try:
            with open(args.index, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}

    repaired = 0
    total = 0
    out_log = open(args.log, "a", encoding="utf-8") if args.log else None
    try:
        for root, _, files in os.walk(args.base):
            for fn in files:
                if not fn.lower().endswith(".json"):
                    continue
                path = os.path.join(root, fn)
                total += 1
                updated = repair_file(path, index)
                if updated:
                    repaired += 1
                    msg = f"Repaired: {path}\n"
                    if out_log:
                        out_log.write(msg)
                    else:
                        print(msg, end="")
        # save index
        try:
            tmp = args.index + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(index, f, ensure_ascii=False, indent=2)
            os.replace(tmp, args.index)
        except Exception:
            pass
    finally:
        if out_log:
            out_log.write(f"Total scanned: {total}, repaired: {repaired}\n")
            out_log.close()

    print(f"Total scanned: {total}, repaired: {repaired}")


if __name__ == "__main__":
    main()
