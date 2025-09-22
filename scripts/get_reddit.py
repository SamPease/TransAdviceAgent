#!/usr/bin/env python3
import os
import json
import time
from pathlib import Path
import praw

ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
CHECKPOINT_DIR = os.path.join(ROOT, "data", "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# NOTE: The user is expected to set REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT as env vars
def make_reddit():
    client_id = os.environ.get("REDDIT_CLIENT_ID")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET")
    user_agent = os.environ.get("REDDIT_USER_AGENT", "trans-advice-agent/0.1")
    if not client_id or not client_secret:
        raise RuntimeError("Please set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET environment variables")
    return praw.Reddit(client_id=client_id, client_secret=client_secret, user_agent=user_agent)

def reddit_post_to_dict(submission):
    data = {
        "id": submission.id,
        "title": submission.title,
        "selftext": submission.selftext,
        "url": submission.url,
        "author": str(submission.author) if submission.author else None,
        "score": submission.score,
        "num_comments": submission.num_comments,
        "created_utc": submission.created_utc,
        "comments": []
    }
    submission.comments.replace_more(limit=0)
    for comment in submission.comments.list():
        data["comments"].append({"id": comment.id, "body": comment.body, "author": str(comment.author)})
    return data

def fetch_and_save(reddit, subreddit, limit=100):
    outdir = os.path.join(DATA_DIR, subreddit)
    os.makedirs(outdir, exist_ok=True)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{subreddit}.json")
    seen = set()
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r") as f:
                seen = set(json.load(f))
        except Exception:
            seen = set()

    sr = reddit.subreddit(subreddit)
    for submission in sr.new(limit=limit):
        if submission.id in seen:
            continue
        d = reddit_post_to_dict(submission)
        with open(os.path.join(outdir, f"{submission.id}.json"), "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)
        seen.add(submission.id)
        with open(checkpoint_path, "w") as f:
            json.dump(list(seen), f)
        print(f"Saved {submission.id} from r/{subreddit}")
        time.sleep(1)

def main():
    reddit = make_reddit()
    targets = ["asktransgender"]
    for t in targets:
        print(f"Fetching subreddit: {t}")
        fetch_and_save(reddit, t, limit=200)

if __name__ == "__main__":
    main()
