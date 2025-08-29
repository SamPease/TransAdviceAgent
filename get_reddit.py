import os
import json
from dotenv import load_dotenv
import praw
from datetime import datetime, timezone

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def reddit_post_to_dict(post, comment_limit=100):
    # Expand up to 5 "load more comments" links, focusing on the ones with the most comments
    post.comments.replace_more(limit=5)
    
    # Get all comments and sort by score
    all_comments = post.comments.list()
    all_comments.sort(key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
    
    comments = []
    for comment in all_comments[:comment_limit]:
        try:
            comments.append({
                "id": comment.id,
                "author": comment.author.name if comment.author else "[deleted]",
                "body": comment.body,
                "score": comment.score,
                "created_utc": datetime.fromtimestamp(comment.created_utc, timezone.utc).isoformat().replace('+00:00','Z')
            })
        except Exception as e:
            print(f"[WARNING] Failed to process comment {comment.id}: {e}")

    return {
        "id": post.id,
        "title": post.title,
        "author": post.author.name if post.author else "[deleted]",
        "selftext": post.selftext,
        "score": post.score,
        "url": post.url,
        "num_comments": post.num_comments,
    "created_utc": datetime.fromtimestamp(post.created_utc, timezone.utc).isoformat().replace('+00:00','Z'),
        "comments": comments
    }

def fetch_and_save(subreddit_name="Transgender_Surgeries", post_limit=1000, collect=None, comment_limit=100, sort="hot", time_filter=None):
    """Fetch up to `post_limit` unique posts from `subreddit_name`, skipping already-saved posts.

    Behavior:
    - Checks the ./data folder for files named like `<subreddit>_post_<id>.json` and counts them.
    - If already at or above post_limit, it skips that subreddit.
    - Streams posts from the subreddit (hot) and saves only new post IDs until the target is met.
    - Uses a simple exponential backoff on transient errors.
    """
    import time

    os.makedirs("data", exist_ok=True)
    checkpoints_dir = os.path.join("data", "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # Load checkpoint file if present (stores saved_ids, counts, and done_sorts)
    checkpoint_path = os.path.join(checkpoints_dir, f"{subreddit_name}.json")
    checkpoint = {"saved_ids": [], "saved_count": 0, "done_sorts": []}
    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as cf:
                checkpoint = json.load(cf)
        except Exception as e:
            print(f"[WARNING] Failed to read checkpoint for r/{subreddit_name}: {e}")

    # Discover already-saved post ids for this subreddit (case-insensitive) and merge with checkpoint
    existing_ids = set(checkpoint.get("saved_ids", []))
    done_sorts = set(checkpoint.get("done_sorts", []))
    prefix = f"{subreddit_name}_post_"
    prefix_lower = prefix.lower()
    for fname in os.listdir("data"):
        fname_lower = fname.lower()
        if fname_lower.startswith(prefix_lower) and fname_lower.endswith(".json"):
            pid = fname[len(prefix):-5]
            existing_ids.add(pid)

    existing_count = len(existing_ids)
    # If collect is provided, we treat it as the number of new unique posts to fetch for this call.
    if collect is None:
        if existing_count >= post_limit:
            print(f"[INFO] r/{subreddit_name} already has {existing_count} saved posts (>= {post_limit}), skipping.")
            return
        need = post_limit - existing_count
        print(f"[INFO] r/{subreddit_name}: {existing_count} saved posts, need {need} more (target {post_limit}).")
    else:
        if collect <= 0:
            print(f"[INFO] r/{subreddit_name}: collect={collect} <= 0, nothing to do.")
            return
        need = collect
        print(f"[INFO] r/{subreddit_name}: {existing_count} already saved, attempting to collect {need} new unique posts for sort={sort} time_filter={time_filter}.")
        # If this sort/time_filter combination was already fully scanned previously, skip it
        sort_key = f"{sort}|{time_filter if time_filter else ''}"
        if sort_key in done_sorts:
            print(f"[INFO] r/{subreddit_name}: sort={sort} time_filter={time_filter} already scanned previously, skipping.")
            return

    subreddit = reddit.subreddit(subreddit_name)
    # Determine fetch limit: if collect provided, only scan up to that many posts for this sort;
    # otherwise stream (limit=None) until need is met.
    fetch_limit = collect if collect is not None else None
    # Choose generator based on sort and time_filter
    if sort == "hot":
        posts = subreddit.hot(limit=fetch_limit)
    elif sort == "new":
        posts = subreddit.new(limit=fetch_limit)
    elif sort == "top":
        # if time_filter is None, default to 'all'
        tf = time_filter if time_filter else "all"
        posts = subreddit.top(limit=fetch_limit, time_filter=tf)
    else:
        print(f"[WARNING] Unknown sort '{sort}' for r/{subreddit_name}, defaulting to hot.")
        posts = subreddit.hot(limit=fetch_limit)

    if collect is not None:
        print(f"[INFO] Scanning up to {fetch_limit} posts for r/{subreddit_name} (sort={sort}, time_filter={time_filter}) and saving unseen ones.")

    saved = 0
    attempts = 0
    max_attempts = 5
    backoff = 1.0

    scanned_count = 0
    for post in posts:
        scanned_count += 1
        if saved >= need:
            break

        # Skip posts we've already saved
        if post.id in existing_ids:
            continue

        filename = f"data/{subreddit_name}_post_{post.id}.json"

        try:
            print(f"[INFO] Fetching new post: {post.title} (ID: {post.id})")
            post_dict = reddit_post_to_dict(post, comment_limit=comment_limit)
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(post_dict, f, ensure_ascii=False, indent=2)

            existing_ids.add(post.id)
            saved += 1
            # update checkpoint on disk after each successful save
            try:
                checkpoint = {"saved_ids": list(existing_ids), "saved_count": len(existing_ids), "last_updated": datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
                with open(checkpoint_path, "w", encoding="utf-8") as cf:
                    json.dump(checkpoint, cf, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[WARNING] Failed to update checkpoint for r/{subreddit_name}: {e}")
            attempts = 0
            backoff = 1.0
            print(f"[INFO] Saved {saved}/{need} new posts for r/{subreddit_name}.")

        except Exception as e:
            attempts += 1
            err_str = str(e)
            is_rate = False
            # heuristic: check for common rate-limit signals
            try:
                import prawcore
                if isinstance(e, prawcore.exceptions.ResponseException) and getattr(e, 'status_code', None) == 429:
                    is_rate = True
            except Exception:
                pass
            if 'RATELIMIT' in err_str.upper() or 'RATE LIMIT' in err_str.upper() or '429' in err_str:
                is_rate = True

            print(f"[WARNING] Error fetching/saving post {getattr(post, 'id', '?')} from r/{subreddit_name}: {e}")

            if is_rate:
                # Detected rate limit: do a longer sleep and then resume
                sleep_time = min(backoff * 10, 600)  # up to 10 minutes
                print(f"[INFO] Detected rate limit; sleeping for {sleep_time:.1f}s before resuming...")
                time.sleep(sleep_time)
                # reset short-term counters and continue
                attempts = 0
                backoff = 1.0
                continue

            if attempts > max_attempts:
                print(f"[ERROR] Too many consecutive errors for r/{subreddit_name}, aborting this subreddit.")
                break

            sleep_time = backoff
            backoff = min(backoff * 2, 60)
            print(f"[INFO] Backing off for {sleep_time:.1f}s before continuing...")
            time.sleep(sleep_time)

    if saved < need:
        # mark sort as done if we scanned the full fetch_limit (i.e., there is nothing more in this window to collect)
        try:
            scanned_full = (collect is not None and fetch_limit is not None and scanned_count >= fetch_limit)
            if collect is not None and scanned_full:
                done_sorts.add(sort_key)
                checkpoint = {"saved_ids": list(existing_ids), "saved_count": len(existing_ids), "done_sorts": list(done_sorts), "last_updated": datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
                with open(checkpoint_path, "w", encoding="utf-8") as cf:
                    json.dump(checkpoint, cf, ensure_ascii=False, indent=2)
                print(f"[INFO] Marked sort={sort} time_filter={time_filter} as completed for r/{subreddit_name}.")
        except Exception:
            pass
        print(f"[INFO] Finished streaming for r/{subreddit_name}. Collected {saved} new posts, target still needs {need - saved} more (feed may be exhausted or rate-limited).")
    else:
        # If we reached the requested number for this sort, mark it done so it won't be rescanned
        try:
            if collect is not None:
                done_sorts.add(sort_key)
                checkpoint = {"saved_ids": list(existing_ids), "saved_count": len(existing_ids), "done_sorts": list(done_sorts), "last_updated": datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}
                with open(checkpoint_path, "w", encoding="utf-8") as cf:
                    json.dump(checkpoint, cf, ensure_ascii=False, indent=2)
        except Exception:
            pass
        print(f"[INFO] Reached target for r/{subreddit_name}: collected {saved} new posts.")

if __name__ == "__main__":
    # List of subreddits to fetch from
    subreddits = [
        "Transgender_Surgeries",
        "mtf",
        "asktransgender",
        "trans",
        "TransDIY",
        "ftm"
    ]
    
    # Per-subreddit strategy: collect N posts for each sort/time_filter tuple
    # Example plan: 200 hot, 200 new, 200 top week, 200 top month, 200 top all
    per_sort_plan = [
        ("hot", None, 200),
        ("new", None, 200),
        ("top", "week", 200),
        ("top", "month", 200),
        ("top", "year", 200),
        ("top", "all", 200),
    ]

    for subreddit in subreddits:
        print(f"\n[INFO] Processing subreddit: r/{subreddit}")
        plan = per_sort_plan
        for sort, tf, collect in plan:
            print(f"[INFO] Collecting {collect} posts for r/{subreddit} sort={sort} time_filter={tf}")
            try:
                fetch_and_save(subreddit_name=subreddit, collect=collect, comment_limit=100, sort=sort, time_filter=tf)
            except Exception as e:
                print(f"[ERROR] Failed collecting sort={sort} for r/{subreddit}: {e}")
