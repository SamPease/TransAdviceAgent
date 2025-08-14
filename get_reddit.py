import os
import json
from dotenv import load_dotenv
import praw
from datetime import datetime

load_dotenv()

reddit = praw.Reddit(
    client_id=os.getenv("REDDIT_CLIENT_ID"),
    client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
    user_agent=os.getenv("REDDIT_USER_AGENT")
)

def reddit_post_to_dict(post, comment_limit=20):
    post.comments.replace_more(limit=0)
    comments = []
    for comment in post.comments.list()[:comment_limit]:
        comments.append({
            "id": comment.id,
            "author": comment.author.name if comment.author else "[deleted]",
            "body": comment.body,
            "score": comment.score,
            "created_utc": datetime.utcfromtimestamp(comment.created_utc).isoformat() + "Z"
        })

    return {
        "id": post.id,
        "title": post.title,
        "author": post.author.name if post.author else "[deleted]",
        "selftext": post.selftext,
        "score": post.score,
        "url": post.url,
        "num_comments": post.num_comments,
        "created_utc": datetime.utcfromtimestamp(post.created_utc).isoformat() + "Z",
        "comments": comments
    }

def fetch_and_save(subreddit_name="Transgender_Surgeries", post_limit=20, comment_limit=20):
    subreddit = reddit.subreddit(subreddit_name)
    posts = subreddit.hot(limit=post_limit)

    os.makedirs("data", exist_ok=True)

    for i, post in enumerate(posts, 1):
        filename = f"data/{subreddit_name}_post_{post.id}.json"
        if os.path.exists(filename):
            print(f"Skipping already downloaded post {i}: {post.title} (ID: {post.id})")
            continue
        print(f"Fetching post {i}: {post.title}")
        post_dict = reddit_post_to_dict(post, comment_limit=comment_limit)
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(post_dict, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    # Increase default post_limit to 200
    fetch_and_save(post_limit=200)
