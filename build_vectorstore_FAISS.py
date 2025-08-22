import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

DATA_DIR = "data"
INDEX_FILE = "app/faiss_index"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

def load_json_files(data_dir, existing_ids=None):
    """Load JSON files and return new Document chunks that are not in existing_ids"""
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
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata=metadata))
    return documents

def get_existing_post_ids(faiss_index):
    """Return set of post IDs already in the index"""
    ids = set()
    for doc in faiss_index.docstore._dict.values():
        if "id" in doc.metadata:
            ids.add(doc.metadata["id"])
    return ids

def main():
    # Load existing index if it exists
    if os.path.exists(INDEX_FILE):
        print(f"Loading existing FAISS index from {INDEX_FILE}")
        faiss_index = FAISS.load_local(INDEX_FILE, embedding_model)
        existing_ids = get_existing_post_ids(faiss_index)
    else:
        print("Creating new FAISS index")
        faiss_index = None
        existing_ids = set()

    # Load new documents
    new_docs = load_json_files(DATA_DIR, existing_ids=existing_ids)
    print(f"Found {len(new_docs)} new document chunks")

    if not new_docs:
        print("No new documents to add. Exiting.")
        return

    # Add to index
    if faiss_index:
        faiss_index.add_documents(new_docs)
    else:
        faiss_index = FAISS.from_documents(new_docs, embedding_model)

    # Save updated index
    faiss_index.save_local(INDEX_FILE)
    print(f"FAISS index updated and saved to {INDEX_FILE}")

if __name__ == "__main__":
    main()
