import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PROCESSED_FILE = "processed_files.json"

def load_processed_files():
    if os.path.exists(PROCESSED_FILE):
        with open(PROCESSED_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_processed_files(processed_files):
    with open(PROCESSED_FILE, "w") as f:
        json.dump(list(processed_files), f)

def load_new_documents(data_dir="data", processed_files=set()):
    documents = []
    new_files = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json") and filename not in processed_files:
            path = os.path.join(data_dir, filename)
            with open(path, "r", encoding="utf-8") as f:
                post = json.load(f)
                text_parts = [post.get("title", ""), post.get("selftext", "")]
                for comment in post.get("comments", []):
                    text_parts.append(comment.get("body", ""))
                full_text = "\n\n".join(text_parts)
                metadata = {
                    "post_id": post.get("id"),
                    "author": post.get("author"),
                    "created_utc": post.get("created_utc"),
                    "source_file": filename,
                }
                documents.append(Document(page_content=full_text, metadata=metadata))
                new_files.append(filename)
    return documents, new_files

def main():
    processed_files = load_processed_files()
    print(f"Previously processed files: {len(processed_files)}")

    new_docs, new_files = load_new_documents(processed_files=processed_files)
    print(f"New documents to process: {len(new_docs)}")

    if not new_docs:
        print("No new documents to process.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(new_docs)
    print(f"Split new docs into {len(chunks)} chunks.")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)

    vectordb.add_documents(chunks)

    print("Vectorstore updated and persisted.")

    processed_files.update(new_files)
    save_processed_files(processed_files)
    print(f"Processed files updated: {len(processed_files)}")


if __name__ == "__main__":
    main()
