import os
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

load_dotenv()

def load_json_documents(data_dir="data"):
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
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
                }
                documents.append(Document(page_content=full_text, metadata=metadata))
    return documents

def main():
    print("Loading documents from JSON...")
    docs = load_json_documents()

    # Debug: confirm Dr. Houtmeyers is in loaded docs
    found = False
    for doc in docs:
        if "houtmeyers" in doc.page_content.lower():
            print("Found 'houtmeyers' in document metadata:", doc.metadata)
            found = True
    if not found:
        print("Warning: 'Dr. Houtmeyers' not found in loaded documents!")

    print(f"Loaded {len(docs)} documents, chunking...")

    # Increase chunk size to 1000 for more context per chunk
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    # Debug: check chunks containing 'houtmeyers'
    houtmeyers_chunks = [c for c in chunks if "houtmeyers" in c.page_content.lower()]
    print(f"Chunks mentioning 'houtmeyers': {len(houtmeyers_chunks)}")
    for i, c in enumerate(houtmeyers_chunks[:3], 1):
        print(f"Chunk {i} preview:\n{c.page_content[:400]}\n---\n")

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir):
        print("Loading existing vectorstore...")
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    else:
        print("Creating new vectorstore...")
        vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=persist_dir)

    llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)

    # Increase retrieval count k to 10 for more candidate chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    print("\nAsk your trans advice agent (type 'exit' to quit):")
    while True:
        query = input("\n> ")
        if query.lower() in ["exit", "quit"]:
            break

        # Debug: show all retrieved chunks for this query
        retrieved_docs = retriever.get_relevant_documents(query)
        print(f"\nRetrieved {len(retrieved_docs)} chunks. Sample content:\n")
        for i, doc in enumerate(retrieved_docs, 1):
            preview = doc.page_content.replace('\n',' ')[:400]
            print(f"Chunk {i} preview: {preview}\n---\n")

        answer = qa.invoke(query)
        print("\nAnswer:\n", answer)

if __name__ == "__main__":
    main()
