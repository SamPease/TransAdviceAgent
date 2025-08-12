from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

def main():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding_model)
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})

    llm = ChatAnthropic(model="claude-3-5-haiku-latest", temperature=0)
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
