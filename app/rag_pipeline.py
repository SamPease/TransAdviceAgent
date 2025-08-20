import asyncio
from typing import TypedDict, List
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load env variables (for Anthropic API key, etc.)
load_dotenv()

# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
DOC_BATCH_SIZE = 20           # docs per first-level summary
SUMMARY_BATCH_SIZE = 10       # summaries per next-level summary
DOCS_FETCH_LIMIT = 200           # max docs to fetch from Chroma
DOCS_KEEP_LIMIT = 100             # max docs to keep in Chroma

# --------------------------
# Prompts
# --------------------------
map_prompt = PromptTemplate(
    input_variables=["question", "context"],
    template="""
You are an assistant reading documents to answer a question.
Question: {question}

Read the following documents carefully and produce a concise summary that captures all key information relevant to the question. 

Documents:
{context}

Summary:
"""
)

reduce_prompt = PromptTemplate(
    input_variables=["question", "summaries"],
    template="""
You are an assistant tasked with combining summaries into a single, coherent summary. 

Question: {question}

Batch summaries:
{summaries}

Combined summary:
"""
)

final_prompt = PromptTemplate(
    input_variables=["question", "final_summary"],
    template="""
You are an expert assistant. Using the following summary, answer the question as accurately and clearly as possible.

Question: {question}

Summary:
{final_summary}

Answer:
"""
)

# --------------------------
# Streaming callback (optional)
# --------------------------
from langchain.callbacks.base import BaseCallbackHandler
class StreamingPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

# --------------------------
# Global cached models
# --------------------------
_embedding_model = None

def get_vectorstore():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return Chroma(
        persist_directory="./app/chroma_db",
        embedding_function=_embedding_model
    )

def get_llm(streaming=False):
    callbacks = [StreamingPrintHandler()] if streaming else []
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=0,
        streaming=streaming,
        callbacks=callbacks,
    )

# --------------------------
# Utilities
# --------------------------
async def summarize_batch(batch: List[Document], llm, question: str) -> str:
    batch_text = "\n\n".join(doc.page_content for doc in batch)
    prompt = map_prompt.format(question=question, context=batch_text)
    result = await llm.ainvoke(prompt)
    return result.content

# --------------------------
# State definition
# --------------------------
class ChatAgentState(TypedDict):
    query: str
    doc_ids: List[str]
    metadata_map: dict
    final_answer: str

# --------------------------
# LangGraph nodes with lazy-loading
# --------------------------
async def retrieve_node(state):
    query = state["query"]
    vectorstore = get_vectorstore()

    # Step 1: compute query embedding
    query_emb = _embedding_model.embed_query(query)

    # Step 2: fetch top-k candidate embeddings + IDs + metadata (similarity search)
    candidate_data = vectorstore._collection.query(
        query_embeddings=[query_emb],
        n_results=DOCS_FETCH_LIMIT,
        include=["embeddings", "metadatas"]
    )

    candidate_ids = candidate_data["ids"][0]  # Chroma query returns nested lists
    candidate_embeddings = np.array(candidate_data["embeddings"][0])
    metadata_map = {doc_id: meta for doc_id, meta in zip(candidate_ids, candidate_data["metadatas"][0])}

    # Build embedding lookup map to avoid index() calls
    embedding_map = {doc_id: emb for doc_id, emb in zip(candidate_ids, candidate_embeddings)}

    # Step 3: Max Marginal Relevance selection
    lambda_mult = 0.7
    k = DOCS_KEEP_LIMIT
    selected_ids = []

    if len(candidate_ids) <= k:
        selected_ids = candidate_ids
    else:
        # Initialize with most similar to query
        sim_to_query = cosine_similarity([query_emb], candidate_embeddings)[0]
        remaining_ids = candidate_ids.copy()
        selected_idx = int(np.argmax(sim_to_query))

        selected_ids.append(remaining_ids[selected_idx])
        selected_embeddings = [candidate_embeddings[selected_idx]]
        remaining_ids.pop(selected_idx)

        while len(selected_ids) < k and remaining_ids:
            remaining_embeddings = np.array([embedding_map[rid] for rid in remaining_ids])
            sim_to_query = cosine_similarity([query_emb], remaining_embeddings)[0]
            sim_to_selected = cosine_similarity(remaining_embeddings, selected_embeddings).max(axis=1)
            mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected

            next_idx = int(np.argmax(mmr_score))
            selected_ids.append(remaining_ids[next_idx])
            selected_embeddings.append(remaining_embeddings[next_idx])
            remaining_ids.pop(next_idx)

    return {
        "doc_ids": selected_ids,
        "metadata_map": metadata_map,
        "query": query
    }



def fetch_docs_by_ids(ids: List[str], metadata_map=None) -> List[Document]:
    """Lazy-load a batch of documents by ID from Chroma."""
    vectorstore = get_vectorstore()
    out = vectorstore._collection.get(ids=ids, include=["documents", "metadatas"])
    return [
        Document(
            page_content=text,
            metadata=metadata_map.get(doc_id) if metadata_map else meta
        )
        for doc_id, text, meta in zip(ids, out["documents"], out["metadatas"])
    ]


async def summarize_node(state):
    doc_ids = state["doc_ids"]
    metadata_map = state.get("metadata_map")
    llm = get_llm(streaming=False)

    # Summarize batch by batch
    level_summaries = []
    for i in range(0, len(doc_ids), DOC_BATCH_SIZE):
        batch_ids = doc_ids[i:i+DOC_BATCH_SIZE]
        batch_docs = fetch_docs_by_ids(batch_ids, metadata_map)
        summary = await summarize_batch(batch_docs, llm, state["query"])
        level_summaries.append(summary)
        del batch_docs  # free memory

    # Higher-level summaries
    while len(level_summaries) > 1:
        next_level = []
        for i in range(0, len(level_summaries), SUMMARY_BATCH_SIZE):
            batch = level_summaries[i:i+SUMMARY_BATCH_SIZE]
            batch_text = "\n\n".join(batch)
            prompt = reduce_prompt.format(question=state["query"], summaries=batch_text)
            result = await llm.ainvoke(prompt)
            next_level.append(result.content)
        level_summaries = next_level

    # Final summary to answer
    final_prompt_text = final_prompt.format(
        question=state["query"],
        final_summary=level_summaries[0]
    )
    final_answer = await llm.ainvoke(final_prompt_text)
    return {
        "final_answer": final_answer,
        "query": state["query"],
        "used_doc_ids": doc_ids,       # return IDs of docs used
        "metadata_map": metadata_map   # return metadata for those docs
    }


async def output_node(state):
    return {
        "final_answer": state["final_answer"],
        "used_doc_ids": state.get("used_doc_ids"),
        "metadata_map": state.get("metadata_map")
    }

# --------------------------
# Workflow setup
# --------------------------
workflow = StateGraph(ChatAgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("summarize", summarize_node)
workflow.add_node("output", output_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "summarize")
workflow.add_edge("summarize", "output")
workflow.add_edge("output", END)
workflow = workflow.compile()

# --------------------------
# Async wrapper for FastAPI
# --------------------------
async def run_rag(user_query: str) -> dict:
    result = await workflow.ainvoke({"query": user_query})
    # Attach metadata info as sources
    result["sources"] = [
        result["metadata_map"][doc_id] for doc_id in result.get("used_doc_ids", [])
    ]
    return result

# --------------------------
# Terminal debugging
# --------------------------
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()
    while True:
        q = input("Enter question (or 'exit'): ")
        if q.lower() == "exit":
            break
        answer = asyncio.run(run_rag(q))
        print("\nFinal Answer:\n", answer["final_answer"], "\n")
        print("Sources metadata:\n", answer["sources"], "\n")
