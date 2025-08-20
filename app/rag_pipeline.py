import asyncio
from typing import TypedDict, List
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Load env variables
load_dotenv()

# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
DOC_BATCH_SIZE = 10
SUMMARY_BATCH_SIZE = 10
DOCS_FETCH_LIMIT = 200
DOCS_KEEP_LIMIT = 100

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
_vectorstore = None

def get_vectorstore():
    global _embedding_model, _vectorstore
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    if _vectorstore is None:
        if os.path.exists("./app/faiss_index/index.faiss") and os.path.exists("./app/faiss_index/index.pkl"):
            _vectorstore = FAISS.load_local(
                "./app/faiss_index",
                _embedding_model,
                allow_dangerous_deserialization=True
            )
        else:
            raise RuntimeError("FAISS index not found. Please build it first.")
    return _vectorstore

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
# LangGraph nodes
# --------------------------
async def retrieve_node(state):
    query = state["query"]
    vectorstore = get_vectorstore()

    # Step 1: compute query embedding
    query_emb = _embedding_model.embed_query(query)

    # Step 2: FAISS search
    D, I = vectorstore.index.search(np.array([query_emb], dtype=np.float32), DOCS_FETCH_LIMIT)
    candidate_indices = I[0].tolist()  # numeric positions in FAISS

    # Step 3: map numeric indices to docstore IDs (UUIDs)
    candidate_doc_ids = [vectorstore.index_to_docstore_id[idx] for idx in candidate_indices
                         if idx in vectorstore.index_to_docstore_id]

    if not candidate_doc_ids:
        return {
            "doc_ids": [],
            "metadata_map": {},
            "query": query
        }

    # Optional: MMR selection using embeddings
    candidate_embeddings = np.array([vectorstore.index.reconstruct(int(idx)) for idx in candidate_indices])
    sim_to_query = cosine_similarity([query_emb], candidate_embeddings)[0]

    lambda_mult = 0.7
    k = min(DOCS_KEEP_LIMIT, len(candidate_doc_ids))
    selected_doc_ids = []

    remaining_ids = candidate_doc_ids.copy()
    remaining_embeddings = candidate_embeddings.copy()
    selected_idx = int(np.argmax(sim_to_query))
    selected_doc_ids.append(remaining_ids[selected_idx])

    selected_embeddings = [remaining_embeddings[selected_idx]]
    del remaining_ids[selected_idx]
    remaining_embeddings = np.delete(remaining_embeddings, selected_idx, axis=0)

    while len(selected_doc_ids) < k and remaining_ids:
        sim_to_query = cosine_similarity([query_emb], remaining_embeddings)[0]
        sim_to_selected = cosine_similarity(remaining_embeddings, selected_embeddings).max(axis=1)
        mmr_score = lambda_mult * sim_to_query - (1 - lambda_mult) * sim_to_selected
        next_idx = int(np.argmax(mmr_score))
        selected_doc_ids.append(remaining_ids[next_idx])
        selected_embeddings.append(remaining_embeddings[next_idx])
        del remaining_ids[next_idx]
        remaining_embeddings = np.delete(remaining_embeddings, next_idx, axis=0)

    # Step 4: build metadata map for selected docs
    metadata_map = {doc_id: vectorstore.docstore._dict[doc_id].metadata for doc_id in selected_doc_ids}

    return {
        "doc_ids": selected_doc_ids,
        "metadata_map": metadata_map,
        "query": query
    }

async def summarize_node(state):
    doc_ids = state["doc_ids"]
    if not doc_ids:
        return {
            "final_answer": "No documents found for this query.",
            "query": state["query"],
            "used_doc_ids": [],
            "metadata_map": {}
        }

    vectorstore = get_vectorstore()
    llm = get_llm(streaming=False)
    level_summaries = []

    for i in range(0, len(doc_ids), DOC_BATCH_SIZE):
        batch_ids = doc_ids[i:i + DOC_BATCH_SIZE]

        # Lazy load documents for this batch only
        batch_docs = vectorstore.get_by_ids(batch_ids)
        if not batch_docs:
            continue

        summary = await summarize_batch(batch_docs, llm, state["query"])
        level_summaries.append(summary)

    if not level_summaries:
        return {
            "final_answer": "No summaries could be generated.",
            "query": state["query"],
            "used_doc_ids": doc_ids,
            "metadata_map": state.get("metadata_map")
        }

    # Recursive reduce summaries
    while len(level_summaries) > 1:
        next_level = []
        for i in range(0, len(level_summaries), SUMMARY_BATCH_SIZE):
            batch = level_summaries[i:i + SUMMARY_BATCH_SIZE]
            batch_text = "\n\n".join(batch)
            prompt = reduce_prompt.format(question=state["query"], summaries=batch_text)
            result = await llm.ainvoke(prompt)
            next_level.append(result.content)
        level_summaries = next_level

    final_prompt_text = final_prompt.format(
        question=state["query"],
        final_summary=level_summaries[0]
    )
    final_answer = await llm.ainvoke(final_prompt_text)

    return {
        "final_answer": final_answer,
        "query": state["query"],
        "used_doc_ids": doc_ids,
        "metadata_map": state.get("metadata_map")
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
