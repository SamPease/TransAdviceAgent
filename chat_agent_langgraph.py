import asyncio
from typing import TypedDict, List
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
import tiktoken

load_dotenv()

# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
INPUT_PERCENT = 0.7  # Use 70% of context for input

# Model context dictionary (max tokens per model)
MODEL_CONTEXT = {
    "claude-3-5-haiku-latest": 100_000,
    "claude-2": 100_000,
}

MAX_CONTEXT = MODEL_CONTEXT[MODEL_NAME]
MAX_TOKENS_PER_BATCH = int(MAX_CONTEXT * INPUT_PERCENT)
HIERARCHICAL_THRESHOLD = 50  # If more than 50 summaries, use extra reduction layer

# --------------------------
# Streaming callback handler
# --------------------------
class StreamingPrintHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

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
# Token encoder
# --------------------------
enc = tiktoken.get_encoding("r50k_base")

# --------------------------
# State definition
# --------------------------
class ChatAgentState(TypedDict):
    query: str
    docs: List
    batch_summaries: List
    final_answer: str
    batch_metadata: List

# --------------------------
# LLM and vectorstore setup
# --------------------------
def get_llm():
    return ChatAnthropic(
        model=MODEL_NAME,
        temperature=0,
        streaming=True,
        callbacks=[StreamingPrintHandler()],
    )

def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# --------------------------
# Utilities
# --------------------------
def batch_docs_by_tokens(docs, max_tokens=MAX_TOKENS_PER_BATCH):
    batches = []
    current_batch = []
    current_tokens = 0
    for doc in docs:
        doc_tokens = len(enc.encode(doc.page_content))
        if current_tokens + doc_tokens > max_tokens and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(doc)
        current_tokens += doc_tokens
    if current_batch:
        batches.append(current_batch)
    return batches

async def summarize_batch(batch, llm, prompt_template, query):
    batch_text = "\n\n".join([doc.page_content for doc in batch])
    prompt = prompt_template.format(question=query, context=batch_text)
    result = await llm.ainvoke(prompt)
    return {"summary": Document(page_content=result.content), "source_docs": batch}

# --------------------------
# LangGraph nodes
# --------------------------
async def retrieve_node(state):
    query = state["query"]
    vectorstore = get_vectorstore()
    docs = vectorstore.max_marginal_relevance_search(
        query=query, k=100, fetch_k=200, lambda_mult=0.7
    )
    return {"docs": docs, "query": query}

async def batch_summarize_node(state):
    docs = state["docs"]
    llm = get_llm()
    batches = batch_docs_by_tokens(docs)
    summaries = await asyncio.gather(
        *[summarize_batch(batch, llm, map_prompt, state["query"]) for batch in batches]
    )
    batch_summaries = [s["summary"] for s in summaries]
    return {"batch_summaries": batch_summaries, "query": state["query"], "batch_metadata": summaries}

# Hierarchical reduction node
async def aggregate_node(state):
    llm = get_llm()
    batch_metadata = state.get("batch_metadata", [])

    # Single-step reduction if few summaries
    if len(batch_metadata) <= HIERARCHICAL_THRESHOLD:
        summaries_text = ""
        for i, batch in enumerate(batch_metadata):
            summary_doc = batch["summary"]
            source_docs = batch["source_docs"]
            source_refs = "\n".join(
                [doc.page_content[:150].replace("\n", " ") + "..." for doc in source_docs]
            )
            summaries_text += f"Summary {i+1}:\n{summary_doc.page_content}\nReferences:\n{source_refs}\n\n"
        prompt = reduce_prompt.format(question=state["query"], summaries=summaries_text)
        final_answer = await llm.ainvoke(prompt)
        return {"final_answer": final_answer.content, "query": state["query"]}

    # Hierarchical reduction for many summaries
    intermediate_summaries = []
    chunked = [batch_metadata[i:i+HIERARCHICAL_THRESHOLD] for i in range(0, len(batch_metadata), HIERARCHICAL_THRESHOLD)]
    for chunk in chunked:
        summaries_text = ""
        for i, batch in enumerate(chunk):
            summary_doc = batch["summary"]
            source_docs = batch["source_docs"]
            source_refs = "\n".join(
                [doc.page_content[:150].replace("\n", " ") + "..." for doc in source_docs]
            )
            summaries_text += f"Summary {i+1}:\n{summary_doc.page_content}\nReferences:\n{source_refs}\n\n"
        prompt = reduce_prompt.format(question=state["query"], summaries=summaries_text)
        intermediate_summary = await llm.ainvoke(prompt)
        intermediate_summaries.append(Document(page_content=intermediate_summary.content))

    # Final reduction step
    final_text = "\n\n".join([doc.page_content for doc in intermediate_summaries])
    prompt = final_prompt.format(question=state["query"], final_summary=final_text)
    final_answer = await llm.ainvoke(prompt)
    return {"final_answer": final_answer.content, "query": state["query"]}

async def output_node(state):
    print("\n\n--- Final synthesized answer ---")
    print(state["final_answer"])
    return {}

# --------------------------
# Workflow setup
# --------------------------
workflow = StateGraph(ChatAgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("batch_summarize", batch_summarize_node)
workflow.add_node("aggregate", aggregate_node)
workflow.add_node("output", output_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "batch_summarize")
workflow.add_edge("batch_summarize", "aggregate")
workflow.add_edge("aggregate", "output")
workflow.add_edge("output", END)
workflow = workflow.compile()

# --------------------------
# Main async invocation
# --------------------------
async def run_query(user_query):
    await workflow.ainvoke({"query": user_query})

def main():
    while True:
        user_query = input("\nEnter your question (or 'exit' to quit): ")
        if user_query.strip().lower() == "exit":
            print("Goodbye!")
            break
        asyncio.run(run_query(user_query))

if __name__ == "__main__":
    main()
