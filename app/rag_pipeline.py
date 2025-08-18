import asyncio
from typing import TypedDict, List
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from dotenv import load_dotenv

# Load env variables (for Anthropic API key, etc.)
load_dotenv()

# --------------------------
# Configurable parameters
# --------------------------
MODEL_NAME = "claude-3-5-haiku-latest"
DOC_BATCH_SIZE = 10           # docs per first-level summary
SUMMARY_BATCH_SIZE = 10       # summaries per next-level summary

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

async def hierarchical_summarize(docs: List[Document], question: str, llm) -> str:
    """
    Strict hierarchical async summarization:
    - Level 0: docs -> first-order summaries
    - Level 1+: summarize batches of summaries
    """
    # Level 0: summarize docs in batches
    level_summaries = []
    for i in range(0, len(docs), DOC_BATCH_SIZE):
        batch = docs[i:i+DOC_BATCH_SIZE]
        summary = await summarize_batch(batch, llm, question)
        level_summaries.append(summary)

    # Higher-level summaries
    while len(level_summaries) > 1:
        next_level = []
        for i in range(0, len(level_summaries), SUMMARY_BATCH_SIZE):
            batch = level_summaries[i:i+SUMMARY_BATCH_SIZE]
            batch_text = "\n\n".join(batch)
            prompt = reduce_prompt.format(question=question, summaries=batch_text)
            result = await llm.ainvoke(prompt)
            next_level.append(result.content)
        level_summaries = next_level

    # Final summary to answer
    final_prompt_text = final_prompt.format(question=question, final_summary=level_summaries[0])
    final_answer = await llm.ainvoke(final_prompt_text)
    return final_answer.content

# --------------------------
# State definition
# --------------------------
class ChatAgentState(TypedDict):
    query: str
    docs: List
    final_answer: str

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

async def summarize_node(state):
    docs = state["docs"]
    llm = get_llm(streaming=False)
    final_answer = await hierarchical_summarize(docs, state["query"], llm)
    return {"final_answer": final_answer, "query": state["query"]}

async def output_node(state):
    return {"final_answer": state["final_answer"]}

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
    result["sources"] = "None :)"  # TODO: return doc metadata if needed
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
