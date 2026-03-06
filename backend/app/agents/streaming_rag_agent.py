from typing import TypedDict, List
from threading import Thread

from app.ingestion import repo_graph
from app.retrieval.graph_retrieval import expand_with_graph
from transformers import pipeline, AutoTokenizer, TextIteratorStreamer
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from app.retrieval.vector_store import search_documents


# ------------------------------
# Local LLM
# ------------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

llm = pipeline(
    "text-generation",
    model=model_name,
    max_new_tokens=512,
    temperature=0.2
)

tokenizer = AutoTokenizer.from_pretrained(model_name)


# ------------------------------
# Agent State
# ------------------------------

class AgentState(TypedDict):

    question: str
    rewritten_question: str
    documents: List[Document]
    context: str
    retry: int
    relevant: bool


# ------------------------------
# Retrieve Documents
# ------------------------------

def retrieve_docs(state: AgentState):

    query = state.get("rewritten_question") or state["question"]

    docs = search_documents(query, k=4)
    
    expanded_symbols = expand_with_graph(docs)
    additional_docs = search_documents(" ".join(expanded_symbols))
    docs.extend(additional_docs)

    return {"documents": docs}


# ------------------------------
# Evaluate Relevance
# ------------------------------

def evaluate_docs(state: AgentState):

    docs = state["documents"]

    if len(docs) == 0:
        return {"relevant": False}

    total_length = sum(len(d.page_content) for d in docs)

    if total_length < 200:
        return {"relevant": False}

    return {"relevant": True}


# ------------------------------
# Rewrite Query
# ------------------------------

def rewrite_query(state: AgentState):

    question = state["question"]

    prompt = f"""
Rewrite the following engineering question to improve document search.

Question:
{question}

Better search query:
"""

    rewritten = llm(prompt)[0]["generated_text"]

    return {
        "rewritten_question": rewritten,
        "retry": state.get("retry", 0) + 1
    }


# ------------------------------
# Build Context
# ------------------------------

def build_context(state: AgentState):

    docs = state["documents"]

    context = ""

    for d in docs:
        source = d.metadata.get("source", "unknown")

        context += f"\nSOURCE: {source}\n"
        context += d.page_content
        context += "\n\n"

    return {"context": context}


# ------------------------------
# Routing Logic
# ------------------------------

def decide_next(state: AgentState):

    retry = state.get("retry", 0)

    if state.get("relevant"):
        return "build_context"

    if retry >= 2:
        return "build_context"

    return "rewrite"


# ------------------------------
# Build LangGraph Workflow
# ------------------------------

def create_rag_graph():

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("evaluate", evaluate_docs)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("build_context", build_context)

    workflow.set_entry_point("retrieve")

    workflow.add_edge("retrieve", "evaluate")

    workflow.add_conditional_edges(
        "evaluate",
        decide_next,
        {
            "rewrite": "rewrite",
            "build_context": "build_context"
        }
    )

    workflow.add_edge("rewrite", "retrieve")

    workflow.add_edge("build_context", END)

    return workflow.compile()


graph = create_rag_graph()


# ------------------------------
# Streaming Answer Generator
# ------------------------------

def stream_answer(question: str):

    state = graph.invoke({
        "question": question,
        "retry": 0
    })

    context = state["context"]

    prompt = f"""
You are an AI engineering assistant that explains codebases.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Explain clearly and reference source files when possible.
"""

    # truncate prompt for GPT2
    max_length = 1024 - 512

    inputs = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        return_tensors="pt"
    )

    truncated_prompt = tokenizer.decode(inputs["input_ids"][0])

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )

    generation_kwargs = dict(
        text_inputs=truncated_prompt,
        streamer=streamer,
        max_new_tokens=300
    )

    thread = Thread(target=llm, kwargs=generation_kwargs)
    thread.start()

    for token in streamer:
        yield token