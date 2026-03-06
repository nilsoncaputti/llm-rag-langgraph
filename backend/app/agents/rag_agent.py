from typing import TypedDict, List
from transformers import pipeline, AutoTokenizer
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END

from app.retrieval.vector_store import search_documents


# ------------------------------
# Local LLM
# ------------------------------

llm = pipeline(
    "text-generation",
    model="gpt2",
    max_new_tokens=512,
    temperature=0.2
)


# ------------------------------
# Agent State
# ------------------------------

class AgentState(TypedDict):

    question: str
    rewritten_question: str
    documents: List[Document]
    context: str
    answer: str
    retry: int


# ------------------------------
# Retrieve Documents
# ------------------------------

def retrieve_docs(state: AgentState):

    query = state.get("rewritten_question") or state["question"]

    docs = search_documents(query, k=4)

    return {
        "documents": docs
    }


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
# Generate Answer
# ------------------------------

def generate_answer(state: AgentState):

    question = state["question"]
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

    # Truncate the entire prompt to fit GPT-2's max input length (1024 tokens)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    max_length = 1024 - 512  # Max model length - max new tokens

    inputs = tokenizer(prompt, max_length=max_length, truncation=True, return_tensors="pt")
    
    # The pipeline expects a string, so we decode the truncated input back to a string
    truncated_prompt = tokenizer.decode(inputs["input_ids"][0])

    result = llm(truncated_prompt)[0]["generated_text"]

    return {
        "answer": result
    }

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

def create_rag_agent():

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_docs)
    workflow.add_node("evaluate", evaluate_docs)
    workflow.add_node("rewrite", rewrite_query)
    workflow.add_node("build_context", build_context)
    workflow.add_node("generate", generate_answer)

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

    workflow.add_edge("build_context", "generate")

    workflow.add_edge("generate", END)

    return workflow.compile()


# ------------------------------
# Run Agent
# ------------------------------

rag_agent = create_rag_agent()


def run_agent(question: str):

    result = rag_agent.invoke({
        "question": question,
        "retry": 0
    })

    docs = result.get("documents", [])

    sources = []

    for d in docs:
        sources.append(d.metadata.get("source", "unknown"))

    return {
        "answer": result.get("answer", ""),
        "sources": sources
    }