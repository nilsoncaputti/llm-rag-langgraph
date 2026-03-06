from transformers import pipeline
from app.retrieval.vector_store import search_documents


def build_context(docs):
    context = ""

    for d in docs:
        context += f"\nSOURCE: {d.metadata}\n"
        context += d.page_content
        context += "\n\n"

    return context


def answer_question(query):
    docs = search_documents(query)
    context = build_context(docs)
    prompt = f"""
You are an AI engineering assistant.

Use the following codebase context to answer the question.

Context:
{context}

Question:
{query}

Provide a clear explanation and reference source files.
"""

    # Use Hugging Face Transformers pipeline for local inference
    llm = pipeline("text-generation", model="gpt2")  # You can change 'gpt2' to another local model
    result = llm(prompt, max_length=512)[0]["generated_text"]

    return result, docs