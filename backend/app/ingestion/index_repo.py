from app.ingestion.github_loader import load_github_repo
from app.ingestion.text_splitter import split_documents
from app.retrieval.vector_store import create_vector_store

repo_url = "https://github.com/tiangolo/fastapi"
repo_path = "./repo"

docs = load_github_repo(repo_url, repo_path)

chunks = split_documents(docs)

create_vector_store(chunks)

print("Repository indexed successfully")