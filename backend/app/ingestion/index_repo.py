from app.ingestion.github_loader import load_github_repo
from app.ingestion.code_chunker import code_aware_chunk
from app.retrieval.vector_store import create_vector_store

repo_url = "https://github.com/tiangolo/fastapi"
repo_path = "./repo"

docs = load_github_repo(repo_url, repo_path)

chunks = code_aware_chunk(docs)

create_vector_store(chunks)

print("Repository indexed successfully")