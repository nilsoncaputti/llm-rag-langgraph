from app.ingestion.github_loader import load_github_repo
from app.ingestion.text_splitter import split_documents
from app.ingestion.code_chunker import code_aware_chunk

repo_url = "https://github.com/tiangolo/fastapi"
repo_path = "./repo"

docs = load_github_repo(repo_url, repo_path)

# chunks = split_documents(docs)
chunks = code_aware_chunk(docs)

print(chunks[0])