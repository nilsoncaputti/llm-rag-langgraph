import os
from langchain_community.document_loaders import GitLoader

ALLOWED_EXTENSIONS = [
    ".py",
    ".js",
    ".ts",
    ".md",
    ".go",
    ".java",
]

def file_filter(file_path: str):

    _, ext = os.path.splitext(file_path)

    return ext in ALLOWED_EXTENSIONS


def load_github_repo(repo_url: str, repo_path: str):

    loader = GitLoader(
        clone_url=repo_url,
        repo_path=repo_path,
        branch="master",
        file_filter=file_filter
    )

    return loader.load()