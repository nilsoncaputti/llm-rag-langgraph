import ast
from langchain_core.documents import Document


def chunk_python_code(doc: Document):

    source = doc.metadata.get("source", "")
    code = doc.page_content

    chunks = []

    try:
        tree = ast.parse(code)
    except:
        return []

    for node in tree.body:

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):

            start = node.lineno - 1
            end = node.end_lineno

            lines = code.splitlines()[start:end]

            chunk = "\n".join(lines)

            chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,
                        "symbol": node.name
                    }
                )
            )

    return chunks


def code_aware_chunk(documents):

    chunks = []

    for doc in documents:

        source = doc.metadata.get("source", "")

        if source.endswith(".py"):

            chunks.extend(chunk_python_code(doc))

        else:
            chunks.append(doc)

    return chunks