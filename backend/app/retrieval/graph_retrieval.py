import networkx as nx
from langchain_core.documents import Document

def expand_with_graph(retrieved_docs: list[Document]):

    try:
        graph = nx.read_gml("repo_graph.gml")
    except:
        return []

    expanded = []

    for doc in retrieved_docs:

        symbol = doc.metadata.get("symbol")

        if not symbol:
            continue

        if not graph.has_node(symbol):
            continue

        neighbors = list(graph.successors(symbol))

        expanded.append(symbol)

        expanded.extend(neighbors)

    return expanded