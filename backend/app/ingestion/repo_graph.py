import ast
import networkx as nx
from langchain_core.documents import Document


def build_repo_graph(documents: list[Document]):

    graph = nx.DiGraph()

    for doc in documents:

        source = doc.metadata.get("source", "")
        code = doc.page_content

        if not source.endswith(".py"):
            continue

        try:
            tree = ast.parse(code)
        except:
            continue

        for node in ast.walk(tree):

            if isinstance(node, ast.FunctionDef):

                func_name = node.name

                graph.add_node(func_name, file=source)

                for child in ast.walk(node):

                    if isinstance(child, ast.Call):

                        if isinstance(child.func, ast.Name):

                            called_func = child.func.id

                            graph.add_edge(func_name, called_func)
    
    nx.write_gml(graph, "repo_graph.gml")

    return graph