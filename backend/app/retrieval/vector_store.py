from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_embedding_model():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vector_store(chunks):

    embeddings = get_embedding_model()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    vector_store.persist()

    return vector_store

def load_vector_store():

    embeddings = get_embedding_model()

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )


def search_documents(query, k=4):

    db = load_vector_store()

    results = db.similarity_search(query, k=k)

    return results