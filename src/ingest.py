import os
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse
from dotenv import load_dotenv
load_dotenv()
PERSISTENT_DIR = "./chroma_db"
COLLECTION_NAME = "rag_mcp"
DATA_DIR = r"D:\Narwal\mcp_rag\data"

def get_collection():
    client = chromadb.PersistentClient(path=PERSISTENT_DIR)
    ef = OpenAIEmbeddingFunction(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="text-embedding-ada-002",  # if it errors, switch to "text-embedding-3-small"
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )
    return client, collection

def ingest_data_dir():
    # recreate collection cleanly so embedding function matches
    client = chromadb.PersistentClient(path=PERSISTENT_DIR)
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass

    client, collection = get_collection()

    parser = LlamaParse(api_key=os.environ["LLAMA_CLOUD_API_KEY"], result_type="text")
    docs = SimpleDirectoryReader(DATA_DIR, file_extractor={".pdf": parser}).load_data()

    for d in docs:
        # IMPORTANT: avoid massive single-doc payloads → truncate or chunk later
        text = d.text or ""
        collection.add(
            documents=[text],
            metadatas=[d.metadata],
            ids=[d.doc_id],
        )

    print("Ingested:", collection.count())

if __name__ == "__main__":
    ingest_data_dir()
