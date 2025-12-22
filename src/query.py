import chromadb
from dotenv import load_dotenv
load_dotenv()
PERSISTENT_DIR = "./chroma_db"
COLLECTION_NAME = "rag_mcp"

chroma_client = chromadb.PersistentClient(path=PERSISTENT_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

res = collection.query(
    query_texts=["PhysX-Anything?"],
    n_results=2
)

print(res)                       # <-- this prints the full output dict
print(res["documents"][0])       # <-- just the retrieved texts
print(res["metadatas"][0])       # <-- metadata
print(res["distances"][0])       # <-- similarity distances
