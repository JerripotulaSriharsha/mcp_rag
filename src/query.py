import chromadb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import pprint
load_dotenv()
PERSISTENT_DIR = "./chroma_db"
COLLECTION_NAME = "rag_mcp"

mcp  = FastMCP("mcp rag server")

@mcp.tool()
def query_chroma(query_text: str, n_results: int = 2):
    """Query the ChromaDB vector database for semantically similar documents.

    Args:
        query_text: The text query to search for similar documents
        n_results: Number of similar results to return (default: 2)

    Returns:
        Query results containing the most similar documents from the collection
    """
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    return collection.query(
        query_texts=[query_text],
        n_results=n_results
    )



