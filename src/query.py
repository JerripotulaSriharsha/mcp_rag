import chromadb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

import pprint
RENDER_HOST = "mcp-rag-6fz6.onrender.com"

load_dotenv()
PERSISTENT_DIR = "./chroma_db"
COLLECTION_NAME = "rag_mcp"

mcp = FastMCP(
    "mcp rag server",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            RENDER_HOST,
            f"{RENDER_HOST}:*",
        ],
    ),
)
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



