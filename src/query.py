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
    chroma_client = chromadb.PersistentClient(path=PERSISTENT_DIR)
    collection = chroma_client.get_collection(name=COLLECTION_NAME)

    return collection.query(
        query_texts=[query_text],
        n_results=n_results
    )



