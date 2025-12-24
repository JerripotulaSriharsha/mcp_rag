# query.py (only the tool function changed)
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI
from qdrant_client import QdrantClient
import os

RENDER_HOST = "mcp-rag-6fz6.onrender.com"
COLLECTION_NAME = "rag_mcp"

load_dotenv()

mcp = FastMCP(
    "mcp rag server",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[
            "localhost:*",
            "127.0.0.1:*",
            "0.0.0.0:*",
            RENDER_HOST,
            f"{RENDER_HOST}:*",
        ],
    ),
)

@mcp.tool()
def query_qdrant(query_text: str, n_results: int = 3):
    # 1) embed the query
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    embedding = openai_client.embeddings.create(
        model="text-embedding-ada-002",  # use this (ada-002 is legacy)
        input=query_text,
    ).data[0].embedding

    # 2) query qdrant
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ["QDRANT_API_KEY"],
    )

    resp = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,          # <-- dense vector
        limit=n_results,
        with_payload=True,
    )

    context = ""
    for i, p in enumerate(resp.points, start=1):
        payload = p.payload or {}
        meta = payload.get("metadata", {})

        context += f"""
### Chunk {i}
Source: {meta.get('file_name')}
Created: {meta.get('creation_date')}
Score: {p.score}

{payload.get('text')}
"""

    # --- send to GPT-4o-mini ---
    completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Answer using ONLY the provided context. Be precise."
            },
            {
                "role": "user",
                "content": f"""
Context:
{context}

Question:
{query_text}
"""
            }
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content