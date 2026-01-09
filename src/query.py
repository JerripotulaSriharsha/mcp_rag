# query.py (only the tool function changed)
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from openai import OpenAI
from qdrant_client import QdrantClient
import os
load_dotenv()

COLLECTION_NAME = "rag_mcp"

render_host = os.environ.get("RENDER_EXTERNAL_HOSTNAME")  

allowed = [
    "localhost:*",
    "127.0.0.1:*",
    "0.0.0.0:*",
]

if render_host:
    allowed += [render_host, f"{render_host}:*"]

mcp = FastMCP(
    "mcp rag server",
    transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=allowed,
    ),
)


@mcp.tool()
def query_qdrant(query_text: str):
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
        limit=10,
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