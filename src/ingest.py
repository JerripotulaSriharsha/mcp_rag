import os
import uuid
from dotenv import load_dotenv

from llama_index.core import SimpleDirectoryReader
from llama_parse import LlamaParse

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

import tiktoken


load_dotenv()

DATA_DIR = r"D:\Narwal\mcp_rag\data"
COLLECTION_NAME = "rag_mcp"

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]


openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)


tokenizer = tiktoken.get_encoding("cl100k_base")

TOKEN_SIZE = 1000

def chunk_by_tokens(text):
    tokens = tokenizer.encode(text)
    chunks = []

    for i in range(0, len(tokens), TOKEN_SIZE):
        chunk_tokens = tokens[i:i + TOKEN_SIZE]
        chunk_text = tokenizer.decode(chunk_tokens)
        if chunk_text.strip():
            chunks.append(chunk_text)

    return chunks


def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=qm.VectorParams(
                size=1536,  # text-embedding-3-small
                distance=qm.Distance.COSINE,
            ),
        )


def ingest_data_dir():
    ensure_collection()

    parser = LlamaParse(
        api_key=os.environ["LLAMA_CLOUD_API_KEY"],
        result_type="text"
    )

    docs = SimpleDirectoryReader(
        DATA_DIR,
        file_extractor={".pdf": parser}
    ).load_data()

    for doc in docs:
        text = doc.text or ""
        chunks = chunk_by_tokens(text)

        for idx, chunk in enumerate(chunks):
            embedding = openai_client.embeddings.create(
                model="text-embedding-ada-002",
                input=chunk,
            ).data[0].embedding

            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=[
                    qm.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "metadata": doc.metadata,
                            "chunk_index": idx,
                            "source_doc": doc.doc_id,
                        },
                    )
                ],
            )

    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_data_dir()
