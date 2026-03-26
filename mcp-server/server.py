"""
Milvus MCP Server
Exposes semantic search and document retrieval over a Milvus collection.
"""

import json
import os

import httpx
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pymilvus import Collection, connections, utility

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "documents")

EMBEDDING_API_TYPE = os.getenv("EMBEDDING_API_TYPE", "ollama")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL", "http://localhost:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "768"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

mcp = FastMCP("Milvus Document Search")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _connect() -> Collection:
    """Return a loaded Milvus collection, opening the connection if needed."""
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    if not utility.has_collection(MILVUS_COLLECTION):
        raise RuntimeError(f"Collection '{MILVUS_COLLECTION}' does not exist in Milvus.")
    col = Collection(MILVUS_COLLECTION)
    col.load()
    return col


def _get_embedding(text: str) -> list[float]:
    """Generate a vector embedding for the given text using the configured API."""
    if EMBEDDING_API_TYPE == "ollama":
        response = httpx.post(
            f"{EMBEDDING_API_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["embedding"]

    elif EMBEDDING_API_TYPE == "openai":
        response = httpx.post(
            f"{EMBEDDING_API_URL}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text},
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    else:
        raise ValueError(
            f"Unknown EMBEDDING_API_TYPE '{EMBEDDING_API_TYPE}'. Use 'ollama' or 'openai'."
        )


# ── MCP Tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def search_documents(query: str, limit: int = 5) -> str:
    """
    Perform a semantic search over the document collection.

    Args:
        query: Natural-language search query.
        limit: Maximum number of results to return (default 5).

    Returns:
        JSON array of matching documents with id, title, author, content, and similarity score.
    """
    col = _connect()
    embedding = _get_embedding(query)

    results = col.search(
        data=[embedding],
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=limit,
        output_fields=["id", "title", "author", "content"],
    )

    hits = []
    for result in results:
        for hit in result:
            hits.append(
                {
                    "id": hit.entity.get("id"),
                    "title": hit.entity.get("title"),
                    "author": hit.entity.get("author"),
                    "content": hit.entity.get("content"),
                    "score": round(hit.score, 4),
                }
            )

    return json.dumps(hits, indent=2, ensure_ascii=False)


@mcp.tool()
def get_document(document_id: int) -> str:
    """
    Retrieve a single document by its numeric ID.

    Args:
        document_id: The integer ID of the document.

    Returns:
        JSON object with id, title, author, and content, or an error message.
    """
    col = _connect()
    results = col.query(
        expr=f"id == {document_id}",
        output_fields=["id", "title", "author", "content"],
    )

    if not results:
        return json.dumps({"error": f"No document found with id={document_id}"})

    return json.dumps(results[0], indent=2, ensure_ascii=False)


@mcp.tool()
def collection_info() -> str:
    """
    Return statistics and schema information about the Milvus collection.

    Returns:
        JSON object with collection name, entity count, and field definitions.
    """
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)

    if not utility.has_collection(MILVUS_COLLECTION):
        return json.dumps({"error": f"Collection '{MILVUS_COLLECTION}' does not exist."})

    col = Collection(MILVUS_COLLECTION)
    schema = col.schema

    fields = [
        {
            "name": f.name,
            "type": str(f.dtype.name),
            "is_primary": f.is_primary,
            **({"max_length": f.params.get("max_length")} if "max_length" in f.params else {}),
            **({"dimension": f.params.get("dim")} if "dim" in f.params else {}),
        }
        for f in schema.fields
    ]

    info = {
        "collection_name": MILVUS_COLLECTION,
        "description": schema.description,
        "num_entities": col.num_entities,
        "fields": fields,
    }

    return json.dumps(info, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
