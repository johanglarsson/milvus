"""
Milvus Data Loader
Reads a CSV file, generates embeddings, and inserts documents into Milvus.
"""

import os
import re
import sys

import httpx
from dotenv import load_dotenv
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────

MILVUS_HOST       = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT       = int(os.getenv("MILVUS_PORT", "19530"))
MILVUS_COLLECTION = os.getenv("MILVUS_COLLECTION", "documents")

EMBEDDING_API_TYPE = os.getenv("EMBEDDING_API_TYPE", "ollama")
EMBEDDING_API_URL  = os.getenv("EMBEDDING_API_URL", "http://localhost:11434")
EMBEDDING_MODEL    = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM      = int(os.getenv("EMBEDDING_DIMENSION", "768"))
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY", "")

DATA_FILE = os.getenv("DATA_FILE_PATH", "../data/sample_documents.toon")

BATCH_SIZE = 50


# ── Embedding ─────────────────────────────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    if EMBEDDING_API_TYPE == "ollama":
        r = httpx.post(
            f"{EMBEDDING_API_URL}/api/embeddings",
            json={"model": EMBEDDING_MODEL, "prompt": text},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["embedding"]

    elif EMBEDDING_API_TYPE == "openai":
        r = httpx.post(
            f"{EMBEDDING_API_URL}/v1/embeddings",
            json={"model": EMBEDDING_MODEL, "input": text},
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            timeout=60,
        )
        r.raise_for_status()
        return r.json()["data"][0]["embedding"]

    else:
        raise ValueError(f"Unknown EMBEDDING_API_TYPE '{EMBEDDING_API_TYPE}'. Use 'ollama' or 'openai'.")


# ── .toon parser ─────────────────────────────────────────────────────────────
# Format:
#   [<id>]           — starts a new entry
#   key: value       — field assignment (title, creator, content)
#   # comment        — ignored
#   blank lines      — ignored

def read_toon(path: str) -> list[dict]:
    documents = []
    current: dict | None = None

    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")

            # Skip comments and blank lines
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue

            # New entry header: [id]
            m = re.fullmatch(r"\[(\d+)\]", stripped)
            if m:
                if current:
                    documents.append(current)
                current = {"id": int(m.group(1))}
                continue

            # Key: value pair
            if current is not None and ":" in line:
                key, _, value = line.partition(":")
                key = key.strip().lower()
                value = value.strip()
                if key == "creator":
                    key = "author"          # normalise to schema field name
                if key in ("title", "author", "content"):
                    current[key] = value[:{"title": 500, "author": 200, "content": 5000}[key]]

    if current:
        documents.append(current)

    print(f"Read {len(documents)} toons from {path}")
    return documents


# ── Milvus ────────────────────────────────────────────────────────────────────

def connect():
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    print(f"Connected to Milvus at {MILVUS_HOST}:{MILVUS_PORT}")


def create_collection_if_not_exists() -> Collection:
    if utility.has_collection(MILVUS_COLLECTION):
        print(f"Collection already exists: {MILVUS_COLLECTION}")
        return Collection(MILVUS_COLLECTION)

    fields = [
        FieldSchema("id",        DataType.INT64,        is_primary=True, auto_id=False),
        FieldSchema("title",     DataType.VARCHAR,      max_length=500),
        FieldSchema("author",    DataType.VARCHAR,      max_length=200),
        FieldSchema("content",   DataType.VARCHAR,      max_length=5000),
        FieldSchema("embedding", DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    ]
    schema = CollectionSchema(fields, description="Cartoon document collection")
    col = Collection(MILVUS_COLLECTION, schema)

    col.create_index(
        "embedding",
        {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}},
    )
    print(f"Collection created: {MILVUS_COLLECTION} (IVF_FLAT / COSINE, dim={EMBEDDING_DIM})")
    return col


def insert_batch(col: Collection, batch: list[dict]):
    col.insert([
        [d["id"]        for d in batch],
        [d["title"]     for d in batch],
        [d["author"]    for d in batch],
        [d["content"]   for d in batch],
        [d["embedding"] for d in batch],
    ])


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data_file = sys.argv[1] if len(sys.argv) > 1 else DATA_FILE

    print("=== Milvus Data Loader ===")
    print(f"Collection : {MILVUS_COLLECTION}")
    print(f"Embedding  : {EMBEDDING_API_TYPE} / {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"Data file  : {data_file}")
    print()

    documents = read_toon(data_file)

    print(f"Generating embeddings for {len(documents)} documents...")
    for i, doc in enumerate(documents):
        text = f"{doc['title']}. {doc['content']}"
        doc["embedding"] = get_embedding(text)
        print(f"  [{i+1}/{len(documents)}] {doc['title']}")

    connect()
    col = create_collection_if_not_exists()

    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        insert_batch(col, batch)
        print(f"Inserted batch {i // BATCH_SIZE + 1} ({len(batch)} documents)")

    col.flush()
    print(f"\nDone! {len(documents)} documents loaded into '{MILVUS_COLLECTION}'.")


if __name__ == "__main__":
    main()
