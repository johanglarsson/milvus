# Milvus Toon Loader + MCP Search Server

Load cartoon data from a CSV into Milvus using Python, then query it semantically via an MCP server.

## Prerequisites

- [Podman](https://podman.io/) + [podman-compose](https://github.com/containers/podman-compose)
- Python 3.11+

```bash
pip install podman-compose
```

## 1. Start the stack

```bash
podman-compose up -d
```

This starts **etcd**, **MinIO**, **Milvus**, and **Ollama** as containers.

## 2. Pull the embedding model (first time only)

```bash
podman exec milvus-ollama ollama pull nomic-embed-text
```

## 3. Load data into Milvus

```bash
cd python-loader
cp .env.example .env      # edit if needed
pip install -r requirements.txt
python loader.py
```

The loader reads `data/sample_documents.csv` (20 classic cartoons), generates
768-dimensional embeddings via Ollama, and inserts them into the `documents`
collection in Milvus.

Pass a custom CSV as an argument:

```bash
python loader.py /path/to/your/data.csv
```

## 4. Start the MCP server

```bash
cd mcp-server
cp .env.example .env      # edit if needed
pip install -r requirements.txt
python server.py
```

### MCP tools exposed

| Tool | Description |
|------|-------------|
| `search_documents` | Semantic search — finds documents closest to a natural-language query |
| `get_document` | Retrieve a document by numeric ID |
| `collection_info` | Schema and entity count for the collection |

## Configuration

Both the loader and MCP server share the same environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus gRPC port |
| `MILVUS_COLLECTION` | `documents` | Collection name |
| `EMBEDDING_API_TYPE` | `ollama` | `ollama` or `openai` |
| `EMBEDDING_API_URL` | `http://localhost:11434` | Embedding API base URL |
| `EMBEDDING_MODEL` | `nomic-embed-text` | Model name |
| `EMBEDDING_DIMENSION` | `768` | Vector dimension (must match model) |

## Stop the stack

```bash
podman-compose down
```
