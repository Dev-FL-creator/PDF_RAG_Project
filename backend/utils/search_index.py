"""
Search Index Module
===================
Handles Azure AI Search index schema creation and management.
"""

import json
import requests
from fastapi import HTTPException

# Vector field name for embeddings in search index
VECTOR_FIELD = "combined_vector"


def build_index_schema(name: str, dims: int, metric: str) -> dict:
    """
    Build Azure AI Search index schema for PDF RAG.

    This schema defines:
    - Document fields (content, metadata, vectors)
    - Vector search configuration (HNSW algorithm)
    - Semantic ranking configuration (for Azure Semantic Ranker)

    Args:
        name: Index name
        dims: Embedding vector dimensions (e.g., 1536 for ada-002)
        metric: Vector similarity metric ("cosine", "euclidean", or "dotProduct")

    Returns:
        dict: Complete index schema for Azure Search API
    """
    return {
        "name": name,
        "fields": [
            # Unique document identifier
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False, "filterable": True, "retrievable": True},
            # Main content field (searchable text)
            {"name": "content", "type": "Edm.String", "searchable": True, "retrievable": True},
            # Source PDF filename
            {"name": "source", "type": "Edm.String", "searchable": False, "retrievable": True},
            # Page number in PDF
            {"name": "page", "type": "Edm.Int32", "searchable": False, "retrievable": True},
            # Chunk index within the document
            {"name": "chunk_id", "type": "Edm.Int32", "searchable": False, "retrievable": True},
            # Type of chunk: "text", "image", "table", etc.
            {"name": "chunk_type", "type": "Edm.String", "searchable": False, "retrievable": True, "filterable": True},
            # Vector embedding field for semantic search
            {
                "name": VECTOR_FIELD,
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": False,
                "dimensions": dims,
                "vectorSearchProfile": "pdf-vector-profile"
            },
        ],
        # Vector search configuration using HNSW algorithm
        "vectorSearch": {
            "algorithms": [{
                "name": "hnsw-config",
                "kind": "hnsw",  # Hierarchical Navigable Small World algorithm
                "hnswParameters": {
                    "m": 8,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": metric
                }
            }],
            "profiles": [{"name": "pdf-vector-profile", "algorithm": "hnsw-config"}]
        },
        # Semantic ranking configuration (for Azure Semantic Ranker)
        "semantic": {
            "configurations": [{
                "name": "semantic-config",
                "prioritizedFields": {
                    "titleField": {"fieldName": "content"},
                    "prioritizedContentFields": [{"fieldName": "content"}],
                    "prioritizedKeywordsFields": [{"fieldName": "source"}]
                }
            }]
        }
    }


def http_error(status: int, message: str):
    """
    Log and raise HTTP exception.

    Args:
        status: HTTP status code
        message: Error message
    """
    print(f"[ERROR] HTTP {status} -> {message}")
    raise HTTPException(status_code=status, detail=message)


def ensure_index(cfg: dict, index_name: str, dims: int, metric: str):
    """
    Ensure that the specified Azure Search index exists, create if not.

    Args:
        cfg: Configuration dictionary with Azure Search credentials
        index_name: Name of the index to ensure
        dims: Vector embedding dimensions
        metric: Vector similarity metric
    """
    service = cfg["search_service_name"]
    api_key = cfg["search_api_key"]
    api_version = cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    headers = {"Content-Type": "application/json", "api-key": api_key}

    # Check if index already exists
    get_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
    r = requests.get(get_url, headers=headers)
    if r.status_code == 200:
        print(f"[INFO] Index '{index_name}' already exists")
        return
    if r.status_code != 404:
        http_error(r.status_code, f"Index check failed: {r.text}")

    # Create new index
    payload = build_index_schema(index_name, dims, metric)
    print(f"[INFO] Creating new index: {index_name}")
    c = requests.post(f"{endpoint}/indexes?api-version={api_version}", headers=headers, data=json.dumps(payload))
    if c.status_code not in (200, 201):
        http_error(c.status_code, f"Create index failed: {c.text}")
