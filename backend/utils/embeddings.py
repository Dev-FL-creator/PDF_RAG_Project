"""
Embeddings Module
=================
Handles text embedding generation with caching and retry logic.
"""

import os
import re
import time
import random
from typing import List, Dict
from hashlib import sha1
from fastapi import HTTPException
from openai import AzureOpenAI, RateLimitError, APIError

# ============================================================
# EMBEDDING CACHE
# ============================================================

# In-memory cache for embeddings to reduce API calls
# Key: hash(model_name + text), Value: embedding vector
_EMBED_CACHE: Dict[str, List[float]] = {}


def _cache_key(model: str, text: str) -> str:
    """
    Generate a unique cache key for embedding lookup.

    Args:
        model: Embedding model name (e.g., "text-embedding-ada-002")
        text: Input text to embed

    Returns:
        SHA1 hash of "model||text" as cache key
    """
    h = sha1()
    h.update((model + "||" + text).encode("utf-8", errors="ignore"))
    return h.hexdigest()


def safe_doc_id(filename: str, i: int) -> str:
    """
    Generate a safe document ID for Azure Search index.

    Azure Search has restrictions on document ID characters, so we need to
    sanitize the filename and append a unique index.

    Args:
        filename: Original PDF filename
        i: Chunk index

    Returns:
        Safe document ID (e.g., "my_document-42")
    """
    # Remove file extension
    name = os.path.splitext(filename)[0]
    # Replace invalid characters with underscore
    name = re.sub(r'[^0-9a-zA-Z_\-=]', "_", name)
    name = name.strip('_')
    # Ensure it starts with alphanumeric character
    if not name or not name[0].isalnum():
        name = f"doc_{name}" if name else "doc"
    return f"{name}-{i}"


def embed_text(
    aoai: AzureOpenAI,
    model: str,
    texts: List[str],
    batch_size: int = 16,
    max_retries: int = 6,
    base_delay: float = 1.0,
    jitter: float = 0.25,
) -> List[List[float]]:
    """
    Generate embeddings for text with caching and robust retry logic.

    This function implements:
    - In-memory caching to avoid redundant API calls
    - Batch processing to optimize throughput
    - Exponential backoff with jitter for rate limit handling
    - Automatic retry on transient errors

    Args:
        aoai: Azure OpenAI client instance
        model: Embedding model name (e.g., "text-embedding-ada-002")
        texts: List of texts to embed
        batch_size: Number of texts to embed in one API call
        max_retries: Maximum number of retry attempts for rate limits
        base_delay: Initial delay in seconds for exponential backoff
        jitter: Random jitter factor (0-1) to avoid thundering herd

    Returns:
        List of embedding vectors (one per input text)
    """
    if not texts:
        return []

    n = len(texts)
    outputs: List[List[float]] = [None] * n  # type: ignore
    uncached_indices: List[int] = []
    uncached_payload: List[str] = []

    # Check cache first to avoid redundant API calls
    for i, t in enumerate(texts):
        k = _cache_key(model, t)
        if k in _EMBED_CACHE:
            outputs[i] = _EMBED_CACHE[k]
        else:
            uncached_indices.append(i)
            uncached_payload.append(t)

    # If everything was cached, return immediately
    if not uncached_payload:
        return outputs  # type: ignore

    # Process uncached texts in batches
    for b in range(0, len(uncached_payload), batch_size):
        sub_payload = uncached_payload[b: b + batch_size]
        sub_indices = uncached_indices[b: b + batch_size]

        attempt = 0
        delay = base_delay

        while True:
            try:
                # Call Azure OpenAI embedding API
                resp = aoai.embeddings.create(model=model, input=sub_payload)
                vecs = [d.embedding for d in resp.data]

                # Sanity check: ensure we got the expected number of embeddings
                if len(vecs) != len(sub_payload):
                    raise RuntimeError(f"Embedding count mismatch")

                # Store embeddings in cache and output array
                for local_i, vec in enumerate(vecs):
                    src_idx = sub_indices[local_i]
                    k = _cache_key(model, texts[src_idx])
                    _EMBED_CACHE[k] = vec
                    outputs[src_idx] = vec
                break  # Success, exit retry loop

            except RateLimitError as e:
                # Handle rate limit (429) with exponential backoff
                attempt += 1
                if attempt > max_retries:
                    raise HTTPException(status_code=429, detail=f"Rate limit: {str(e)}")
                sleep_s = delay * (1.0 + random.uniform(0, jitter))
                print(f"[429] Retry {attempt}/{max_retries} in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                delay = min(delay * 2.0, 60.0)  # Exponential backoff, capped at 60s

            except APIError as e:
                # Handle 5xx server errors with retry
                attempt += 1
                if getattr(e, "status_code", 500) >= 500 and attempt <= max_retries:
                    sleep_s = delay * (1.0 + random.uniform(0, jitter))
                    print(f"[5xx] Retry {attempt}/{max_retries} in {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    delay = min(delay * 2.0, 60.0)
                else:
                    raise HTTPException(status_code=502, detail=f"API error: {str(e)}")

            except Exception as e:
                # Unexpected error, fail fast
                raise HTTPException(status_code=500, detail=f"Embedding failed: {repr(e)}")

    return outputs  # type: ignore
