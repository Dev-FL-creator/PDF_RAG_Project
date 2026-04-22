"""
PDF RAG Router Module
====================
FastAPI router for PDF-based Retrieval Augmented Generation (RAG) system.

Key Features:
- PDF document upload and processing
- Hybrid search (BM25 keyword + vector semantic + Azure Semantic Ranker)
- Image analysis using GPT-4o
- AI Agent for adaptive retrieval and reasoning
- Real-time agent trace streaming via SSE
"""

import os
import json
import tempfile
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from openai import AzureOpenAI
import re
import time
import random
from hashlib import sha1
from openai import RateLimitError, APIError
from azure.ai.formrecognizer import DocumentAnalysisClient            
from azure.core.credentials import AzureKeyCredential
import threading

# Import PDF Agent Engine for adaptive retrieval
from AI_AGENT.pdf_agent_engine import run_pdf_agent

# Import utility modules for semantic chunking and image analysis
from utils.semantic_chunker import create_semantic_chunks
from utils.image_analyzer import PDFImageAnalyzer, format_image_analysis_as_complete_text

# ============================================================
# GLOBAL VARIABLES
# ============================================================

# In-memory cache for embeddings to reduce API calls
# Key: hash(model_name + text), Value: embedding vector
_EMBED_CACHE: Dict[str, List[float]] = {}

# Storage for agent reasoning traces (for real-time SSE streaming)
# Key: trace_id, Value: list of trace log lines
_agent_traces: Dict[str, List[str]] = {}
_trace_lock = threading.Lock()  # Thread-safe access to traces

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

router = APIRouter(prefix="/api/pdf_rag", tags=["pdf-rag"])

# ============================================================
# SSE STREAMING ENDPOINT
# ============================================================

@router.get("/agent_trace_stream/{trace_id}")
async def pdf_agent_trace_stream(trace_id: str):
    """
    Server-Sent Events (SSE) endpoint for streaming PDF RAG agent trace in real-time.
    
    This allows the frontend to receive agent reasoning steps as they happen,
    providing transparency into the agent's decision-making process.
    
    Args:
        trace_id: Unique identifier for the trace session
    
    Yields:
        SSE events containing trace lines or status updates
    """
    async def event_generator():
        last_index = 0
        max_wait_time = 120  # 2 minutes timeout
        start_time = time.time()
        
        while True:
            # Check for timeout
            if time.time() - start_time > max_wait_time:
                yield f"data: {json.dumps({'status': 'timeout'})}\n\n"
                break
            
            # Thread-safe access to trace storage
            with _trace_lock:
                if trace_id in _agent_traces:
                    trace_lines = _agent_traces[trace_id]
                    
                    # Send new lines since last_index
                    while last_index < len(trace_lines):
                        line = trace_lines[last_index]
                        yield f"data: {json.dumps({'line': line})}\n\n"
                        last_index += 1
                        
                        # Check if this is the end marker
                        if "=== Agent Decision End ===" in line or "=== PDF Agent Decision End ===" in line:
                            yield f"data: {json.dumps({'status': 'done'})}\n\n"
                            # Clean up trace storage
                            del _agent_traces[trace_id]
                            return
            
            await asyncio.sleep(0.1)  # Poll every 100ms
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# ============================================================
# CONFIGURATION LOADER
# ============================================================

def load_config() -> dict:
    """
    Load configuration from config.json file.
    
    Returns:
        dict: Configuration dictionary containing API keys, endpoints, etc.
    """
    cfg_path = os.getenv("CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "..", "config.json")
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Configuration constants
DEFAULT_EMBED_DIM = 1536  # Default embedding dimension for text-embedding-ada-002
DEFAULT_METRIC = "cosine"  # Vector similarity metric (cosine, euclidean, dotProduct)
VECTOR_FIELD = "combined_vector"  # Field name for vector embeddings in search index

# ============================================================
# PDF EXTRACTION FUNCTIONS
# ============================================================

def extract_pages_with_docint(
    pdf_path: str, 
    endpoint: str, 
    key: str,
    aoai: Optional[AzureOpenAI] = None,
    enable_image_analysis: bool = True
) -> List[tuple]:
    """
    Extract text, tables, and images from PDF using Azure Document Intelligence.
    
    This function uses Azure's prebuilt-document model to extract structured content
    from PDFs, including paragraphs, tables, and key-value pairs. It also optionally
    analyzes images using GPT-4o vision capabilities.
    
    Args:
        pdf_path: Path to the PDF file
        endpoint: Azure Document Intelligence endpoint URL
        key: Azure Document Intelligence API key
        aoai: Optional Azure OpenAI client for image analysis
        enable_image_analysis: Whether to analyze images with GPT-4o
    
    Returns:
        List of (page_number, page_content) tuples with proper page tracking
    """
    # Initialize Document Intelligence client
    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()

    # Build page-based structure to organize content by page number
    page_contents = {}  # {page_num: [content_items]}
    
    # 1) Extract paragraphs with page tracking
    for p in (getattr(result, "paragraphs", None) or []):
        content = (getattr(p, "content", "") or "").strip()
        if not content:
            continue
        
        # Get page number from bounding regions
        page_num = 1
        if hasattr(p, "bounding_regions") and p.bounding_regions:
            page_num = p.bounding_regions[0].page_number
        
        if page_num not in page_contents:
            page_contents[page_num] = []
        
        page_contents[page_num].append({
            "type": "paragraph",
            "content": content,
            "order": len(page_contents[page_num])  # Preserve document structure order
        })

    # 2) Extract tables with page tracking
    table_id = 0
    for t in (getattr(result, "tables", None) or []):
        table_id += 1
        
        # Get page number
        page_num = 1
        if hasattr(t, "bounding_regions") and t.bounding_regions:
            page_num = t.bounding_regions[0].page_number
        
        # Build table structure cell by cell
        rows_map = {}
        for cell in t.cells:
            rows_map.setdefault(cell.row_index, {})
            rows_map[cell.row_index][cell.column_index] = (cell.content or "").strip()

        if rows_map:
            # Calculate table dimensions
            try:
                r_cnt = int(getattr(t, "row_count", None) or (max(rows_map.keys()) + 1))
            except Exception:
                r_cnt = max(rows_map.keys()) + 1
            try:
                c_cnt = int(getattr(t, "column_count", None) or (max(max(r.keys()) for r in rows_map.values()) + 1))
            except Exception:
                c_cnt = max(max(r.keys()) for r in rows_map.values()) + 1

            # Create TSV (tab-separated values) representation
            tsv_lines = []
            for r in range(r_cnt):
                row = rows_map.get(r, {})
                cols = [row.get(c, "") for c in range(c_cnt)]
                tsv_lines.append("\t".join(cols).rstrip())

            # Format table with markers for easy identification
            table_content = (
                f"[[TABLE {table_id} rows={r_cnt} cols={c_cnt}]]\n" +
                "\n".join(tsv_lines) +
                "\n[[/TABLE]]"
            )
            
            if page_num not in page_contents:
                page_contents[page_num] = []
            
            page_contents[page_num].append({
                "type": "table",
                "content": table_content,
                "order": len(page_contents[page_num])
            })

    # 3) Extract key-value pairs with page tracking
    for kv in (getattr(result, "key_value_pairs", None) or []):
        k = getattr(getattr(kv, "key", None), "content", None)
        v = getattr(getattr(kv, "value", None), "content", None)
        line = f"{(k or '').strip()} : {(v or '').strip()}".strip(" :")
        
        if not line:
            continue
        
        # Get page number
        page_num = 1
        if hasattr(kv, "key") and hasattr(kv.key, "bounding_regions") and kv.key.bounding_regions:
            page_num = kv.key.bounding_regions[0].page_number
        
        if page_num not in page_contents:
            page_contents[page_num] = []
        
        page_contents[page_num].append({
            "type": "kv",
            "content": f"[[KV]] {line}",
            "order": len(page_contents[page_num])
        })

    # 4) Analyze images if enabled
    image_results = []
    if enable_image_analysis and aoai:
        try:
            print("[INFO] Analyzing images in PDF...")
            
            # Initialize GPT-4o vision analyzer
            analyzer = PDFImageAnalyzer(
                openai_client=aoai,
                deployment_name="gpt-4o",
                min_image_size=100  # Skip images smaller than 100x100 pixels
            )
            
            # Get document context for better image understanding
            doc_context = ""
            if hasattr(result, "content"):
                doc_context = result.content[:500]  # First 500 chars as context
            
            # Analyze all images in the PDF
            image_results = analyzer.analyze_all_images(pdf_path, context=doc_context)
            print(f"[INFO] Analyzed {len(image_results)} images")
            
            # Store image results separately - they will become standalone chunks
            # This prevents long image descriptions from being split across multiple chunks
            if not hasattr(extract_pages_with_docint, '_image_results'):
                extract_pages_with_docint._image_results = {}
            extract_pages_with_docint._image_results[pdf_path] = image_results
                
        except Exception as e:
            print(f"[WARNING] Image analysis failed: {e}")
            if not hasattr(extract_pages_with_docint, '_image_results'):
                extract_pages_with_docint._image_results = {}
            extract_pages_with_docint._image_results[pdf_path] = []

    # 5) Build final page-based output
    pages_output = []
    
    if not page_contents:
        # Fallback: use full document content if no structured content found
        full = (getattr(result, "content", "") or "").strip()
        if full:
            pages_output.append((1, full))
    else:
        # Combine content for each page in document order
        for page_num in sorted(page_contents.keys()):
            items = page_contents[page_num]
            
            # Sort by order (preserve document structure)
            items.sort(key=lambda x: x["order"])
            
            # Combine all content for this page with double newlines as separators
            page_text = "\n\n".join(item["content"] for item in items)
            
            if page_text.strip():
                pages_output.append((page_num, page_text.strip()))
    
    return pages_output


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

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
        sub_payload = uncached_payload[b : b + batch_size]
        sub_indices = uncached_indices[b : b + batch_size]

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


def http_error(status: int, message: str):
    """
    Log and raise HTTP exception.
    
    Args:
        status: HTTP status code
        message: Error message
    """
    print(f"[ERROR] HTTP {status} -> {message}")
    raise HTTPException(status_code=status, detail=message)

# ============================================================
# AZURE SEARCH INDEX SCHEMA
# ============================================================

def build_minimal_index_schema(name: str, dims: int, metric: str) -> dict:
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
                    "m": 8,  # Number of bi-directional links (higher = better recall, slower)
                    "efConstruction": 400,  # Size of dynamic candidate list for construction
                    "efSearch": 500,  # Size of dynamic candidate list for search
                    "metric": metric  # Similarity metric (cosine, euclidean, dotProduct)
                }
            }],
            "profiles": [{"name": "pdf-vector-profile", "algorithm": "hnsw-config"}]
        },
        # Semantic ranking configuration (for Azure Semantic Ranker)
        "semantic": {
            "configurations": [{
                "name": "semantic-config",
                "prioritizedFields": {
                    # Highest priority field (treated as document title)
                    "titleField": {"fieldName": "content"},
                    # Main content fields for deep semantic analysis
                    "prioritizedContentFields": [{"fieldName": "content"}],
                    # Keyword/metadata fields for additional context
                    "prioritizedKeywordsFields": [{"fieldName": "source"}]
                }
            }]
        }
    }

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
    payload = build_minimal_index_schema(index_name, dims, metric)
    print(f"[INFO] Creating new index: {index_name}")
    c = requests.post(f"{endpoint}/indexes?api-version={api_version}", headers=headers, data=json.dumps(payload))
    if c.status_code not in (200, 201):
        http_error(c.status_code, f"Create index failed: {c.text}")

# ============================================================
# PYDANTIC MODELS
# ============================================================

class QueryBody(BaseModel):
    """Request body for PDF query endpoint"""
    index_name: str  # Name of the Azure Search index to query
    query: str  # User's question
    top_k: int = 5  # Number of top results to retrieve
    use_agent: bool = False  # Whether to use AI agent for adaptive retrieval
    trace_id: Optional[str] = None  # Optional trace ID for SSE streaming

class CreateIndexBody(BaseModel):
    """Request body for index creation endpoint"""
    index_name: str  # Name of the index to create
    vector_dimensions: int = DEFAULT_EMBED_DIM  # Embedding vector dimensions
    recreate: bool = False  # Whether to recreate if index already exists
    metric: str = DEFAULT_METRIC  # Vector similarity metric

# ============================================================
# API ENDPOINTS
# ============================================================

@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), index_name: str = Form(...)):
    """
    Upload and process a PDF document for RAG.
    
    This endpoint:
    1. Extracts text, tables, and images from PDF using Azure Document Intelligence
    2. Analyzes images with GPT-4o vision (if enabled)
    3. Performs semantic chunking to create optimal-sized chunks
    4. Generates embeddings for all chunks
    5. Uploads chunks to Azure AI Search index
    
    Args:
        file: PDF file to upload
        index_name: Name of the Azure Search index to upload to
    
    Returns:
        dict: Upload status with chunk count and embedding dimensions
    """
    cfg = load_config()
    service, api_key, api_version = cfg["search_service_name"], cfg["search_api_key"], cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"

    embedding_model = cfg.get("embedding_model", "text-embedding-ada-002")
    embed_dims = int(cfg.get("embedding_dimensions", DEFAULT_EMBED_DIM))
    metric = cfg.get("vector_metric", DEFAULT_METRIC)

    # Ensure index exists before upload
    ensure_index(cfg, index_name, embed_dims, metric)

    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Get Azure Document Intelligence credentials
        docint_endpoint = cfg.get("docint_endpoint")
        docint_key = cfg.get("docint_key")

        if not docint_endpoint or not docint_key:
            http_error(500, "Missing docint_endpoint or docint_key in config.json")

        # Initialize Azure OpenAI client (needed for embeddings and image analysis)
        aoai = AzureOpenAI(
            api_key=cfg["openai_api_key"],
            azure_endpoint=cfg["openai_endpoint"],
            api_version=cfg["openai_api_version"]
        )

        # Extract pages with optional image analysis
        enable_images = cfg.get("enable_image_analysis", True)
        pages = extract_pages_with_docint(
            tmp_path, 
            docint_endpoint, 
            docint_key,
            aoai=aoai,
            enable_image_analysis=enable_images
        )

        if not pages:
            http_error(400, "PDF has no extractable text")

        # Perform semantic chunking on extracted text
        contents, meta = [], []
        for page_no, page_text in pages:
            # Get chunking parameters from config
            target_size = cfg.get("chunk_target_size", 800)
            min_size = cfg.get("chunk_min_size", 200)
            max_size = cfg.get("chunk_max_size", 1500)
            
            # Create semantic chunks that respect sentence boundaries
            semantic_chunks = create_semantic_chunks(
                text=page_text, 
                page_number=page_no,
                target_size=target_size,
                min_size=min_size,
                max_size=max_size
            )
            
            for ci, chunk_dict in enumerate(semantic_chunks):
                contents.append(chunk_dict["content"])
                meta.append({
                    "page": chunk_dict["page_number"], 
                    "chunk_id": ci,
                    "chunk_type": chunk_dict.get("chunk_type", "text")
                })

        # Add image analysis results as standalone chunks
        # Images are kept whole to prevent splitting long descriptions
        if hasattr(extract_pages_with_docint, '_image_results'):
            image_results = extract_pages_with_docint._image_results.get(tmp_path, [])
            
            for img_idx, img_result in enumerate(image_results):
                # Use complete image description (not truncated)
                image_text = format_image_analysis_as_complete_text(img_result)
                contents.append(image_text)
                meta.append({
                    "page": img_result.page_number,
                    "chunk_id": len(contents) - 1,
                    "chunk_type": "image",
                    "image_index": img_idx
                })
            
            print(f"[INFO] Added {len(image_results)} image chunks")

        print(f"[INFO] Extracted {len(contents)} total chunks from {file.filename}")

        # Generate embeddings for all chunks
        vectors = embed_text(aoai, embedding_model, contents)
        dim = len(vectors[0]) if vectors else 0
        print(f"[INFO] Embedding dimension = {dim}")
        
        # Verify embedding dimensions match index configuration
        if dim != embed_dims:
            http_error(500, f"Embedding dim mismatch: got {dim}, expected {embed_dims}")

        # Upload chunks to Azure Search in batches
        headers = {"Content-Type": "application/json", "api-key": api_key}
        index_url = f"{endpoint}/indexes/{index_name}/docs/index?api-version={api_version}"

        batch_size, total = 256, len(contents)
        for b in range(0, total, batch_size):
            upper = min(b + batch_size, total)
            docs = []
            for i in range(b, upper):
                docs.append({
                    "@search.action": "upload",
                    "id": safe_doc_id(file.filename, i),
                    "content": contents[i],
                    "source": file.filename,
                    "page": meta[i]["page"],
                    "chunk_id": meta[i]["chunk_id"],
                    "chunk_type": meta[i]["chunk_type"],
                    VECTOR_FIELD: vectors[i]
                })
            payload = {"value": docs}
            r = requests.post(index_url, headers=headers, data=json.dumps(payload))
            if r.status_code not in (200, 201):
                http_error(r.status_code, f"Batch upload failed: {r.text}")

        return {"ok": True, "chunks_uploaded": total, "index_name": index_name, "embedding_dim": dim}
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/query")
def query_pdf(body: QueryBody):
    """
    Query PDF documents using hybrid search and optional AI agent.
    
    This endpoint supports two modes:
    1. Standard Mode (use_agent=False):
       - Performs hybrid search (BM25 + Vector + Semantic Reranking)
       - Returns top-K results
       - Generates answer using GPT with retrieved context
    
    2. Agent Mode (use_agent=True):
       - Uses adaptive retrieval with query quality assessment
       - Performs query rewriting if needed
       - Broadens search iteratively until sufficient evidence found
       - Generates reasoning-based answer with confidence scores
       - Streams reasoning trace in real-time via SSE
    
    Args:
        body: Query request containing index_name, query, top_k, use_agent, trace_id
    
    Returns:
        dict: Query results with retrieved contexts, answer, and optional agent trace
    """
    cfg = load_config()
    service, api_key, api_version = cfg["search_service_name"], cfg["search_api_key"], cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    embedding_model = cfg.get("embedding_model", "text-embedding-ada-002")
    headers = {"Content-Type": "application/json", "api-key": api_key}

    # Initialize Azure OpenAI client
    aoai = AzureOpenAI(
        api_key=cfg["openai_api_key"],
        azure_endpoint=cfg["openai_endpoint"],
        api_version=cfg["openai_api_version"]
    )
    
    # Generate query embedding for vector search
    q_vec = embed_text(aoai, embedding_model, [body.query])[0]
    search_url = f"{endpoint}/indexes/{body.index_name}/docs/search?api-version={api_version}"
    
    k = max(1, body.top_k)
    
    # Hybrid search payload combining three retrieval methods:
    # 1. BM25 keyword search
    # 2. Vector semantic search  
    # 3. Azure Semantic Ranker for reranking
    payload = {
        "search": body.query or "",  # BM25 full-text search query
        "queryType": "semantic",  # Enable Azure Semantic Ranker
        "semanticConfiguration": "semantic-config",  # Use semantic config from index schema
        "vectorQueries": [{  # Vector similarity search
            "kind": "vector",
            "vector": q_vec,  # Query embedding vector
            "fields": VECTOR_FIELD,  # Vector field name in index
            "k": k  # Number of nearest neighbors to retrieve
        }],
        "select": "content,source,page,chunk_id,chunk_type",  # Fields to return
        "top": k,  # Total number of results to return
        "count": True  # Include total count of matching documents
    }
    
    # Execute hybrid search
    r = requests.post(search_url, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        http_error(r.status_code, f"Search failed: {r.text}")

    hits = r.json().get("value", [])
    print(f"[INFO] Retrieved {len(hits)} hits")

    # Format retrieved chunks with metadata prefixes
    contexts = []
    for h in hits:
        c = h.get("content", "")
        src = h.get("source")
        page = h.get("page")
        cid = h.get("chunk_id")
        ctype = h.get("chunk_type", "text")
        
        # Add chunk type indicator to prefix (e.g., [IMAGE] for image chunks)
        type_indicator = f"[{ctype.upper()}]" if ctype == "image" else ""
        prefix = f"{type_indicator}[{src} p.{page} #{cid}] " if src and page is not None else ""
        contexts.append(prefix + c)

    agent_trace = ""
    
    # === AGENT MODE ===
    if body.use_agent:
        trace_id = body.trace_id
        
        # Callback function for real-time trace streaming
        def log_trace(line: str):
            if trace_id:
                with _trace_lock:
                    if trace_id not in _agent_traces:
                        _agent_traces[trace_id] = []
                    _agent_traces[trace_id].append(line)
        
        # Search function wrapper for agent to call
        def search_func(query_text: str, top_k_val: int) -> List[Dict]:
            """
            Perform hybrid search with custom query and top-k.
            Used by agent for adaptive retrieval.
            """
            query_vec = embed_text(aoai, embedding_model, [query_text])[0]
            search_payload = {
                "search": query_text or "",
                "queryType": "semantic",
                "semanticConfiguration": "semantic-config",
                "vectorQueries": [{
                    "kind": "vector",
                    "vector": query_vec,
                    "fields": VECTOR_FIELD,
                    "k": top_k_val
                }],
                "select": "content,source,page,chunk_id,chunk_type",
                "top": max(1, top_k_val),
                "count": True
            }
            
            search_response = requests.post(search_url, headers=headers, data=json.dumps(search_payload))
            if search_response.status_code == 200:
                return search_response.json().get("value", [])
            else:
                return []
        
        try:
            # Run PDF agent with adaptive retrieval
            agent_result = run_pdf_agent(
                query=body.query,
                search_func=search_func,
                llm=aoai,
                deployment_name=cfg["openai_deployment"],
                top_k=k,
                min_score=0.85,
                trace_callback=log_trace
            )
            
            answer = agent_result.final_answer
            agent_trace = agent_result.agent_trace
            
            # Use agent's selected evidence (may differ from initial retrieval)
            if agent_result.evidence_used:
                contexts = []
                for ev in agent_result.evidence_used:
                    src = ev.get("source", "")
                    page = ev.get("page", "")
                    cid = ev.get("chunk_id", "")
                    content = ev.get("content", "")
                    prefix = f"[{src} p.{page} #{cid}] " if src else ""
                    contexts.append(prefix + content)
            
        except Exception as e:
            log_trace(f"\n⚠️ Agent error: {str(e)}")
            print(f"[ERROR] Agent failed: {e}")
            answer = f"Error in agent processing: {str(e)}"
            agent_trace = "\n".join(_agent_traces.get(trace_id, [])) if trace_id else ""
    
    # === STANDARD MODE ===
    else:
        # Simple prompt-based answer generation
        prompt = (
            "You are given the following document excerpts:\n"
            + "\n---\n".join(contexts)
            + f"\n\nQuestion: {body.query}\n\n"
            "Instructions:\n"
            "1. Base your answer only on the retrieved excerpts.\n"
            "2. Summarize what is known and what is missing.\n"
            "3. Do not hallucinate information.\n"
        )

        completion = aoai.chat.completions.create(
            model=cfg["openai_deployment"],
            messages=[{"role": "user", "content": prompt}]
        )
        answer = completion.choices[0].message.content

    return {
        "ok": True,
        "retrieved": contexts,  # List of retrieved chunks with metadata
        "answer": answer,  # Generated answer
        "agent_trace": agent_trace if body.use_agent else ""  # Agent reasoning trace (if enabled)
    }

@router.post("/create_index")
def create_pdf_index(body: CreateIndexBody):
    """
    Create or recreate an Azure AI Search index for PDF RAG.
    
    This endpoint creates a new search index with:
    - Document fields for content and metadata
    - Vector search configuration using HNSW algorithm
    - Semantic ranking configuration for Azure Semantic Ranker
    
    Args:
        body: Index configuration (name, dimensions, metric, recreate flag)
    
    Returns:
        dict: Creation status with index name
    """
    cfg = load_config()
    service = cfg["search_service_name"]
    api_key = cfg["search_api_key"]
    api_version = cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    headers = {"Content-Type": "application/json", "api-key": api_key}

    index_name = body.index_name.strip()
    if not index_name:
        raise HTTPException(status_code=400, detail="index_name is required")

    # Check if index already exists
    get_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
    r = requests.get(get_url, headers=headers)
    
    if r.status_code == 200:
        # Index exists
        if not body.recreate:
            # Return existing index without modification
            return {"ok": True, "status": "exists", "index_name": index_name}
        
        # Delete existing index for recreation
        del_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
        d = requests.delete(del_url, headers=headers)
        if d.status_code not in (204, 202):
            raise HTTPException(status_code=500, detail=f"Delete failed: {d.text}")
    
    elif r.status_code != 404:
        # Unexpected error
        raise HTTPException(status_code=500, detail=f"Check failed: {r.text}")

    # Create new index with specified configuration
    payload = build_minimal_index_schema(
        name=index_name,
        dims=int(body.vector_dimensions or DEFAULT_EMBED_DIM),
        metric=(body.metric or DEFAULT_METRIC)
    )
    create_url = f"{endpoint}/indexes?api-version={api_version}"
    c = requests.post(create_url, headers=headers, data=json.dumps(payload))
    
    if c.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Create failed: {c.text}")

    return {"ok": True, "status": "created", "index_name": index_name}
