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
import time
import threading
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import requests
from openai import AzureOpenAI

# Import PDF Agent Engine for adaptive retrieval
from AI_AGENT.pdf_agent_engine import run_pdf_agent

# Import utility modules
from utils.semantic_chunker import create_semantic_chunks
from utils.image_analyzer import format_image_analysis_as_complete_text
from utils.embeddings import embed_text, safe_doc_id
from utils.search_index import VECTOR_FIELD, build_index_schema, ensure_index, http_error
from utils.pdf_extractor import extract_pages_with_docint

# ============================================================
# GLOBAL VARIABLES
# ============================================================

# Storage for agent reasoning traces (for real-time SSE streaming)
# Key: trace_id, Value: list of trace log lines
_agent_traces: Dict[str, List[str]] = {}
_trace_lock = threading.Lock()  # Thread-safe access to traces

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
    payload = build_index_schema(
        name=index_name,
        dims=int(body.vector_dimensions or DEFAULT_EMBED_DIM),
        metric=(body.metric or DEFAULT_METRIC)
    )
    create_url = f"{endpoint}/indexes?api-version={api_version}"
    c = requests.post(create_url, headers=headers, data=json.dumps(payload))
    
    if c.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Create failed: {c.text}")

    return {"ok": True, "status": "created", "index_name": index_name}