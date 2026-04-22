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

# Import PDF Agent Engine
from AI_AGENT.pdf_agent_engine import run_pdf_agent

# Import utility modules for semantic chunking and image analysis
from utils.semantic_chunker import create_semantic_chunks
from utils.image_analyzer import PDFImageAnalyzer, format_image_analysis_as_complete_text

# ---- Simple in-memory embedding cache (keyed by model+text) ----
_EMBED_CACHE: Dict[str, List[float]] = {}

# Agent trace storage for SSE streaming
_agent_traces: Dict[str, List[str]] = {}  # {trace_id: [trace_lines]}
_trace_lock = threading.Lock()

def _cache_key(model: str, text: str) -> str:
    h = sha1()
    h.update((model + "||" + text).encode("utf-8", errors="ignore"))
    return h.hexdigest()

router = APIRouter(prefix="/api/pdf_rag", tags=["pdf-rag"])

@router.get("/agent_trace_stream/{trace_id}")
async def pdf_agent_trace_stream(trace_id: str):
    """
    Server-Sent Events (SSE) endpoint for streaming PDF RAG agent trace in real-time.
    Frontend connects with EventSource to receive trace lines as they're generated.
    """
    async def event_generator():
        last_index = 0
        max_wait_time = 120  # 2 minutes timeout
        start_time = time.time()
        
        while True:
            if time.time() - start_time > max_wait_time:
                yield f"data: {json.dumps({'status': 'timeout'})}\n\n"
                break
            
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

# ----------------- Config -----------------
def load_config() -> dict:
    cfg_path = os.getenv("CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "..", "config.json")
    cfg_path = os.path.abspath(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)

DEFAULT_EMBED_DIM = 1536
DEFAULT_METRIC = "cosine"
VECTOR_FIELD = "combined_vector"

# ----------------- Helpers -----------------
def extract_pages_with_docint(
    pdf_path: str, 
    endpoint: str, 
    key: str,
    aoai: Optional[AzureOpenAI] = None,
    enable_image_analysis: bool = True
) -> List[tuple]:
    """
    Extract text, tables, and images from PDF using Azure Document Intelligence.
    
    Returns:
        List of (page_number, page_content) tuples with proper page tracking
    """
    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()

    # Build page-based structure
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
            "order": len(page_contents[page_num])
        })

    # 2) Extract tables with page tracking
    table_id = 0
    for t in (getattr(result, "tables", None) or []):
        table_id += 1
        
        # Get page number
        page_num = 1
        if hasattr(t, "bounding_regions") and t.bounding_regions:
            page_num = t.bounding_regions[0].page_number
        
        # Build table structure
        rows_map = {}
        for cell in t.cells:
            rows_map.setdefault(cell.row_index, {})
            rows_map[cell.row_index][cell.column_index] = (cell.content or "").strip()

        if rows_map:
            try:
                r_cnt = int(getattr(t, "row_count", None) or (max(rows_map.keys()) + 1))
            except Exception:
                r_cnt = max(rows_map.keys()) + 1
            try:
                c_cnt = int(getattr(t, "column_count", None) or (max(max(r.keys()) for r in rows_map.values()) + 1))
            except Exception:
                c_cnt = max(max(r.keys()) for r in rows_map.values()) + 1

            # Create TSV representation
            tsv_lines = []
            for r in range(r_cnt):
                row = rows_map.get(r, {})
                cols = [row.get(c, "") for c in range(c_cnt)]
                tsv_lines.append("\t".join(cols).rstrip())

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
            
            analyzer = PDFImageAnalyzer(
                openai_client=aoai,
                deployment_name="gpt-4o",
                min_image_size=100
            )
            
            # Get document context for better image understanding
            doc_context = ""
            if hasattr(result, "content"):
                doc_context = result.content[:500]  # First 500 chars as context
            
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
        # Fallback: use full document content
        full = (getattr(result, "content", "") or "").strip()
        if full:
            pages_output.append((1, full))
    else:
        # Combine content for each page
        for page_num in sorted(page_contents.keys()):
            items = page_contents[page_num]
            
            # Sort by order (preserve document structure)
            items.sort(key=lambda x: x["order"])
            
            # Combine all content for this page
            page_text = "\n\n".join(item["content"] for item in items)
            
            if page_text.strip():
                pages_output.append((page_num, page_text.strip()))
    
    return pages_output


def safe_doc_id(filename: str, i: int) -> str:
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[^0-9a-zA-Z_\-=]', "_", name)
    name = name.strip('_')
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
    """Robust embedding with caching and retry logic."""
    if not texts:
        return []

    n = len(texts)
    outputs: List[List[float]] = [None] * n  # type: ignore
    uncached_indices: List[int] = []
    uncached_payload: List[str] = []

    for i, t in enumerate(texts):
        k = _cache_key(model, t)
        if k in _EMBED_CACHE:
            outputs[i] = _EMBED_CACHE[k]
        else:
            uncached_indices.append(i)
            uncached_payload.append(t)

    if not uncached_payload:
        return outputs  # type: ignore

    for b in range(0, len(uncached_payload), batch_size):
        sub_payload = uncached_payload[b : b + batch_size]
        sub_indices = uncached_indices[b : b + batch_size]

        attempt = 0
        delay = base_delay
        while True:
            try:
                resp = aoai.embeddings.create(model=model, input=sub_payload)
                vecs = [d.embedding for d in resp.data]
                if len(vecs) != len(sub_payload):
                    raise RuntimeError(f"Embedding count mismatch")
                
                for local_i, vec in enumerate(vecs):
                    src_idx = sub_indices[local_i]
                    k = _cache_key(model, texts[src_idx])
                    _EMBED_CACHE[k] = vec
                    outputs[src_idx] = vec
                break

            except RateLimitError as e:
                attempt += 1
                if attempt > max_retries:
                    raise HTTPException(status_code=429, detail=f"Rate limit: {str(e)}")
                sleep_s = delay * (1.0 + random.uniform(0, jitter))
                print(f"[429] Retry {attempt}/{max_retries} in {sleep_s:.1f}s")
                time.sleep(sleep_s)
                delay = min(delay * 2.0, 60.0)

            except APIError as e:
                attempt += 1
                if getattr(e, "status_code", 500) >= 500 and attempt <= max_retries:
                    sleep_s = delay * (1.0 + random.uniform(0, jitter))
                    print(f"[5xx] Retry {attempt}/{max_retries} in {sleep_s:.1f}s")
                    time.sleep(sleep_s)
                    delay = min(delay * 2.0, 60.0)
                else:
                    raise HTTPException(status_code=502, detail=f"API error: {str(e)}")
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Embedding failed: {repr(e)}")

    return outputs  # type: ignore


def http_error(status: int, message: str):
    print(f"[ERROR] HTTP {status} -> {message}")
    raise HTTPException(status_code=status, detail=message)

def build_minimal_index_schema(name: str, dims: int, metric: str) -> dict:
    return {
        "name": name,
        "fields": [
            {"name": "id", "type": "Edm.String", "key": True, "searchable": False, "filterable": True, "retrievable": True},
            {"name": "content", "type": "Edm.String", "searchable": True, "retrievable": True},
            {"name": "source", "type": "Edm.String", "searchable": False, "retrievable": True},
            {"name": "page", "type": "Edm.Int32", "searchable": False, "retrievable": True},
            {"name": "chunk_id", "type": "Edm.Int32", "searchable": False, "retrievable": True},
            {"name": "chunk_type", "type": "Edm.String", "searchable": False, "retrievable": True, "filterable": True},
            {
                "name": VECTOR_FIELD,
                "type": "Collection(Edm.Single)",
                "searchable": True,
                "retrievable": False,
                "dimensions": dims,
                "vectorSearchProfile": "pdf-vector-profile"
            },
        ],
        "vectorSearch": {
            "algorithms": [{
                "name": "hnsw-config",
                "kind": "hnsw",
                "hnswParameters": {"m": 8, "efConstruction": 400, "efSearch": 500, "metric": metric}
            }],
            "profiles": [{"name": "pdf-vector-profile", "algorithm": "hnsw-config"}]
        },
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

def ensure_index(cfg: dict, index_name: str, dims: int, metric: str):
    service = cfg["search_service_name"]
    api_key = cfg["search_api_key"]
    api_version = cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    headers = {"Content-Type": "application/json", "api-key": api_key}

    get_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
    r = requests.get(get_url, headers=headers)
    if r.status_code == 200:
        print(f"[INFO] Index '{index_name}' already exists")
        return
    if r.status_code != 404:
        http_error(r.status_code, f"Index check failed: {r.text}")

    payload = build_minimal_index_schema(index_name, dims, metric)
    print(f"[INFO] Creating new index: {index_name}")
    c = requests.post(f"{endpoint}/indexes?api-version={api_version}", headers=headers, data=json.dumps(payload))
    if c.status_code not in (200, 201):
        http_error(c.status_code, f"Create index failed: {c.text}")

# ----------------- Models -----------------
class QueryBody(BaseModel):
    index_name: str
    query: str
    top_k: int = 5
    use_agent: bool = False
    trace_id: Optional[str] = None

class CreateIndexBody(BaseModel):
    index_name: str
    vector_dimensions: int = DEFAULT_EMBED_DIM
    recreate: bool = False
    metric: str = DEFAULT_METRIC

# ----------------- Routes -----------------
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), index_name: str = Form(...)):
    cfg = load_config()
    service, api_key, api_version = cfg["search_service_name"], cfg["search_api_key"], cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"

    embedding_model = cfg.get("embedding_model", "text-embedding-ada-002")
    embed_dims = int(cfg.get("embedding_dimensions", DEFAULT_EMBED_DIM))
    metric = cfg.get("vector_metric", DEFAULT_METRIC)

    ensure_index(cfg, index_name, embed_dims, metric)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docint_endpoint = cfg.get("docint_endpoint")
        docint_key = cfg.get("docint_key")

        if not docint_endpoint or not docint_key:
            http_error(500, "Missing docint_endpoint or docint_key in config.json")

        # Initialize Azure OpenAI client (needed for image analysis)
        aoai = AzureOpenAI(
            api_key=cfg["openai_api_key"],
            azure_endpoint=cfg["openai_endpoint"],
            api_version=cfg["openai_api_version"]
        )

        # Extract pages with image analysis
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

        # Use semantic chunking
        contents, meta = [], []
        for page_no, page_text in pages:
            # Get chunking parameters from config
            target_size = cfg.get("chunk_target_size", 800)
            min_size = cfg.get("chunk_min_size", 200)
            max_size = cfg.get("chunk_max_size", 1500)
            
            # Semantic chunking using utility function
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

        # Add image analysis results as standalone chunks (prevents splitting)
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

        vectors = embed_text(aoai, embedding_model, contents)
        dim = len(vectors[0]) if vectors else 0
        print(f"[INFO] Embedding dimension = {dim}")
        if dim != embed_dims:
            http_error(500, f"Embedding dim mismatch: got {dim}, expected {embed_dims}")

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
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@router.post("/query")
def query_pdf(body: QueryBody):
    cfg = load_config()
    service, api_key, api_version = cfg["search_service_name"], cfg["search_api_key"], cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    embedding_model = cfg.get("embedding_model", "text-embedding-ada-002")
    headers = {"Content-Type": "application/json", "api-key": api_key}

    aoai = AzureOpenAI(
        api_key=cfg["openai_api_key"],
        azure_endpoint=cfg["openai_endpoint"],
        api_version=cfg["openai_api_version"]
    )
    
    q_vec = embed_text(aoai, embedding_model, [body.query])[0]
    search_url = f"{endpoint}/indexes/{body.index_name}/docs/search?api-version={api_version}"
    
    k = max(1, body.top_k)
    payload = {
        "search": body.query or "",
        "queryType": "semantic",
        "semanticConfiguration": "semantic-config",
        "vectorQueries": [{
            "kind": "vector",
            "vector": q_vec,
            "fields": VECTOR_FIELD,
            "k": k
        }],
        "select": "content,source,page,chunk_id,chunk_type",
        "top": k,
        "count": True
    }
    
    r = requests.post(search_url, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        http_error(r.status_code, f"Search failed: {r.text}")

    hits = r.json().get("value", [])
    print(f"[INFO] Retrieved {len(hits)} hits")

    contexts = []
    for h in hits:
        c = h.get("content", "")
        src = h.get("source")
        page = h.get("page")
        cid = h.get("chunk_id")
        ctype = h.get("chunk_type", "text")
        
        # Add chunk type indicator to prefix
        type_indicator = f"[{ctype.upper()}]" if ctype == "image" else ""
        prefix = f"{type_indicator}[{src} p.{page} #{cid}] " if src and page is not None else ""
        contexts.append(prefix + c)

    agent_trace = ""
    
    if body.use_agent:
        trace_id = body.trace_id
        
        def log_trace(line: str):
            if trace_id:
                with _trace_lock:
                    if trace_id not in _agent_traces:
                        _agent_traces[trace_id] = []
                    _agent_traces[trace_id].append(line)
        
        def search_func(query_text: str, top_k_val: int) -> List[Dict]:
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
        
    else:
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
        "retrieved": contexts,
        "answer": answer,
        "agent_trace": agent_trace if body.use_agent else ""
    }

@router.post("/create_index")
def create_pdf_index(body: CreateIndexBody):
    """Create or recreate a PDF RAG index"""
    cfg = load_config()
    service = cfg["search_service_name"]
    api_key = cfg["search_api_key"]
    api_version = cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    headers = {"Content-Type": "application/json", "api-key": api_key}

    index_name = body.index_name.strip()
    if not index_name:
        raise HTTPException(status_code=400, detail="index_name is required")

    get_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
    r = requests.get(get_url, headers=headers)
    if r.status_code == 200:
        if not body.recreate:
            return {"ok": True, "status": "exists", "index_name": index_name}
        del_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
        d = requests.delete(del_url, headers=headers)
        if d.status_code not in (204, 202):
            raise HTTPException(status_code=500, detail=f"Delete failed: {d.text}")
    elif r.status_code != 404:
        raise HTTPException(status_code=500, detail=f"Check failed: {r.text}")

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
