import os
import json
import tempfile
import requests
from fastapi import UploadFile, HTTPException
from openai import AzureOpenAI

from utils.common.config_loader import load_config, DEFAULT_EMBED_DIM, DEFAULT_METRIC
from utils.upload.semantic_chunker import create_semantic_chunks
from utils.upload.image_analyzer import format_image_analysis_as_complete_text
from utils.common.embeddings import embed_text, safe_doc_id
from utils.common.ai_search_index import VECTOR_FIELD, build_index_schema, ensure_index, http_error
from utils.upload.pdf_extractor import extract_pages_with_docint


# Main upload pipeline: parse PDF → chunk → embed → push to Azure AI Search
async def process_pdf_upload(file: UploadFile, index_name: str) -> dict:
    cfg = load_config()
    service, api_key, api_version = cfg["search_service_name"], cfg["search_api_key"], cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"

    embedding_model = cfg.get("embedding_model", "text-embedding-ada-002")
    embed_dims = int(cfg.get("embedding_dimensions", DEFAULT_EMBED_DIM))
    metric = cfg.get("vector_metric", DEFAULT_METRIC)

    # Create the index if it doesn't exist yet (safe no-op if it already does)
    ensure_index(cfg, index_name, embed_dims, metric)

    # Save the uploaded PDF to a temp file so downstream tools can read it from disk
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        docint_endpoint = cfg.get("docint_endpoint")
        docint_key = cfg.get("docint_key")

        if not docint_endpoint or not docint_key:
            http_error(500, "Missing docint_endpoint or docint_key in config.json")

        aoai = AzureOpenAI(
            api_key=cfg["openai_api_key"],
            azure_endpoint=cfg["openai_endpoint"],
            api_version=cfg["openai_api_version"]
        )

        # Step 1: extract text, tables, and (optionally) image descriptions from the PDF
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

        # Step 2: split each page into semantic chunks (respecting sentence boundaries)
        contents, meta = [], []
        for page_no, page_text in pages:
            target_size = cfg.get("chunk_target_size", 800)
            min_size = cfg.get("chunk_min_size", 200)
            max_size = cfg.get("chunk_max_size", 1500)

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

        # Step 3: append image analysis results as standalone chunks (kept whole to preserve descriptions)
        if hasattr(extract_pages_with_docint, '_image_results'):
            image_results = extract_pages_with_docint._image_results.get(tmp_path, [])

            for img_idx, img_result in enumerate(image_results):
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

        # Step 4: generate an embedding vector for every chunk
        vectors = embed_text(aoai, embedding_model, contents)
        dim = len(vectors[0]) if vectors else 0
        print(f"[INFO] Embedding dimension = {dim}")

        # Safety check: embedding dim must match what the index schema was created with
        if dim != embed_dims:
            http_error(500, f"Embedding dim mismatch: got {dim}, expected {embed_dims}")

        # Step 5: upload chunks + vectors to the Azure AI Search index in batches of 256
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
        # Always clean up the temp file, even if upload failed partway through
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# Create a new Azure AI Search index; if it already exists, either keep it or delete+recreate
def create_or_recreate_index(index_name: str, vector_dimensions: int, metric: str, recreate: bool) -> dict:
    cfg = load_config()
    service = cfg["search_service_name"]
    api_key = cfg["search_api_key"]
    api_version = cfg["search_api_version"]
    endpoint = f"https://{service}.search.windows.net"
    headers = {"Content-Type": "application/json", "api-key": api_key}

    index_name = index_name.strip()
    if not index_name:
        raise HTTPException(status_code=400, detail="index_name is required")

    # Check whether the index already exists (200 = exists, 404 = missing, anything else = error)
    get_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
    r = requests.get(get_url, headers=headers)

    if r.status_code == 200:
        # Already exists: either return it as-is, or delete it so we can rebuild fresh
        if not recreate:
            return {"ok": True, "status": "exists", "index_name": index_name}

        del_url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
        d = requests.delete(del_url, headers=headers)
        if d.status_code not in (204, 202):
            raise HTTPException(status_code=500, detail=f"Delete failed: {d.text}")

    elif r.status_code != 404:
        raise HTTPException(status_code=500, detail=f"Check failed: {r.text}")

    # Build the index schema (fields + HNSW vector config + semantic ranker config) and POST it
    payload = build_index_schema(
        name=index_name,
        dims=int(vector_dimensions or DEFAULT_EMBED_DIM),
        metric=(metric or DEFAULT_METRIC)
    )
    create_url = f"{endpoint}/indexes?api-version={api_version}"
    c = requests.post(create_url, headers=headers, data=json.dumps(payload))

    if c.status_code not in (200, 201):
        raise HTTPException(status_code=500, detail=f"Create failed: {c.text}")

    return {"ok": True, "status": "created", "index_name": index_name}