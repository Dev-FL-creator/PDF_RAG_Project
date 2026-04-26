"""
PDF RAG Project - Main Application
FastAPI backend for PDF document analysis with RAG
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers import pdf_rag_routes
import requests
import nltk

# Initialize NLTK data on application startup for sentence splitting in semantic chunking. This ensures the necessary tokenizers are available without manual setup.
def init_nltk():
    """Initialize NLTK data (auto-download on first run)"""
    required_packages = ['punkt', 'punkt_tab']
    
    for package in required_packages:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            print(f"[INFO] Downloading NLTK {package} data...")
            nltk.download(package, quiet=True)
            print(f"[INFO] NLTK {package} download complete")

# Initialize NLTK
init_nltk()

app = FastAPI(title="PDF RAG API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include PDF RAG routes
app.include_router(pdf_rag_routes.router)

@app.get("/")
def root():
    return {"message": "PDF RAG API is running", "version": "1.0.0"}

@app.get("/api/indexes")
def list_indexes():
    """List all available search indexes"""
    try:
        import json
        import os
        cfg_path = os.getenv("CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        
        service = cfg["search_service_name"]
        api_key = cfg["search_api_key"]
        api_version = cfg["search_api_version"]
        endpoint = f"https://{service}.search.windows.net"
        headers = {"api-key": api_key}
        
        url = f"{endpoint}/indexes?api-version={api_version}"
        r = requests.get(url, headers=headers)
        
        if r.status_code == 200:
            data = r.json()
            index_names = [idx["name"] for idx in data.get("value", [])]
            return {"ok": True, "indexes": index_names}
        else:
            return {"ok": False, "error": f"Failed to list indexes: {r.status_code}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

@app.delete("/api/indexes/{index_name}")
def delete_index(index_name: str):
    """Delete a search index"""
    try:
        import json
        import os
        cfg_path = os.getenv("CONFIG_PATH") or os.path.join(os.path.dirname(__file__), "config.json")
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        
        service = cfg["search_service_name"]
        api_key = cfg["search_api_key"]
        api_version = cfg["search_api_version"]
        endpoint = f"https://{service}.search.windows.net"
        headers = {"api-key": api_key}
        
        url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
        r = requests.delete(url, headers=headers)
        
        if r.status_code in (204, 200, 202):
            # Drop any cached schema probe so the next query re-checks the (now-deleted) index.
            try:
                from utils.query.pdf_query_service import invalidate_seq_support_cache
                invalidate_seq_support_cache(index_name)
            except Exception:
                pass
            return {"ok": True, "message": f"Index '{index_name}' deleted"}
        else:
            return {"ok": False, "error": f"Failed to delete: {r.status_code} {r.text}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
