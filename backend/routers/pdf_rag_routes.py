from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form
from pydantic import BaseModel

from utils.common.config_loader import DEFAULT_EMBED_DIM, DEFAULT_METRIC
from utils.query.AI_AGENT.agent_trace import get_trace_stream_response
from utils.upload.pdf_upload_service import process_pdf_upload, create_or_recreate_index
from utils.query.pdf_query_service import process_pdf_query


router = APIRouter(prefix="/api/pdf_rag", tags=["pdf-rag"])


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


# Stream agent reasoning steps to the frontend via Server-Sent Events
@router.get("/agent_trace_stream/{trace_id}")
async def pdf_agent_trace_stream(trace_id: str):
    return get_trace_stream_response(trace_id)


# Upload a PDF: extract, chunk, embed, and index it into Azure AI Search
@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), index_name: str = Form(...)):
    return await process_pdf_upload(file, index_name)


# Answer question using hybrid search (+ optional agent)
@router.post("/query")
def query_pdf(body: QueryBody):
    return process_pdf_query(
        index_name=body.index_name,
        query=body.query,
        top_k=body.top_k,
        use_agent=body.use_agent,
        trace_id=body.trace_id
    )


# Create a new Azure AI Search index (or recreate if it already exists)
@router.post("/create_index")
def create_pdf_index(body: CreateIndexBody):
    return create_or_recreate_index(
        index_name=body.index_name,
        vector_dimensions=body.vector_dimensions,
        metric=body.metric,
        recreate=body.recreate
    )