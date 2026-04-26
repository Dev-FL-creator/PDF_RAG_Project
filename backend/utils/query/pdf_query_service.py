import json
import requests
from typing import List, Dict, Optional
from openai import AzureOpenAI

from utils.common.config_loader import load_config
from utils.common.embeddings import embed_text
from utils.common.ai_search_index import VECTOR_FIELD, http_error
from utils.query.AI_AGENT.agent_trace import _agent_traces, _trace_lock
from utils.query.AI_AGENT.pdf_agent_engine import run_pdf_agent


# Minimum Azure Semantic Ranker score to keep a hit (range 0-4, 1.5 = "relevant")
MIN_RERANKER_SCORE = 1.5

# When many PDFs match the query, only fetch top-k from the first 10 to avoid too many API calls.
MAX_DOCS_FOR_PER_DOC_FETCH = 10


# Run a single hybrid search (BM25 + vector + semantic reranker), optionally filtered to one source file
def _execute_search(search_url: str, headers: dict, query_text: str,
                    q_vec: List[float], k: int, doc_filter: Optional[str] = None) -> List[Dict]:
    payload = {
        "search": query_text or "",  # BM25 keyword search (empty string means "match all")
        "queryType": "semantic",     # Use Semantic Ranker
        "semanticConfiguration": "semantic-config",
        "vectorQueries": [{          # Semantic similarity search using the query embedding
            "kind": "vector",
            "vector": q_vec,
            "fields": VECTOR_FIELD,
            "k": k
        }],
        "select": "content,source,page,chunk_id,chunk_type",
        "top": k,
        "count": True
    }
    if doc_filter:
        payload["filter"] = f"source eq '{doc_filter}'"

    r = requests.post(search_url, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        http_error(r.status_code, f"Search failed: {r.text}")
    return r.json().get("value", [])


# Ensure multi-document coverage: if results span multiple PDFs, fetch top-k from EACH and merge
def _retrieve_with_per_doc_balance(search_url: str, headers: dict, query_text: str,
                                   q_vec: List[float], k: int) -> List[Dict]:
    initial_hits = _execute_search(
        search_url, headers, query_text, q_vec,
        k=max(k * 3, 15)   # expand search range, 3* topk(innitial 5) to detect more documents
    )

    if not initial_hits:
        return []

    docs_seen = []
    seen_set = set()
    for h in initial_hits:
        src = h.get("source")
        if src and src not in seen_set:
            seen_set.add(src)
            docs_seen.append(src)

    if len(docs_seen) <= 1:
        print(f"[INFO] Only 1 document in results, using standard top-{k}")
        return initial_hits[:k] # only return top-k if there's just one document, otherwise we will fetch top-k per doc in the next step

    print(f"[INFO] {len(docs_seen)} documents detected, fetching top-{k} per document")
    all_hits = []
    for doc_name in docs_seen[:MAX_DOCS_FOR_PER_DOC_FETCH]:
        doc_hits = _execute_search(
            search_url, headers, query_text, q_vec, k=k, doc_filter=doc_name
        )
        all_hits.extend(doc_hits)

    all_hits.sort(
        key=lambda x: x.get("@search.rerankerScore", 0) or 0,
        reverse=True
    )

    return all_hits


# Drop low-quality hits whose reranker score falls below the threshold
def _filter_by_score(hits: List[Dict], min_score: float) -> List[Dict]:
    filtered = []
    for h in hits:
        score = h.get("@search.rerankerScore", 0) or 0
        if score >= min_score:
            filtered.append(h)

    dropped = len(hits) - len(filtered)
    if dropped > 0:
        print(f"[INFO] Dropped {dropped}/{len(hits)} hits below reranker score {min_score}")
    return filtered


# Build two views of the hits: pretty strings for the UI, structured blocks for the LLM prompt
def _format_contexts(hits: List[Dict]) -> tuple:
    display_contexts = []
    llm_blocks = []

    for idx, h in enumerate(hits, start=1):
        c = h.get("content", "")
        src = h.get("source", "")
        page = h.get("page", "")
        cid = h.get("chunk_id", "")
        ctype = h.get("chunk_type", "text")
        score = h.get("@search.rerankerScore", 0) or 0

        type_indicator = f"[{ctype.upper()}]" if ctype == "image" else ""
        prefix = f"{type_indicator}[{src} p.{page} #{cid}] " if src and page != "" else ""
        display_contexts.append(prefix + c)

        llm_blocks.append(
            f"[EXCERPT {idx}]\n"
            f"SOURCE_DOCUMENT: {src}\n"
            f"PAGE_NUMBER: {page}\n"
            f"RELEVANCE_SCORE: {score:.2f}\n"
            f"CONTENT:\n{c}"
        )

    return display_contexts, llm_blocks


# Build the standard-mode prompt: grounding rules, required citation format, quoted-passage section
def _build_standard_prompt(query: str, llm_blocks: List[str]) -> str:
    excerpts_text = "\n\n---\n\n".join(llm_blocks)
    return (
        "You are a document question-answering assistant. Answer the user's question "
        "strictly based on the provided excerpts from one or more PDF documents.\n\n"
        "=== RETRIEVED EXCERPTS ===\n"
        f"{excerpts_text}\n\n"
        "=== USER QUESTION ===\n"
        f"{query}\n\n"
        "=== RESPONSE REQUIREMENTS ===\n"
        "1. GROUNDING: Use ONLY the information in the excerpts above. Do not use outside knowledge.\n"
        "2. INSUFFICIENT EVIDENCE: If the excerpts do not contain enough information to answer "
        "the question, respond EXACTLY with:\n"
        "   \"I cannot find sufficient evidence in the provided documents to answer this question.\"\n"
        "   Do not speculate or fabricate.\n"
        "3. CITATION: Every factual claim MUST be followed by a citation in this exact format:\n"
        "   (Source: <SOURCE_DOCUMENT>, page <PAGE_NUMBER>)\n"
        "   Example: \"Passwords must be rotated every 90 days (Source: Security_Policy.pdf, page 6).\"\n"
        "4. QUOTED SUPPORT: After your answer, add a \"Supporting passages:\" section that quotes "
        "the exact sentence(s) from the excerpts that support your answer, each followed by its "
        "source and page. Keep each quote under 25 words.\n"
        "5. MULTI-DOCUMENT: If the answer involves multiple documents (e.g. a comparison), "
        "clearly attribute each point to its source document.\n"
        "6. FORMAT: Write the answer as a natural-language paragraph, then the supporting passages list.\n"
    )


# Agent mode: delegate to the PDF agent for adaptive retrieval, query rewriting, and SSE trace streaming
def _run_agent_mode(query: str, k: int, trace_id: Optional[str], cfg: dict,
                    aoai: AzureOpenAI, embedding_model: str,
                    search_url: str, headers: dict,
                    contexts: List[str]) -> tuple:
    # Push each reasoning step to the shared trace store so the SSE endpoint can stream it live
    def log_trace(line: str):
        if trace_id:
            with _trace_lock:
                if trace_id not in _agent_traces:
                    _agent_traces[trace_id] = []
                _agent_traces[trace_id].append(line)

    # Callback the agent uses to re-query the index on its own during iterative search
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
            query=query,
            search_func=search_func,
            llm=aoai,
            deployment_name=cfg["openai_deployment"],
            top_k=k,
            min_score=0.85,
            trace_callback=log_trace
        )

        answer = agent_result.final_answer
        agent_trace = agent_result.agent_trace

        # Override contexts with the evidence the agent actually used (may differ from initial retrieval)
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

    return answer, agent_trace, contexts


# Standard mode: one-shot RAG, hard-gate on evidence quality, then call the LLM
def _run_standard_mode(query: str, filtered_hits: List[Dict], llm_blocks: List[str],
                       cfg: dict, aoai: AzureOpenAI) -> str:
    # If no hits passed the relevance threshold, return a default message instead of calling the LLM
    if not filtered_hits:
        return (
            "I cannot find sufficient evidence in the provided documents to answer this question. "
            f"No retrieved passages met the minimum relevance threshold "
            f"(reranker score >= {MIN_RERANKER_SCORE})."
        )

    prompt = _build_standard_prompt(query, llm_blocks)
    completion = aoai.chat.completions.create(
        model=cfg["openai_deployment"],
        messages=[{"role": "user", "content": prompt}]
    )
    return completion.choices[0].message.content


# Main entry point: orchestrates retrieval, filtering, answer generation, and citation packaging
def process_pdf_query(index_name: str, query: str, top_k: int,
                      use_agent: bool, trace_id: Optional[str]) -> dict:
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

    # Embed the user's question once up-front (reused across all retrieval calls below)
    q_vec = embed_text(aoai, embedding_model, [query])[0]
    search_url = f"{endpoint}/indexes/{index_name}/docs/search?api-version={api_version}"

    k = max(1, top_k)

    # Step 1: retrieve with per-document balancing so one PDF can't dominate the results
    hits = _retrieve_with_per_doc_balance(search_url, headers, query, q_vec, k)
    print(f"[INFO] Retrieved {len(hits)} raw hits (pre-filter)")

    # Step 2: drop anything below the reranker score threshold to cut noise before the LLM sees it
    filtered_hits = _filter_by_score(hits, MIN_RERANKER_SCORE)

    # Step 3: build display + LLM-facing views of the surviving hits
    contexts, llm_blocks = _format_contexts(filtered_hits)

    agent_trace = ""

    # Step 4: answer the question in the chosen mode
    if use_agent:
        answer, agent_trace, contexts = _run_agent_mode(
            query=query,
            k=k,
            trace_id=trace_id,
            cfg=cfg,
            aoai=aoai,
            embedding_model=embedding_model,
            search_url=search_url,
            headers=headers,
            contexts=contexts
        )
    else:
        answer = _run_standard_mode(query, filtered_hits, llm_blocks, cfg, aoai)

    # Step 5: build structured citations list for the frontend (source + page + score + content)
    citations = []
    for h in filtered_hits:
        citations.append({
            "source": h.get("source", ""),
            "page": h.get("page"),
            "chunk_id": h.get("chunk_id"),
            "chunk_type": h.get("chunk_type", "text"),
            "score": round(h.get("@search.rerankerScore", 0) or 0, 3),
            "content": h.get("content", "")
        })

    return {
        "ok": True,  #ok indicates successful processing, even if no answer could be found
        "retrieved": contexts,
        "citations": citations,
        "answer": answer,
        "num_sources": len(filtered_hits),
        "min_score_threshold": MIN_RERANKER_SCORE,
        "agent_trace": agent_trace if (use_agent and not trace_id) else ""
    }