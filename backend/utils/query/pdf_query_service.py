import json
import re
import time
import requests
from typing import List, Dict, Optional, Tuple
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

# When a retrieved hit is a table or image (or a text chunk that explicitly references a figure
# or table by number), fetch this many neighbor chunks on each side (by `seq`) so the LLM can see
# context around the possibly cross-page visual/caption.
NEIGHBOR_WINDOW = 2

# Backwards-compat alias
TABLE_NEIGHBOR_WINDOW = NEIGHBOR_WINDOW

# Detects in-line references like "Figure 5", "Fig. 6", "Table 3", "Diagram 2" so we can also
# expand around text chunks that are talking about a visual element on a nearby page.
FIGURE_REF_PATTERN = re.compile(r'\b(Figure|Fig\.?|Table|Diagram)\s+\d+', re.IGNORECASE)

# Per-index cache: does this index expose the `seq` field? Indexes created before we introduced
# neighbor expansion don't have `seq`, and asking for it in `select` would 400 the entire query.
# We probe periodically and skip neighbor expansion for legacy indexes. The TTL covers the case
# where a user deletes & recreates an index in the same backend run with a different schema.
_INDEX_SEQ_TTL_SECONDS = 300  # 5 min — short enough to catch recreations, long enough to amortize
_index_seq_support: Dict[str, Tuple[bool, float]] = {}  # index_name → (has_seq, probed_at)


def invalidate_seq_support_cache(index_name: Optional[str] = None) -> None:
    """Drop the cached schema probe so the next query re-checks. Call this when an index
    is recreated under the same name. Pass `None` to clear all entries."""
    if index_name is None:
        _index_seq_support.clear()
    else:
        _index_seq_support.pop(index_name, None)


def _index_supports_seq(endpoint: str, index_name: str, api_version: str, headers: dict) -> bool:
    cached = _index_seq_support.get(index_name)
    if cached is not None:
        has_seq, probed_at = cached
        if time.time() - probed_at < _INDEX_SEQ_TTL_SECONDS:
            return has_seq

    url = f"{endpoint}/indexes/{index_name}?api-version={api_version}"
    has_seq = False
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 200:
            fields = r.json().get("fields", [])
            has_seq = any(f.get("name") == "seq" for f in fields)
    except Exception as e:
        print(f"[WARN] Index schema probe failed for {index_name}: {e}")

    _index_seq_support[index_name] = (has_seq, time.time())
    if not has_seq:
        print(f"[INFO] Index '{index_name}' has no `seq` field — neighbor expansion disabled for this index")
    return has_seq


# Run a single hybrid search (BM25 + vector + semantic reranker), optionally filtered to one source file
def _execute_search(search_url: str, headers: dict, query_text: str,
                    q_vec: List[float], k: int, doc_filter: Optional[str] = None,
                    supports_seq: bool = True) -> List[Dict]:
    select_fields = "content,source,page,chunk_id,chunk_type"
    if supports_seq:
        select_fields += ",seq"
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
        "select": select_fields,
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
                                   q_vec: List[float], k: int,
                                   supports_seq: bool = True) -> List[Dict]:
    initial_hits = _execute_search(
        search_url, headers, query_text, q_vec,
        k=max(k * 3, 15),   # expand search range, 3* topk(innitial 5) to detect more documents
        supports_seq=supports_seq
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
            search_url, headers, query_text, q_vec, k=k, doc_filter=doc_name,
            supports_seq=supports_seq
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


# For every "expandable" hit, fetch ±window chunks around it (by seq) from the same source so the
# LLM sees surrounding context. A hit is expandable when:
#   - chunk_type is "table" or "image" (the visual lives in one chunk; context is on adjacent ones), or
#   - chunk_type is "text" AND its content references a figure/table by number (the visual it's
#     talking about may live in a separate chunk on this or an adjacent page).
def _is_expandable(h: Dict) -> bool:
    if h.get("seq") is None:
        return False
    if h.get("chunk_type") in ("table", "image"):
        return True
    # Any non-image, non-table chunk that references a figure/table by number is expandable.
    # The semantic chunker labels text chunks as "paragraph" (not "text"), so we match by
    # exclusion rather than by a hard-coded "text" string here.
    if FIGURE_REF_PATTERN.search(h.get("content") or ""):
        return True
    return False


# Fetch a contiguous run of chunks by `seq` from a single source document. Used by the agent's
# `fetch_section` tool for section traversal — the agent finds a section heading via search, then
# walks forward by seq until it sees the next heading, regardless of how poorly BM25/vector ranks
# the middle chunks of the section.
def _fetch_by_seq(search_url: str, headers: dict, source: str, start_seq: int,
                  direction: str = "next", window: int = 12) -> List[Dict]:
    if start_seq is None:
        return []
    src_escaped = (source or "").replace("'", "''")
    if direction == "prev":
        lo, hi = max(0, start_seq - window), start_seq
    elif direction == "both":
        lo, hi = max(0, start_seq - window), start_seq + window
    else:  # "next" (default)
        lo, hi = start_seq, start_seq + window

    payload = {
        "search": "*",
        "queryType": "simple",
        "filter": f"source eq '{src_escaped}' and seq ge {lo} and seq le {hi}",
        "select": "content,source,page,chunk_id,chunk_type,seq",
        "orderby": "seq asc",
        "top": (hi - lo) + 5,
    }
    r = requests.post(search_url, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        print(f"[WARN] _fetch_by_seq failed (source={source!r}, seq={start_seq}, {direction}): {r.text[:200]}")
        return []
    chunks = r.json().get("value", [])
    chunks.sort(key=lambda x: x.get("seq") if x.get("seq") is not None else 0)
    return chunks


def _expand_neighbors(hits: List[Dict], search_url: str, headers: dict,
                      window: int = NEIGHBOR_WINDOW) -> List[Dict]:
    if not hits:
        return hits

    # Index existing hits by (source, seq) so we don't re-fetch them as neighbors.
    seen = {}
    for h in hits:
        s = h.get("seq")
        if s is not None:
            seen[(h.get("source"), s)] = h

    expandable = [h for h in hits if _is_expandable(h)]
    if not expandable:
        return hits

    # Group neighbor windows by source and merge overlapping ranges → one query per source.
    ranges_by_source: Dict[str, List[tuple]] = {}
    for h in expandable:
        src = h.get("source")
        seq = h.get("seq")
        ranges_by_source.setdefault(src, []).append((max(0, seq - window), seq + window))

    new_chunks: List[Dict] = []
    for src, ranges in ranges_by_source.items():
        ranges.sort()
        merged = [ranges[0]]
        for lo, hi in ranges[1:]:
            if lo <= merged[-1][1] + 1:
                merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
            else:
                merged.append((lo, hi))

        # OData: escape ' as '' inside string literal
        src_escaped = (src or "").replace("'", "''")
        range_clauses = " or ".join(f"(seq ge {lo} and seq le {hi})" for lo, hi in merged)
        filter_expr = f"source eq '{src_escaped}' and ({range_clauses})"

        total_size = sum(hi - lo + 1 for lo, hi in merged)
        payload = {
            "search": "*",
            "queryType": "simple",
            "filter": filter_expr,
            "select": "content,source,page,chunk_id,chunk_type,seq",
            "top": total_size + 5,
        }
        r = requests.post(search_url, headers=headers, data=json.dumps(payload))
        if r.status_code != 200:
            print(f"[WARN] Neighbor fetch failed for source={src!r}: {r.text[:200]}")
            continue

        for chunk in r.json().get("value", []):
            key = (chunk.get("source"), chunk.get("seq"))
            if key in seen or key[1] is None:
                continue
            seen[key] = chunk
            new_chunks.append(chunk)

    if not new_chunks:
        return hits

    merged_hits = list(hits) + new_chunks
    # Sort by document order so the LLM reads chunks in the order they appear in the PDF.
    merged_hits.sort(key=lambda x: (x.get("source") or "", x.get("seq") if x.get("seq") is not None else 0))
    print(f"[INFO] Expanded {len(expandable)} hit(s) with {len(new_chunks)} neighbor chunk(s)")
    return merged_hits


# Build two views of the hits: pretty strings for the UI, structured blocks for the LLM prompt.
# Hits with no `@search.rerankerScore` are neighbor chunks fetched by `_expand_neighbors`.
# We tag those as CONTEXT BLOCKs in the LLM prompt (no relevance score line) so the LLM treats
# them as supporting context rather than low-relevance filler.
def _format_contexts(hits: List[Dict]) -> tuple:
    display_contexts = []
    llm_blocks = []

    for idx, h in enumerate(hits, start=1):
        c = h.get("content", "")
        src = h.get("source", "")
        page = h.get("page", "")
        cid = h.get("chunk_id", "")
        ctype = h.get("chunk_type", "text")
        score = h.get("@search.rerankerScore")  # None for neighbor chunks
        is_context = score is None

        type_indicator = f"[{ctype.upper()}]" if ctype == "image" else ""
        prefix = f"{type_indicator}[{src} p.{page} #{cid}] " if src and page != "" else ""
        display_contexts.append(prefix + c)

        source_type = _source_type_of(ctype)

        if is_context:
            llm_blocks.append(
                f"[EXCERPT {idx} — CONTEXT BLOCK]\n"
                f"SOURCE_DOCUMENT: {src}\n"
                f"PAGE_NUMBER: {page}\n"
                f"SOURCE_TYPE: {source_type}\n"
                f"NOTE: Adjacent chunk included to provide surrounding context for a nearby table, image, or figure-referencing excerpt.\n"
                f"CONTENT:\n{c}"
            )
        else:
            llm_blocks.append(
                f"[EXCERPT {idx}]\n"
                f"SOURCE_DOCUMENT: {src}\n"
                f"PAGE_NUMBER: {page}\n"
                f"SOURCE_TYPE: {source_type}\n"
                f"RELEVANCE_SCORE: {score:.2f}\n"
                f"CONTENT:\n{c}"
            )

    return display_contexts, llm_blocks


# Distinguish chunks that are extracted *from* the PDF (paragraph / table / KV — original text)
# from chunks that are AI-generated *descriptions* of figures (image-analyzer output — not in
# the PDF verbatim). We label both in the LLM prompt and use this to filter what the user sees
# in the Retrieved panel: only original-text quotes are shown, never generated descriptions.
def _source_type_of(chunk_type: str) -> str:
    return "generated_description" if chunk_type == "image" else "original_text"


# Whitespace-normalize for verbatim-quote verification: collapse all whitespace runs to a single
# space and lowercase, so "the quick   brown\nfox" matches "the quick brown fox" exactly when
# substring-tested. Tolerant of PDF newline noise but does NOT allow paraphrase.
def _normalize_ws(s: str) -> str:
    return re.sub(r'\s+', ' ', s or '').strip().lower()


def _verify_quote(quote: str, src: str, page, hits: List[Dict]) -> Optional[str]:
    """If the quote appears verbatim (whitespace-normalized, case-insensitive) inside the
    `content` of some retrieved hit whose source AND page match what the LLM claimed,
    return that hit's `chunk_type` (so the caller can distinguish original vs generated).
    Return None if no match."""
    qn = _normalize_ws(quote)
    if not qn:
        return None
    try:
        page_int = int(page) if page is not None and page != "" else None
    except (TypeError, ValueError):
        page_int = None
    for h in hits:
        if h.get("source") != src:
            continue
        # Page may come in as int or str; compare loosely.
        try:
            hp = int(h.get("page")) if h.get("page") is not None else None
        except (TypeError, ValueError):
            hp = None
        if page_int is not None and hp != page_int:
            continue
        cn = _normalize_ws(h.get("content", ""))
        if qn in cn:
            return h.get("chunk_type") or "text"
    return None


# Shared verification + display formatter used by both standard and agent modes.
# Takes the LLM's claimed `used_passages` and the chunks they should match against;
# returns a list of display-ready strings (with disclaimer prefix for image-derived quotes)
# plus a small log-friendly counts tuple.
def _format_verified_passages(
    passages_raw: List[Dict],
    verification_hits: List[Dict],
    mode_label: str,
) -> List[str]:
    verified: List[str] = []
    kept_original = 0
    kept_generated = 0
    dropped_unverified = 0
    for p in (passages_raw or []):
        if not isinstance(p, dict):
            dropped_unverified += 1
            continue
        src = p.get("source") or ""
        page = p.get("page")
        quote = (p.get("quote") or "").strip()
        if not (src and quote):
            dropped_unverified += 1
            continue
        matched_type = _verify_quote(quote, src, page, verification_hits)
        if matched_type is None:
            dropped_unverified += 1
            print(
                f"[WARN] {mode_label}: used_passage dropped (quote not found verbatim in {src} p.{page}): "
                f"{quote[:80]}{'…' if len(quote) > 80 else ''}"
            )
            continue
        if matched_type == "image":
            verified.append(
                f"[AI-generated description of image on {src} p.{page} — for the actual diagram, view {src} p.{page} directly]\n"
                f"{quote}"
            )
            kept_generated += 1
        else:
            verified.append(f"[{src} p.{page}] {quote}")
            kept_original += 1

    if kept_original or kept_generated or dropped_unverified:
        print(
            f"[INFO] {mode_label}: kept {len(verified)} passage(s) "
            f"({kept_original} original_text + {kept_generated} generated_description), "
            f"dropped {dropped_unverified} unverified"
        )
    return verified


# Build the standard-mode prompt — JSON-object mode. The model returns:
#   { "answer": "<prose with inline citations>",
#     "used_passages": [ {"source": "...", "page": <int>, "quote": "<verbatim text>"}, ... ] }
# We then substring-verify each quote against the original chunks; unverified ones are dropped
# (the LLM cannot smuggle in paraphrased "quotes" past the verifier).
def _build_standard_prompt(query: str, llm_blocks: List[str]) -> str:
    excerpts_text = "\n\n---\n\n".join(llm_blocks)
    return (
        "You are a document question-answering assistant. Answer the user's input "
        "strictly based on the provided excerpts from one or more PDF documents.\n\n"
        "If the user's input is a figure caption, section title, topic name, or other "
        "phrase without a complete question, treat it as 'describe / summarize / explain "
        "that topic' and answer using the relevant excerpts. Do not refuse just because "
        "the input isn't phrased as a complete question.\n\n"
        "Excerpts beginning with '[IMAGE on page N - TYPE]' are descriptions of a figure "
        "or diagram on that page. When the user asks about a figure (by number, caption, "
        "or topic), use these excerpts to describe what the figure shows — its components, "
        "structure, labels, and relationships. Cite them like any other excerpt.\n\n"
        "Each excerpt has a SOURCE_TYPE field with one of two values:\n"
        "- \"original_text\": text extracted directly from the PDF (paragraphs, tables, "
        "key-value pairs). Verbatim PDF content.\n"
        "- \"generated_description\": an AI-generated description of a figure or diagram "
        "(produced by an image-analysis model). The description is ABOUT the PDF but is "
        "NOT itself in the PDF.\n"
        "Both types are equally valid for composing your `answer` AND for inclusion in "
        "`used_passages`. The downstream UI will label generated_description quotes with a "
        "clear disclaimer (\"AI-generated description — view the actual image on page X to "
        "verify\"), so the user can tell them apart. Use whichever excerpts genuinely support "
        "your claims; never refuse to answer just because the only relevant excerpts are "
        "generated_description.\n\n"
        "Some excerpts may be marked as \"CONTEXT BLOCK\" — these are adjacent chunks "
        "included alongside a table excerpt, an image excerpt, or a text excerpt that "
        "references a figure or table by number. Use them to interpret column headers, "
        "captions, continuation rows of split tables, and the visual content described "
        "in nearby figure-analyzer chunks. Cite a context block only when its content "
        "directly supports a claim in your answer.\n\n"
        "=== RETRIEVED EXCERPTS ===\n"
        f"{excerpts_text}\n\n"
        "=== USER QUESTION ===\n"
        f"{query}\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Return a single JSON object with EXACTLY these two fields and nothing else:\n\n"
        "{\n"
        "  \"answer\": \"<your prose answer with inline (Source: <file>, page <N>) citations>\",\n"
        "  \"used_passages\": [\n"
        "    {\"source\": \"<filename>\", \"page\": <integer>, \"quote\": \"<exact verbatim text from that excerpt>\"}\n"
        "  ]\n"
        "}\n\n"
        "=== RULES FOR `answer` ===\n"
        "1. GROUNDING: Use ONLY the information in the excerpts above. Do not use outside knowledge.\n"
        "2. INSUFFICIENT EVIDENCE: Set `answer` to exactly \"I cannot find sufficient evidence "
        "in the provided documents to answer this question.\" ONLY if NONE of the excerpts "
        "contain content relevant to the user's input. If at least one excerpt is on-topic — "
        "INCLUDING any `generated_description` excerpt about the figure being asked about — "
        "give the best answer you can. The absence of `original_text` quotes is NEVER a reason "
        "to refuse: a substantive answer with `used_passages: []` is preferable to refusal.\n"
        "3. CITATION: Every factual claim in `answer` MUST be followed by an inline citation in this exact format: (Source: <SOURCE_DOCUMENT>, page <PAGE_NUMBER>).\n"
        "4. MULTI-DOCUMENT: If the answer involves multiple documents, clearly attribute each point to its source document.\n\n"
        "=== RULES FOR `used_passages` ===\n"
        "5. EXACT VERBATIM: Each `quote` MUST be copy-pasted character-for-character from one "
        "of the excerpts above. Do NOT paraphrase, summarize, shorten, fix typos, or reformat. "
        "Do NOT use ellipsis (`...` or `…`) inside a quote.\n"
        "6. SOURCE & PAGE MATCH: For every passage, `source` and `page` MUST exactly match the "
        "SOURCE_DOCUMENT and PAGE_NUMBER of the excerpt the quote was taken from. `page` must be an integer.\n"
        "7. EITHER SOURCE_TYPE OK: Quotes may come from `original_text` OR `generated_description` "
        "excerpts. The display layer labels each kind appropriately for the user.\n"
        "8. ONLY USED: Include only quotes that DIRECTLY support a specific claim you made in `answer`. "
        "Do NOT include quotes from excerpts you didn't actually rely on. Quality over quantity — "
        "1 to 5 well-chosen quotes is normal; 10+ usually means you're dumping irrelevant excerpts.\n"
        "9. FULL UNIT: A quote should be a complete sentence (or a complete table row, or a complete "
        "bullet point). If you can't include a complete unit verbatim, omit the passage.\n"
        "10. EMPTY OK: Return `used_passages: []` only when no excerpt actually supports your claims "
        "verbatim — for example, when your `answer` is the insufficient-evidence message. Otherwise, "
        "include at least one quote.\n"
    )


# Agent mode: delegate to the PDF agent for adaptive retrieval, query rewriting, and SSE trace streaming
def _run_agent_mode(query: str, k: int, trace_id: Optional[str], cfg: dict,
                    aoai: AzureOpenAI, embedding_model: str,
                    search_url: str, headers: dict,
                    contexts: List[str], supports_seq: bool = True) -> tuple:
    # Push each reasoning step to the shared trace store so the SSE endpoint can stream it live
    def log_trace(line: str):
        if trace_id:
            with _trace_lock:
                if trace_id not in _agent_traces:
                    _agent_traces[trace_id] = []
                _agent_traces[trace_id].append(line)

    # Callback the agent uses to re-query the index on its own during iterative search.
    # When the index supports `seq`, we also expand table neighbors here so the agent's
    # evidence (and final answer) include context around cross-page table fragments.
    #
    # `exclude_keys`, when provided, is a set of (source, chunk_id) tuples of already-seen
    # chunks. We oversample 4x from Azure and filter client-side, so the returned `top_k_val`
    # chunks are guaranteed to be NEW. Without this, repeated searches with rephrased queries
    # would keep returning the same BM25/vector winners, wasting the top-k budget on duplicates
    # the agent already has — and masking the "corpus exhausted" signal that should trigger a
    # stop-early instead of yet another rewrite.
    def search_func(query_text: str, top_k_val: int,
                    exclude_keys: Optional[set] = None) -> List[Dict]:
        query_vec = embed_text(aoai, embedding_model, [query_text])[0]
        select_fields = "content,source,page,chunk_id,chunk_type"
        if supports_seq:
            select_fields += ",seq"

        fetch_k = top_k_val * 4 if exclude_keys else top_k_val
        search_payload = {
            "search": query_text or "",
            "queryType": "semantic",
            "semanticConfiguration": "semantic-config",
            "vectorQueries": [{
                "kind": "vector",
                "vector": query_vec,
                "fields": VECTOR_FIELD,
                "k": fetch_k
            }],
            "select": select_fields,
            "top": max(1, fetch_k),
            "count": True
        }

        search_response = requests.post(search_url, headers=headers, data=json.dumps(search_payload))
        if search_response.status_code != 200:
            return []
        raw_hits = search_response.json().get("value", [])

        if exclude_keys:
            raw_hits = [
                h for h in raw_hits
                if (h.get("source"), h.get("chunk_id")) not in exclude_keys
            ]
        raw_hits = raw_hits[:top_k_val]

        if supports_seq:
            return _expand_neighbors(raw_hits, search_url, headers)
        return raw_hits

    # Section-traversal callback: only meaningful on indexes that have `seq`. The agent uses this
    # to walk a section forward/backward by seq once it has located the heading chunk via
    # search_func. None on legacy indexes — the agent's fetch_section tool is then disabled.
    fetch_by_seq_func = None
    if supports_seq:
        def fetch_by_seq_func(source: str, start_seq: int, direction: str = "next",
                              window: int = 12) -> List[Dict]:
            return _fetch_by_seq(search_url, headers, source, start_seq, direction, window)

    try:
        agent_result = run_pdf_agent(
            query=query,
            search_func=search_func,
            llm=aoai,
            deployment_name=cfg["openai_deployment"],
            top_k=k,
            min_score=0.85,
            trace_callback=log_trace,
            fetch_by_seq_func=fetch_by_seq_func,
        )

        answer = agent_result.final_answer
        agent_trace = agent_result.agent_trace

        # Replace `contexts` with the verified verbatim quotes the agent's answer-generation
        # step claimed to use. Same display rules as standard mode: original-text quotes show
        # plainly; image-chunk quotes show with the AI-generated disclaimer; unverified
        # (hallucinated) quotes are dropped. Verification corpus is the agent's full session
        # (state.all_hits, surfaced as `verification_pool`) so quotes from any chunk the agent
        # ever saw can match — not just the post-filter `evidence_used` subset.
        verified = _format_verified_passages(
            getattr(agent_result, "used_passages", []) or [],
            getattr(agent_result, "verification_pool", []) or [],
            mode_label="agent-mode",
        )
        if verified:
            contexts = verified
        else:
            contexts = ["(No exact-quote evidence — the model either could not find sufficient grounding, or produced quotes that did not match the retrieved chunks verbatim.)"]

    except Exception as e:
        log_trace(f"\n⚠️ Agent error: {str(e)}")
        print(f"[ERROR] Agent failed: {e}")
        answer = f"Error in agent processing: {str(e)}"
        agent_trace = "\n".join(_agent_traces.get(trace_id, [])) if trace_id else ""

    return answer, agent_trace, contexts


# Standard mode: one-shot RAG with JSON-mode response. Returns (answer_text, verified_quotes).
# `verified_quotes` is a list of pre-formatted strings ready for the UI's "Retrieved" panel,
# OR None if JSON parsing failed (caller should fall back to the raw retrieved chunks).
def _run_standard_mode(query: str, filtered_hits: List[Dict], all_hits_for_verify: List[Dict],
                       llm_blocks: List[str], cfg: dict,
                       aoai: AzureOpenAI) -> Tuple[str, Optional[List[str]]]:
    # If no hits passed the relevance threshold, skip the LLM call entirely.
    if not filtered_hits:
        msg = (
            "I cannot find sufficient evidence in the provided documents to answer this question. "
            f"No retrieved passages met the minimum relevance threshold "
            f"(reranker score >= {MIN_RERANKER_SCORE})."
        )
        return msg, []

    prompt = _build_standard_prompt(query, llm_blocks)
    completion = aoai.chat.completions.create(
        model=cfg["openai_deployment"],
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    raw = completion.choices[0].message.content or ""

    # Parse the JSON. If parsing fails, return raw text + None so the caller falls back.
    try:
        data = json.loads(raw)
    except Exception as e:
        print(f"[WARN] standard-mode JSON parse failed: {e}; falling back to raw text")
        return raw, None

    answer = (data.get("answer") or "").strip()
    passages_raw = data.get("used_passages") or []
    if not isinstance(passages_raw, list):
        passages_raw = []

    verified = _format_verified_passages(passages_raw, all_hits_for_verify, mode_label="standard-mode")
    return answer, verified


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

    # Probe the index schema once: legacy indexes built before neighbor expansion don't have
    # `seq`, so we skip both the `select seq` and the expansion step for them.
    supports_seq = _index_supports_seq(endpoint, index_name, api_version, headers)

    # Step 1: retrieve with per-document balancing so one PDF can't dominate the results
    hits = _retrieve_with_per_doc_balance(search_url, headers, query, q_vec, k, supports_seq=supports_seq)
    print(f"[INFO] Retrieved {len(hits)} raw hits (pre-filter)")

    # Step 2: drop anything below the reranker score threshold to cut noise before the LLM sees it
    filtered_hits = _filter_by_score(hits, MIN_RERANKER_SCORE)

    # Step 3: for any table hit, fetch its ±N neighbor chunks (by seq) from the same source so the
    # LLM sees surrounding context — this is how cross-page tables get stitched at retrieval time.
    if supports_seq:
        expanded_hits = _expand_neighbors(filtered_hits, search_url, headers)
    else:
        expanded_hits = filtered_hits

    # Step 4: build display + LLM-facing views of the expanded set
    contexts, llm_blocks = _format_contexts(expanded_hits)

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
            contexts=contexts,
            supports_seq=supports_seq
        )
    else:
        # Standard mode: ask the LLM for JSON with answer + the exact verbatim passages it used,
        # verify each quote against the original chunks, and replace the user-visible retrieved
        # context with just those verified quotes — clean, no noise from unused chunks.
        answer, verified_quotes = _run_standard_mode(
            query, filtered_hits, expanded_hits, llm_blocks, cfg, aoai
        )
        if verified_quotes is not None:
            if verified_quotes:
                contexts = verified_quotes
            else:
                contexts = ["(No exact-quote evidence — the model either could not find sufficient grounding, or produced quotes that did not match the retrieved chunks verbatim.)"]
        # else: JSON parsing failed; keep the original full-chunk contexts as a fallback

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