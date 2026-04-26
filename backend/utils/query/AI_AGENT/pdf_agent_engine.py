"""
Available tools:
  - search_documents
  - assess_query
  - rewrite_query
  - evaluate_evidence
"""

import json
import re
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from openai import AzureOpenAI

from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import AzureChatOpenAI


# ============================================================
# Configuration & Constants
# ============================================================

MIN_HITS = 2
MIN_SCORE_FLOOR = 0.70
LOW_SCORE_WARNING_THRESHOLD = 1.5
MAX_AGENT_ITERATIONS = 10
EXPLORATORY_TOP_K = 3

# After this many failed rewrite attempts the evaluator stops insisting on "MUST rewrite again"
# and instead lets the agent answer with whatever evidence is available. Without this escape
# hatch a query like "what is 7.3 about?" can loop until max_iterations and refuse outright,
# even when the right chunks were already retrieved earlier.
MAX_REWRITES_BEFORE_ANSWER = 2

# Section-walk window: when fetch_section walks a section in document order by `seq`, this caps
# how far forward it goes from the heading chunk. ~12 chunks ≈ 3-4 PDF pages of body text, which
# covers any normal section without risking pulling in unrelated content.
SECTION_WALK_WINDOW = 12

# Detects section numbers ("7.3", "4.2.1") and figure refs ("Figure 6", "Table 3") in the user
# query. The pre-flight step uses these to seed `state.all_hits` with the full section before the
# ReAct loop runs — guaranteeing coverage even if the agent wanders.
SECTION_NUMBER_PATTERN = re.compile(r'\b(\d+(?:\.\d+){1,3})\b')
# Match "<num> <Capitalized title>" on a single line. Critical: use `[ \t]+` (NOT `\s+`) for the
# gap between number and title — `\s+` would match across newlines and accept Table-of-Contents
# layouts where the number sits on its own line above the title:
#     7.3
#     Recommendation
#     on data quality   ← NOT a real section start, just a TOC entry
HEADING_AT_LINESTART_PATTERN = re.compile(
    r'^[ \t]*(\d+(?:\.\d+)*)[ \t]+[-–—]?[ \t]*[A-Z]', re.MULTILINE
)
QUOTED_PHRASE_PATTERN = re.compile(r'"([^"]+)"')
FIGURE_REF_IN_QUERY_PATTERN = re.compile(r'\b(?:Figure|Fig\.?|Table|Diagram)\s+\d+', re.IGNORECASE)

STEP_DIVIDER = "─" * 50

VAGUE_TERMS = [
    "something", "anything", "things", "stuff", "any", "some",
    "various", "several", "many", "few", "multiple", "general",
    "overall", "basic", "detailed", "information", "data", "details"
]

HARD_FAIL_ISSUES = {"too_short", "empty", "single_word"}

DOMAIN_CONTEXT = """
You are analyzing technical PDF documents provided by the user. The rewrite
should make the question specific enough to retrieve relevant passages from
a document index, WITHOUT assuming a particular topic area. Let the actual
document content (if provided below) guide the rewrite — do not inject
generic "technical compliance / system architecture" language unless it
actually appears in the retrieval context.
"""


# ============================================================
# Pydantic Data Structures
# ============================================================

class QueryQualityResult(BaseModel):
    is_clear: bool = Field(...)
    issues: List[str] = Field(default_factory=list)
    needs_rewrite: bool = Field(False)


class QueryRewriteResult(BaseModel):
    strategy: str = Field(...)
    rewritten_queries: List[str] = Field(...)


class EvidenceSufficiencyResult(BaseModel):
    is_sufficient: bool = Field(...)
    reason: str = Field("")
    confidence: str = Field("")


class AnswerResult(BaseModel):
    answer: str = Field("")
    reasoning: str = Field("")
    confidence: str = Field("")
    # LLM-claimed verbatim quotes from the retrieved excerpts; verified downstream by the
    # query service before they reach the user. Each entry is {source, page, quote}.
    used_passages: List[Dict] = Field(default_factory=list)


class PDFAgentResult(BaseModel):
    final_answer: str = Field("")
    reasoning: str = Field("")
    evidence_used: List[Dict] = Field(default_factory=list)
    query_rewrites: List[str] = Field(default_factory=list)
    agent_trace: str = Field("")
    # LLM-claimed verbatim quotes (same shape as AnswerResult.used_passages). Surfaced here so
    # the query-service wrapper can verify them before showing to the user.
    used_passages: List[Dict] = Field(default_factory=list)
    # Full untruncated content of every chunk the agent saw across its session; used as the
    # verification corpus for `used_passages`. Same shape as evidence_used but with full content.
    verification_pool: List[Dict] = Field(default_factory=list)


# ============================================================
# Section / anchor helpers
# ============================================================

def _extract_anchor_tokens(query: str) -> List[str]:
    """Tokens that MUST survive a rewrite — section numbers, figure/table refs, quoted phrases.
    Without anchoring, the rewriter (which sees real document content) tends to drift to whatever
    sub-topic dominates the exploratory hits, losing the user's actual question."""
    anchors: List[str] = []
    for m in SECTION_NUMBER_PATTERN.finditer(query):
        anchors.append(m.group(0))
    for m in FIGURE_REF_IN_QUERY_PATTERN.finditer(query):
        anchors.append(m.group(0))
    for m in QUOTED_PHRASE_PATTERN.finditer(query):
        anchors.append(m.group(1))
    # De-duplicate while preserving order
    seen = set()
    unique = []
    for a in anchors:
        key = a.lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)
    return unique


def _detect_section_number(query: str) -> Optional[str]:
    """If the query references a single document section like '7.3' or '4.2.1', return it."""
    m = SECTION_NUMBER_PATTERN.search(query)
    if m:
        return m.group(1)
    return None


def _looks_like_heading_for_section(content: str, section_num: str) -> bool:
    """Does this chunk contain `<section_num> <Capitalized title>` on a single line that
    does NOT look like a table-of-contents entry? True ≈ this chunk is the section start
    (or contains it mid-chunk because the chunker spanned a section boundary).

    TOC rejection: a TOC line for section 7.3 looks like
        '7.3\tRecommendation\t\ton data quality\t\t\t36'
    — i.e. trailing whitespace + page number. Real body headings don't have that trailing
    number. Without rejection, Document-Intelligence-extracted TOC tables (where the number
    and title sit on the same row separated by tabs) collide with real heading detection."""
    if not content or not section_num:
        return False
    pat = re.compile(
        rf"^[ \t]*{re.escape(section_num)}[ \t]+[-–—]?[ \t]*[A-Z][^\n]*",
        re.MULTILINE,
    )
    for m in pat.finditer(content):
        line = m.group(0).rstrip()
        # Reject TOC-style: trailing whitespace + 1-4 digit page number
        if re.search(r'\s+\d{1,4}\s*$', line):
            continue
        return True
    return False


def _is_within_section(heading_num: str, target: str) -> bool:
    """Is `heading_num` either the target section itself or one of its sub-sections?
    Used to detect when section traversal should stop (e.g., walking 7.3 → 7.3.1 OK,
    7.3 → 7.4 stops, 7.3 → 8 stops)."""
    return heading_num == target or heading_num.startswith(target + ".")


# Regexes for TOC detection. We see TOCs in two layouts:
#   1. PROSE-extracted TOC: section number on its own line, title on next line(s), page number
#      on a later line. Many "8.1\n\nMetadata elements\n\n38" patterns in one chunk.
#   2. TABLE-extracted TOC (from Document Intelligence): tab-separated rows like
#      "\t8.1.1 Conformity\t\t\t\t\t39" — number, title, page on same line.
_TOC_INLINE_LINE = re.compile(
    r'(?:^|\n)[ \t]*\d+(?:\.\d+){0,3}[ \t]+\S[^\n]*?[ \t]+\d{1,4}[ \t]*(?=\n|$)',
)
_TOC_PROSE_PAGENUM_LINE = re.compile(r'^[ \t]*\d{1,4}[ \t]*$', re.MULTILINE)
# Section identifier on its own line. Two valid forms:
#   - "8.1" / "8.1.1" / "8.1.1.1"  — section number with at least one dot, alone on line
#   - "8.1.1 Conformity"           — section number (any depth) followed by title text
# A bare "8" alone on a line is excluded — could be a page number or list item.
_TOC_PROSE_SECTION_LINE = re.compile(
    r'^[ \t]*(?:'
    r'\d+(?:\.\d+){1,3}[ \t]*$'
    r'|\d+(?:\.\d+){0,3}[ \t]+[A-Z<]'
    r')',
    re.MULTILINE,
)


def _is_toc_chunk(content: str) -> bool:
    """Heuristic: does this chunk look like a table of contents / index, rather than real body
    content? TOCs are useless for answering content questions — they list section titles and
    page numbers but don't describe what the sections contain. Filtering them out of the LLM's
    evidence pool prevents the agent from "answering" by paraphrasing the TOC.

    Two patterns are detected:
      (a) ≥4 same-line entries like "8.1.1 Conformity ... 39" (table-format TOC), or
      (b) ≥3 standalone-page-number lines AND ≥3 section-identifier lines (prose-format TOC
          where PDF text extraction split number/title/page onto separate lines).
    Thresholds are tuned for real PDFs: actual body text rarely shows 3+ pure-digit lines
    next to 3+ section identifiers, but a TOC chunk has dozens.
    """
    if not content:
        return False
    if len(_TOC_INLINE_LINE.findall(content)) >= 4:
        return True
    if (len(_TOC_PROSE_PAGENUM_LINE.findall(content)) >= 3
            and len(_TOC_PROSE_SECTION_LINE.findall(content)) >= 3):
        return True
    return False


def _walk_section_from_seed(seed: Dict, fetch_by_seq_func: Callable, section_num: str,
                            window: int = SECTION_WALK_WINDOW) -> List[Dict]:
    """Starting from the seed chunk (the one containing the section heading), pull consecutive
    chunks by `seq` from the same source until either:
      - we encounter a chunk whose body starts with a heading OUTSIDE the target section, or
      - we exhaust `window` chunks.
    Returns the seed plus the body chunks, in document order."""
    source = seed.get("source")
    start_seq = seed.get("seq")
    if start_seq is None or not source:
        return [seed]

    chunks = fetch_by_seq_func(source, start_seq, "next", window) or []
    if not chunks:
        return [seed]

    section_chunks: List[Dict] = []
    for i, c in enumerate(chunks):
        if i == 0:
            # The fetch should return the seed itself first; always include it.
            section_chunks.append(c)
            continue
        content = (c.get("content") or "")[:300]
        m = HEADING_AT_LINESTART_PATTERN.search(content)
        if m:
            num = m.group(1)
            if not _is_within_section(num, section_num):
                # Hit the next section — stop without including this chunk.
                break
        section_chunks.append(c)
    return section_chunks


# ============================================================
# Core tool implementations
# ============================================================

def _check_query_quality(query: str, llm: AzureOpenAI, deployment: str) -> QueryQualityResult:
    issues = []
    query_stripped = query.strip()
    query_lower = query_stripped.lower()

    if not query_stripped:
        issues.append("empty")
    elif len(query_stripped.split()) == 1:
        issues.append("single_word")
    elif len(query_stripped.split()) < 3:
        issues.append("too_short")

    found_vague = []
    for term in VAGUE_TERMS:
        if f" {term} " in f" {query_lower} " or query_lower.startswith(f"{term} ") or query_lower.endswith(f" {term}"):
            found_vague.append(term)
    if len(found_vague) >= 2:
        issues.append(f"vague_terms: {', '.join(found_vague)}")

    if any(issue in HARD_FAIL_ISSUES or issue.split(":")[0] in HARD_FAIL_ISSUES for issue in issues):
        return QueryQualityResult(is_clear=False, issues=issues, needs_rewrite=True)

    if not issues:
        return QueryQualityResult(is_clear=True, issues=[], needs_rewrite=False)

    try:
        prompt = (
            f'Analyze: "{query}"\n\n'
            'Is this question specific enough to retrieve good results from technical docs?\n'
            'Respond "CLEAR" or "UNCLEAR: [brief reason]". Be lenient for normal questions.'
        )
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        response = completion.choices[0].message.content.strip()

        if response.startswith("UNCLEAR"):
            if ":" in response:
                issues.append(f"llm_judgment: {response.split(':', 1)[1].strip()}")
            return QueryQualityResult(is_clear=False, issues=issues, needs_rewrite=True)
        else:
            return QueryQualityResult(is_clear=True, issues=[], needs_rewrite=False)
    except Exception as e:
        print(f"[WARNING] LLM quality check failed: {e}")
        return QueryQualityResult(is_clear=len(issues) == 0, issues=issues, needs_rewrite=len(issues) > 0)


def _rewrite_query(query: str, issues: List[str], llm: AzureOpenAI, deployment: str,
                   initial_hits: Optional[List[Dict]] = None) -> QueryRewriteResult:
    retrieval_context = ""
    if initial_hits:
        retrieval_context = f"\n\nREAL DOCUMENT CONTEXT (from exploratory search — {len(initial_hits)} chunks):\n"
        for idx, hit in enumerate(initial_hits[:3], 1):
            source = hit.get("source", "?")
            page = hit.get("page", "?")
            preview = hit.get("content", "")[:200].replace("\n", " ")
            retrieval_context += f"{idx}. [{source} p.{page}] {preview}...\n"
        retrieval_context += (
            "\nUse terminology from the ACTUAL document content above when rewriting. "
            "Do NOT invent topics that aren't in the retrieved content."
        )

    # Anchor tokens MUST survive in every rewrite. Without this, an exploratory search that
    # surfaces a sub-topic of the asked-about section drags the rewrite onto the sub-topic
    # (e.g., "7.3 Recommendation on data quality" → "topological consistency") and the agent
    # never recovers.
    anchors = _extract_anchor_tokens(query)
    anchor_block = ""
    if anchors:
        anchor_list = ", ".join(repr(a) for a in anchors)
        anchor_block = (
            f"\n\nANCHOR TOKENS — these MUST appear UNCHANGED in EVERY rewrite. "
            f"A rewrite that drops any anchor is INVALID and will be discarded: {anchor_list}\n"
        )

    prompt = f"""{DOMAIN_CONTEXT}

Rewrite this question into a complete, specific question.

ORIGINAL: "{query}"
ISSUES: {', '.join(issues)}
{retrieval_context}{anchor_block}

Rules:
1. If the original is a single word or phrase, turn it into a FULL QUESTION
   grounded in the ACTUAL document content shown above.
2. Maintain core intent of the original. Do NOT pivot to a different topic.
3. Use specific terminology from the document context when available, but only
   to clarify the original question — not to replace it.
4. Break broad questions into 2-3 focused sub-questions if useful.
5. Every anchor token (if listed above) MUST appear verbatim in every rewrite.

Format (exactly these lines, one per rewrite):
REWRITE: [your first rewritten question]
REWRITE: [second variation if useful]"""

    try:
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        response = completion.choices[0].message.content.strip()

        rewritten = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("REWRITE:"):
                rq = line.replace("REWRITE:", "").strip()
                if rq:
                    rewritten.append(rq)

        # Hard-enforce anchors: drop any rewrite that lost a required token. If all rewrites
        # drift, fall back to the original query — better to retry the same search than chase
        # a sub-topic.
        if anchors and rewritten:
            kept = [rq for rq in rewritten if all(a.lower() in rq.lower() for a in anchors)]
            if kept:
                rewritten = kept
            else:
                print(f"[WARN] All rewrites dropped anchor tokens {anchors}; falling back to original query")
                rewritten = [query]

        if not rewritten:
            rewritten = [query]

        strategy = "informed_expand" if initial_hits and len(rewritten) > 1 else (
            "informed_clarify" if initial_hits else "blind_clarify"
        )
        return QueryRewriteResult(strategy=strategy, rewritten_queries=rewritten[:3])

    except Exception as e:
        print(f"[WARNING] Query rewrite failed: {e}")
        return QueryRewriteResult(strategy="fallback", rewritten_queries=[query])


def _check_evidence_sufficiency(query: str, hits: List[Dict], llm: AzureOpenAI,
                                 deployment: str) -> EvidenceSufficiencyResult:
    if not hits:
        return EvidenceSufficiencyResult(is_sufficient=False, reason="No evidence retrieved", confidence="high")

    # 800 chars per hit lets the judge see actual section content, not just a heading.
    # Earlier 200-char previews triggered false-INSUFFICIENT verdicts on legitimate hits
    # because a section-7.3 chunk's first 200 chars are usually just the heading + intro.
    summary = ""
    for idx, hit in enumerate(hits[:5], 1):
        content = hit.get("content", "")[:800]
        source = hit.get("source", "?")
        page = hit.get("page", "?")
        summary += f"\n{idx}. [{source} p.{page}] {content}\n"

    prompt = f"""Question: {query}

Evidence:{summary}

Does this evidence DIRECTLY answer the question? Be lenient: if the evidence covers
the topic the user asked about — even partially — call it SUFFICIENT and let the
answer step decide how complete the answer is. Only return INSUFFICIENT when none
of the evidence is on-topic at all.

Respond "SUFFICIENT" or "INSUFFICIENT: [what's missing]"."""

    try:
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        response = completion.choices[0].message.content.strip()

        if response.startswith("INSUFFICIENT"): #LLM returns "INSUFFICIENT" , trust directly
            return EvidenceSufficiencyResult(
                is_sufficient=False,
                reason=response.split(":", 1)[1].strip() if ":" in response else "gaps identified",
                confidence="high"
            )
        else: #LLM returns "SUFFICIENT", check again with hits count to avoid over-trusting
            is_sufficient = len(hits) >= MIN_HITS
            return EvidenceSufficiencyResult(
                is_sufficient=is_sufficient,
                reason="Evidence addresses the question" if is_sufficient else f"Only {len(hits)} hit(s)",
                confidence="high" if len(hits) >= 3 else "medium"
            )
    except Exception as e:
        print(f"[WARNING] Sufficiency check failed: {e}")
        return EvidenceSufficiencyResult(
            is_sufficient=len(hits) >= MIN_HITS,
            reason=f"Found {len(hits)} chunks (LLM check failed)",
            confidence="low"
        )


def _generate_answer(query: str, hits: List[Dict], llm: AzureOpenAI, deployment: str) -> AnswerResult:
    """
    Final-answer step for the agent. Mirrors the standard-mode JSON-output contract so the
    downstream query-service wrapper can apply the same verbatim-quote verification + display
    rules to agent results as it does to standard results.
    """
    if not hits:
        return AnswerResult(
            answer="I cannot find sufficient evidence in the provided documents to answer this question.",
            reasoning="No matching excerpts retrieved.",
            confidence="high",
            used_passages=[],
        )

    # Label each excerpt with SOURCE_TYPE so the LLM can distinguish PDF-extracted text from
    # AI-generated image descriptions. Both kinds may inform the answer; the display layer
    # adds a clear disclaimer to generated_description quotes.
    blocks = []
    for idx, h in enumerate(hits, start=1):
        content = h.get("content", "")
        source = h.get("source", "")
        page = h.get("page", "")
        chunk_type = h.get("chunk_type", "text")
        source_type = "generated_description" if chunk_type == "image" else "original_text"
        blocks.append(
            f"[EXCERPT {idx}]\n"
            f"SOURCE_DOCUMENT: {source}\n"
            f"PAGE_NUMBER: {page}\n"
            f"SOURCE_TYPE: {source_type}\n"
            f"CONTENT:\n{content}"
        )
    excerpts_text = "\n\n---\n\n".join(blocks)

    prompt = (
        "You are a document question-answering assistant. Answer the user's input strictly "
        "based on the provided excerpts from one or more PDF documents.\n\n"
        "If the user's input is a figure caption, section title, topic name, or other phrase "
        "without a complete question, treat it as 'describe / summarize / explain that topic' "
        "and answer using the relevant excerpts. Do not refuse just because the input isn't "
        "phrased as a complete question.\n\n"
        "Excerpts beginning with '[IMAGE on page N - TYPE]' are descriptions of a figure or "
        "diagram on that page. When the user asks about a figure (by number, caption, or topic), "
        "use these excerpts to describe what the figure shows — its components, structure, "
        "labels, and relationships. Cite them like any other excerpt.\n\n"
        "Each excerpt has a SOURCE_TYPE field with one of two values:\n"
        "- \"original_text\": text extracted directly from the PDF (paragraphs, tables, KV pairs). "
        "Verbatim PDF content.\n"
        "- \"generated_description\": an AI-generated description of a figure/diagram. The "
        "description is ABOUT the PDF but is NOT itself in the PDF.\n"
        "Both types are equally valid for composing your `answer` AND for inclusion in "
        "`used_passages`. The downstream UI labels generated_description quotes with a clear "
        "disclaimer, so the user can tell them apart. Never refuse just because the only "
        "relevant excerpts are generated_description.\n\n"
        "=== RETRIEVED EXCERPTS ===\n"
        f"{excerpts_text}\n\n"
        "=== USER QUESTION ===\n"
        f"{query}\n\n"
        "=== OUTPUT FORMAT ===\n"
        "Return a single JSON object with EXACTLY these three fields and nothing else:\n\n"
        "{\n"
        "  \"answer\": \"<your prose answer with inline (Source: <file>, page <N>) citations>\",\n"
        "  \"reasoning\": \"<one or two sentences explaining how you derived the answer from the excerpts>\",\n"
        "  \"used_passages\": [\n"
        "    {\"source\": \"<filename>\", \"page\": <integer>, \"quote\": \"<exact verbatim text from that excerpt>\"}\n"
        "  ]\n"
        "}\n\n"
        "=== RULES FOR `answer` ===\n"
        "1. GROUNDING: Use ONLY the information in the excerpts. No outside knowledge.\n"
        "2. INSUFFICIENT EVIDENCE: Set `answer` to exactly \"I cannot find sufficient evidence "
        "in the provided documents to answer this question.\" ONLY if NONE of the excerpts contain "
        "content relevant to the user's input. The absence of `original_text` quotes is NEVER a "
        "reason to refuse: if a `generated_description` excerpt is on-topic, give a substantive answer.\n"
        "3. CITATION: Every factual claim in `answer` MUST be followed by an inline citation "
        "in this exact format: (Source: <SOURCE_DOCUMENT>, page <PAGE_NUMBER>).\n"
        "4. MULTI-DOCUMENT: If the answer involves multiple documents, attribute each point to its source.\n\n"
        "=== RULES FOR `used_passages` ===\n"
        "5. EXACT VERBATIM: Each `quote` MUST be copy-pasted character-for-character from one of "
        "the excerpts above. Do NOT paraphrase, summarize, shorten, fix typos, or reformat. Do NOT "
        "use ellipsis (`...` or `…`) inside a quote.\n"
        "6. SOURCE & PAGE MATCH: `source` and `page` MUST exactly match the SOURCE_DOCUMENT and "
        "PAGE_NUMBER of the excerpt the quote came from. `page` is an integer.\n"
        "7. EITHER SOURCE_TYPE OK: Quotes may come from `original_text` OR `generated_description` excerpts.\n"
        "8. ONLY USED: Include only quotes that DIRECTLY support a specific claim in `answer`. "
        "1 to 5 well-chosen quotes is normal.\n"
        "9. FULL UNIT: A quote should be a complete sentence (or row, or bullet). If you can't "
        "include a complete unit verbatim, omit the passage.\n"
        "10. EMPTY OK: Return `used_passages: []` only when no excerpt actually supports your "
        "claims verbatim (e.g., insufficient-evidence answer). Otherwise include at least one quote.\n"
    )

    try:
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        raw = (completion.choices[0].message.content or "").strip()

        try:
            data = json.loads(raw)
        except Exception as parse_err:
            print(f"[WARN] agent _generate_answer JSON parse failed: {parse_err}; using raw text")
            confidence = "high" if len(hits) >= 3 else "medium" if len(hits) >= 2 else "low"
            return AnswerResult(answer=raw, reasoning="Based on retrieved evidence", confidence=confidence, used_passages=[])

        answer = (data.get("answer") or "").strip()
        reasoning = (data.get("reasoning") or "Based on retrieved evidence").strip()
        passages = data.get("used_passages") or []
        if not isinstance(passages, list):
            passages = []
        # Light schema sanity — keep only dict-shaped entries with the required keys.
        used_passages = [
            {
                "source": p.get("source") or "",
                "page": p.get("page"),
                "quote": (p.get("quote") or "").strip(),
            }
            for p in passages
            if isinstance(p, dict) and p.get("source") and p.get("quote")
        ]

        confidence = "high" if len(hits) >= 3 else "medium" if len(hits) >= 2 else "low"
        return AnswerResult(answer=answer, reasoning=reasoning, confidence=confidence, used_passages=used_passages)

    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        return AnswerResult(
            answer="Error generating answer. Please try again.",
            reasoning=f"Error: {str(e)}",
            confidence="low",
            used_passages=[],
        )


# ============================================================
# Agent State
# ============================================================

class AgentState:
    """Tracks what the agent has done so the safety net can verify key steps."""
    def __init__(self, original_query: str):
        self.original_query = original_query
        self.all_hits: List[Dict] = []
        self.query_rewrites: List[str] = []
        self.last_hits: List[Dict] = []
        self.searched_queries: set = set()
        self.tools_used: set = set()
        self.seen_keys = set()
        # Pre-flight (section-walk) chunks. These are guaranteed on-topic and bypass the
        # reranker-score sort in best_hits() so they always reach the answer LLM. Without this,
        # the agent's first BM25-dominant search can fill state.all_hits with off-topic
        # high-rank chunks (e.g., the table-of-contents page) and bury the actual section body.
        self.pinned_keys: set = set()
        # Step counter shared across the callback and tools (for visible step blocks)
        self.step_counter = 0
        # Buffer for sub-step lines (exploratory searches that happen
        # inside a tool); flushed when the agent's tool actually finishes.
        self.pending_substeps: List[str] = []

    def add_hits(self, hits: List[Dict], pinned: bool = False):
        self.last_hits = hits
        for h in hits:
            key = (h.get("source"), h.get("chunk_id"))
            if key not in self.seen_keys:
                self.seen_keys.add(key)
                self.all_hits.append(h)
            if pinned:
                self.pinned_keys.add(key)

    def best_hits(self, n: int) -> List[Dict]:
        """Return up to n hits, ordered:
          1. pinned (pre-flight section chunks) in document/seq order, always included
          2. unpinned NON-TOC chunks sorted by reranker score descending
        TOC chunks are excluded entirely — they describe the document's structure but contain
        no real content the LLM can use to answer a content question. Without this filter,
        the agent's first BM25-favored TOC hit could occupy all 5 final slots and produce
        misleading paraphrases of the index.
        Pinned chunks are kept regardless (a pre-flight seed that happened to be a TOC-ish
        chunk is the user's explicit anchor; we trust it).
        """
        pinned: List[Dict] = []
        unpinned: List[Dict] = []
        for h in self.all_hits:
            key = (h.get("source"), h.get("chunk_id"))
            if key in self.pinned_keys:
                pinned.append(h)
                continue
            if _is_toc_chunk(h.get("content", "")):
                continue
            unpinned.append(h)
        pinned.sort(key=lambda x: (x.get("source") or "", x.get("seq") if x.get("seq") is not None else 0))
        unpinned.sort(key=lambda x: x.get("@search.rerankerScore", 0) or 0, reverse=True)
        if n <= 0:
            return pinned + unpinned
        if len(pinned) >= n:
            return pinned[:n]
        return pinned + unpinned[: n - len(pinned)]

    def next_step_number(self) -> int:
        self.step_counter += 1
        return self.step_counter


# ============================================================
# Trace streaming callback — formatted as clear step blocks
# ============================================================

class TraceStreamCallback(BaseCallbackHandler):
    """Stream the agent's reasoning steps to the frontend as boxed blocks.

    Each agent step becomes:
      ──────────
      📍 STEP N
      ──────────
      💭 Thought: ...
      🔧 Action: ...
         Input: ...
      [optional sub-step lines from inside the tool]
      👁  Observation: ...
    """
    def __init__(self, trace_callback: Optional[Callable[[str], None]],
                 trace_log: List[str], state: AgentState):
        super().__init__()
        self.trace_callback = trace_callback
        self.trace_log = trace_log
        self.state = state

    def _emit(self, line: str):
        self.trace_log.append(line)
        if self.trace_callback:
            self.trace_callback(line)

    @staticmethod
    def _clean_thought(text: str) -> str:
        """Extract real reasoning text from LangChain's action.log envelope."""
        text = text.strip()
        while text.lower().startswith("thought:"):
            text = text[len("thought:"):].strip()
        # Cut everything from "Action:" onwards — that's routing, not thought
        action_match = re.search(r'(?:\n|^)\s*action\s*:', text, re.IGNORECASE)
        if action_match:
            text = text[:action_match.start()].strip()
        text = text.strip(" \t\r\n-*•:")
        return text

    def _emit_step_header(self, step_num: int, label: str = "STEP"):
        self._emit("")
        self._emit(STEP_DIVIDER)
        self._emit(f"📍 {label} {step_num}")
        self._emit(STEP_DIVIDER)

    def on_agent_action(self, action, **kwargs):
        # Open a new step block
        step_num = self.state.next_step_number()
        self._emit_step_header(step_num)

        # Thought (skip if empty)
        if action.log:
            cleaned = self._clean_thought(action.log)
            if cleaned:
                self._emit(f"💭 Thought:")
                # Indent multi-line thoughts
                for line in cleaned.splitlines():
                    self._emit(f"   {line}")

        # Action header
        self._emit("")
        self._emit(f"🔧 Action: {action.tool}")
        self._emit(f"   Input: {str(action.tool_input)[:300]}")
        self.state.tools_used.add(action.tool)

    def on_tool_end(self, output: str, **kwargs):
        # Observation
        self._emit("")
        self._emit("👁  Observation:")
        for line in str(output).splitlines():
            self._emit(f"   {line}")

    def on_agent_finish(self, finish, **kwargs):
        # Final wrap-up "step" — show the closing thought + finish marker
        self._emit("")
        self._emit(STEP_DIVIDER)
        self._emit("🏁 FINAL")
        self._emit(STEP_DIVIDER)

        if hasattr(finish, "log") and finish.log:
            cleaned = self._clean_thought(finish.log)
            # Strip "Final Answer:" trailing block too
            final_match = re.search(r'\s*final\s+answer\s*:', cleaned, re.IGNORECASE)
            if final_match:
                cleaned = cleaned[:final_match.start()].strip()
            if cleaned:
                self._emit("💭 Thought:")
                for line in cleaned.splitlines():
                    self._emit(f"   {line}")

        self._emit("")
        final_output = str(finish.return_values.get('output', ''))[:200]
        self._emit(f"✅ Agent finished")
        self._emit(f"   {final_output}...")


# ============================================================
# Tool factories
# ============================================================

def _build_tools(state: AgentState, search_func: Callable,
                 llm: AzureOpenAI, deployment: str, top_k: int,
                 log_fn: Callable[[str], None],
                 fetch_by_seq_func: Optional[Callable] = None) -> List[Tool]:
    """Create the toolkit the agent can choose from. The `fetch_section` tool is registered only
    when `fetch_by_seq_func` is provided (i.e., the index supports `seq`)."""

    def search_documents(search_query: str) -> str:
        if not search_query or not search_query.strip():
            return "ERROR: empty search query. You MUST call rewrite_query first."

        query_key = search_query.strip().lower()

        if query_key in state.searched_queries:
            return (f"DUPLICATE: you already searched for '{search_query}'. "
                    f"Do NOT repeat the same query. You MUST either: "
                    f"(a) call rewrite_query to get a new phrasing, or "
                    f"(b) call evaluate_evidence on what you have, or "
                    f"(c) produce Final Answer if evidence is enough.")
        state.searched_queries.add(query_key)

        if search_query.strip() != state.original_query.strip() and search_query not in state.query_rewrites:
            state.query_rewrites.append(search_query)

        # Snapshot pre-search state so we can detect "this search added nothing new" later.
        prev_seen = set(state.seen_keys)
        prev_useful_count = sum(
            1 for h in state.all_hits if not _is_toc_chunk(h.get("content", ""))
        )

        # Exclude chunks the agent has already retrieved so each search returns fresh content.
        # Without exclusion, BM25/vector winners get re-fetched every time the query is rephrased,
        # wasting the top-k slots on duplicates and masking the corpus-exhausted signal. The
        # underlying search_func oversamples and filters client-side; if it can't produce top_k
        # fresh hits, that is itself the signal we want — and STUCK detection will pick it up.
        try:
            hits = search_func(search_query, top_k, exclude_keys=prev_seen)
        except TypeError:
            # Fallback for any search_func without the exclude_keys kwarg (e.g., custom test stubs).
            hits = search_func(search_query, top_k)
        state.add_hits(hits)

        # Classify hits: useful (real content) vs TOC (structure-only). Both are tracked in
        # state.all_hits so dedup works, but TOC chunks never reach the answer LLM.
        toc_flags = [_is_toc_chunk(h.get("content", "")) for h in hits]
        n_toc = sum(toc_flags)
        n_useful = len(hits) - n_toc
        new_keys_this_search = {(h.get("source"), h.get("chunk_id")) for h in hits} - prev_seen
        new_useful_this_search = sum(
            1 for h in hits
            if (h.get("source"), h.get("chunk_id")) in new_keys_this_search
            and not _is_toc_chunk(h.get("content", ""))
        )

        # STUCK detection runs BEFORE the no-hits early return: when exclusion drains the
        # candidate pool, hits=0 is itself a strong "corpus exhausted" signal — without this
        # ordering the agent would be told "MUST rewrite" and loop forever on a query whose
        # remaining matches are all already-seen chunks. Once 2+ prior searches have happened,
        # any zero-new-useful round means more rewrites won't help.
        if new_useful_this_search == 0 and len(state.searched_queries) >= 2:
            if not hits:
                why = (f"search returned 0 hits — after excluding the chunks you've already "
                       f"retrieved, nothing new in the index matches this query.")
            else:
                why = (f"search returned {len(hits)} hit(s) but ZERO new useful chunks "
                       f"({n_toc}/{len(hits)} were table-of-contents/index entries; the rest "
                       f"were already in your evidence pool).")
            return (
                f"STUCK: {why} You currently have {prev_useful_count} useful chunk(s) total. "
                f"Further rewrites will not help. NEXT STEP: call evaluate_evidence and then "
                f"produce Final Answer with what you have — do NOT rewrite again."
            )

        if not hits:
            return (f"NO RESULTS for '{search_query}'. "
                    f"You MUST call rewrite_query next with a different phrasing, "
                    f"then search again.")

        scores = [h.get("@search.rerankerScore", 0) or 0 for h in hits]
        avg_score = sum(scores) / len(scores) if scores else 0

        lines = [
            f"Found {len(hits)} chunks for '{search_query}' "
            f"(useful: {n_useful}, TOC filtered: {n_toc}, avg reranker score: {avg_score:.2f})"
        ]
        for i, h in enumerate(hits[:5], 1):
            src = h.get("source", "?")
            page = h.get("page", "?")
            score = h.get("@search.rerankerScore", 0) or 0
            toc_tag = " [TOC — will be excluded from final evidence]" if toc_flags[i - 1] else ""
            preview = h.get("content", "")[:120].replace("\n", " ")
            lines.append(f"  {i}. [score={score:.2f}] [{src} p.{page}]{toc_tag} {preview}...")

        warnings = []
        if avg_score < LOW_SCORE_WARNING_THRESHOLD:
            warnings.append(
                f"⚠️ LOW RELEVANCE: avg score {avg_score:.2f} < {LOW_SCORE_WARNING_THRESHOLD}. "
                f"Results may be keyword matches without real semantic relevance."
            )
        if len(search_query.split()) <= 1:
            warnings.append(
                f"⚠️ SINGLE-WORD QUERY: '{search_query}' cannot express a question's intent. "
                f"Any keyword match is suspicious."
            )

        if warnings:
            lines.append("")
            lines.extend(warnings)
            lines.append(
                "RECOMMENDED NEXT STEP: call rewrite_query to form a more specific question, "
                "then search again with the rewrite."
            )
        else:
            lines.append("")
            lines.append("NEXT STEP: call evaluate_evidence to verify these chunks answer the question.")

        return "\n".join(lines)

    def assess_query(query: str) -> str:
        result = _check_query_quality(query, llm, deployment)
        if result.is_clear:
            return ("Query is CLEAR and specific enough. "
                    "NEXT STEP: call search_documents with this query.")
        issues_str = ", ".join(result.issues) if result.issues else "quality issues"
        return (f"Query is UNCLEAR. Issues: {issues_str}. "
                f"MANDATORY NEXT STEP: you MUST call rewrite_query before searching.")

    def rewrite_query(query: str) -> str:
        """Rewrite with informed context from an internal exploratory search.

        The exploratory search is shown as its own visible sub-step block so
        the user can see the full chain of operations.
        """
        if not state.last_hits:
            issues = ["no_results_yet"]
        elif len(state.last_hits) < MIN_HITS:
            issues = ["insufficient_results"]
        else:
            issues = ["needs_refinement"]

        if len(query.split()) <= 2:
            issues.append("too_short")

        context_hits = state.last_hits
        used_exploratory = False

        if not context_hits:
            # Run exploratory search and emit it as a visible sub-step block.
            # We log directly via log_fn (synchronously) instead of buffering,
            # because relying on on_tool_end to flush is fragile across
            # LangChain versions.
            try:
                sub_step_num = state.next_step_number()
                log_fn("")
                log_fn(STEP_DIVIDER)
                log_fn(f"📍 STEP {sub_step_num}  [sub-step inside rewrite_query]")
                log_fn(STEP_DIVIDER)
                log_fn("💭 Thought:")
                log_fn("   No prior search results, so I'll do a quick exploratory search")
                log_fn("   to see what the documents actually contain about this term.")
                log_fn("")
                log_fn("🔧 Action: search_documents  [exploratory]")
                log_fn(f"   Input: {query}")

                context_hits = search_func(query, EXPLORATORY_TOP_K)
                used_exploratory = True

                log_fn("")
                log_fn("👁  Observation:")
                log_fn(f"   Found {len(context_hits)} chunks (will be used to ground the rewrite,")
                log_fn("   but NOT added to the main evidence pool):")
                for i, h in enumerate(context_hits[:3], 1):
                    src = h.get("source", "?")
                    page = h.get("page", "?")
                    preview = h.get("content", "")[:80].replace("\n", " ")
                    log_fn(f"   {i}. [{src} p.{page}] {preview}...")

            except Exception as e:
                print(f"[WARNING] Exploratory search during rewrite failed: {e}")
                log_fn(f"   ⚠️ exploratory search failed: {e}")
                context_hits = []

        result = _rewrite_query(query, issues, llm, deployment, initial_hits=context_hits)
        if not result.rewritten_queries:
            return "Could not generate rewrites. Try a different approach."

        lines = []
        if used_exploratory:
            lines.append(
                f"(Exploratory search done — see sub-step above. "
                f"Rewrites are grounded in {len(context_hits)} sample chunks.)"
            )
        lines.append(f"Strategy: {result.strategy}. Rewrites (pick one and pass to search_documents):")
        for i, rq in enumerate(result.rewritten_queries, 1):
            lines.append(f"  {i}. {rq}")
        lines.append("")
        lines.append("MANDATORY NEXT STEP: call search_documents with one of these rewrites.")
        return "\n".join(lines)

    def evaluate_evidence(question: str) -> str:
        if not state.all_hits:
            return ("No evidence gathered yet. "
                    "MANDATORY NEXT STEP: call search_documents first.")
        # Use best_hits so the judge sees pre-flight section chunks first and unpinned chunks
        # sorted by reranker score — NOT first-added order, which can be dominated by an
        # off-topic but BM25-friendly chunk like the table-of-contents page.
        judge_hits = state.best_hits(5)
        result = _check_evidence_sufficiency(question, judge_hits, llm, deployment)
        verdict = "SUFFICIENT" if result.is_sufficient else "INSUFFICIENT"
        base = (f"Evidence is {verdict} (confidence: {result.confidence}). "
                f"Reason: {result.reason}. "
                f"Total chunks gathered: {len(state.all_hits)}. ")
        if result.is_sufficient:
            return base + "NEXT STEP: produce Final Answer now."

        # Escape hatch: after several rewrite attempts, stop forcing more rewrites and let the
        # answer step handle the partial evidence. The downstream JSON-mode answer prompt
        # already knows how to refuse on truly insufficient evidence — there's no need for the
        # ReAct loop to also enforce that, especially since the loop tends to drift.
        if len(state.query_rewrites) >= MAX_REWRITES_BEFORE_ANSWER:
            return base + (
                f"You have already tried {len(state.query_rewrites)} rewrite(s) without finding "
                "stronger evidence. Further rewrites are unlikely to help. NEXT STEP: produce "
                "Final Answer now using the available evidence — the answer step will decide "
                "whether to give a substantive answer or refuse based on what's actually there."
            )

        return base + (
            "NEXT STEP: try rewrite_query for a different phrasing, OR (if a section number "
            "or figure number is in the original question) call fetch_section to walk the "
            "section in document order — a single search rarely captures multi-page sections. "
            "Do NOT produce Final Answer yet."
        )

    def fetch_section(arg: str) -> str:
        """Walk consecutive chunks by `seq` from a found heading. Use this when the user asked
        about a section (by number like '7.3', '4.2.1', or by Figure/Table number) and a previous
        search hit the section heading — sections often span multiple pages and only one chunk
        usually ranks well in semantic search.

        Input format: 'source.pdf | start_seq | section_number'
          - source: the source filename you saw in a previous search hit
          - start_seq: the seq value of the chunk that contains the heading
          - section_number: the section number to walk through (e.g., '7.3'). Optional —
            if omitted, walks `SECTION_WALK_WINDOW` chunks forward without a stop condition.
        """
        if fetch_by_seq_func is None:
            return ("ERROR: fetch_section is unavailable on this index (legacy index without "
                    "`seq` field). Use search_documents with multiple rephrasings instead.")

        parts = [p.strip() for p in (arg or "").split("|")]
        if len(parts) < 2 or not parts[0]:
            return ("ERROR: input format is 'source.pdf | start_seq | section_number(optional)'. "
                    "Get source and start_seq from a prior search_documents result line like "
                    "'[score=2.10] [Test1.pdf p.47] 7.3 Recommendation on data quality...' — "
                    "use the source from the bracketed prefix and call this tool with the "
                    "matching seq from the same hit.")

        source = parts[0]
        try:
            start_seq = int(parts[1])
        except (TypeError, ValueError):
            return f"ERROR: start_seq must be an integer, got '{parts[1]}'"
        section_num = parts[2] if len(parts) >= 3 and parts[2] else None

        if section_num:
            seed = {"source": source, "seq": start_seq}
            chunks = _walk_section_from_seed(seed, fetch_by_seq_func, section_num,
                                             window=SECTION_WALK_WINDOW)
        else:
            chunks = fetch_by_seq_func(source, start_seq, "next", SECTION_WALK_WINDOW) or []

        if not chunks:
            return (f"NO CHUNKS found for source='{source}' starting at seq={start_seq}. "
                    f"Verify the source filename and seq from a previous search_documents result.")

        state.add_hits(chunks)

        lines = [
            f"Fetched {len(chunks)} consecutive chunk(s) from '{source}' starting at seq={start_seq}"
            + (f" within section {section_num}" if section_num else "")
            + ":"
        ]
        for c in chunks:
            page = c.get("page", "?")
            seq_v = c.get("seq", "?")
            ctype = c.get("chunk_type", "text")
            preview = (c.get("content") or "")[:120].replace("\n", " ")
            lines.append(f"  [seq={seq_v} p.{page} type={ctype}] {preview}...")
        lines.append("")
        lines.append("These chunks are now in the evidence pool. NEXT STEP: call evaluate_evidence "
                     "to see whether the section content answers the question.")
        return "\n".join(lines)

    tools = [
        Tool(
            name="search_documents",
            func=search_documents,
            description=(
                "Search the indexed PDF documents. Input: a search query string. "
                "This is your PRIMARY tool for gathering evidence. "
                "Call multiple times with different phrasings if needed. "
                "Never call it twice with the same exact query."
            )
        ),
        Tool(
            name="assess_query",
            func=assess_query,
            description=(
                "Check if a medium-length question is specific enough for good retrieval. "
                "Input: the question string. "
                "Use this only when you're UNCERTAIN whether a multi-word question is specific enough. "
                "Do NOT use it for single-word queries — those are obviously unclear; go to rewrite_query directly."
            )
        ),
        Tool(
            name="rewrite_query",
            func=rewrite_query,
            description=(
                "Expand a vague/short query into clearer questions. "
                "Input: the query to rewrite. "
                "This tool does its own exploratory mini-search internally to ground the rewrite "
                "in real document content. "
                "MUST be called when: query is a single word, search returns NO RESULTS, "
                "search returns LOW RELEVANCE, or evaluate_evidence says INSUFFICIENT."
            )
        ),
        Tool(
            name="evaluate_evidence",
            func=evaluate_evidence,
            description=(
                "Judge whether gathered evidence is enough to answer. "
                "Input: the original user question. "
                "MUST be called before producing Final Answer."
            )
        ),
    ]

    if fetch_by_seq_func is not None:
        tools.append(Tool(
            name="fetch_section",
            func=fetch_section,
            description=(
                "Walk a section in document order by `seq`. Use when the question asks about "
                "a specific section (e.g. '7.3', '4.2.1') or a Figure/Table that may span "
                "multiple pages — a single search_documents hit rarely captures the whole section. "
                "Input format: 'source.pdf | start_seq | section_number'. Get source + start_seq "
                "from the chunk in a previous search result that contains the section heading."
            )
        ))

    return tools


# ============================================================
# ReAct prompt
# ============================================================

REACT_PROMPT_TEMPLATE = """You are a document-analysis agent answering questions about indexed PDFs.

You have access to these tools:
{tools}

MANDATORY WORKFLOW:

Rule 1: If the input question is a SINGLE WORD or under 3 words, your FIRST
        action MUST be rewrite_query (the query is obviously unclear, so you
        don't need to call assess_query first). Then search with the rewrite.

Rule 2: For longer questions, you may call assess_query if you're uncertain
        about specificity. Otherwise, go straight to search_documents.

Rule 3: You MUST call evaluate_evidence at least once BEFORE producing Final Answer.

Rule 4: If evaluate_evidence returns INSUFFICIENT, follow the NEXT STEP it
        recommends. After 2+ rewrite attempts the evaluator will tell you to
        finalize with the available evidence — do that, do not loop forever.

Rule 5: If search_documents returns LOW RELEVANCE or SINGLE-WORD WARNING,
        you MUST call rewrite_query before concluding.

Rule 6: Never call search_documents twice with the exact same query.

Rule 7: You MUST make at least 2 tool calls total (one search + one evaluate minimum).

Rule 8: SECTION & FIGURE QUESTIONS — if the question contains a section number
        ("7.3", "4.2.1"), a figure number ("Figure 6", "Table 3"), or any other
        structural reference: after your first search_documents hit lands on a
        chunk that contains the heading or caption, immediately call
        fetch_section with that chunk's source and seq. Sections often span
        multiple PDF pages and a single semantic search rarely surfaces the
        body chunks — fetch_section walks them in document order. (If
        fetch_section is not in the tool list, your index is legacy; skip this
        rule.)

Rule 9: ANCHOR PRESERVATION — when you call rewrite_query, the rewrites you
        get back will preserve any section numbers, figure numbers, and
        quoted phrases from the original question. Always pass one of these
        rewrites to search_documents — never invent a new question that
        drops the original anchors.

Rule 10: STUCK signals — search_documents may return a "STUCK:" message
         when a new search produces no new useful chunks (everything was
         either already seen or is just table-of-contents/index junk).
         When this happens, STOP rewriting. Call evaluate_evidence and
         then produce Final Answer with what you have. Continuing to
         rewrite will only re-fetch the same chunks.

Rule 11: TOC IS NOT EVIDENCE — search results may include hits tagged
         "[TOC — will be excluded from final evidence]". These are
         table-of-contents pages and contain no real information about
         section content (just titles + page numbers). Do NOT treat them
         as relevant evidence; they are excluded from your final answer
         pool automatically. If a search returns ONLY TOC hits, you need
         a different search angle, not just a rephrase.

REASONING STYLE:
Every "Thought:" line MUST contain at least one full sentence explaining WHY you
chose the next action. Never write an empty "Thought:" line. Never write
"Thought:" immediately followed by "Action:" — always include reasoning first.

Use this EXACT format — do not deviate:

Question: the input question
Thought: your reasoning about what to do next (at least one sentence)
Action: the tool name, must be one of [{tool_names}]
Action Input: the input to the tool (a plain string)
Observation: the tool's result
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: I have enough evidence and have called evaluate_evidence, so I can finalize.
Final Answer: a short confirmation (the detailed answer is generated separately)

IMPORTANT:
- Do NOT include Markdown code fences around your output.
- Do NOT write "Thought: Thought:" — write "Thought:" only once per step.
- Do NOT write an empty Thought line — always explain your reasoning.
- Read tool output warnings carefully — they tell you the mandatory next step.
- After Final Answer, stop producing further Thought/Action lines.

Question: {input}
Thought:{agent_scratchpad}"""





# ============================================================
# Main entry point
# ============================================================

def run_pdf_agent(
    query: str,
    search_func: Callable,
    llm: AzureOpenAI,
    deployment_name: str,
    top_k: int = 5,
    min_score: float = 0.85,
    trace_callback: Optional[Callable[[str], None]] = None,
    fetch_by_seq_func: Optional[Callable] = None,
) -> PDFAgentResult:
    """Run a true ReAct agent over the indexed PDFs.

    `fetch_by_seq_func`, when provided, enables the agent's `fetch_section` tool and the
    pre-flight section seeding step. It must be a callable
    `(source: str, start_seq: int, direction: str, window: int) -> List[Dict]`
    that returns chunks ordered by `seq` ascending. None on legacy indexes without `seq`.
    """
    trace_log: List[str] = []

    def log(line: str):
        trace_log.append(line)
        if trace_callback:
            trace_callback(line)

    log("=== PDF Agent Decision Start ===")
    log(f"Question: {query}")
    log("Mode: LangChain ReAct agent (LLM-driven tool selection)")

    state = AgentState(original_query=query)

    # Pre-flight: if the user's question references a specific section number, seed
    # `state.all_hits` with the full section content BEFORE the ReAct loop starts. This
    # guarantees coverage of multi-page sections even when the agent's later searches don't
    # surface every body chunk. The agent can still search/rewrite further; this just gives
    # it a strong starting evidence pool.
    if fetch_by_seq_func is not None:
        section_num = _detect_section_number(query)
        if section_num:
            log("")
            log(STEP_DIVIDER)
            log(f"📍 PRE-FLIGHT — section reference '{section_num}' detected")
            log(STEP_DIVIDER)
            log("💭 Thought:")
            log(f"   The question mentions section {section_num}. Sections usually span multiple")
            log("   chunks/pages, so I'll find the heading and pull the full section in document")
            log("   order before starting the normal ReAct loop.")
            try:
                # Find the chunk that contains the section heading via semantic search.
                seed_candidates = search_func(section_num, max(top_k, 5)) or []

                # Helper: pick the first candidate that is a real heading (not TOC), preferring
                # non-table chunks. Real section starts are nearly always in body paragraphs;
                # tables almost always mean we matched a TOC row that slipped past the regex.
                def _pick_seed(candidates):
                    # Pass 1: paragraph chunks only
                    for h in candidates:
                        if h.get("chunk_type") == "table":
                            continue
                        if h.get("seq") is None:
                            continue
                        if _looks_like_heading_for_section(h.get("content", ""), section_num):
                            return h
                    # Pass 2: allow tables (rare — only if no paragraph matched)
                    for h in candidates:
                        if h.get("seq") is None:
                            continue
                        if _looks_like_heading_for_section(h.get("content", ""), section_num):
                            return h
                    return None

                seed = _pick_seed(seed_candidates)
                # Fallback: also try the original query phrasing — it sometimes ranks the
                # heading chunk higher than a bare section-number search does.
                if seed is None:
                    fallback_candidates = search_func(query, max(top_k, 5)) or []
                    seed = _pick_seed(fallback_candidates)

                if seed:
                    log(f"   Found heading at [{seed.get('source')} p.{seed.get('page')} seq={seed.get('seq')}]")
                    section_chunks = _walk_section_from_seed(
                        seed, fetch_by_seq_func, section_num, window=SECTION_WALK_WINDOW
                    )
                    # Pin these chunks so best_hits() always surfaces them, regardless of
                    # subsequent BM25 hits that might score higher but be off-topic.
                    state.add_hits(section_chunks, pinned=True)
                    # Mark the seed-search query as used so the agent doesn't repeat it.
                    state.searched_queries.add(section_num.strip().lower())
                    log(f"   Loaded {len(section_chunks)} chunk(s) for section {section_num} into evidence pool.")
                else:
                    log(f"   No heading chunk found for section {section_num} via pre-flight search; "
                        f"agent will continue normally.")
            except Exception as e:
                log(f"   ⚠️ Pre-flight section seed failed: {e}")

    try:
        chat_llm = AzureChatOpenAI(
            azure_endpoint=str(llm.base_url).rstrip("/").replace("/openai", ""),
            api_key=llm.api_key,
            api_version=llm._api_version if hasattr(llm, "_api_version") else "2024-02-15-preview",
            azure_deployment=deployment_name,
            temperature=0,
        )
    except Exception as e:
        log(f"⚠️  Failed to build LangChain LLM wrapper: {e}")
        log("Falling back to direct retrieval + answer generation.")
        hits = search_func(query, top_k)
        state.add_hits(hits)
        answer_result = _generate_answer(query, state.all_hits, llm, deployment_name)
        log("=== PDF Agent Decision End ===")
        return PDFAgentResult(
            final_answer=answer_result.answer,
            reasoning=answer_result.reasoning,
            evidence_used=[{
                "source": h.get("source"),
                "page": h.get("page"),
                "chunk_id": h.get("chunk_id"),
                "content": h.get("content", "")[:200]
            } for h in state.all_hits[:5]],
            query_rewrites=[],
            agent_trace="\n".join(trace_log),
            used_passages=answer_result.used_passages,
            verification_pool=[{
                "source": h.get("source"),
                "page": h.get("page"),
                "chunk_id": h.get("chunk_id"),
                "chunk_type": h.get("chunk_type", "text"),
                "content": h.get("content", ""),  # full content, no truncation — needed for substring verification
            } for h in state.all_hits],
        )

    tools = _build_tools(state, search_func, llm, deployment_name, top_k, log,
                         fetch_by_seq_func=fetch_by_seq_func)

    prompt = PromptTemplate.from_template(REACT_PROMPT_TEMPLATE)
    agent = create_react_agent(chat_llm, tools, prompt)

    trace_handler = TraceStreamCallback(
        trace_callback=trace_callback, trace_log=trace_log, state=state
    )
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_AGENT_ITERATIONS,
        handle_parsing_errors=True,
        verbose=False,
        callbacks=[trace_handler],
        return_intermediate_steps=True,
    )

    try:
        executor.invoke({"input": query})
    except Exception as e:
        log(f"⚠️  Agent loop error: {e}")

    log("")
    log(STEP_DIVIDER)
    log("📊 SUMMARY")
    log(STEP_DIVIDER)
    log(f"Tools used: {sorted(state.tools_used)}")
    log(f"Queries searched: {len(state.searched_queries)}")
    log(f"Evidence chunks gathered: {len(state.all_hits)}")
    log(f"Query rewrites used: {len(state.query_rewrites)}")





    # =========================================================
    # Add safety mechanisms to catch common failure modes and guide the agent back on track if it missed key steps.
    # =========================================================
    if len(state.searched_queries) == 0:
        log("")
        log("[Safety Net] Agent did not call search_documents. Forcing retrieval.")
        hits = search_func(query, top_k)
        state.add_hits(hits)

    if len(query.strip().split()) <= 1 and not state.query_rewrites:
        log("")
        log("[Safety Net] Single-word query with no rewrite. Forcing rewrite + re-search.")
        rewrite_result = _rewrite_query(query, ["single_word"], llm, deployment_name, state.all_hits)
        for rq in rewrite_result.rewritten_queries[:2]:
            if rq.lower().strip() not in state.searched_queries:
                state.searched_queries.add(rq.lower().strip())
                state.query_rewrites.append(rq)
                log(f"  Searching with rewrite: '{rq}'")
                more_hits = search_func(rq, top_k)
                state.add_hits(more_hits)

    if "evaluate_evidence" not in state.tools_used and state.all_hits:
        log("")
        log("[Safety Net] Agent skipped evaluate_evidence. Running post-hoc check.")
        sufficiency = _check_evidence_sufficiency(query, state.best_hits(5), llm, deployment_name)
        log(f"  Sufficiency: {'SUFFICIENT' if sufficiency.is_sufficient else 'INSUFFICIENT'} "
            f"(confidence: {sufficiency.confidence})")
        log(f"  Reason: {sufficiency.reason}")

        if not sufficiency.is_sufficient and len(state.query_rewrites) < 2:
            log("  Insufficient → triggering rewrite + re-search")
            rewrite_result = _rewrite_query(
                query, ["insufficient_evidence"], llm, deployment_name, state.all_hits
            )
            for rq in rewrite_result.rewritten_queries[:1]:
                if rq.lower().strip() not in state.searched_queries:
                    state.searched_queries.add(rq.lower().strip())
                    state.query_rewrites.append(rq)
                    log(f"  Searching with rewrite: '{rq}'")
                    more_hits = search_func(rq, top_k)
                    state.add_hits(more_hits)



    # =========================================================
    # Final answer generation
    # =========================================================
    log("")
    log(STEP_DIVIDER)
    log("📝 FINAL ANSWER GENERATION")
    log(STEP_DIVIDER)
    log("Generating grounded answer from gathered evidence...")

    # best_hits puts pre-flight section chunks first (in document order) and ranks the rest by
    # reranker score. This is critical: without it, a BM25-favored TOC chunk added by the very
    # first search can occupy all 5 final slots and force a refusal even when the actual section
    # body was retrieved later in the loop.
    final_hits = state.best_hits(max(top_k, 5))
    answer_result = _generate_answer(query, final_hits, llm, deployment_name)

    log(f"✓ Answer generated (confidence: {answer_result.confidence})")
    log("")
    log("=== PDF Agent Decision End ===")

    return PDFAgentResult(
        final_answer=answer_result.answer,
        reasoning=answer_result.reasoning,
        evidence_used=[{
            "source": h.get("source"),
            "page": h.get("page"),
            "chunk_id": h.get("chunk_id"),
            "content": h.get("content", "")[:200]
        } for h in final_hits[:5]],
        query_rewrites=state.query_rewrites,
        agent_trace="\n".join(trace_log),
        used_passages=answer_result.used_passages,
        verification_pool=[{
            "source": h.get("source"),
            "page": h.get("page"),
            "chunk_id": h.get("chunk_id"),
            "chunk_type": h.get("chunk_type", "text"),
            "content": h.get("content", ""),  # full content, no truncation — needed for substring verification
        } for h in state.all_hits],
    )