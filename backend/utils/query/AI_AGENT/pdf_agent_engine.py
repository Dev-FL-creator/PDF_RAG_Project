"""
Available tools:
  - search_documents
  - assess_query
  - rewrite_query
  - evaluate_evidence
"""

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


class PDFAgentResult(BaseModel):
    final_answer: str = Field("")
    reasoning: str = Field("")
    evidence_used: List[Dict] = Field(default_factory=list)
    query_rewrites: List[str] = Field(default_factory=list)
    agent_trace: str = Field("")


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

    prompt = f"""{DOMAIN_CONTEXT}

Rewrite this question into a complete, specific question.

ORIGINAL: "{query}"
ISSUES: {', '.join(issues)}
{retrieval_context}

Rules:
1. If the original is a single word or phrase, turn it into a FULL QUESTION
   grounded in the ACTUAL document content shown above.
2. Maintain core intent of the original
3. Use specific terminology from the document context when available
4. Break broad questions into 2-3 focused sub-questions if useful

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

    summary = ""
    for idx, hit in enumerate(hits[:5], 1):
        content = hit.get("content", "")[:200]
        source = hit.get("source", "?")
        page = hit.get("page", "?")
        summary += f"\n{idx}. [{source} p.{page}] {content}...\n"

    prompt = f"""Question: {query}

Evidence:{summary}

Does this evidence DIRECTLY answer the question? Consider:
- Is the information specific and on-topic?
- Are there key aspects of the question left unanswered?

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
    if not hits:
        return AnswerResult(
            answer="I could not find relevant information in the documents to answer this question.",
            reasoning="No matching excerpts retrieved.",
            confidence="high"
        )

    contexts = []
    for h in hits:
        content = h.get("content", "")
        source = h.get("source", "")
        page = h.get("page", "")
        chunk_id = h.get("chunk_id", "")
        prefix = f"[{source} p.{page} #{chunk_id}] " if source else ""
        contexts.append(prefix + content)

    prompt = f"""Answer the question based on the retrieved document excerpts.

Excerpts:
{chr(10).join(['---', *contexts, '---'])}

Question: {query}

Structure your response:
**Evidence Found:** [what the docs explicitly state]
**Evidence Missing:** [what is unclear or missing]
**Reasoning:** [your thought process]
**Final Answer:** [clear, concise answer with source/page citations]

Base your answer ONLY on the excerpts. Cite source + page for every factual claim."""

    try:
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        answer = completion.choices[0].message.content.strip()

        reasoning_match = re.search(r'\*\*Reasoning:\*\*\s*\n(.+?)(?=\*\*Final Answer:\*\*|$)', answer, re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else "Based on retrieved evidence"

        confidence = "high" if len(hits) >= 3 else "medium" if len(hits) >= 2 else "low"
        return AnswerResult(answer=answer, reasoning=reasoning, confidence=confidence)

    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        return AnswerResult(
            answer="Error generating answer. Please try again.",
            reasoning=f"Error: {str(e)}",
            confidence="low"
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
        # Step counter shared across the callback and tools (for visible step blocks)
        self.step_counter = 0
        # Buffer for sub-step lines (exploratory searches that happen
        # inside a tool); flushed when the agent's tool actually finishes.
        self.pending_substeps: List[str] = []

    def add_hits(self, hits: List[Dict]):
        self.last_hits = hits
        for h in hits:
            key = (h.get("source"), h.get("chunk_id"))
            if key not in self.seen_keys:
                self.seen_keys.add(key)
                self.all_hits.append(h)

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
                 log_fn: Callable[[str], None]) -> List[Tool]:
    """Create the toolkit the agent can choose from."""

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

        hits = search_func(search_query, top_k)
        state.add_hits(hits)

        if not hits:
            return (f"NO RESULTS for '{search_query}'. "
                    f"You MUST call rewrite_query next with a different phrasing, "
                    f"then search again.")

        scores = [h.get("@search.rerankerScore", 0) or 0 for h in hits]
        avg_score = sum(scores) / len(scores) if scores else 0

        lines = [f"Found {len(hits)} chunks for '{search_query}' (avg reranker score: {avg_score:.2f})"]
        for i, h in enumerate(hits[:5], 1):
            src = h.get("source", "?")
            page = h.get("page", "?")
            score = h.get("@search.rerankerScore", 0) or 0
            preview = h.get("content", "")[:120].replace("\n", " ")
            lines.append(f"  {i}. [score={score:.2f}] [{src} p.{page}] {preview}...")

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
        result = _check_evidence_sufficiency(question, state.all_hits, llm, deployment)
        verdict = "SUFFICIENT" if result.is_sufficient else "INSUFFICIENT"
        base = (f"Evidence is {verdict} (confidence: {result.confidence}). "
                f"Reason: {result.reason}. "
                f"Total chunks gathered: {len(state.all_hits)}. ")
        if result.is_sufficient:
            return base + "NEXT STEP: produce Final Answer now."
        else:
            return base + (
                "MANDATORY NEXT STEP: call rewrite_query to try a different angle, "
                "then search again. Do NOT produce Final Answer with insufficient evidence."
            )

    return [
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

Rule 4: If evaluate_evidence returns INSUFFICIENT, you MUST call rewrite_query
        and search again. Do not produce Final Answer with insufficient evidence.

Rule 5: If search_documents returns LOW RELEVANCE or SINGLE-WORD WARNING,
        you MUST call rewrite_query before concluding.

Rule 6: Never call search_documents twice with the exact same query.

Rule 7: You MUST make at least 2 tool calls total (one search + one evaluate minimum).

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
    trace_callback: Optional[Callable[[str], None]] = None
) -> PDFAgentResult:
    """Run a true ReAct agent over the indexed PDFs."""
    trace_log: List[str] = []

    def log(line: str):
        trace_log.append(line)
        if trace_callback:
            trace_callback(line)

    log("=== PDF Agent Decision Start ===")
    log(f"Question: {query}")
    log("Mode: LangChain ReAct agent (LLM-driven tool selection)")

    state = AgentState(original_query=query)

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
            agent_trace="\n".join(trace_log)
        )

    tools = _build_tools(state, search_func, llm, deployment_name, top_k, log)

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
        sufficiency = _check_evidence_sufficiency(query, state.all_hits, llm, deployment_name)
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

    final_hits = state.all_hits[:max(top_k, 5)]
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
        agent_trace="\n".join(trace_log)
    )