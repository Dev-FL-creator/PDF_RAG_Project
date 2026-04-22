"""
PDF RAG Agent Engine for Enhanced Document Analysis
====================================================

This module provides an intelligent agent for PDF document question answering
with adaptive retrieval strategies.

Key Features:
- Query quality assessment
- Query rewriting and optimization
- Adaptive retrieval with broadening strategies
- Evidence sufficiency evaluation
- Reasoning-based answer generation
"""

import re
from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel, Field
from openai import AzureOpenAI

# ============================================================
# Configuration & Constants
# ============================================================

MIN_HITS = 2  # Minimum number of evidence hits required
MIN_SCORE_FLOOR = 0.70  # Minimum similarity score floor
MAX_RETRIEVAL_ATTEMPTS = 5  # Maximum retrieval broadening attempts

# Fuzzy/vague terms that indicate poor query quality
VAGUE_TERMS = [
    "something", "anything", "things", "stuff", "any", "some",
    "various", "several", "many", "few", "multiple", "general",
    "overall", "basic", "detailed", "information", "data", "details"
]

# Domain-specific context
DOMAIN_CONTEXT = """
DOMAIN: Technical Documentation Analysis

Common topics include:
- System specifications and requirements
- Technical standards and compliance
- Testing and validation procedures
- Environmental conditions and constraints
- System architecture and components
"""

# ============================================================
# Pydantic Data Structures
# ============================================================

class QueryQualityResult(BaseModel):
    """Result of query quality check"""
    is_clear: bool = Field(..., description="Whether the query is clear and specific")
    issues: List[str] = Field(default_factory=list, description="List of quality issues found")
    needs_rewrite: bool = Field(False, description="Whether query needs rewriting")


class QueryRewriteResult(BaseModel):
    """Result of query rewriting"""
    strategy: str = Field(..., description="Strategy used: 'clarify', 'expand', or 'simplify'")
    rewritten_queries: List[str] = Field(..., description="List of rewritten queries")


class EvidenceSufficiencyResult(BaseModel):
    """Result of evidence sufficiency check"""
    is_sufficient: bool = Field(..., description="Whether evidence is sufficient")
    reason: str = Field("", description="Explanation of sufficiency judgment")
    confidence: str = Field("", description="Confidence level: high, medium, low")


class AnswerResult(BaseModel):
    """Result of answer generation"""
    answer: str = Field("", description="Generated answer")
    reasoning: str = Field("", description="Reasoning for the answer")
    confidence: str = Field("", description="Confidence level")


class PDFAgentResult(BaseModel):
    """Final output from PDF Agent"""
    final_answer: str = Field("", description="Final answer")
    reasoning: str = Field("", description="Reasoning trace")
    evidence_used: List[Dict] = Field(default_factory=list, description="Evidence chunks used")
    query_rewrites: List[str] = Field(default_factory=list, description="Query rewrites attempted")
    agent_trace: str = Field("", description="Agent reasoning trace (for logging)")


# ============================================================
# Tool Functions
# ============================================================

def check_query_quality_impl(query: str, llm: AzureOpenAI, deployment: str) -> QueryQualityResult:
    """Tool: Check if query is clear and specific enough"""
    issues = []
    query_lower = query.lower()
    found_vague = []
    
    for term in VAGUE_TERMS:
        if f" {term} " in f" {query_lower} " or query_lower.startswith(f"{term} ") or query_lower.endswith(f" {term}"):
            found_vague.append(term)
    
    if len(found_vague) >= 2:
        issues.append(f"vague_terms: {', '.join(found_vague)}")
    
    word_count = len(query.split())
    if word_count < 2:
        issues.append("too_short")
    
    if not issues:
        return QueryQualityResult(is_clear=True, issues=[], needs_rewrite=False)
    
    try:
        prompt = f"""Analyze the following question for clarity and specificity.

Question: "{query}"

Is this question clear, specific, and answerable from technical documentation?

Respond ONLY with:
- "CLEAR" - if the question is understandable and answerable
- "UNCLEAR: [brief reason]" - only if the question is truly confusing

Be lenient - mark as CLEAR unless the question is genuinely problematic."""

        completion = llm.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a lenient query analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        response = completion.choices[0].message.content.strip()
        
        if response.startswith("UNCLEAR"):
            is_clear = False
            if ":" in response:
                llm_issues = response.split(":", 1)[1].strip()
                if llm_issues:
                    issues.append(f"llm_judgment: {llm_issues}")
        else:
            is_clear = True
            issues = []
            
    except Exception as e:
        print(f"[WARNING] LLM quality check failed: {e}")
        is_clear = len(issues) <= 1
    
    return QueryQualityResult(is_clear=is_clear, issues=issues, needs_rewrite=not is_clear)


def rewrite_query_impl(
    query: str, 
    issues: List[str], 
    llm: AzureOpenAI, 
    deployment: str,
    initial_hits: Optional[List[Dict]] = None
) -> QueryRewriteResult:
    """Tool: Rewrite query to be clearer and more specific"""
    
    retrieval_context = ""
    if initial_hits:
        retrieval_context = "\n\nINITIAL RETRIEVAL RESULTS:\n"
        if len(initial_hits) == 0:
            retrieval_context += "No relevant documents found. Consider broadening the query.\n"
        else:
            retrieval_context += f"Found {len(initial_hits)} document(s). Sample content:\n"
            for idx, hit in enumerate(initial_hits[:3], 1):
                source = hit.get("source", "Unknown")
                page = hit.get("page", "?")
                content_preview = hit.get("content", "")[:150].replace("\n", " ")
                retrieval_context += f"{idx}. [{source} p.{page}] {content_preview}...\n"
    
    prompt = f"""You are a query optimization assistant.

{DOMAIN_CONTEXT}

Your task is to rewrite the following question to make it clearer and more specific.

ORIGINAL QUESTION: {query}

IDENTIFIED ISSUES: {', '.join(issues)}
{retrieval_context}

INSTRUCTIONS:
1. MAINTAIN THE CORE TOPIC AND INTENT
2. Add technical context if helpful
3. Replace vague terms with specific terminology
4. If too broad, break into 2-3 focused sub-questions
5. Use terminology from retrieval results if provided

Format your response as:
REWRITE: [your rewritten question 1]
REWRITE: [your rewritten question 2] (if applicable)"""

    try:
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": "You are a precise query rewriting assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        response = completion.choices[0].message.content.strip()
        
        rewritten = []
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("REWRITE:"):
                rewritten_query = line.replace("REWRITE:", "").strip()
                if rewritten_query:
                    rewritten.append(rewritten_query)
        
        if not rewritten:
            rewritten = [query]
            
        strategy = "expand" if len(rewritten) > 1 else "clarify"
        
        return QueryRewriteResult(strategy=strategy, rewritten_queries=rewritten[:3])
        
    except Exception as e:
        print(f"[WARNING] Query rewrite failed: {e}")
        return QueryRewriteResult(strategy="fallback", rewritten_queries=[query])


def check_evidence_sufficiency_impl(
    query: str,
    hits: List[Dict],
    llm: AzureOpenAI,
    deployment: str
) -> EvidenceSufficiencyResult:
    """Tool: Check if retrieved evidence is sufficient"""
    
    if not hits:
        return EvidenceSufficiencyResult(
            is_sufficient=False,
            reason="No evidence found",
            confidence="high"
        )
    
    evidence_summary = ""
    for idx, hit in enumerate(hits[:5], 1):
        content = hit.get("content", "")[:200]
        source = hit.get("source", "Unknown")
        page = hit.get("page", "?")
        evidence_summary += f"\n{idx}. [{source} p.{page}] {content}...\n"
    
    prompt = f"""Evaluate if the retrieved document excerpts are sufficient to answer the question.

Question: {query}

Retrieved Evidence:
{evidence_summary}

Is this evidence sufficient? Consider:
1. Does it directly address the question?
2. Is there enough context and detail?

Respond with:
- "SUFFICIENT" if adequate
- "INSUFFICIENT: [reason]" if more needed"""

    try:
        completion = llm.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        response = completion.choices[0].message.content.strip()
        
        if response.startswith("INSUFFICIENT"):
            is_sufficient = False
            reason = response.split(":", 1)[1].strip() if ":" in response else "Evidence gaps identified"
            confidence = "high"
        else:
            is_sufficient = len(hits) >= MIN_HITS
            reason = "Evidence addresses the question" if is_sufficient else f"Only {len(hits)} hit(s)"
            confidence = "high" if len(hits) >= 3 else "medium"
            
    except Exception as e:
        print(f"[WARNING] Evidence sufficiency check failed: {e}")
        is_sufficient = len(hits) >= MIN_HITS
        reason = f"Found {len(hits)} evidence chunks"
        confidence = "medium"
    
    return EvidenceSufficiencyResult(
        is_sufficient=is_sufficient,
        reason=reason,
        confidence=confidence
    )


def generate_answer_impl(
    query: str,
    hits: List[Dict],
    llm: AzureOpenAI,
    deployment: str
) -> AnswerResult:
    """Tool: Generate intelligent answer from evidence"""
    
    if not hits:
        return AnswerResult(
            answer="I could not find relevant information in the document to answer this question.",
            reasoning="No matching excerpts were retrieved.",
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
    
    prompt = f"""You are an intelligent document analysis agent. Provide a comprehensive answer based on the retrieved evidence.

Retrieved Document Excerpts:
{chr(10).join(['---', *contexts, '---'])}

Question: {query}

Provide your answer in this structure:

**Evidence Found:**
[Summarize what the document explicitly states]

**Evidence Missing:**
[Point out what is not specified or unclear]

**Reasoning:**
[Explain your thought process]

**Final Answer:**
[Provide a clear, concise answer]

Important: Base your answer ONLY on the retrieved excerpts."""

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
# Main Agent Function
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
    """
    Main PDF RAG Agent with adaptive retrieval strategies
    
    Args:
        query: User's question
        search_func: Function to search index: (query_text, top_k) -> List[Dict]
        llm: Azure OpenAI client
        deployment_name: Model deployment name
        top_k: Initial top-K for retrieval
        min_score: Initial minimum score threshold
        trace_callback: Optional callback for real-time trace logging
    
    Returns:
        PDFAgentResult with answer and reasoning trace
    """
    trace_log = []
    
    def log_trace(line: str):
        """Log trace line"""
        trace_log.append(line)
        if trace_callback:
            trace_callback(line)
    
    log_trace("=== PDF Agent Decision Start ===")
    log_trace(f"Question: {query}")
    log_trace("")
    
    # Step 1: Query Quality Check
    log_trace("[Step 1] Query Quality Assessment")
    quality_result = check_query_quality_impl(query, llm, deployment_name)
    log_trace(f"  is_clear: {quality_result.is_clear}")
    if quality_result.issues:
        log_trace(f"  issues: {quality_result.issues}")
    
    queries_to_try = [query]
    query_rewrites = []
    
    # Step 2: Initial Retrieval
    log_trace("")
    log_trace("[Step 2] Initial Retrieval")
    log_trace(f"  Performing initial search with original query...")
    initial_hits = search_func(query, top_k)
    log_trace(f"  Retrieved: {len(initial_hits)} chunks")
    
    if initial_hits:
        for i, h in enumerate(initial_hits[:3], 1):
            src = h.get("source", "Unknown")
            page = h.get("page", "?")
            content_preview = h.get("content", "")[:80].replace("\n", " ")
            log_trace(f"    {i}. [{src} p.{page}] {content_preview}...")
    else:
        log_trace("    ⚠️ No results found with original query")
    
    # Step 3: Query Rewriting if needed
    if quality_result.needs_rewrite or len(initial_hits) == 0:
        log_trace("")
        log_trace("[Step 3] Query Rewriting")
        
        if quality_result.needs_rewrite:
            log_trace(f"  Reason: Query has quality issues")
        if len(initial_hits) == 0:
            log_trace(f"  Reason: Initial retrieval returned no results")
        
        log_trace(f"  Using initial retrieval results to inform rewrite...")
        
        rewrite_result = rewrite_query_impl(
            query, 
            quality_result.issues if quality_result.issues else ["no_results"],
            llm, 
            deployment_name,
            initial_hits=initial_hits
        )
        log_trace(f"  strategy: {rewrite_result.strategy}")
        log_trace(f"  Generated {len(rewrite_result.rewritten_queries)} variation(s):")
        for i, rq in enumerate(rewrite_result.rewritten_queries, 1):
            log_trace(f"    {i}. {rq}")
        
        queries_to_try = rewrite_result.rewritten_queries
        query_rewrites = rewrite_result.rewritten_queries
    else:
        log_trace("")
        log_trace("[Step 3] Query Rewriting")
        log_trace("  Query quality OK and initial retrieval successful, proceeding with original")
    
    # Step 4: Adaptive Retrieval
    log_trace("")
    log_trace("[Step 4] Adaptive Retrieval")
    
    best_hits = initial_hits if not query_rewrites else []
    best_query = query
    evidence_sufficient = False
    
    if initial_hits and not query_rewrites:
        log_trace(f"  Checking sufficiency of initial retrieval results...")
        sufficiency = check_evidence_sufficiency_impl(query, initial_hits, llm, deployment_name)
        evidence_sufficient = sufficiency.is_sufficient
        log_trace(f"  LLM judgment: {evidence_sufficient} (confidence: {sufficiency.confidence})")
        log_trace(f"  Reason: {sufficiency.reason}")
        
        if evidence_sufficient:
            log_trace(f"  ✓ Initial results are sufficient, skipping further retrieval")
            best_hits = initial_hits
            best_query = query
    
    if not evidence_sufficient:
        for query_idx, current_query in enumerate(queries_to_try):
            if query_idx > 0 or query_rewrites:
                log_trace(f"\n[Step 4.{query_idx+1}] Trying {'rewritten' if query_rewrites else 'original'} query: {current_query[:80]}...")
            
            current_top_k = top_k
            attempt = 0
            
            while attempt < MAX_RETRIEVAL_ATTEMPTS:
                attempt += 1
                log_trace(f"  Retrieval attempt {attempt}: top_k={current_top_k}")
                
                if query_idx == 0 and attempt == 1 and initial_hits and not query_rewrites:
                    hits = initial_hits
                    log_trace(f"    Using cached initial results: {len(hits)} chunks")
                else:
                    hits = search_func(current_query, current_top_k)
                    log_trace(f"    Retrieved: {len(hits)} chunks")
                
                if hits:
                    for i, h in enumerate(hits[:3], 1):
                        src = h.get("source", "Unknown")
                        page = h.get("page", "?")
                        content_preview = h.get("content", "")[:80].replace("\n", " ")
                        log_trace(f"      {i}. [{src} p.{page}] {content_preview}...")
                
                log_trace(f"    Checking evidence sufficiency with LLM...")
                sufficiency = check_evidence_sufficiency_impl(current_query, hits, llm, deployment_name)
                evidence_sufficient = sufficiency.is_sufficient
                
                log_trace(f"    LLM judgment: {evidence_sufficient} (confidence: {sufficiency.confidence})")
                log_trace(f"    Reason: {sufficiency.reason}")
                
                if evidence_sufficient:
                    log_trace(f"    ✓ Evidence sufficient")
                    best_hits = hits
                    best_query = current_query
                    break
                
                log_trace(f"    ✗ Evidence insufficient, broadening...")
                
                if attempt < MAX_RETRIEVAL_ATTEMPTS:
                    current_top_k = min(current_top_k * 2, 20)
                    log_trace(f"      Increasing top_k to {current_top_k}")
                    continue
                else:
                    log_trace(f"      Max attempts reached, using current results")
                    if len(hits) > len(best_hits):
                        best_hits = hits
                        best_query = current_query
                    break
            
            if evidence_sufficient:
                break
    
    log_trace("")
    log_trace(f"[Step 5] Evidence Evaluation Summary")
    log_trace(f"  Best query used: {best_query}")
    log_trace(f"  Total evidence chunks: {len(best_hits)}")
    log_trace(f"  Evidence sufficient: {evidence_sufficient}")
    
    # Step 6: Generate Answer
    log_trace("")
    log_trace("[Step 6] Intelligent Answer Generation")
    log_trace("  🤖 Invoking LLM for reasoning-based answer...")
    
    answer_result = generate_answer_impl(best_query, best_hits, llm, deployment_name)
    
    log_trace(f"  ✓ Answer generated")
    log_trace(f"  Confidence: {answer_result.confidence}")
    log_trace("")
    log_trace("=== PDF Agent Decision End ===")
    
    return PDFAgentResult(
        final_answer=answer_result.answer,
        reasoning=answer_result.reasoning,
        evidence_used=[{
            "source": h.get("source"),
            "page": h.get("page"),
            "chunk_id": h.get("chunk_id"),
            "content": h.get("content", "")[:200]
        } for h in best_hits[:5]],
        query_rewrites=query_rewrites,
        agent_trace="\n".join(trace_log)
    )
