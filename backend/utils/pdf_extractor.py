"""
PDF Extractor Module
====================
Handles PDF text, table, image extraction using Azure Document Intelligence.
"""

from typing import List, Optional
from openai import AzureOpenAI
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from utils.image_analyzer import PDFImageAnalyzer, format_image_analysis_as_complete_text


def extract_pages_with_docint(
    pdf_path: str,
    endpoint: str,
    key: str,
    aoai: Optional[AzureOpenAI] = None,
    enable_image_analysis: bool = True
) -> List[tuple]:
    """
    Extract text, tables, and images from PDF using Azure Document Intelligence.

    This function uses Azure's prebuilt-document model to extract structured content
    from PDFs, including paragraphs, tables, and key-value pairs. It also optionally
    analyzes images using GPT-4o vision capabilities.

    Args:
        pdf_path: Path to the PDF file
        endpoint: Azure Document Intelligence endpoint URL
        key: Azure Document Intelligence API key
        aoai: Optional Azure OpenAI client for image analysis
        enable_image_analysis: Whether to analyze images with GPT-4o

    Returns:
        List of (page_number, page_content) tuples with proper page tracking
    """
    # Initialize Document Intelligence client
    client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with open(pdf_path, "rb") as f:
        poller = client.begin_analyze_document("prebuilt-document", document=f)
        result = poller.result()

    # Build page-based structure to organize content by page number
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

        # Build table structure cell by cell
        rows_map = {}
        for cell in t.cells:
            rows_map.setdefault(cell.row_index, {})
            rows_map[cell.row_index][cell.column_index] = (cell.content or "").strip()

        if rows_map:
            # Calculate table dimensions
            try:
                r_cnt = int(getattr(t, "row_count", None) or (max(rows_map.keys()) + 1))
            except Exception:
                r_cnt = max(rows_map.keys()) + 1
            try:
                c_cnt = int(getattr(t, "column_count", None) or (max(max(r.keys()) for r in rows_map.values()) + 1))
            except Exception:
                c_cnt = max(max(r.keys()) for r in rows_map.values()) + 1

            # Create TSV (tab-separated values) representation
            tsv_lines = []
            for r in range(r_cnt):
                row = rows_map.get(r, {})
                cols = [row.get(c, "") for c in range(c_cnt)]
                tsv_lines.append("\t".join(cols).rstrip())

            # Format table with markers for easy identification
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

            # Initialize GPT-4o vision analyzer
            analyzer = PDFImageAnalyzer(
                openai_client=aoai,
                deployment_name="gpt-4o",
                min_image_size=100  # Skip images smaller than 100x100 pixels
            )

            # Get document context for better image understanding
            doc_context = ""
            if hasattr(result, "content"):
                doc_context = result.content[:500]  # First 500 chars as context

            # Analyze all images in the PDF
            image_results = analyzer.analyze_all_images(pdf_path, context=doc_context)
            print(f"[INFO] Analyzed {len(image_results)} images")

            # Store image results separately - they will become standalone chunks
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
        # Fallback: use full document content if no structured content found
        full = (getattr(result, "content", "") or "").strip()
        if full:
            pages_output.append((1, full))
    else:
        # Combine content for each page in document order
        for page_num in sorted(page_contents.keys()):
            items = page_contents[page_num]

            # Sort by order (preserve document structure)
            items.sort(key=lambda x: x["order"])

            # Combine all content for this page with double newlines as separators
            page_text = "\n\n".join(item["content"] for item in items)

            if page_text.strip():
                pages_output.append((page_num, page_text.strip()))

    return pages_output
