"""
Extract and analyze images from PDFs:
- PyMuPDF (fitz) for raster image extraction and whole-page rendering of vector figures
- GPT-4o Vision for understanding image content (NOT just OCR)
- Caption injection: detects nearby `Figure N – …` / `Table N – …` lines via PyMuPDF text
  blocks (with y-coordinates) and feeds the matched caption into the vision prompt so the
  description chunk literally starts with the figure title — making it directly retrievable
  by BM25 when a user queries the figure by name.
- Vector-figure fallback: pages that mention a figure but have no raster image (vector
  diagrams, drawn-with-PDF-graphics figures) are rendered to PNG and analyzed too.
"""

import io
import re
import base64
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
from PIL import Image
import fitz  # PyMuPDF
from openai import AzureOpenAI


# Anchored at line start so passing references like "see Figure 5" don't qualify.
# Group 1: keyword. Group 2: number (e.g. "4" or "4.1"). Group 3: rest of caption.
CAPTION_PATTERN = re.compile(
    r'^\s*(Figure|Fig\.?|Table|Diagram)\s+(\d+(?:\.\d+)?)\s*[–—\-:]\s*(.+?)\s*$',
    re.IGNORECASE
)

# When we see a caption first-line, we look at subsequent lines in the same PyMuPDF block
# to capture wrapped caption text (e.g. "Figure 6 – UML class diagram: Overview of the\n
# implementation model for the <Theme Name> application schema"). We stop appending once a
# continuation line starts with one of these — almost certainly the next paragraph or section.
_CAPTION_CONTINUATION_BREAK_PREFIXES = re.compile(
    r'^\s*(Figure|Fig\.?|Table|Diagram|Section|Annex|Recommendation|Note|Example|'
    r'\d+(?:\.\d+)*\s+[A-Z]|[A-Z]\.\d+)',
    re.IGNORECASE
)
_MAX_CAPTION_CONTINUATION_LINES = 3
_MAX_CAPTION_TOTAL_CHARS = 400


@dataclass
class ImageInfo:
    """Information about an extracted image (raster) or rendered page (vector)."""
    image_index: int
    page_number: int                  # 1-indexed
    bbox: Tuple[float, float, float, float]   # x0, y0, x1, y1 on the page
    width: int
    height: int
    image_bytes: bytes
    format: str = "png"
    is_full_page: bool = False        # True when this is a whole-page render for a vector figure


@dataclass
class CaptionInfo:
    """A `Figure N – …` / `Table N – …` line detected in the PDF, with its position."""
    page_number: int                  # 1-indexed
    y_top: float                      # top-y of the caption line on its page
    figure_id: str                    # "Figure 4" / "Table 3"
    full_text: str                    # full caption line as it appears


@dataclass
class ImageAnalysisResult:
    """Result of vision analysis."""
    image_index: int
    page_number: int
    description: str
    content_type: str                 # 'diagram', 'chart', 'photo', 'flowchart', 'other', …
    key_elements: List[str]
    text_detected: str
    confidence: str                   # 'high' / 'medium' / 'low'
    matched_caption: Optional[str] = None  # the caption we injected into the prompt, if any


# ─────────────────────────────────────────────────────────────────────────────────
# Caption detection (PyMuPDF text blocks → caption-line regex with y-positions)
# ─────────────────────────────────────────────────────────────────────────────────

def find_captions_with_positions(pdf_path: str) -> List[CaptionInfo]:
    """Walk every page, parse text blocks, return all caption lines with their y-positions.

    Captures wrapped multi-line captions: when a line matches the caption regex, we append
    subsequent lines from the same PyMuPDF block until we hit a structural break (next caption,
    next section/annex/recommendation, all-caps heading, or 3+ extra lines / 400+ total chars).
    """
    captions: List[CaptionInfo] = []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[WARN] caption scan: cannot open {pdf_path}: {e}")
        return captions
    try:
        for page_idx in range(len(doc)):
            page = doc[page_idx]
            try:
                blocks = page.get_text("dict").get("blocks", [])
            except Exception:
                continue
            for b in blocks:
                if b.get("type") != 0:
                    continue
                lines = b.get("lines", [])
                line_texts = [
                    "".join(span.get("text", "") for span in ln.get("spans", [])).strip()
                    for ln in lines
                ]
                for li, line_text in enumerate(line_texts):
                    if not line_text:
                        continue
                    m = CAPTION_PATTERN.match(line_text)
                    if not m:
                        continue

                    # Try to extend through continuation lines within the same block.
                    full_text = line_text
                    for offset in range(1, _MAX_CAPTION_CONTINUATION_LINES + 1):
                        nxt_idx = li + offset
                        if nxt_idx >= len(line_texts):
                            break
                        nxt = line_texts[nxt_idx]
                        if not nxt:
                            break
                        if _CAPTION_CONTINUATION_BREAK_PREFIXES.match(nxt):
                            break
                        if len(full_text) + 1 + len(nxt) > _MAX_CAPTION_TOTAL_CHARS:
                            break
                        full_text = full_text + " " + nxt

                    captions.append(CaptionInfo(
                        page_number=page_idx + 1,
                        y_top=lines[li]["bbox"][1],
                        figure_id=f"{m.group(1)} {m.group(2)}",
                        full_text=full_text,
                    ))
    finally:
        doc.close()
    return captions


def match_caption_for_image(image: ImageInfo, captions: List[CaptionInfo]) -> Tuple[Optional[CaptionInfo], List[CaptionInfo]]:
    """
    Pick the most likely caption for an image, plus a short list of alternates GPT-4o
    can fall back on if the visual content disagrees.

    Heuristic:
      Signal 1 — first `Figure N – …` line on the same page whose top-y is below the
                 image's bottom-y (figures with captions directly below them).
      Fallback — first caption at the top of the next page (cross-page captions).
      For full-page renders (vector figures), use the first caption on the page.

    Returns (primary, alternates). `alternates` is at most 3 captions from page ±1.
    """
    img_y_bottom = image.bbox[3]
    img_page = image.page_number

    if image.is_full_page:
        same_page = [c for c in captions if c.page_number == img_page]
        same_page.sort(key=lambda c: c.y_top)
        primary = same_page[0] if same_page else None
        alts_pool = [c for c in captions if c.page_number in (img_page - 1, img_page, img_page + 1) and c is not primary]
        return primary, alts_pool[:3]

    same_page_below = sorted(
        (c for c in captions if c.page_number == img_page and c.y_top > img_y_bottom),
        key=lambda c: c.y_top
    )
    primary: Optional[CaptionInfo] = same_page_below[0] if same_page_below else None

    if primary is None:
        next_page_caps = sorted(
            (c for c in captions if c.page_number == img_page + 1),
            key=lambda c: c.y_top
        )
        if next_page_caps:
            primary = next_page_caps[0]

    alts_pool = [c for c in captions if c.page_number in (img_page, img_page + 1) and c is not primary]
    return primary, alts_pool[:3]


# ─────────────────────────────────────────────────────────────────────────────────
# Main analyzer
# ─────────────────────────────────────────────────────────────────────────────────

class PDFImageAnalyzer:
    def __init__(
        self,
        openai_client: AzureOpenAI,
        deployment_name: str = "gpt-4o",
        min_image_size: int = 100,
        max_images_per_page: int = 10,
        page_render_zoom: float = 2.0,   # 2x scale for legible vector renders
    ):
        self.llm = openai_client
        self.deployment = deployment_name
        self.min_image_size = min_image_size
        self.max_images_per_page = max_images_per_page
        self.page_render_zoom = page_render_zoom

    # ── Raster image extraction (unchanged behaviour) ───────────────────────────

    def extract_images_from_pdf(self, pdf_path: str, page_numbers: Optional[List[int]] = None) -> List[ImageInfo]:
        images: List[ImageInfo] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[ERROR] Failed to open PDF {pdf_path}: {e}")
            return images
        try:
            pages_to_process = (
                [p for p in page_numbers if 0 <= p < len(doc)] if page_numbers else range(len(doc))
            )
            global_image_index = 0
            for page_num in pages_to_process:
                page = doc[page_num]
                image_list = page.get_images(full=True)[: self.max_images_per_page]
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        width, height = img_pil.size
                        if width < self.min_image_size or height < self.min_image_size:
                            continue
                        img_instances = page.get_image_rects(xref)
                        bbox = img_instances[0] if img_instances else (0, 0, width, height)
                        if hasattr(bbox, "x0"):
                            bbox_t = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
                        else:
                            bbox_t = bbox
                        images.append(ImageInfo(
                            image_index=global_image_index,
                            page_number=page_num + 1,
                            bbox=bbox_t,
                            width=width,
                            height=height,
                            image_bytes=image_bytes,
                            format=image_ext,
                            is_full_page=False,
                        ))
                        global_image_index += 1
                    except Exception as e:
                        print(f"[WARN] Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
        finally:
            doc.close()
        return images

    # ── Vision call (with caption injection) ────────────────────────────────────

    def analyze_image(
        self,
        image_info: ImageInfo,
        primary_caption: Optional[CaptionInfo] = None,
        alt_captions: Optional[List[CaptionInfo]] = None,
        surrounding_text: str = "",
    ) -> ImageAnalysisResult:
        try:
            image_base64 = base64.b64encode(image_info.image_bytes).decode("utf-8")
            prompt = self._build_analysis_prompt(
                primary_caption=primary_caption,
                alt_captions=alt_captions or [],
                surrounding_text=surrounding_text,
                is_full_page=image_info.is_full_page,
            )
            response = self.llm.chat.completions.create(
                model=self.deployment,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_info.format};base64,{image_base64}",
                                "detail": "high",
                            },
                        },
                    ],
                }],
                max_tokens=2000,
                temperature=0.0,
            )
            analysis_text = response.choices[0].message.content.strip()
            result = self._parse_analysis_response(
                analysis_text, image_info.image_index, image_info.page_number
            )
            if primary_caption:
                result.matched_caption = primary_caption.full_text
            return result
        except Exception as e:
            print(f"[ERROR] Image analysis failed for image {image_info.image_index}: {e}")
            return ImageAnalysisResult(
                image_index=image_info.image_index,
                page_number=image_info.page_number,
                description=f"[Image analysis failed: {str(e)}]",
                content_type="other",
                key_elements=[],
                text_detected="",
                confidence="low",
            )

    def _build_analysis_prompt(
        self,
        primary_caption: Optional[CaptionInfo],
        alt_captions: List[CaptionInfo],
        surrounding_text: str,
        is_full_page: bool,
    ) -> str:
        parts: List[str] = []

        if is_full_page:
            parts.append(
                "This is a whole-page render of a PDF page. The page contains a figure or "
                "diagram drawn with vector graphics, plus surrounding text (page header, "
                "paragraphs). Describe ONLY the figure/diagram visible on this page; ignore "
                "the page header, footer, and unrelated paragraph text."
            )
            parts.append("")

        if primary_caption:
            parts.append(
                f"Most likely caption (matched by position — first 'Figure N – …' below the image):"
            )
            parts.append(f'  "{primary_caption.full_text}"  (page {primary_caption.page_number})')
            parts.append("")
            if alt_captions:
                parts.append(
                    "Other candidate captions on adjacent pages (use these only if the primary "
                    "match above does NOT visually match what you see):"
                )
                for c in alt_captions:
                    parts.append(f'  - "{c.full_text}"  (page {c.page_number})')
                parts.append("")

        if surrounding_text:
            parts.append("Document text adjacent to this image (for thematic context):")
            parts.append(surrounding_text[:800].strip())
            parts.append("")

        parts.append(
            "Now analyze the image with MAXIMUM DETAIL. Provide the following six fields, "
            "each on its own labelled line."
        )
        parts.append("")

        if primary_caption:
            parts.append(
                "1. DESCRIPTION:\n"
                "   BEGIN your description with the figure caption you can confidently identify\n"
                "   from the candidates above (start with the literal text 'Figure N – …' or\n"
                "   'Table N – …' as written). If NONE of the candidate captions visually matches\n"
                "   what you see, say so explicitly (e.g. \"This image is NOT 'Figure 5 – …' as\n"
                "   suggested; what is actually shown is …\") and then describe what is visible.\n"
                "   After the caption sentence, give a comprehensive description including:"
            )
        else:
            parts.append(
                "1. DESCRIPTION: A VERY comprehensive description including:"
            )
        parts.append(
            "   - What type of visual it is (diagram, chart, flowchart, etc.)\n"
            "   - The main subject or topic\n"
            "   - ALL visible boxes, sections, or components with their EXACT text labels\n"
            "   - RELATIONSHIPS: How elements connect to each other (e.g., 'Box A flows to Box B via an arrow')\n"
            "   - ARROWS: What each arrow represents (flow, dependency, relationship, etc.)\n"
            "   - HIERARCHY: Top-to-bottom or left-to-right structure\n"
            "   - GROUPINGS: Any sections divided by lines, colors, or labels\n"
            "   - ANNOTATIONS: Any side notes, labels, or explanatory text\n"
            "   IMPORTANT: Don't just list components — EXPLAIN how they relate to each other."
        )
        parts.append("")
        parts.append(
            "2. CONTENT_TYPE: Classify as one of: diagram, chart, photo, table, schematic, "
            "flowchart, screenshot, architecture_diagram, other"
        )
        parts.append("")
        parts.append("3. KEY_ELEMENTS: List ALL visible text labels, boxes, and components (comma-separated).")
        parts.append("")
        parts.append("4. TEXT_DETECTED: Extract ALL readable text EXACTLY as it appears.")
        parts.append("")
        parts.append("5. STRUCTURE: Describe levels, dividers, arrow meaning, branches.")
        parts.append("")
        parts.append("6. CONFIDENCE: Your confidence in this analysis (high/medium/low).")
        parts.append("")
        parts.append("Format your response as:")
        parts.append("DESCRIPTION: [comprehensive description, beginning with the figure caption]")
        parts.append("CONTENT_TYPE: [type]")
        parts.append("KEY_ELEMENTS: [element1, element2, …]")
        parts.append("TEXT_DETECTED: [all readable text]")
        parts.append("STRUCTURE: [visual structure explanation]")
        parts.append("CONFIDENCE: [high/medium/low]")
        return "\n".join(parts)

    def _parse_analysis_response(self, response_text: str, image_index: int, page_number: int) -> ImageAnalysisResult:
        # Clean up markdown markers GPT-4o sometimes emits.
        response_text = re.sub(r'\*\*+', '', response_text)
        response_text = re.sub(r'^\s*-\s*', '', response_text, flags=re.MULTILINE)
        response_text = re.sub(r'--\s*###\s*', '', response_text)
        response_text = re.sub(r'###\s*', '', response_text)
        response_text = re.sub(r'--\s*', '', response_text)

        description, content_type, text_detected = "", "other", ""
        key_elements: List[str] = []
        confidence = "medium"

        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if desc_match:
            description = re.sub(r'\s+', ' ', desc_match.group(1)).strip()
            description = re.sub(r'CONTENT_TYPE:|KEY_ELEMENTS:|TEXT_DETECTED:|STRUCTURE:|CONFIDENCE:', '', description).strip()

        type_match = re.search(r'CONTENT_TYPE:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if type_match:
            content_type = re.sub(r'[^\w\s-]', '', type_match.group(1)).strip().lower()

        elements_match = re.search(r'KEY_ELEMENTS:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if elements_match:
            raw = re.sub(r'\s+', ' ', elements_match.group(1).strip())
            key_elements = [re.sub(r'["\']', '', e).strip() for e in raw.split(',')]
            key_elements = [e for e in key_elements if e and len(e) > 1]

        text_match = re.search(r'TEXT_DETECTED:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if text_match:
            text_detected = re.sub(r'\s+', ' ', text_match.group(1).strip())
            text_detected = re.sub(r'["""]', '"', text_detected)

        structure_match = re.search(r'STRUCTURE:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if structure_match:
            structure = re.sub(r'\s+', ' ', structure_match.group(1).strip())
            if structure and structure not in description:
                description = description + " STRUCTURE: " + structure

        conf_match = re.search(r'CONFIDENCE:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).strip().lower()

        if not description:
            description = response_text

        return ImageAnalysisResult(
            image_index=image_index,
            page_number=page_number,
            description=description,
            content_type=content_type,
            key_elements=key_elements,
            text_detected=text_detected,
            confidence=confidence,
        )

    # ── High-level entry points ─────────────────────────────────────────────────

    def analyze_all_images(
        self,
        pdf_path: str,
        page_text_map: Optional[Dict[int, str]] = None,
        captions: Optional[List[CaptionInfo]] = None,
        page_numbers: Optional[List[int]] = None,
    ) -> List[ImageAnalysisResult]:
        """Extract every raster image, match each to its caption, run vision."""
        page_text_map = page_text_map or {}
        captions = captions if captions is not None else find_captions_with_positions(pdf_path)

        images = self.extract_images_from_pdf(pdf_path, page_numbers)
        if not images:
            print("[INFO] No raster images found in PDF")
            return []
        print(f"[INFO] Found {len(images)} raster images, analyzing with caption injection…")

        results: List[ImageAnalysisResult] = []
        for img_info in images:
            primary, alts = match_caption_for_image(img_info, captions)
            surrounding = page_text_map.get(img_info.page_number, "")
            result = self.analyze_image(
                img_info,
                primary_caption=primary,
                alt_captions=alts,
                surrounding_text=surrounding,
            )
            results.append(result)
            cap_msg = f" (caption: {primary.figure_id})" if primary else ""
            print(f"[INFO] Analyzed raster image {img_info.image_index + 1}/{len(images)} on page {img_info.page_number}{cap_msg}")
        return results

    def analyze_vector_figures(
        self,
        pdf_path: str,
        raster_pages: Set[int],
        captions: List[CaptionInfo],
        page_text_map: Dict[int, str],
        starting_index: int = 0,
    ) -> List[ImageAnalysisResult]:
        """
        For pages that have a `Figure N – …` / `Diagram N – …` caption AND no raster image was
        extracted from them AND their previous page has no raster image either (so the figure
        isn't already captured), render the page to PNG and run vision on it.

        Tables are intentionally skipped here: Azure Document Intelligence already extracts
        `[[TABLE …]]` chunks (chunk_type=table, source_type=original_text) with the real cell
        data, so re-describing them with vision would duplicate content and waste API calls.
        """
        captions_by_page: Dict[int, List[CaptionInfo]] = {}
        for c in captions:
            if c.figure_id.lower().startswith("table"):
                continue
            captions_by_page.setdefault(c.page_number, []).append(c)

        results: List[ImageAnalysisResult] = []
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"[WARN] vector-figure render: cannot open {pdf_path}: {e}")
            return results

        try:
            img_idx = starting_index
            for page_no in sorted(captions_by_page.keys()):
                if page_no in raster_pages:
                    continue                     # raster image on this page already covers it
                if (page_no - 1) in raster_pages:
                    continue                     # caption is just continuation of prev page's figure
                if not (1 <= page_no <= len(doc)):
                    continue
                page = doc[page_no - 1]
                page_caps = sorted(captions_by_page[page_no], key=lambda c: c.y_top)
                primary = page_caps[0]
                alts = page_caps[1:][:3]
                try:
                    pix = page.get_pixmap(matrix=fitz.Matrix(self.page_render_zoom, self.page_render_zoom))
                    png_bytes = pix.tobytes("png")
                except Exception as e:
                    print(f"[WARN] page render failed for page {page_no}: {e}")
                    continue
                virtual_img = ImageInfo(
                    image_index=img_idx,
                    page_number=page_no,
                    bbox=(0.0, 0.0, float(page.rect.width), float(page.rect.height)),
                    width=int(pix.width),
                    height=int(pix.height),
                    image_bytes=png_bytes,
                    format="png",
                    is_full_page=True,
                )
                img_idx += 1
                surrounding = page_text_map.get(page_no, "")
                result = self.analyze_image(
                    virtual_img,
                    primary_caption=primary,
                    alt_captions=alts,
                    surrounding_text=surrounding,
                )
                results.append(result)
                print(f"[INFO] Analyzed vector figure on page {page_no} ({primary.figure_id})")
        finally:
            doc.close()
        return results


# ─────────────────────────────────────────────────────────────────────────────────
# Chunk text formatting (page-based label, no numeric collision with `Figure N`)
# ─────────────────────────────────────────────────────────────────────────────────

def format_image_analysis_as_complete_text(result: ImageAnalysisResult) -> str:
    """Build the chunk content stored in Azure Search.

    We label by page (`[IMAGE on page 68 - DIAGRAM]`) instead of by extraction index
    (`[IMAGE 7 - DIAGRAM]`) because the numeric form collided with `Figure N` queries
    via BM25 — `[IMAGE 7]` would top-rank for a "Figure 7" search even though the 7th
    extracted image is unrelated to Figure 7 in the PDF.
    """
    page = result.page_number
    ctype = (result.content_type or "other").upper()
    parts = [
        f"[IMAGE on page {page} - {ctype}]",
        "",
        result.description,
        "",
    ]
    if result.key_elements:
        parts.append(f"Key elements: {', '.join(result.key_elements)}")
        parts.append("")
    if result.text_detected:
        parts.append(f"Text in image: {result.text_detected}")
        parts.append("")
    parts.append(f"[/IMAGE on page {page}]")
    return "\n".join(parts)
