"""
 extracts and analyzes images from PDFs using:
- PyMuPDF (fitz) for image extraction
- GPT-4o Vision for understanding image content and context
- NOT just OCR - understands diagrams, charts, photos, etc.

Key Features:
- Extract images from PDF pages with location tracking
- Analyze images using GPT-4o Vision for semantic understanding
- Generate descriptive captions and explanations
- Track which page each image came from
"""

import io
import base64
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from PIL import Image
import fitz  # PyMuPDF
from openai import AzureOpenAI


@dataclass
class ImageInfo:
    """Information about an extracted image"""
    image_index: int
    page_number: int
    bbox: Tuple[float, float, float, float]  # x0, y0, x1, y1
    width: int
    height: int
    image_bytes: bytes
    format: str = "png"


@dataclass
class ImageAnalysisResult:
    """Result of image analysis"""
    image_index: int
    page_number: int
    description: str
    content_type: str  # 'diagram', 'chart', 'photo', 'table', 'mixed', 'other'
    key_elements: List[str]
    text_detected: str  # Any text found in the image
    confidence: str  # 'high', 'medium', 'low'


class PDFImageAnalyzer:
    """
    Analyzer for extracting and understanding images in PDFs
    """
    
    def __init__(
        self,
        openai_client: AzureOpenAI,
        deployment_name: str = "gpt-4o",
        min_image_size: int = 100,
        max_images_per_page: int = 10
    ):
        """
        Initialize image analyzer
        
        Args:
            openai_client: Azure OpenAI client
            deployment_name: Model deployment name (must support vision)
            min_image_size: Minimum image dimension to process (pixels)
            max_images_per_page: Maximum images to extract per page
        """
        self.llm = openai_client
        self.deployment = deployment_name
        self.min_image_size = min_image_size
        self.max_images_per_page = max_images_per_page
    
    def extract_images_from_pdf(
        self, 
        pdf_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[ImageInfo]:
        """
        Extract images from PDF file
        
        Args:
            pdf_path: Path to PDF file
            page_numbers: Specific pages to extract from (None = all pages)
            
        Returns:
            List of ImageInfo objects
        """
        images = []
        
        try:
            doc = fitz.open(pdf_path)
            
            # Determine which pages to process
            if page_numbers:
                pages_to_process = [p for p in page_numbers if 0 <= p < len(doc)]
            else:
                pages_to_process = range(len(doc))
            
            global_image_index = 0
            
            for page_num in pages_to_process:
                page = doc[page_num]
                image_list = page.get_images(full=True)
                
                # Limit images per page
                image_list = image_list[:self.max_images_per_page]
                
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        base_image = doc.extract_image(xref)
                        
                        image_bytes = base_image["image"]
                        image_ext = base_image["ext"]
                        
                        # Check image size
                        img_pil = Image.open(io.BytesIO(image_bytes))
                        width, height = img_pil.size
                        
                        if width < self.min_image_size or height < self.min_image_size:
                            continue
                        
                        # Get image bounding box on page
                        img_instances = page.get_image_rects(xref)
                        bbox = img_instances[0] if img_instances else (0, 0, width, height)
                        
                        images.append(ImageInfo(
                            image_index=global_image_index,
                            page_number=page_num + 1,  # 1-indexed
                            bbox=(bbox.x0, bbox.y0, bbox.x1, bbox.y1) if hasattr(bbox, 'x0') else bbox,
                            width=width,
                            height=height,
                            image_bytes=image_bytes,
                            format=image_ext
                        ))
                        
                        global_image_index += 1
                        
                    except Exception as e:
                        print(f"[WARNING] Failed to extract image {img_index} from page {page_num + 1}: {e}")
                        continue
            
            doc.close()
            
        except Exception as e:
            print(f"[ERROR] Failed to extract images from PDF: {e}")
            return []
        
        return images
    
    def analyze_image(self, image_info: ImageInfo, context: str = "") -> ImageAnalysisResult:
        """
        Analyze image using GPT-4o Vision
        
        Args:
            image_info: ImageInfo object with image data
            context: Optional context about the document (helps with analysis)
            
        Returns:
            ImageAnalysisResult with analysis details
        """
        try:
            # Convert image to base64
            image_base64 = base64.b64encode(image_info.image_bytes).decode('utf-8')
            
            # Construct prompt
            prompt = self._build_analysis_prompt(context)
            
            # Call GPT-4o Vision
            response = self.llm.chat.completions.create(
                model=self.deployment,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_info.format};base64,{image_base64}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,  # Increased for very detailed structural analysis
                temperature=0.0   # Deterministic for consistent results
            )
            
            analysis_text = response.choices[0].message.content.strip()
            
            # Parse the structured response
            result = self._parse_analysis_response(
                analysis_text, 
                image_info.image_index, 
                image_info.page_number
            )
            
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
                confidence="low"
            )
    
    def _build_analysis_prompt(self, context: str = "") -> str:
        """Build prompt for image analysis"""
        base_prompt = """Analyze this image from a technical document with MAXIMUM DETAIL. You must provide:

1. DESCRIPTION: A VERY comprehensive description including:
   - What type of visual it is (diagram, chart, flowchart, etc.)
   - The main subject or topic
   - ALL visible boxes, sections, or components with their EXACT text labels
   - RELATIONSHIPS: How elements connect to each other (e.g., "Box A flows to Box B via an arrow")
   - ARROWS: What each arrow represents (flow, dependency, relationship, etc.)
   - HIERARCHY: Top-to-bottom or left-to-right structure
   - GROUPINGS: Any sections divided by lines, colors, or labels (e.g., "upper section labeled 'X', lower section labeled 'Y'")
   - ANNOTATIONS: Any side notes, labels, or explanatory text
   
   IMPORTANT: Don't just list components - EXPLAIN how they relate to each other!

2. CONTENT_TYPE: Classify as one of: diagram, chart, photo, table, schematic, flowchart, screenshot, architecture_diagram, other

3. KEY_ELEMENTS: List ALL visible text labels, boxes, and components (comma-separated). Be exhaustive.

4. TEXT_DETECTED: Extract ALL readable text EXACTLY as it appears, including:
   - Titles and headings
   - Labels on ALL boxes and components
   - Annotations and notes
   - Side labels and categories
   - Any descriptive text
   
5. STRUCTURE: Describe the visual structure:
   - How many levels or layers?
   - What divides the sections (dashed lines, solid lines, colors)?
   - What do arrows indicate?
   - Are there parallel branches or sequential steps?

6. CONFIDENCE: Your confidence in this analysis (high/medium/low)

CRITICAL: Your goal is to help someone UNDERSTAND the diagram without seeing it. Explain relationships, not just components!

Format your response as:
DESCRIPTION: [comprehensive description explaining relationships and structure]
CONTENT_TYPE: [type]
KEY_ELEMENTS: [element1, element2, element3, ...]
TEXT_DETECTED: [all readable text]
STRUCTURE: [visual structure explanation]
CONFIDENCE: [high/medium/low]"""
        
        if context:
            base_prompt = f"Document context: {context}\n\n" + base_prompt
        
        return base_prompt
    
    def _parse_analysis_response(
        self, 
        response_text: str, 
        image_index: int, 
        page_number: int
    ) -> ImageAnalysisResult:
        """Parse structured response from LLM"""
        
        # Extract fields using regex
        import re
        
        # Clean up markdown and formatting first
        response_text = re.sub(r'\*\*+', '', response_text)  # Remove ** markers
        response_text = re.sub(r'^\s*-\s*', '', response_text, flags=re.MULTILINE)  # Remove bullet points
        response_text = re.sub(r'--\s*###\s*', '', response_text)  # Remove -- ### markers
        response_text = re.sub(r'###\s*', '', response_text)  # Remove ### markers
        response_text = re.sub(r'--\s*', '', response_text)  # Remove -- markers
        
        description = ""
        content_type = "other"
        key_elements = []
        text_detected = ""
        confidence = "medium"
        
        # Parse DESCRIPTION
        desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if desc_match:
            description = desc_match.group(1).strip()
            # Clean description - remove multiple spaces and newlines
            description = re.sub(r'\s+', ' ', description)
            # Remove any remaining field markers that might be embedded
            description = re.sub(r'CONTENT_TYPE:|KEY_ELEMENTS:|TEXT_DETECTED:|STRUCTURE:|CONFIDENCE:', '', description)
            description = description.strip()
        
        # Parse CONTENT_TYPE
        type_match = re.search(r'CONTENT_TYPE:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if type_match:
            content_type = type_match.group(1).strip().lower()
            # Remove any extra markers
            content_type = re.sub(r'[^\w\s-]', '', content_type).strip()
        
        # Parse KEY_ELEMENTS
        elements_match = re.search(r'KEY_ELEMENTS:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if elements_match:
            elements_str = elements_match.group(1).strip()
            # Clean and split
            elements_str = re.sub(r'\s+', ' ', elements_str)
            key_elements = [e.strip() for e in elements_str.split(',') if e.strip()]
            # Remove quotes and extra markers
            key_elements = [re.sub(r'["\']', '', e).strip() for e in key_elements]
            key_elements = [e for e in key_elements if e and len(e) > 1]  # Remove empty and single chars
        
        # Parse TEXT_DETECTED
        text_match = re.search(r'TEXT_DETECTED:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if text_match:
            text_detected = text_match.group(1).strip()
            # Clean up formatting
            text_detected = re.sub(r'\s+', ' ', text_detected)
            # Remove quotes
            text_detected = re.sub(r'["""]', '"', text_detected)
        
        # Parse STRUCTURE
        structure_match = re.search(r'STRUCTURE:\s*(.+?)(?=\n[A-Z_]+:|$)', response_text, re.DOTALL | re.IGNORECASE)
        if structure_match:
            structure = structure_match.group(1).strip()
            structure = re.sub(r'\s+', ' ', structure)
            # Append structure to description for richer context
            if structure and structure not in description:  # Avoid duplication
                description = description + " STRUCTURE: " + structure
        
        # Parse CONFIDENCE
        conf_match = re.search(r'CONFIDENCE:\s*(.+?)(?=\n|$)', response_text, re.IGNORECASE)
        if conf_match:
            confidence = conf_match.group(1).strip().lower()
        
        # If parsing failed, use the whole response as description
        if not description:
            description = response_text
        
        return ImageAnalysisResult(
            image_index=image_index,
            page_number=page_number,
            description=description,
            content_type=content_type,
            key_elements=key_elements,
            text_detected=text_detected,
            confidence=confidence
        )
    
    def analyze_all_images(
        self, 
        pdf_path: str,
        context: str = "",
        page_numbers: Optional[List[int]] = None
    ) -> List[ImageAnalysisResult]:
        """
        Extract and analyze all images from PDF
        
        Args:
            pdf_path: Path to PDF file
            context: Document context for better analysis
            page_numbers: Specific pages to process
            
        Returns:
            List of ImageAnalysisResult objects
        """
        # Extract images
        images = self.extract_images_from_pdf(pdf_path, page_numbers)
        
        if not images:
            print("[INFO] No images found in PDF")
            return []
        
        print(f"[INFO] Found {len(images)} images, analyzing...")
        
        # Analyze each image
        results = []
        for img_info in images:
            result = self.analyze_image(img_info, context)
            results.append(result)
            print(f"[INFO] Analyzed image {img_info.image_index + 1}/{len(images)} on page {img_info.page_number}")
        
        return results


def format_image_analysis_as_complete_text(result: ImageAnalysisResult) -> str:
    """
    Format COMPLETE image analysis as text for storage and retrieval.
    Returns the full description without truncation.
    
    Args:
        result: ImageAnalysisResult object
        
    Returns:
        Formatted text with complete information (used as standalone chunk)
    """
    parts = [
        f"[IMAGE {result.image_index} - {result.content_type.upper()}]",
        "",
        result.description,  # Complete description, NOT truncated
        ""
    ]
    
    # Add key elements if present
    if result.key_elements:
        parts.append(f"Key elements: {', '.join(result.key_elements)}")
        parts.append("")
    
    # Add detected text if present
    if result.text_detected:
        parts.append(f"Text in image: {result.text_detected}")
        parts.append("")
    
    parts.append(f"[/IMAGE {result.image_index}]")
    
    return "\n".join(parts)
