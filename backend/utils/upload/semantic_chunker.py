"""
Chunking strategy (priority cascade):

Raw text
  │
  ├─ Any [[TABLE]] markers?
  │     │
  │     ├─ Yes → extract tables → tables go through _chunk_table (row-level splitting)
  │     │                         remaining text goes to next step
  │     └─ No  → whole text goes to next step
  │
  ▼
_chunk_regular_text
  │
  ├─ Split on \\n\\n into paragraphs
  │
  ├─ Pack paragraphs into chunks:
  │     ├─ Paragraph < max_size 1500 and fits under target 800 → append to current chunk
  │     ├─ Adding paragraph would exceed target 800       → close current chunk, start new one
  │     └─ Paragraph itself > max_size                → fall back to _split_large_paragraph
  │
  ▼
_split_large_paragraph (only for oversized paragraphs, >1500 chars)
  │
  ├─ Tokenize into sentences with NLTK
  │     ├─ Success → pack sentences into chunks
  │     └─ Failure → fall back to _split_by_characters (char-split with 100-char overlap)

Features:
- Sentence and paragraph boundary detection
- Table and list structure preservation
- Metadata tracking (page numbers, chunk types)
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import nltk
from openai import AzureOpenAI


@dataclass
class SemanticChunk:
    """Represents a semantically meaningful text chunk"""
    content: str
    page_number: int
    chunk_type: str  # 'paragraph', 'table', 'list', 'mixed'
    start_char: int
    end_char: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class SemanticChunker:
    """
    Intelligent text chunker that preserves semantic boundaries
    """
    
    def __init__(
        self, 
        target_chunk_size: int = 800,
        min_chunk_size: int = 200,
        max_chunk_size: int = 1500,
        enable_sentence_splitting: bool = True
    ):
        """
        Initialize semantic chunker
        
        Args:
            target_chunk_size: Target size for chunks (800 characters)
            min_chunk_size: Minimum chunk size (200 characters)
            max_chunk_size: Maximum chunk size (1500 characters)
            enable_sentence_splitting: Whether to split on sentence boundaries
        """
        self.target_chunk_size = target_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.enable_sentence_splitting = enable_sentence_splitting
        
    def chunk_text(
        self, 
        text: str, 
        page_number: int = 1,
        preserve_tables: bool = True
    ) -> List[SemanticChunk]:
        """
        Chunk text using semantic boundaries
        
        Args:
            text: Text to chunk
            page_number: Page number for tracking
            preserve_tables: Whether to keep tables intact
            
        Returns:
            List of SemanticChunk objects
        """
        if not text or not text.strip():
            return []
        
        chunks = []
        
        # Step 1: Extract tables and process them separately
        if preserve_tables:
            text_parts = self._extract_tables(text)
        else:
            text_parts = [("text", text, 0)]
        
        # Step 2: Process each part
        current_position = 0
        for part_type, content, offset in text_parts:
            if part_type == "table":
                # Tables are kept as single chunks (or split if too large)
                table_chunks = self._chunk_table(content, page_number, offset)
                chunks.extend(table_chunks)
            else:
                # Regular text: use semantic chunking
                text_chunks = self._chunk_regular_text(content, page_number, offset)
                chunks.extend(text_chunks)
            current_position += len(content)
        
        return chunks
    
    def _extract_tables(self, text: str) -> List[Tuple[str, str, int]]:
        """
        Extract tables from text and separate them
        
        Returns:
            List of (type, content, offset) tuples
        """
        parts = []
        pattern = r'\[\[TABLE.*?\]\].*?\[\[/TABLE\]\]'
        
        matches = list(re.finditer(pattern, text, re.DOTALL))
        
        if not matches:
            return [("text", text, 0)]
        
        last_end = 0
        for match in matches:
            # Text before table
            if match.start() > last_end:
                before_text = text[last_end:match.start()].strip()
                if before_text:
                    parts.append(("text", before_text, last_end))
            
            # Table itself
            table_text = match.group(0)
            parts.append(("table", table_text, match.start()))
            
            last_end = match.end()
        
        # Text after last table
        if last_end < len(text):
            after_text = text[last_end:].strip()
            if after_text:
                parts.append(("text", after_text, last_end))
        
        return parts
    
    def _chunk_table(
        self, 
        table_text: str, 
        page_number: int, 
        offset: int
    ) -> List[SemanticChunk]:
        """
        Chunk table content (keep intact if possible, split if too large)
        """
        chunks = []
        
        if len(table_text) <= self.max_chunk_size:
            # Table fits in one chunk
            chunks.append(SemanticChunk(
                content=table_text,
                page_number=page_number,
                chunk_type="table",
                start_char=offset,
                end_char=offset + len(table_text),
                metadata={"preserved": True}
            ))
        else:
            # Table is too large, split by rows
            lines = table_text.split('\n')
            current_chunk = []
            current_size = 0
            
            for line in lines:
                line_size = len(line) + 1  # +1 for newline
                
                if current_size + line_size > self.max_chunk_size and current_chunk:
                    # Flush current chunk
                    chunk_content = '\n'.join(current_chunk)
                    chunks.append(SemanticChunk(
                        content=chunk_content,
                        page_number=page_number,
                        chunk_type="table",
                        start_char=offset,
                        end_char=offset + len(chunk_content),
                        metadata={"split": True}
                    ))
                    current_chunk = [line]
                    current_size = line_size
                    offset += len(chunk_content)
                else:
                    current_chunk.append(line)
                    current_size += line_size
            
            # Flush remaining
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    page_number=page_number,
                    chunk_type="table",
                    start_char=offset,
                    end_char=offset + len(chunk_content),
                    metadata={"split": True, "final": True}
                ))
        
        return chunks
    
    def _chunk_regular_text(
        self, 
        text: str, 
        page_number: int, 
        offset: int
    ) -> List[SemanticChunk]:
        """
        Chunk regular text using paragraph and sentence boundaries
        """
        chunks = []
        
        # Step 1: Split into paragraphs
        paragraphs = self._split_into_paragraphs(text)
        
        current_chunk = []
        current_size = 0
        chunk_start = offset
        
        for para in paragraphs:
            para_size = len(para)
            
            # If paragraph itself is too large, split it further
            if para_size > self.max_chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunk_content = '\n\n'.join(current_chunk)
                    chunks.append(SemanticChunk(
                        content=chunk_content,
                        page_number=page_number,
                        chunk_type="paragraph",
                        start_char=chunk_start,
                        end_char=chunk_start + len(chunk_content),
                        metadata={}
                    ))
                    chunk_start += len(chunk_content)
                    current_chunk = []
                    current_size = 0
                
                # Split large paragraph by sentences
                para_chunks = self._split_large_paragraph(para, page_number, chunk_start)
                chunks.extend(para_chunks)
                chunk_start += para_size
                
            elif current_size + para_size > self.target_chunk_size and current_chunk:
                # Current chunk is full, flush it
                chunk_content = '\n\n'.join(current_chunk)
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    page_number=page_number,
                    chunk_type="paragraph",
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_content),
                    metadata={}
                ))
                chunk_start += len(chunk_content)
                current_chunk = [para]
                current_size = para_size
            else:
                # Add to current chunk
                current_chunk.append(para)
                current_size += para_size + 2  # +2 for \n\n
        
        # Flush remaining
        if current_chunk:
            chunk_content = '\n\n'.join(current_chunk)
            chunks.append(SemanticChunk(
                content=chunk_content,
                page_number=page_number,
                chunk_type="paragraph",
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_content),
                metadata={}
            ))
        
        return chunks
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """
        Split text into paragraphs
        """
        # Split on double newlines or multiple spaces
        paragraphs = re.split(r'\n\s*\n', text)
        
        # Clean and filter
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    def _split_large_paragraph(
        self, 
        paragraph: str, 
        page_number: int, 
        offset: int
    ) -> List[SemanticChunk]:
        """
        Split a large paragraph into smaller chunks using sentence boundaries
        """
        chunks = []
        
        if not self.enable_sentence_splitting:
            # Fallback to character-based splitting
            return self._split_by_characters(paragraph, page_number, offset)
        
        try:
            # Split into sentences
            sentences = nltk.sent_tokenize(paragraph)
        except Exception as e:
            print(f"[WARNING] NLTK sentence tokenization failed: {e}")
            return self._split_by_characters(paragraph, page_number, offset)
        
        current_chunk = []
        current_size = 0
        chunk_start = offset
        
        for sentence in sentences:
            sentence_size = len(sentence) + 1  # +1 for space
            
            if current_size + sentence_size > self.max_chunk_size and current_chunk:
                # Flush current chunk
                chunk_content = ' '.join(current_chunk)
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    page_number=page_number,
                    chunk_type="paragraph",
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_content),
                    metadata={"sentence_split": True}
                ))
                chunk_start += len(chunk_content)
                current_chunk = [sentence]
                current_size = sentence_size
            elif current_size + sentence_size > self.target_chunk_size and current_chunk:
                # Target reached, flush
                chunk_content = ' '.join(current_chunk)
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    page_number=page_number,
                    chunk_type="paragraph",
                    start_char=chunk_start,
                    end_char=chunk_start + len(chunk_content),
                    metadata={"sentence_split": True}
                ))
                chunk_start += len(chunk_content)
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Flush remaining
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            chunks.append(SemanticChunk(
                content=chunk_content,
                page_number=page_number,
                chunk_type="paragraph",
                start_char=chunk_start,
                end_char=chunk_start + len(chunk_content),
                metadata={"sentence_split": True}
            ))
        
        return chunks
    
    def _split_by_characters(
        self, 
        text: str, 
        page_number: int, 
        offset: int
    ) -> List[SemanticChunk]:
        """
        Fallback: split by character count with overlap
        """
        chunks = []
        overlap = 100
        start = 0
        
        while start < len(text):
            end = min(start + self.target_chunk_size, len(text))
            chunk_content = text[start:end].strip()
            
            if chunk_content:
                chunks.append(SemanticChunk(
                    content=chunk_content,
                    page_number=page_number,
                    chunk_type="text",
                    start_char=offset + start,
                    end_char=offset + end,
                    metadata={"character_split": True}
                ))
            
            start += self.target_chunk_size - overlap
        
        return chunks


def create_semantic_chunks(
    text: str,
    page_number: int = 1,
    target_size: int = 800,
    min_size: int = 200,
    max_size: int = 1500
) -> List[Dict]:
    """
    Convenience function to create semantic chunks
    
    Args:
        text: Text to chunk
        page_number: Page number for tracking
        target_size: Target chunk size
        min_size: Minimum chunk size
        max_size: Maximum chunk size
        
    Returns:
        List of chunk dictionaries with content, page_number, and metadata
    """
    chunker = SemanticChunker(
        target_chunk_size=target_size,
        min_chunk_size=min_size,
        max_chunk_size=max_size
    )
    
    chunks = chunker.chunk_text(text, page_number)
    
    # Convert to dictionaries
    return [
        {
            "content": chunk.content,
            "page_number": chunk.page_number,
            "chunk_type": chunk.chunk_type,
            "metadata": chunk.metadata
        }
        for chunk in chunks
    ]
