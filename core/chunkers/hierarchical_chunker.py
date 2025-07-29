from typing import List, Dict, Any, Optional
import re

from core.chunkers.base_chunker import BaseChunker
from core.data_models import Document, Chunk, ChunkLocation


class HierarchicalChunker(BaseChunker):
    """Hierarchical chunker that respects document structure (sections, chapters, etc.)."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Default configuration
        self.respect_sections = self.config.get('respect_sections', True)
        self.max_chunk_size = self.config.get('max_chunk_size', 2000)
        self.min_chunk_size = self.config.get('min_chunk_size', 200)
        self.section_overlap = self.config.get('section_overlap', 100)
        self.preserve_headers = self.config.get('preserve_headers', True)
        
    @property
    def strategy_name(self) -> str:
        return 'hierarchical'
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document respecting hierarchical structure."""
        if not document.content:
            return []
        
        text = document.content.strip()
        
        # Try to use document structure if available
        if document.structure and self.respect_sections:
            return self._chunk_with_structure(document, text)
        else:
            return self._chunk_by_headers(document, text)
    
    def _chunk_with_structure(self, document: Document, text: str) -> List[Chunk]:
        """Use existing document structure for chunking."""
        chunks = []
        chunk_index = 0
        
        # Handle different document types
        if 'chapters' in document.structure:
            # ePub or book-like structure
            chunks = self._chunk_by_chapters(document, chunk_index)
        elif 'headings' in document.structure:
            # HTML or Markdown structure
            chunks = self._chunk_by_headings(document, text, chunk_index)
        elif 'paragraphs' in document.structure:
            # DOCX or structured text
            chunks = self._chunk_by_paragraphs_structure(document, text, chunk_index)
        else:
            # Fallback to header detection
            chunks = self._chunk_by_headers(document, text)
        
        return chunks
    
    def _chunk_by_chapters(self, document: Document, start_index: int) -> List[Chunk]:
        """Chunk using chapter structure from ePub."""
        chunks = []
        chunk_index = start_index
        
        chapters = document.structure.get('chapters', [])
        
        for chapter_info in chapters:
            chapter_title = chapter_info.get('title', f"Chapter {chapter_info.get('chapter_number', '?')}")
            
            # Find chapter content in full text
            # This is a simplified approach - in practice, you'd want to store
            # chapter content separately during extraction
            chapter_content = self._extract_chapter_content(document.content, chapter_title)
            
            if len(chapter_content) <= self.max_chunk_size:
                # Chapter fits in one chunk
                location = ChunkLocation(
                    section=chapter_title,
                    char_start=0,  # Would need proper calculation
                    char_end=len(chapter_content)
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chapter_content,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                chunk.semantic_tags.extend([
                    'chapter',
                    f'chapter_title:{chapter_title}',
                    f'chapter_number:{chapter_info.get("chapter_number", "unknown")}'
                ])
                
                chunks.append(chunk)
                chunk_index += 1
            else:
                # Split large chapter into smaller chunks
                chapter_chunks = self._split_large_section(
                    document, chapter_content, chapter_title, chunk_index
                )
                chunks.extend(chapter_chunks)
                chunk_index += len(chapter_chunks)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_headings(self, document: Document, text: str, start_index: int) -> List[Chunk]:
        """Chunk using heading structure from HTML/Markdown."""
        chunks = []
        chunk_index = start_index
        
        headings = document.structure.get('headings', [])
        
        if not headings:
            return self._chunk_by_headers(document, text)
        
        # Sort headings by line/position if available
        headings_sorted = sorted(headings, key=lambda h: h.get('line', 0))
        
        current_section = {
            'title': 'Introduction',
            'level': 0,
            'content': '',
            'start_pos': 0
        }
        
        for i, heading in enumerate(headings_sorted):
            heading_text = heading.get('text', '')
            heading_level = heading.get('level', 1)
            
            # Finalize current section
            if current_section['content']:
                section_chunk = self._create_section_chunk(
                    document, current_section, chunk_index
                )
                
                if len(section_chunk.text) >= self.min_chunk_size:
                    chunks.append(section_chunk)
                    chunk_index += 1
            
            # Start new section
            current_section = {
                'title': heading_text,
                'level': heading_level,
                'content': '',
                'start_pos': heading.get('line', 0)
            }
            
            # Extract content between this heading and the next
            if i < len(headings_sorted) - 1:
                next_heading = headings_sorted[i + 1]
                section_content = self._extract_section_content(
                    text, heading, next_heading
                )
            else:
                # Last section - take everything to the end
                section_content = self._extract_section_content(
                    text, heading, None
                )
            
            current_section['content'] = section_content
        
        # Handle final section
        if current_section['content']:
            section_chunk = self._create_section_chunk(
                document, current_section, chunk_index
            )
            
            if len(section_chunk.text) >= self.min_chunk_size:
                chunks.append(section_chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_headers(self, document: Document, text: str) -> List[Chunk]:
        """Detect and chunk by headers in the text."""
        # Common header patterns
        header_patterns = [
            r'^#+\s+(.+)$',  # Markdown headers
            r'^(.+)\n=+$',   # Underlined headers
            r'^(.+)\n-+$',   # Underlined subheaders
            r'^\d+\.\s+(.+)$',  # Numbered sections
            r'^[A-Z][A-Za-z\s]+:$',  # Colon-ended headers
            r'^\[(.+)\]$'    # Bracketed sections
        ]
        
        chunks = []
        lines = text.splitlines()
        current_section = {
            'title': 'Content',
            'content': [],
            'start_line': 0
        }
        
        chunk_index = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check if line matches header pattern
            is_header = False
            header_title = None
            
            for pattern in header_patterns:
                match = re.match(pattern, line_stripped, re.MULTILINE)
                if match:
                    is_header = True
                    header_title = match.group(1).strip()
                    break
            
            if is_header and len(current_section['content']) > 0:
                # Finalize current section
                section_text = '\n'.join(current_section['content'])
                
                if len(section_text.strip()) >= self.min_chunk_size:
                    location = ChunkLocation(
                        section=current_section['title'],
                        line_start=current_section['start_line'],
                        line_end=i - 1
                    )
                    
                    chunk = self._create_chunk(
                        document=document,
                        text=section_text.strip(),
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        location=location
                    )
                    
                    chunk.semantic_tags.extend([
                        'section',
                        f'section_title:{current_section["title"]}'
                    ])
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new section
                current_section = {
                    'title': header_title or 'Section',
                    'content': [],
                    'start_line': i
                }
                
                if self.preserve_headers:
                    current_section['content'].append(line)
            else:
                current_section['content'].append(line)
        
        # Handle final section
        if current_section['content']:
            section_text = '\n'.join(current_section['content'])
            
            if len(section_text.strip()) >= self.min_chunk_size:
                location = ChunkLocation(
                    section=current_section['title'],
                    line_start=current_section['start_line'],
                    line_end=len(lines) - 1
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=section_text.strip(),
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                chunk.semantic_tags.extend([
                    'section',
                    f'section_title:{current_section["title"]}'
                ])
                
                chunks.append(chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _split_large_section(self, document: Document, content: str, section_title: str, start_index: int) -> List[Chunk]:
        """Split a large section into smaller chunks."""
        # Use paragraph boundaries for splitting
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            if current_chunk and (current_size + paragraph_size > self.max_chunk_size):
                # Finalize current chunk
                chunk_text = '\n\n'.join(current_chunk)
                
                location = ChunkLocation(
                    section=section_title,
                    subsection=f"Part {chunk_index - start_index + 1}"
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                chunk.semantic_tags.extend([
                    'section_split',
                    f'section_title:{section_title}',
                    f'part:{chunk_index - start_index + 1}'
                ])
                
                chunks.append(chunk)
                chunk_index += 1
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(paragraph)
            current_size += paragraph_size + 2  # +2 for \n\n
        
        # Handle remaining content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            
            location = ChunkLocation(
                section=section_title,
                subsection=f"Part {chunk_index - start_index + 1}"
            )
            
            chunk = self._create_chunk(
                document=document,
                text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                location=location
            )
            
            chunk.semantic_tags.extend([
                'section_split',
                f'section_title:{section_title}',
                f'part:{chunk_index - start_index + 1}'
            ])
            
            chunks.append(chunk)
        
        return chunks
    
    def _extract_chapter_content(self, text: str, chapter_title: str) -> str:
        """Extract content for a specific chapter (simplified implementation)."""
        # This is a placeholder - in practice, you'd want to store
        # chapter boundaries during extraction
        return text  # Return full text for now
    
    def _extract_section_content(self, text: str, heading: Dict, next_heading: Optional[Dict]) -> str:
        """Extract content between two headings."""
        # This is a simplified implementation
        # In practice, you'd need proper position tracking
        return ""  # Placeholder
    
    def _create_section_chunk(self, document: Document, section: Dict, chunk_index: int) -> Chunk:
        """Create a chunk from section information."""
        location = ChunkLocation(
            section=section['title'],
            line_start=section.get('start_pos', 0)
        )
        
        chunk = self._create_chunk(
            document=document,
            text=section['content'],
            chunk_index=chunk_index,
            total_chunks=0,  # Will be updated later
            location=location
        )
        
        chunk.semantic_tags.extend([
            'hierarchical_section',
            f'section_title:{section["title"]}',
            f'section_level:{section.get("level", 1)}'
        ])
        
        return chunk
    
    def validate_config(self) -> List[str]:
        """Validate chunker configuration."""
        warnings = []
        
        if self.max_chunk_size <= 0:
            warnings.append("max_chunk_size must be positive")
        
        if self.min_chunk_size <= 0:
            warnings.append("min_chunk_size must be positive")
        
        if self.min_chunk_size >= self.max_chunk_size:
            warnings.append("min_chunk_size must be less than max_chunk_size")
        
        if self.section_overlap < 0:
            warnings.append("section_overlap cannot be negative")
        
        return warnings