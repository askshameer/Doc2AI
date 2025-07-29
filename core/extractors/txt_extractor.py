from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None

from core.base_extractor import ExtractorPlugin
from core.data_models import Document, ProcessingInfo

logger = logging.getLogger(__name__)


class TextExtractor(ExtractorPlugin):
    """Plain text and Markdown document extractor."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.txt', '.md', '.markdown', '.rst', '.text']
    
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.supported_extensions
    
    def extract(self, file_path: Path) -> Document:
        """Extract text and metadata from plain text file."""
        doc = Document.from_file(file_path)
        
        processing_info = ProcessingInfo(
            extraction_method='plain_text',
            chunking_strategy='none',
            processing_timestamp=datetime.now()
        )
        
        try:
            # Detect encoding
            encoding = 'utf-8'
            confidence = 1.0
            
            if HAS_CHARDET:
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                    encoding_result = chardet.detect(raw_data)
                    encoding = encoding_result['encoding'] or 'utf-8'
                    confidence = encoding_result['confidence'] or 0.0
            else:
                # Fallback without chardet
                processing_info.warnings.append("chardet not available, assuming UTF-8 encoding")
            
            # Read file with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                content = f.read()
            
            doc.content = content
            doc.metadata.encoding = encoding
            doc.metadata.char_count = len(content)
            doc.metadata.word_count = len(content.split()) if content else 0
            
            # Analyze structure for Markdown files
            if file_path.suffix.lower() in ['.md', '.markdown']:
                structure = self._analyze_markdown_structure(content)
                doc.structure = structure
                processing_info.extraction_method = 'markdown'
            else:
                structure = self._analyze_text_structure(content)
                doc.structure = structure
            
            # Add encoding confidence to processing info
            if confidence < 0.8:
                processing_info.warnings.append(f"Low encoding confidence: {confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Failed to extract text content from {file_path}: {e}")
            processing_info.errors.append(str(e))
            doc.content = ""
            doc.structure = {}
        
        return doc
    
    def _analyze_markdown_structure(self, content: str) -> Dict[str, Any]:
        """Analyze Markdown document structure."""
        lines = content.splitlines()
        structure = {
            'headings': [],
            'paragraphs': [],
            'code_blocks': [],
            'lists': [],
            'links': [],
            'images': []
        }
        
        current_paragraph = []
        in_code_block = False
        code_block_content = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Code blocks
            if line_stripped.startswith('```'):
                if in_code_block:
                    # End of code block
                    structure['code_blocks'].append({
                        'line_start': i - len(code_block_content),
                        'line_end': i,
                        'content': '\n'.join(code_block_content),
                        'language': code_block_content[0] if code_block_content else ''
                    })
                    code_block_content = []
                    in_code_block = False
                else:
                    # Start of code block
                    in_code_block = True
                    if current_paragraph:
                        self._add_paragraph(structure, current_paragraph, i)
                        current_paragraph = []
                continue
            
            if in_code_block:
                code_block_content.append(line)
                continue
            
            # Headings
            if line_stripped.startswith('#'):
                if current_paragraph:
                    self._add_paragraph(structure, current_paragraph, i)
                    current_paragraph = []
                
                level = len(line_stripped) - len(line_stripped.lstrip('#'))
                heading_text = line_stripped.lstrip('#').strip()
                structure['headings'].append({
                    'text': heading_text,
                    'level': level,
                    'line': i
                })
            
            # Lists
            elif line_stripped.startswith(('- ', '* ', '+ ')) or (line_stripped and line_stripped[0].isdigit() and '. ' in line_stripped[:5]):
                if current_paragraph:
                    self._add_paragraph(structure, current_paragraph, i)
                    current_paragraph = []
                
                # This is simplified - a full parser would handle nested lists
                list_item = line_stripped.lstrip('- *+ ').split('. ', 1)[-1]
                structure['lists'].append({
                    'item': list_item,
                    'line': i,
                    'type': 'ordered' if line_stripped[0].isdigit() else 'unordered'
                })
            
            # Regular paragraphs
            elif line_stripped:
                current_paragraph.append(line_stripped)
            
            # Empty line - end of paragraph
            elif current_paragraph:
                self._add_paragraph(structure, current_paragraph, i)
                current_paragraph = []
        
        # Add final paragraph if exists
        if current_paragraph:
            self._add_paragraph(structure, current_paragraph, len(lines))
        
        # Extract links and images using regex
        import re
        
        # Markdown links: [text](url)
        link_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        for match in re.finditer(link_pattern, content):
            structure['links'].append({
                'text': match.group(1),
                'url': match.group(2)
            })
        
        # Markdown images: ![alt](url)
        image_pattern = r'!\[([^\]]*)\]\(([^)]+)\)'
        for match in re.finditer(image_pattern, content):
            structure['images'].append({
                'alt': match.group(1),
                'url': match.group(2)
            })
        
        return structure
    
    def _analyze_text_structure(self, content: str) -> Dict[str, Any]:
        """Analyze plain text document structure."""
        lines = content.splitlines()
        structure = {
            'paragraphs': [],
            'line_count': len(lines),
            'blank_lines': 0,
            'avg_line_length': 0
        }
        
        current_paragraph = []
        line_lengths = []
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            line_lengths.append(len(line))
            
            if not line_stripped:
                structure['blank_lines'] += 1
                if current_paragraph:
                    self._add_paragraph(structure, current_paragraph, i)
                    current_paragraph = []
            else:
                current_paragraph.append(line_stripped)
        
        # Add final paragraph if exists
        if current_paragraph:
            self._add_paragraph(structure, current_paragraph, len(lines))
        
        # Calculate average line length
        if line_lengths:
            structure['avg_line_length'] = sum(line_lengths) / len(line_lengths)
        
        return structure
    
    def _add_paragraph(self, structure: Dict[str, Any], paragraph_lines: List[str], end_line: int):
        """Add a paragraph to the structure."""
        paragraph_text = ' '.join(paragraph_lines)
        structure['paragraphs'].append({
            'text': paragraph_text,
            'line_start': end_line - len(paragraph_lines),
            'line_end': end_line - 1,
            'word_count': len(paragraph_text.split()),
            'char_count': len(paragraph_text)
        })