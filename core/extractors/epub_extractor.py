from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

try:
    import ebooklib
    from ebooklib import epub
    HAS_EBOOKLIB = True
except ImportError:
    HAS_EBOOKLIB = False
    ebooklib = None
    epub = None

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None

from core.base_extractor import ExtractorPlugin
from core.data_models import Document, ProcessingInfo

logger = logging.getLogger(__name__)


class EPubExtractor(ExtractorPlugin):
    """ePub document extractor preserving chapter structure."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.epub']
    
    def can_handle(self, file_path: Path) -> bool:
        if not file_path.suffix.lower() == '.epub':
            return False
        
        if not (HAS_EBOOKLIB and HAS_BS4):
            logger.warning(f"Cannot handle ePub {file_path}: Missing dependencies (ebooklib, beautifulsoup4)")
            return False
            
        return True
    
    def extract(self, file_path: Path) -> Document:
        """Extract text and metadata from ePub file."""
        if not (HAS_EBOOKLIB and HAS_BS4):
            raise RuntimeError("ePub processing requires ebooklib and beautifulsoup4")
            
        doc = Document.from_file(file_path)
        
        processing_info = ProcessingInfo(
            extraction_method='ebooklib',
            chunking_strategy='none',
            processing_timestamp=datetime.now()
        )
        
        try:
            book = epub.read_epub(str(file_path))
            
            # Extract metadata
            metadata_updates = self._extract_metadata(book)
            for key, value in metadata_updates.items():
                setattr(doc.metadata, key, value)
            
            # Extract content with structure
            content, structure = self._extract_content_with_structure(book)
            doc.content = content
            doc.structure = structure
            
            # Update character and word counts
            doc.metadata.char_count = len(content)
            doc.metadata.word_count = len(content.split()) if content else 0
            
        except Exception as e:
            logger.error(f"Failed to extract ePub content from {file_path}: {e}")
            processing_info.errors.append(str(e))
            doc.content = ""
            doc.structure = {}
        
        return doc
    
    def _extract_metadata(self, book) -> Dict[str, Any]:
        """Extract metadata from ePub book."""
        metadata = {}
        
        # Basic metadata
        if book.get_metadata('DC', 'title'):
            metadata['title'] = book.get_metadata('DC', 'title')[0][0]
        
        if book.get_metadata('DC', 'creator'):
            authors = [creator[0] for creator in book.get_metadata('DC', 'creator')]
            metadata['author'] = ', '.join(authors)
        
        if book.get_metadata('DC', 'language'):
            metadata['language'] = book.get_metadata('DC', 'language')[0][0]
        
        if book.get_metadata('DC', 'publisher'):
            metadata['publisher'] = book.get_metadata('DC', 'publisher')[0][0]
        
        if book.get_metadata('DC', 'date'):
            metadata['publication_date'] = book.get_metadata('DC', 'date')[0][0]
        
        if book.get_metadata('DC', 'description'):
            metadata['description'] = book.get_metadata('DC', 'description')[0][0]
        
        if book.get_metadata('DC', 'subject'):
            subjects = [subject[0] for subject in book.get_metadata('DC', 'subject')]
            metadata['subjects'] = subjects
        
        return metadata
    
    def _extract_content_with_structure(self, book) -> tuple[str, Dict[str, Any]]:
        """Extract content while preserving chapter/section structure."""
        content_parts = []
        structure = {
            'chapters': [],
            'toc': [],
            'spine_order': []
        }
        
        # Get table of contents
        toc_items = self._extract_toc(book)
        structure['toc'] = toc_items
        
        # Process spine items (reading order)
        chapter_num = 1
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            try:
                # Parse HTML content
                soup = BeautifulSoup(item.get_content(), 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Extract text
                text = soup.get_text()
                
                # Clean up whitespace
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                
                if text.strip():
                    # Try to determine chapter title
                    chapter_title = self._extract_chapter_title(soup) or f"Chapter {chapter_num}"
                    
                    content_parts.append(f"[{chapter_title}]\n{text}\n")
                    
                    structure['chapters'].append({
                        'chapter_number': chapter_num,
                        'title': chapter_title,
                        'item_id': item.get_id(),
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                    
                    structure['spine_order'].append(item.get_id())
                    chapter_num += 1
                    
            except Exception as e:
                logger.warning(f"Failed to process ePub item {item.get_id()}: {e}")
        
        return '\n\n'.join(content_parts), structure
    
    def _extract_toc(self, book) -> List[Dict[str, Any]]:
        """Extract table of contents structure."""
        toc_items = []
        
        def process_toc_item(item, level=0):
            if isinstance(item, tuple):
                # Item is (section, children)
                section, children = item
                toc_entry = {
                    'title': section.title,
                    'href': section.href,
                    'level': level
                }
                toc_items.append(toc_entry)
                
                # Process children
                for child in children:
                    process_toc_item(child, level + 1)
            else:
                # Item is just a section
                toc_entry = {
                    'title': item.title,
                    'href': item.href,
                    'level': level
                }
                toc_items.append(toc_entry)
        
        # Process TOC
        for item in book.toc:
            process_toc_item(item)
        
        return toc_items
    
    def _extract_chapter_title(self, soup) -> str:
        """Try to extract chapter title from HTML content."""
        # Look for common heading tags
        for tag in ['h1', 'h2', 'h3']:
            heading = soup.find(tag)
            if heading and heading.get_text().strip():
                return heading.get_text().strip()
        
        # Look for title tag
        title = soup.find('title')
        if title and title.get_text().strip():
            return title.get_text().strip()
        
        return None