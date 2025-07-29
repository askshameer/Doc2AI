from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

try:
    from bs4 import BeautifulSoup, Tag
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False
    BeautifulSoup = None
    Tag = None

try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False
    chardet = None

from core.base_extractor import ExtractorPlugin
from core.data_models import Document, ProcessingInfo

logger = logging.getLogger(__name__)


class HTMLExtractor(ExtractorPlugin):
    """HTML document extractor with semantic structure preservation."""
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.html', '.htm']
    
    def can_handle(self, file_path: Path) -> bool:
        if not file_path.suffix.lower() in ['.html', '.htm']:
            return False
        
        if not HAS_BS4:
            logger.warning(f"Cannot handle HTML {file_path}: Missing beautifulsoup4 dependency")
            return False
            
        return True
    
    def extract(self, file_path: Path) -> Document:
        """Extract text and metadata from HTML file."""
        if not HAS_BS4:
            raise RuntimeError("HTML processing requires beautifulsoup4")
            
        doc = Document.from_file(file_path)
        
        processing_info = ProcessingInfo(
            extraction_method='beautifulsoup4',
            chunking_strategy='none',
            processing_timestamp=datetime.now()
        )
        
        try:
            # Read file with encoding detection
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                encoding = chardet.detect(raw_data)['encoding'] or 'utf-8'
            
            with open(file_path, 'r', encoding=encoding) as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract metadata from HTML head
            metadata_updates = self._extract_metadata(soup)
            for key, value in metadata_updates.items():
                setattr(doc.metadata, key, value)
            
            # Extract content with structure
            content, structure = self._extract_content_with_structure(soup)
            doc.content = content
            doc.structure = structure
            
            # Update character and word counts
            doc.metadata.char_count = len(content)
            doc.metadata.word_count = len(content.split()) if content else 0
            doc.metadata.encoding = encoding
            
        except Exception as e:
            logger.error(f"Failed to extract HTML content from {file_path}: {e}")
            processing_info.errors.append(str(e))
            doc.content = ""
            doc.structure = {}
        
        return doc
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract metadata from HTML head section."""
        metadata = {}
        
        # Title
        title_tag = soup.find('title')
        if title_tag:
            metadata['title'] = title_tag.get_text().strip()
        
        # Meta tags
        meta_tags = soup.find_all('meta')
        for meta in meta_tags:
            name = meta.get('name', '').lower()
            property_attr = meta.get('property', '').lower()
            content = meta.get('content', '')
            
            if name == 'author':
                metadata['author'] = content
            elif name == 'description':
                metadata['description'] = content
            elif name == 'keywords':
                metadata['keywords'] = [k.strip() for k in content.split(',')]
            elif name == 'language' or name == 'lang':
                metadata['language'] = content
            elif property_attr == 'og:title':
                metadata['og_title'] = content
            elif property_attr == 'og:description':
                metadata['og_description'] = content
            elif property_attr == 'og:type':
                metadata['og_type'] = content
        
        # Language from html tag
        html_tag = soup.find('html')
        if html_tag and html_tag.get('lang'):
            metadata['language'] = html_tag.get('lang')
        
        return metadata
    
    def _extract_content_with_structure(self, soup: BeautifulSoup) -> tuple[str, Dict[str, Any]]:
        """Extract content while preserving semantic structure."""
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        structure = {
            'headings': [],
            'paragraphs': [],
            'lists': [],
            'tables': [],
            'links': [],
            'images': []
        }
        
        content_parts = []
        
        # Find main content area
        main_content = self._find_main_content(soup)
        
        # Process elements in order
        for element in main_content.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'ul', 'ol', 'table', 'article', 'section']):
            
            if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Heading
                text = element.get_text().strip()
                if text:
                    level = int(element.name[1])
                    content_parts.append(f"\n{'#' * level} {text}\n")
                    
                    structure['headings'].append({
                        'text': text,
                        'level': level,
                        'tag': element.name
                    })
            
            elif element.name == 'p':
                # Paragraph
                text = element.get_text().strip()
                if text:
                    content_parts.append(text)
                    structure['paragraphs'].append({
                        'text': text,
                        'char_count': len(text),
                        'word_count': len(text.split())
                    })
            
            elif element.name in ['ul', 'ol']:
                # List
                list_text = self._extract_list_text(element)
                if list_text:
                    content_parts.append(f"\n{list_text}\n")
                    structure['lists'].append({
                        'type': element.name,
                        'items': len(element.find_all('li')),
                        'text': list_text
                    })
            
            elif element.name == 'table':
                # Table
                table_text = self._extract_table_text(element)
                if table_text:
                    content_parts.append(f"\n[Table]\n{table_text}\n")
                    structure['tables'].append({
                        'rows': len(element.find_all('tr')),
                        'text': table_text
                    })
            
            elif element.name in ['div', 'article', 'section']:
                # Container elements - extract text if not empty
                text = element.get_text().strip()
                if text and len(text.split()) > 5:  # Only include substantial text blocks
                    # Check if this div contains mainly text (not other structural elements)
                    child_structural = element.find_all(['div', 'section', 'article', 'header', 'footer'])
                    if len(child_structural) == 0:
                        content_parts.append(text)
        
        # Extract links and images for structure info
        for link in main_content.find_all('a', href=True):
            if link.get_text().strip():
                structure['links'].append({
                    'text': link.get_text().strip(),
                    'href': link['href']
                })
        
        for img in main_content.find_all('img'):
            structure['images'].append({
                'src': img.get('src', ''),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        
        return '\n\n'.join(content_parts), structure
    
    def _find_main_content(self, soup: BeautifulSoup) -> Tag:
        """Find the main content area of the HTML document."""
        # Try common main content selectors
        main_selectors = [
            'main',
            '[role="main"]',
            '#main',
            '#content',
            '.main-content',
            '.content',
            'article',
            '.post-content',
            '.entry-content'
        ]
        
        for selector in main_selectors:
            main = soup.select_one(selector)
            if main:
                return main
        
        # Fallback to body
        body = soup.find('body')
        return body if body else soup
    
    def _extract_list_text(self, list_element: Tag) -> str:
        """Extract text from HTML list elements."""
        items = []
        for li in list_element.find_all('li'):
            text = li.get_text().strip()
            if text:
                prefix = "- " if list_element.name == 'ul' else f"{len(items) + 1}. "
                items.append(f"{prefix}{text}")
        
        return '\n'.join(items)
    
    def _extract_table_text(self, table_element: Tag) -> str:
        """Extract text from HTML table elements."""
        rows = []
        for tr in table_element.find_all('tr'):
            cells = []
            for cell in tr.find_all(['td', 'th']):
                cell_text = cell.get_text().strip().replace('\n', ' ')
                cells.append(cell_text)
            if cells:
                rows.append(' | '.join(cells))
        
        return '\n'.join(rows)