from pathlib import Path
from typing import List, Optional, Type
import mimetypes
import logging

from core.data_models import Document
from core.base_extractor import ExtractorPlugin
from core.extractors.pdf_extractor import PDFExtractor
from core.extractors.epub_extractor import EPubExtractor
from core.extractors.docx_extractor import DocxExtractor
from core.extractors.html_extractor import HTMLExtractor
from core.extractors.txt_extractor import TextExtractor

logger = logging.getLogger(__name__)


class DocumentLoader:
    """Main document loader that routes to appropriate extractors."""
    
    def __init__(self):
        self.extractors: List[ExtractorPlugin] = [
            PDFExtractor(),
            EPubExtractor(),
            DocxExtractor(),
            HTMLExtractor(),  
            TextExtractor(),  # Keep as fallback
        ]
        
    def load_document(self, file_path: Path) -> Optional[Document]:
        """Load a document using the appropriate extractor."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return None
            
        if not file_path.is_file():
            logger.error(f"Path is not a file: {file_path}")
            return None
            
        # Find appropriate extractor
        extractor = self._find_extractor(file_path)
        if not extractor:
            logger.error(f"No extractor found for file: {file_path}")
            return None
            
        try:
            logger.info(f"Extracting content from {file_path} using {extractor.__class__.__name__}")
            return extractor.extract(file_path)
        except Exception as e:
            logger.error(f"Failed to extract content from {file_path}: {e}")
            return None
    
    def load_documents(self, file_paths: List[Path]) -> List[Document]:
        """Load multiple documents."""
        documents = []
        for file_path in file_paths:
            doc = self.load_document(file_path)
            if doc:
                documents.append(doc)
        return documents
    
    def _find_extractor(self, file_path: Path) -> Optional[ExtractorPlugin]:
        """Find the appropriate extractor for a file."""
        for extractor in self.extractors:
            if extractor.can_handle(file_path):
                return extractor
        return None
    
    def _detect_format(self, file_path: Path) -> str:
        """Detect file format using extension and MIME type."""
        # Primary detection by extension
        extension = file_path.suffix.lower()
        
        # Secondary detection by MIME type
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        # Extension-based mapping
        extension_map = {
            '.pdf': 'pdf',
            '.epub': 'epub',
            '.docx': 'docx',
            '.doc': 'doc',
            '.html': 'html',
            '.htm': 'html',
            '.txt': 'txt',
            '.md': 'markdown',
        }
        
        if extension in extension_map:
            return extension_map[extension]
            
        # MIME type-based mapping as fallback
        mime_map = {
            'application/pdf': 'pdf',
            'application/epub+zip': 'epub',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
            'application/msword': 'doc',
            'text/html': 'html',
            'text/plain': 'txt',
        }
        
        if mime_type in mime_map:
            return mime_map[mime_type]
            
        return 'unknown'
    
    def get_supported_formats(self) -> List[str]:
        """Get list of all supported file formats."""
        formats = set()
        for extractor in self.extractors:
            formats.update(extractor.supported_extensions)
        return sorted(list(formats))
    
    def register_extractor(self, extractor: ExtractorPlugin):
        """Register a new extractor plugin."""
        self.extractors.insert(0, extractor)  # Insert at beginning for priority