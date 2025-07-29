from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
from datetime import datetime

# Import PDF libraries with graceful fallbacks
try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False
    pdfplumber = None

try:
    import PyPDF2
    HAS_PYPDF2 = True
except ImportError:
    HAS_PYPDF2 = False
    PyPDF2 = None

try:
    from PIL import Image
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    Image = None
    pytesseract = None

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    fitz = None

import io

from core.base_extractor import ExtractorPlugin
from core.data_models import Document, DocumentMetadata, ProcessingInfo

logger = logging.getLogger(__name__)


class PDFExtractor(ExtractorPlugin):
    """PDF document extractor with OCR fallback support."""
    
    def __init__(self, ocr_enabled: bool = True, ocr_language: str = 'eng'):
        self.ocr_enabled = ocr_enabled and HAS_OCR
        self.ocr_language = ocr_language
        
        # Check if we have any PDF processing capability
        if not (HAS_PDFPLUMBER or HAS_PYPDF2):
            logger.error("No PDF processing libraries available. Install pdfplumber or PyPDF2.")
        
    @property
    def supported_extensions(self) -> List[str]:
        return ['.pdf']
    
    def can_handle(self, file_path: Path) -> bool:
        if not file_path.suffix.lower() == '.pdf':
            return False
        
        # Only handle PDFs if we have at least one extraction method
        if not (HAS_PDFPLUMBER or HAS_PYPDF2):
            logger.warning(f"Cannot handle PDF {file_path}: No PDF libraries available")
            return False
            
        return True
    
    def extract(self, file_path: Path) -> Document:
        """Extract text and metadata from PDF file."""
        if not (HAS_PDFPLUMBER or HAS_PYPDF2):
            raise RuntimeError("No PDF processing libraries available. Please install pdfplumber or PyPDF2.")
        
        doc = Document.from_file(file_path)
        
        processing_info = ProcessingInfo(
            extraction_method='unknown',
            chunking_strategy='none',
            processing_timestamp=datetime.now()
        )
        
        content = ""
        metadata_updates = {}
        structure = {}
        
        # Try pdfplumber first if available
        if HAS_PDFPLUMBER:
            try:
                content, metadata_updates, structure = self._extract_with_pdfplumber(file_path)
                processing_info.extraction_method = 'pdfplumber'
                
                if not content.strip() and self.ocr_enabled:
                    logger.info(f"No extractable text found, falling back to OCR for {file_path}")
                    content, ocr_metadata = self._extract_with_ocr(file_path)
                    processing_info.extraction_method = 'ocr_tesseract'
                    metadata_updates.update(ocr_metadata)
                    
            except Exception as e:
                logger.warning(f"pdfplumber failed for {file_path}: {e}")
                processing_info.errors.append(f"pdfplumber failed: {str(e)}")
                content = ""
        
        # Try PyPDF2 as fallback if pdfplumber failed or isn't available
        if not content.strip() and HAS_PYPDF2:
            try:
                content, metadata_updates, structure = self._extract_with_pypdf2(file_path)
                processing_info.extraction_method = 'pypdf2'
                
                if not content.strip() and self.ocr_enabled:
                    logger.info(f"PyPDF2 yielded no text, falling back to OCR for {file_path}")
                    content, ocr_metadata = self._extract_with_ocr(file_path)
                    processing_info.extraction_method = 'ocr_tesseract'
                    metadata_updates.update(ocr_metadata)
                    
            except Exception as e:
                logger.error(f"PyPDF2 failed for {file_path}: {e}")
                processing_info.errors.append(f"pypdf2 failed: {str(e)}")
                
        # Final OCR attempt if no text was extracted
        if not content.strip() and self.ocr_enabled:
            try:
                content, ocr_metadata = self._extract_with_ocr(file_path)
                processing_info.extraction_method = 'ocr_tesseract'
                metadata_updates.update(ocr_metadata)
            except Exception as e:
                logger.error(f"OCR failed for {file_path}: {e}")
                processing_info.errors.append(f"OCR failed: {str(e)}")
        
        if not content.strip():
            logger.error(f"Failed to extract any text from {file_path}")
            processing_info.errors.append("No text content extracted")
                
        # Update document with extracted content and metadata
        doc.content = content
        doc.structure = structure
        
        # Update metadata
        for key, value in metadata_updates.items():
            setattr(doc.metadata, key, value)
            
        doc.metadata.char_count = len(content)
        doc.metadata.word_count = len(content.split()) if content else 0
        
        return doc
    
    def _extract_with_pdfplumber(self, file_path: Path) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Extract text using pdfplumber for better formatting preservation."""
        if not HAS_PDFPLUMBER:
            raise ImportError("pdfplumber not available")
            
        content_parts = []
        structure = {
            'pages': [],
            'tables': [],
            'figures': []
        }
        metadata_updates = {}
        
        with pdfplumber.open(file_path) as pdf:
            metadata_updates['page_count'] = len(pdf.pages)
            
            for i, page in enumerate(pdf.pages):
                page_info = {
                    'page_number': i + 1,
                    'width': page.width,
                    'height': page.height,
                    'has_text': False,
                    'has_tables': False,
                    'has_images': False
                }
                
                # Extract text
                text = page.extract_text()
                if text:
                    content_parts.append(f"[Page {i + 1}]\n{text}\n")
                    page_info['has_text'] = True
                
                # Extract tables
                tables = page.extract_tables()
                if tables:
                    page_info['has_tables'] = True
                    for j, table in enumerate(tables):
                        table_text = self._format_table(table)
                        content_parts.append(f"[Table {j + 1} on Page {i + 1}]\n{table_text}\n")
                        structure['tables'].append({
                            'page': i + 1,
                            'table_index': j + 1,
                            'rows': len(table),
                            'cols': len(table[0]) if table else 0
                        })
                
                # Check for images
                if page.images:
                    page_info['has_images'] = True
                    structure['figures'].extend([
                        {'page': i + 1, 'image_index': idx + 1} 
                        for idx in range(len(page.images))
                    ])
                
                structure['pages'].append(page_info)
        
        return '\n'.join(content_parts), metadata_updates, structure
    
    def _extract_with_pypdf2(self, file_path: Path) -> tuple[str, Dict[str, Any], Dict[str, Any]]:
        """Extract text using PyPDF2 as fallback."""
        if not HAS_PYPDF2:
            raise ImportError("PyPDF2 not available")
            
        content_parts = []
        structure = {'pages': []}
        metadata_updates = {}
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            metadata_updates['page_count'] = len(pdf_reader.pages)
            
            # Extract PDF metadata
            if pdf_reader.metadata:
                if pdf_reader.metadata.get('/Title'):
                    metadata_updates['title'] = pdf_reader.metadata['/Title']
                if pdf_reader.metadata.get('/Author'):
                    metadata_updates['author'] = pdf_reader.metadata['/Author']
                if pdf_reader.metadata.get('/CreationDate'):
                    metadata_updates['creation_date'] = pdf_reader.metadata['/CreationDate']
            
            for i, page in enumerate(pdf_reader.pages):
                try:
                    text = page.extract_text()
                    if text.strip():
                        content_parts.append(f"[Page {i + 1}]\n{text}\n")
                        structure['pages'].append({
                            'page_number': i + 1,
                            'has_text': True
                        })
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i + 1}: {e}")
                    structure['pages'].append({
                        'page_number': i + 1,
                        'has_text': False,
                        'error': str(e)
                    })
        
        return '\n'.join(content_parts), metadata_updates, structure
    
    def _extract_with_ocr(self, file_path: Path) -> tuple[str, Dict[str, Any]]:
        """Extract text using OCR as last resort."""
        if not self.ocr_enabled or not HAS_OCR:
            return "", {}
            
        if not HAS_PYMUPDF:
            logger.error("PyMuPDF not available for OCR extraction")
            return "", {}
            
        content_parts = []
        metadata_updates = {'extraction_method': 'ocr'}
        
        doc = fitz.open(file_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            
            # Convert page to image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x scale for better OCR
            img_data = pix.tobytes("png")
            
            # Save temporarily and run OCR
            img = Image.open(io.BytesIO(img_data))
            
            try:
                ocr_text = pytesseract.image_to_string(img, lang=self.ocr_language)
                if ocr_text.strip():
                    content_parts.append(f"[Page {page_num + 1} - OCR]\n{ocr_text}\n")
            except Exception as e:
                logger.warning(f"OCR failed for page {page_num + 1}: {e}")
        
        doc.close()
        
        return '\n'.join(content_parts), metadata_updates
    
    def _format_table(self, table: List[List[str]]) -> str:
        """Format extracted table as readable text."""
        if not table:
            return ""
            
        # Filter out None values and convert to strings
        formatted_rows = []
        for row in table:
            formatted_row = [str(cell) if cell is not None else "" for cell in row]
            formatted_rows.append(" | ".join(formatted_row))
        
        return "\n".join(formatted_rows)