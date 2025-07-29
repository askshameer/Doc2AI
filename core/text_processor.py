import re
import unicodedata
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text cleaning and normalization pipeline."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.default_config = {
            'remove_extra_whitespace': True,
            'fix_encoding_issues': True,
            'normalize_unicode': True,
            'remove_control_characters': True,
            'preserve_paragraph_breaks': True,
            'remove_headers_footers': False,
            'remove_page_numbers': False,
            'expand_abbreviations': False,
            'fix_hyphenation': True,
            'standardize_quotes': True,
            'normalize_line_endings': True
        }
        
        # Merge user config with defaults
        self.settings = {**self.default_config, **self.config}
        
        # Compile regex patterns for performance
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns used in text processing."""
        # Multiple whitespace
        self.whitespace_pattern = re.compile(r'\s+')
        
        # Page numbers and headers/footers patterns
        self.page_number_pattern = re.compile(r'^\s*-?\s*\d+\s*-?\s*$', re.MULTILINE)
        self.header_footer_pattern = re.compile(
            r'^.*(?:Page \d+|Chapter \d+|\d+/\d+|©.*|\d{4}.*All rights reserved).*$',
            re.MULTILINE | re.IGNORECASE
        )
        
        # Hyphenation pattern (word split across lines)
        self.hyphenation_pattern = re.compile(r'(\w+)-\s*\n\s*(\w+)')
        
        # Quote standardization
        self.smart_quotes_pattern = re.compile(r'[""''`´]')
        
        # Control characters (except line breaks and tabs)
        self.control_chars_pattern = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]')
        
        # Common encoding issues
        self.encoding_fixes = {
            'â€™': "'",
            'â€œ': '"',
            'â€\x9d': '"',
            'â€"': '—',
            'â€"': '–',
            'Ã¡': 'á',
            'Ã©': 'é',
            'Ã­': 'í',
            'Ã³': 'ó',
            'Ãº': 'ú',
            'Ã±': 'ñ',
            'Ã¼': 'ü',
        }
        
        # Abbreviation expansion dictionary
        self.abbreviations = {
            'e.g.': 'for example',
            'i.e.': 'that is',
            'etc.': 'et cetera',
            'vs.': 'versus',
            'Dr.': 'Doctor',
            'Mr.': 'Mister',
            'Mrs.': 'Missus',
            'Ms.': 'Miss',
            'Prof.': 'Professor',
            'Inc.': 'Incorporated',
            'Corp.': 'Corporation',
            'Ltd.': 'Limited',
            'Co.': 'Company',
            'U.S.': 'United States',
            'U.K.': 'United Kingdom',
        }
    
    def process(self, text: str, custom_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Process text using the configured cleaning pipeline.
        
        Args:
            text: Input text to process
            custom_config: Optional config overrides for this specific processing
            
        Returns:
            Processed text
        """
        if not text:
            return text
        
        # Use custom config if provided
        settings = {**self.settings, **(custom_config or {})}
        
        processed_text = text
        
        # Step 1: Normalize line endings
        if settings['normalize_line_endings']:
            processed_text = self._normalize_line_endings(processed_text)
        
        # Step 2: Fix encoding issues
        if settings['fix_encoding_issues']:
            processed_text = self._fix_encoding_issues(processed_text)
        
        # Step 3: Unicode normalization
        if settings['normalize_unicode']:
            processed_text = self._normalize_unicode(processed_text)
        
        # Step 4: Remove control characters
        if settings['remove_control_characters']:
            processed_text = self._remove_control_characters(processed_text)
        
        # Step 5: Fix hyphenation
        if settings['fix_hyphenation']:
            processed_text = self._fix_hyphenation(processed_text)
        
        # Step 6: Standardize quotes
        if settings['standardize_quotes']:
            processed_text = self._standardize_quotes(processed_text)
        
        # Step 7: Remove headers and footers
        if settings['remove_headers_footers']:
            processed_text = self._remove_headers_footers(processed_text)
        
        # Step 8: Remove page numbers
        if settings['remove_page_numbers']:
            processed_text = self._remove_page_numbers(processed_text)
        
        # Step 9: Expand abbreviations
        if settings['expand_abbreviations']:
            processed_text = self._expand_abbreviations(processed_text)
        
        # Step 10: Clean whitespace (but preserve paragraph breaks if configured)
        if settings['remove_extra_whitespace']:
            processed_text = self._clean_whitespace(
                processed_text, 
                preserve_paragraphs=settings['preserve_paragraph_breaks']
            )
        
        return processed_text
    
    def _normalize_line_endings(self, text: str) -> str:
        """Normalize different line ending formats to \n."""
        # Convert Windows (\r\n) and old Mac (\r) line endings to Unix (\n)
        return text.replace('\r\n', '\n').replace('\r', '\n')
    
    def _fix_encoding_issues(self, text: str) -> str:
        """Fix common encoding issues."""
        for wrong, correct in self.encoding_fixes.items():
            text = text.replace(wrong, correct)
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters."""
        # Normalize to NFC (Canonical Decomposition followed by Canonical Composition)
        return unicodedata.normalize('NFC', text)
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove control characters except line breaks and tabs."""
        return self.control_chars_pattern.sub('', text)
    
    def _fix_hyphenation(self, text: str) -> str:
        """Fix words hyphenated across line breaks."""
        return self.hyphenation_pattern.sub(r'\1\2', text)
    
    def _standardize_quotes(self, text: str) -> str:
        """Standardize various quote characters to straight quotes."""
        return self.smart_quotes_pattern.sub('"', text)
    
    def _remove_headers_footers(self, text: str) -> str:
        """Remove likely headers and footers."""
        return self.header_footer_pattern.sub('', text)
    
    def _remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers."""
        return self.page_number_pattern.sub('', text)
    
    def _expand_abbreviations(self, text: str) -> str:
        """Expand common abbreviations."""
        for abbrev, expansion in self.abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        return text
    
    def _clean_whitespace(self, text: str, preserve_paragraphs: bool = True) -> str:
        """Clean excessive whitespace while optionally preserving paragraph breaks."""
        if preserve_paragraphs:
            # Split into paragraphs first
            paragraphs = re.split(r'\n\s*\n', text)
            cleaned_paragraphs = []
            
            for paragraph in paragraphs:
                # Clean whitespace within paragraph
                cleaned = self.whitespace_pattern.sub(' ', paragraph).strip()
                if cleaned:
                    cleaned_paragraphs.append(cleaned)
            
            return '\n\n'.join(cleaned_paragraphs)
        else:
            # Simple whitespace cleaning
            return self.whitespace_pattern.sub(' ', text).strip()
    
    def get_processing_stats(self, original: str, processed: str) -> Dict[str, Any]:
        """Get statistics about the processing changes."""
        return {
            'original_length': len(original),
            'processed_length': len(processed),
            'chars_removed': len(original) - len(processed),
            'original_words': len(original.split()),
            'processed_words': len(processed.split()),
            'words_changed': len(original.split()) - len(processed.split()),
            'original_lines': len(original.splitlines()),
            'processed_lines': len(processed.splitlines())
        }
    
    def validate_processing(self, text: str) -> List[str]:
        """Validate that processing won't cause issues."""
        warnings = []
        
        if len(text) > 1_000_000:  # 1MB
            warnings.append("Text is very large, processing may be slow")
        
        # Check for unusual character distributions
        control_chars = len(self.control_chars_pattern.findall(text))
        if control_chars > len(text) * 0.01:  # More than 1% control characters
            warnings.append(f"High number of control characters found: {control_chars}")
        
        # Check for potential encoding issues
        encoding_issues = sum(1 for issue in self.encoding_fixes.keys() if issue in text)
        if encoding_issues > 0:
            warnings.append(f"Potential encoding issues detected: {encoding_issues} patterns found")
        
        return warnings