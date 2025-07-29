from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from core.data_models import Document, Chunk, RAGEntry, FineTuneEntry, ConversationTurn

logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages metadata enrichment and formatting for different output formats."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.include_fields = self.config.get('include_fields', 'all')
        self.custom_fields = self.config.get('custom_fields', {})
        
    def enrich_chunk_metadata(self, chunk: Chunk, document: Document) -> Dict[str, Any]:
        """Enrich chunk metadata with comprehensive information."""
        metadata = {
            'chunk_id': chunk.chunk_id,
            'document_id': chunk.document_id,
            'source': self._get_source_metadata(document),
            'location': self._get_location_metadata(chunk),
            'processing': self._get_processing_metadata(chunk),
            'content_stats': self._get_content_stats(chunk),
            'semantic_info': self._get_semantic_info(chunk)
        }
        
        # Add document-level metadata if requested
        if self.include_fields == 'all' or 'document' in self.include_fields:
            metadata['document_metadata'] = self._get_document_metadata(document)
        
        # Add custom fields
        for field_name, field_value in self.custom_fields.items():
            metadata[field_name] = field_value
        
        return metadata
    
    def create_rag_entry(self, chunk: Chunk, document: Document, embedding: Optional[List[float]] = None) -> RAGEntry:
        """Create a RAG entry from a chunk."""
        metadata = self.enrich_chunk_metadata(chunk, document)
        
        return RAGEntry(
            id=chunk.chunk_id,
            text=chunk.text,
            embedding=embedding,
            metadata=metadata
        )
    
    def create_finetune_entry(
        self, 
        chunk: Chunk, 
        document: Document, 
        model_id: str = 'gpt-3.5-turbo',
        question_type: str = 'general',
        system_message: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a fine-tuning entry from a chunk using model-specific templates."""
        from core.model_templates import ModelTemplateManager
        
        template_manager = ModelTemplateManager()
        
        # Format using the appropriate model template
        entry = template_manager.format_for_model(
            model_id=model_id,
            chunk=chunk,
            document=document,
            question_type=question_type,
            system_message=system_message,
            **kwargs
        )
        
        # Add DocToAI-specific metadata
        doctoai_metadata = self.enrich_chunk_metadata(chunk, document)
        doctoai_metadata.update({
            'model_id': model_id,
            'question_type': question_type,
            'template_version': '2.0'
        })
        
        # Merge with template metadata
        if 'metadata' in entry:
            entry['metadata'].update(doctoai_metadata)
        else:
            entry['metadata'] = doctoai_metadata
        
        # Validate the entry
        warnings = template_manager.validate_for_model(model_id, entry)
        if warnings:
            entry['metadata']['validation_warnings'] = warnings
        
        return entry
    
    def _get_source_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract source metadata from document."""
        return {
            'filename': document.metadata.filename,
            'file_hash': document.metadata.file_hash,
            'file_size': document.metadata.file_size,
            'format': document.metadata.format,
            'created_date': document.metadata.created_date.isoformat() if document.metadata.created_date else None,
            'modified_date': document.metadata.modified_date.isoformat() if document.metadata.modified_date else None,
            'encoding': document.metadata.encoding,
            'language': document.metadata.language
        }
    
    def _get_location_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract location metadata from chunk."""
        location = chunk.location
        return {
            'page': location.page,
            'section': location.section,
            'subsection': location.subsection,
            'paragraph': location.paragraph,
            'char_start': location.char_start,
            'char_end': location.char_end,
            'line_start': location.line_start,
            'line_end': location.line_end
        }
    
    def _get_processing_metadata(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract processing metadata from chunk."""
        processing = chunk.processing
        return {
            'extraction_method': processing.extraction_method,
            'chunking_strategy': processing.chunking_strategy,
            'processing_timestamp': processing.processing_timestamp.isoformat(),
            'quality_score': processing.quality_score,
            'confidence_score': processing.confidence_score,
            'errors': processing.errors,
            'warnings': processing.warnings
        }
    
    def _get_content_stats(self, chunk: Chunk) -> Dict[str, Any]:
        """Calculate content statistics for chunk."""
        text = chunk.text
        words = text.split()
        
        # Basic stats
        stats = {
            'char_count': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'paragraph_count': len([p for p in text.split('\n\n') if p.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'chunk_index': chunk.chunk_index,
            'total_chunks': chunk.total_chunks
        }
        
        # Reading level estimation (simple Flesch approximation)
        if stats['sentence_count'] > 0 and stats['word_count'] > 0:
            avg_sentence_length = stats['word_count'] / stats['sentence_count']
            syllable_count = self._estimate_syllables(text)
            avg_syllables_per_word = syllable_count / stats['word_count']
            
            flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
            stats['estimated_reading_level'] = max(0, min(100, flesch_score))
        
        return stats
    
    def _get_semantic_info(self, chunk: Chunk) -> Dict[str, Any]:
        """Extract semantic information from chunk."""
        return {
            'semantic_tags': chunk.semantic_tags,
            'contains_code': 'code' in chunk.text.lower() or '```' in chunk.text,
            'contains_table': 'table' in chunk.semantic_tags or '|' in chunk.text,
            'contains_list': any(tag.startswith('list') for tag in chunk.semantic_tags),
            'is_header': any(tag.startswith('header') for tag in chunk.semantic_tags),
            'section_type': self._infer_section_type(chunk)
        }
    
    def _get_document_metadata(self, document: Document) -> Dict[str, Any]:
        """Extract comprehensive document metadata."""
        metadata = {
            'title': getattr(document.metadata, 'title', None),
            'author': getattr(document.metadata, 'author', None),
            'subject': getattr(document.metadata, 'subject', None),
            'keywords': getattr(document.metadata, 'keywords', None),
            'page_count': document.metadata.page_count,
            'word_count': document.metadata.word_count,
            'char_count': document.metadata.char_count,
            'structure_info': self._summarize_structure(document.structure)
        }
        
        return {k: v for k, v in metadata.items() if v is not None}
    
    def _generate_qa_pair(self, chunk: Chunk, document: Document) -> tuple[str, str]:
        """Generate a question-answer pair from chunk content."""
        text = chunk.text
        
        # Simple heuristics for question generation
        if chunk.location.section:
            question = f"What information is provided in the {chunk.location.section} section?"
        elif any('table' in tag for tag in chunk.semantic_tags):
            question = "What data is presented in this table?"
        elif any('list' in tag for tag in chunk.semantic_tags):
            question = "What items are listed in this section?"
        elif len(text.split()) > 100:
            question = "Please summarize the key points from this content."
        else:
            question = "What does this text explain?"
        
        # Answer is the chunk content, potentially truncated
        answer = text
        if len(answer) > 2000:  # Truncate very long answers
            answer = answer[:2000] + "..."
        
        return question, answer
    
    def _estimate_syllables(self, text: str) -> int:
        """Estimate syllable count for readability calculation."""
        # Simple syllable estimation
        words = text.lower().split()
        syllable_count = 0
        
        for word in words:
            # Remove punctuation
            word = ''.join(c for c in word if c.isalpha())
            if not word:
                continue
            
            # Count vowel groups
            vowels = 'aeiouy'
            prev_was_vowel = False
            word_syllables = 0
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_was_vowel:
                    word_syllables += 1
                prev_was_vowel = is_vowel
            
            # Handle silent e
            if word.endswith('e') and word_syllables > 1:
                word_syllables -= 1
            
            # Every word has at least one syllable
            syllable_count += max(1, word_syllables)
        
        return syllable_count
    
    def _infer_section_type(self, chunk: Chunk) -> str:
        """Infer the type of section from chunk content and tags."""
        text_lower = chunk.text.lower()
        tags = chunk.semantic_tags
        
        if any('header' in tag for tag in tags):
            return 'header'
        elif any('table' in tag for tag in tags):
            return 'table'
        elif any('list' in tag for tag in tags):
            return 'list'
        elif any('code' in tag for tag in tags) or '```' in chunk.text:
            return 'code'
        elif 'introduction' in text_lower or 'overview' in text_lower:
            return 'introduction'
        elif 'conclusion' in text_lower or 'summary' in text_lower:
            return 'conclusion'
        elif 'method' in text_lower or 'approach' in text_lower:
            return 'methodology'
        elif 'result' in text_lower or 'finding' in text_lower:
            return 'results'
        elif 'reference' in text_lower or 'bibliography' in text_lower:
            return 'references'
        else:
            return 'content'
    
    def _summarize_structure(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Summarize document structure information."""
        summary = {}
        
        if 'chapters' in structure:
            summary['chapter_count'] = len(structure['chapters'])
        
        if 'headings' in structure:
            headings = structure['headings']
            summary['heading_count'] = len(headings)
            summary['max_heading_level'] = max((h.get('level', 1) for h in headings), default=1)
        
        if 'tables' in structure:
            summary['table_count'] = len(structure['tables'])
        
        if 'paragraphs' in structure:
            summary['paragraph_count'] = len(structure['paragraphs'])
        
        if 'pages' in structure:
            summary['page_count'] = len(structure['pages'])
        
        return summary
    
    def filter_metadata(self, metadata: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
        """Filter metadata to include only specified fields."""
        if self.include_fields == 'all':
            return metadata
        
        if isinstance(self.include_fields, list):
            fields = self.include_fields
        
        filtered = {}
        for field in fields:
            if field in metadata:
                filtered[field] = metadata[field]
        
        return filtered