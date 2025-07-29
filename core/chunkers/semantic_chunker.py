from typing import List, Dict, Any, Optional
import re

try:
    import nltk
    from nltk.tokenize import sent_tokenize, word_tokenize
    HAS_NLTK = True
    
    # Download required NLTK data
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt')
        except Exception:
            HAS_NLTK = False
            
except ImportError:
    HAS_NLTK = False
    nltk = None
    sent_tokenize = None
    word_tokenize = None

from core.chunkers.base_chunker import BaseChunker
from core.data_models import Document, Chunk, ChunkLocation


class SemanticChunker(BaseChunker):
    """Semantic-aware chunker that respects sentence and paragraph boundaries."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Default configuration
        self.method = self.config.get('method', 'sentence_boundary')  # 'sentence_boundary', 'paragraph_boundary'
        self.min_size = self.config.get('min_size', 100)
        self.max_size = self.config.get('max_size', 1000)
        self.preserve_paragraphs = self.config.get('preserve_paragraphs', True)
        self.target_size = self.config.get('target_size', 500)
        self.language = self.config.get('language', 'english')
        
    def _tokenize_sentences(self, text: str) -> List[str]:
        """Tokenize text into sentences, with fallback if NLTK is not available."""
        if HAS_NLTK:
            try:
                return sent_tokenize(text, language=self.language)
            except Exception:
                pass
        
        # Fallback simple sentence tokenizer
        # Split on common sentence endings, but be careful with abbreviations
        sentences = []
        current_sentence = ""
        
        # Simple regex-based sentence splitting
        sentence_endings = re.split(r'([.!?]+)', text)
        
        for i in range(0, len(sentence_endings), 2):
            if i + 1 < len(sentence_endings):
                sentence = sentence_endings[i] + sentence_endings[i + 1]
            else:
                sentence = sentence_endings[i]
            
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
        
        return sentences
        
    @property
    def strategy_name(self) -> str:
        return 'semantic'
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document using semantic boundaries."""
        if not document.content:
            return []
        
        text = document.content.strip()
        
        if self.method == 'paragraph_boundary':
            return self._chunk_by_paragraphs(document, text)
        else:  # sentence_boundary
            return self._chunk_by_sentences(document, text)
    
    def _chunk_by_paragraphs(self, document: Document, text: str) -> List[Chunk]:
        """Chunk by paragraph boundaries with size constraints."""
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        char_position = 0
        
        for paragraph in paragraphs:
            paragraph_size = len(paragraph)
            
            # If adding this paragraph would exceed max_size, finalize current chunk
            if current_chunk and (current_size + paragraph_size > self.max_size):
                chunk_text = '\n\n'.join(current_chunk)
                
                if len(chunk_text) >= self.min_size:
                    location = ChunkLocation(
                        char_start=char_position - current_size,
                        char_end=char_position
                    )
                    
                    chunk = self._create_chunk(
                        document=document,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        location=location
                    )
                    
                    # Add semantic tags
                    chunk.semantic_tags.extend([
                        'paragraph_boundary',
                        f'paragraphs:{len(current_chunk)}'
                    ])
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = []
                current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
            char_position += paragraph_size + 2  # +2 for \n\n
            
            # If single paragraph exceeds max_size, split it
            if paragraph_size > self.max_size:
                # Split large paragraph into sentences
                sentence_chunks = self._split_large_paragraph(document, paragraph, chunk_index)
                chunks.extend(sentence_chunks)
                chunk_index += len(sentence_chunks)
                current_chunk = []
                current_size = 0
        
        # Handle remaining content
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            
            if len(chunk_text) >= self.min_size:
                location = ChunkLocation(
                    char_start=char_position - current_size,
                    char_end=char_position
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                chunk.semantic_tags.extend([
                    'paragraph_boundary',
                    f'paragraphs:{len(current_chunk)}'
                ])
                
                chunks.append(chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_sentences(self, document: Document, text: str) -> List[Chunk]:
        """Chunk by sentence boundaries with size constraints."""
        sentences = self._tokenize_sentences(text)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        char_position = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max_size, finalize current chunk
            if current_chunk and (current_size + sentence_size > self.max_size):
                chunk_text = ' '.join(current_chunk)
                
                if len(chunk_text) >= self.min_size:
                    location = ChunkLocation(
                        char_start=char_position - current_size,
                        char_end=char_position
                    )
                    
                    chunk = self._create_chunk(
                        document=document,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        total_chunks=0,  # Will be updated later
                        location=location
                    )
                    
                    # Add semantic tags
                    chunk.semantic_tags.extend([
                        'sentence_boundary',
                        f'sentences:{len(current_chunk)}'
                    ])
                    
                    chunks.append(chunk)
                    chunk_index += 1
                
                current_chunk = []
                current_size = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1  # +1 for space
            char_position += sentence_size + 1
            
            # If we're near target size and have multiple sentences, consider finalizing
            if (current_size >= self.target_size and 
                len(current_chunk) > 1 and 
                current_size < self.max_size):
                
                chunk_text = ' '.join(current_chunk)
                location = ChunkLocation(
                    char_start=char_position - current_size,
                    char_end=char_position
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                chunk.semantic_tags.extend([
                    'sentence_boundary',
                    f'sentences:{len(current_chunk)}'
                ])
                
                chunks.append(chunk)
                chunk_index += 1
                
                current_chunk = []
                current_size = 0
        
        # Handle remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            if len(chunk_text) >= self.min_size:
                location = ChunkLocation(
                    char_start=char_position - current_size,
                    char_end=char_position
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                chunk.semantic_tags.extend([
                    'sentence_boundary',
                    f'sentences:{len(current_chunk)}'
                ])
                
                chunks.append(chunk)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _split_large_paragraph(self, document: Document, paragraph: str, start_index: int) -> List[Chunk]:
        """Split a large paragraph into sentence-based chunks."""
        sentences = self._tokenize_sentences(paragraph)
        
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = start_index
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            sentence_size = len(sentence)
            
            # If adding this sentence would exceed max_size, finalize current chunk
            if current_chunk and (current_size + sentence_size > self.max_size):
                chunk_text = ' '.join(current_chunk)
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=ChunkLocation()
                )
                
                chunk.semantic_tags.extend([
                    'large_paragraph_split',
                    f'sentences:{len(current_chunk)}'
                ])
                
                chunks.append(chunk)
                chunk_index += 1
                
                current_chunk = []
                current_size = 0
            
            current_chunk.append(sentence)
            current_size += sentence_size + 1
        
        # Handle remaining content
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            chunk = self._create_chunk(
                document=document,
                text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                location=ChunkLocation()
            )
            
            chunk.semantic_tags.extend([
                'large_paragraph_split',
                f'sentences:{len(current_chunk)}'
            ])
            
            chunks.append(chunk)
        
        return chunks
    
    def validate_config(self) -> List[str]:
        """Validate chunker configuration."""
        warnings = []
        
        if self.min_size <= 0:
            warnings.append("min_size must be positive")
        
        if self.max_size <= self.min_size:
            warnings.append("max_size must be greater than min_size")
        
        if self.target_size < self.min_size or self.target_size > self.max_size:
            warnings.append("target_size should be between min_size and max_size")
        
        if self.method not in ['sentence_boundary', 'paragraph_boundary']:
            warnings.append("method must be 'sentence_boundary' or 'paragraph_boundary'")
        
        return warnings