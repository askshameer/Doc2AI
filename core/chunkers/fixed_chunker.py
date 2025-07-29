from typing import List, Dict, Any, Optional
import re

from core.chunkers.base_chunker import BaseChunker
from core.data_models import Document, Chunk, ChunkLocation


class FixedChunker(BaseChunker):
    """Fixed-size chunker with configurable overlap."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        
        # Default configuration
        self.chunk_size = self.config.get('chunk_size', 512)
        self.overlap = self.config.get('overlap', 50)
        self.split_on = self.config.get('split_on', 'tokens')  # 'tokens', 'characters', 'words'
        self.preserve_sentences = self.config.get('preserve_sentences', True)
        
    @property
    def strategy_name(self) -> str:
        return 'fixed_size'
    
    def chunk(self, document: Document) -> List[Chunk]:
        """Chunk document into fixed-size pieces with overlap."""
        if not document.content:
            return []
        
        text = document.content.strip()
        
        if self.split_on == 'characters':
            return self._chunk_by_characters(document, text)
        elif self.split_on == 'words':
            return self._chunk_by_words(document, text)
        else:  # tokens (approximate)
            return self._chunk_by_tokens(document, text)
    
    def _chunk_by_characters(self, document: Document, text: str) -> List[Chunk]:
        """Chunk by character count."""
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If we're preserving sentences, try to end at sentence boundary
            if self.preserve_sentences and end < len(text):
                end = self._find_sentence_boundary(text, end)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                location = ChunkLocation(
                    char_start=start,
                    char_end=end
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - self.overlap)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_words(self, document: Document, text: str) -> List[Chunk]:
        """Chunk by word count."""
        words = text.split()
        chunks = []
        chunk_index = 0
        
        start_word = 0
        while start_word < len(words):
            end_word = start_word + self.chunk_size
            
            # Get word slice
            chunk_words = words[start_word:end_word]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions
            char_start = len(' '.join(words[:start_word]))
            if start_word > 0:
                char_start += 1  # Account for space before first word
            char_end = char_start + len(chunk_text)
            
            location = ChunkLocation(
                char_start=char_start,
                char_end=char_end
            )
            
            chunk = self._create_chunk(
                document=document,
                text=chunk_text,
                chunk_index=chunk_index,
                total_chunks=0,  # Will be updated later
                location=location
            )
            chunks.append(chunk)
            chunk_index += 1
            
            # Move start position with overlap
            start_word = max(start_word + 1, end_word - self.overlap)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_tokens(self, document: Document, text: str) -> List[Chunk]:
        """
        Chunk by approximate token count.
        Uses a simple heuristic: 1 token ≈ 4 characters on average.
        """
        char_size = self.chunk_size * 4  # Approximate token-to-char conversion
        char_overlap = self.overlap * 4
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            end = start + char_size
            
            # If we're preserving sentences, try to end at sentence boundary
            if self.preserve_sentences and end < len(text):
                end = self._find_sentence_boundary(text, end)
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Estimate token count
                estimated_tokens = len(chunk_text) // 4
                
                location = ChunkLocation(
                    char_start=start,
                    char_end=end
                )
                
                chunk = self._create_chunk(
                    document=document,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    total_chunks=0,  # Will be updated later
                    location=location
                )
                
                # Add token count as semantic tag
                chunk.semantic_tags.append(f"estimated_tokens:{estimated_tokens}")
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            start = max(start + 1, end - char_overlap)
        
        # Update total_chunks for all chunks
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _find_sentence_boundary(self, text: str, position: int) -> int:
        """Find the nearest sentence boundary after the given position."""
        # Look for sentence-ending punctuation followed by space/newline
        sentence_end_pattern = r'[.!?]\s+'
        
        # Search for sentence boundaries near the position
        search_start = max(0, position - 100)  # Look back up to 100 chars
        search_end = min(len(text), position + 100)  # Look ahead up to 100 chars
        search_text = text[search_start:search_end]
        
        # Find all sentence boundaries in the search area
        matches = list(re.finditer(sentence_end_pattern, search_text))
        
        if not matches:
            return position
        
        # Find the match closest to our target position
        target_relative = position - search_start
        best_match = None
        best_distance = float('inf')
        
        for match in matches:
            match_pos = match.end()
            distance = abs(match_pos - target_relative)
            
            if distance < best_distance:
                best_distance = distance
                best_match = match
        
        if best_match:
            return search_start + best_match.end()
        
        return position
    
    def validate_config(self) -> List[str]:
        """Validate chunker configuration."""
        warnings = []
        
        if self.chunk_size <= 0:
            warnings.append("chunk_size must be positive")
        
        if self.overlap < 0:
            warnings.append("overlap cannot be negative")
        
        if self.overlap >= self.chunk_size:
            warnings.append("overlap should be less than chunk_size")
        
        if self.split_on not in ['tokens', 'characters', 'words']:
            warnings.append("split_on must be 'tokens', 'characters', or 'words'")
        
        return warnings