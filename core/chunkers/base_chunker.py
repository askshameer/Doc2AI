from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime

from core.data_models import Document, Chunk, ChunkLocation, ProcessingInfo


class BaseChunker(ABC):
    """Abstract base class for document chunkers."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """
        Chunk a document into smaller pieces.
        
        Args:
            document: The document to chunk
            
        Returns:
            List of chunks
        """
        pass
    
    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Name of the chunking strategy."""
        pass
    
    def _create_chunk(
        self,
        document: Document,
        text: str,
        chunk_index: int,
        total_chunks: int,
        location: Optional[ChunkLocation] = None,
        semantic_tags: Optional[List[str]] = None
    ) -> Chunk:
        """Create a chunk with proper metadata."""
        chunk_id = f"{document.document_id}_chunk_{chunk_index:04d}"
        
        if location is None:
            location = ChunkLocation()
        
        processing_info = ProcessingInfo(
            extraction_method=self.strategy_name,
            chunking_strategy=self.strategy_name,
            processing_timestamp=datetime.now()
        )
        
        return Chunk(
            chunk_id=chunk_id,
            document_id=document.document_id,
            text=text,
            location=location,
            processing=processing_info,
            semantic_tags=semantic_tags or [],
            chunk_index=chunk_index,
            total_chunks=total_chunks
        )
    
    def _calculate_text_positions(self, text: str, full_text: str) -> tuple[int, int]:
        """Calculate character start and end positions of text within full_text."""
        start_pos = full_text.find(text)
        if start_pos == -1:
            return 0, len(text)
        
        end_pos = start_pos + len(text)
        return start_pos, end_pos
    
    def validate_config(self) -> List[str]:
        """Validate chunker configuration and return any warnings."""
        return []