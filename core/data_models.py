from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import hashlib


@dataclass
class DocumentMetadata:
    filename: str
    file_hash: str
    file_size: int
    created_date: Optional[datetime] = None
    modified_date: Optional[datetime] = None
    format: Optional[str] = None
    language: Optional[str] = None
    encoding: Optional[str] = None
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    char_count: Optional[int] = None


@dataclass
class ChunkLocation:
    page: Optional[int] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    paragraph: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None


@dataclass
class ProcessingInfo:
    extraction_method: str
    chunking_strategy: str
    processing_timestamp: datetime
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    location: ChunkLocation
    processing: ProcessingInfo
    semantic_tags: List[str] = field(default_factory=list)
    chunk_index: int = 0
    total_chunks: int = 0
    
    
@dataclass
class Document:
    document_id: str
    content: str
    metadata: DocumentMetadata
    structure: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Chunk] = field(default_factory=list)
    
    @classmethod
    def from_file(cls, file_path: Path) -> 'Document':
        """Create a Document from a file path with basic metadata."""
        stat = file_path.stat()
        
        # Calculate file hash
        with open(file_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        metadata = DocumentMetadata(
            filename=file_path.name,
            file_hash=file_hash,
            file_size=stat.st_size,
            created_date=datetime.fromtimestamp(stat.st_ctime),
            modified_date=datetime.fromtimestamp(stat.st_mtime),
            format=file_path.suffix.lower()
        )
        
        return cls(
            document_id=f"doc_{file_hash[:8]}",
            content="",
            metadata=metadata
        )


@dataclass
class ConversationTurn:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class FineTuneEntry:
    id: str
    conversations: List[ConversationTurn]
    metadata: Dict[str, Any]


@dataclass
class RAGEntry:
    id: str
    text: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)