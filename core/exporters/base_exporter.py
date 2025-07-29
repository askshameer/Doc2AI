from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging

from core.data_models import RAGEntry, FineTuneEntry

logger = logging.getLogger(__name__)


class BaseExporter(ABC):
    """Abstract base class for data exporters."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.compression = self.config.get('compression', None)  # 'gzip', 'bz2', 'xz'
        
    @abstractmethod
    def export_rag_data(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Export RAG entries to the specified format."""
        pass
    
    @abstractmethod
    def export_finetune_data(self, entries: List[FineTuneEntry], output_path: Path) -> None:
        """Export fine-tuning entries to the specified format."""
        pass
    
    @property
    @abstractmethod
    def file_extension(self) -> str:
        """File extension for this format."""
        pass
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Human-readable format name."""
        pass
    
    def _ensure_output_dir(self, output_path: Path) -> None:
        """Ensure the output directory exists."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def _get_compression_extension(self) -> str:
        """Get the compression extension."""
        if self.compression == 'gzip':
            return '.gz'
        elif self.compression == 'bz2':
            return '.bz2'
        elif self.compression == 'xz':
            return '.xz'
        return ''
    
    def _open_file(self, file_path: Path, mode: str = 'w'):
        """Open file with optional compression."""
        if self.compression == 'gzip':
            import gzip
            return gzip.open(file_path, mode + 't', encoding='utf-8')
        elif self.compression == 'bz2':
            import bz2
            return bz2.open(file_path, mode + 't', encoding='utf-8')
        elif self.compression == 'xz':
            import lzma
            return lzma.open(file_path, mode + 't', encoding='utf-8')
        else:
            return open(file_path, mode, encoding='utf-8')
    
    def validate_entries(self, entries: List[Union[RAGEntry, FineTuneEntry]]) -> List[str]:
        """Validate entries and return any warnings."""
        warnings = []
        
        if not entries:
            warnings.append("No entries to export")
            return warnings
        
        # Check for duplicate IDs
        ids = [entry.id for entry in entries]
        if len(ids) != len(set(ids)):
            warnings.append("Duplicate entry IDs found")
        
        # Check for empty content
        empty_entries = 0
        for entry in entries:
            if isinstance(entry, RAGEntry):
                if not entry.text.strip():
                    empty_entries += 1
            elif isinstance(entry, FineTuneEntry):
                if not entry.conversations:
                    empty_entries += 1
        
        if empty_entries > 0:
            warnings.append(f"{empty_entries} entries have empty content")
        
        return warnings