"""
Base extractor interface for DocToAI document extractors.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List
from core.data_models import Document


class ExtractorPlugin(ABC):
    """Abstract base class for document extractors."""
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this extractor can handle the file."""
        pass
    
    @abstractmethod
    def extract(self, file_path: Path) -> Document:
        """Extract text and metadata from the file."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions."""
        pass