"""
DocToAI - Document to AI Dataset Converter

A comprehensive tool for converting various document formats (PDF, ePub, DOCX, HTML, TXT) 
into structured datasets optimized for AI model fine-tuning and RAG systems.

Key Features:
- Multiple input format support (PDF, ePub, DOCX, HTML, TXT, Markdown)
- Intelligent text chunking strategies (fixed, semantic, hierarchical)
- Comprehensive metadata preservation
- Multiple output formats (JSONL, JSON, CSV, Parquet)
- Text cleaning and normalization
- Extensible plugin architecture

Usage:
    from doctoai import DocToAI
    
    app = DocToAI()
    app.process_documents(
        input_paths=[Path("document.pdf")],
        output_path=Path("dataset.jsonl"),
        mode="rag",
        chunk_strategy="semantic"
    )

Command Line:
    doctoai convert document.pdf --output dataset.jsonl --mode rag
"""

__version__ = "1.0.0"
__author__ = "DocToAI Team"
__email__ = "contact@doctoai.dev"

# Import main classes for easy access
from .cli import DocToAI
from .core.document_loader import DocumentLoader
from .core.base_extractor import ExtractorPlugin
from .core.text_processor import TextProcessor
from .utils.metadata_manager import MetadataManager

# Import data models
from .core.data_models import (
    Document,
    Chunk,
    RAGEntry,
    FineTuneEntry,
    DocumentMetadata,
    ChunkLocation,
    ProcessingInfo
)

# Import chunkers
from .core.chunkers.fixed_chunker import FixedChunker
from .core.chunkers.semantic_chunker import SemanticChunker
from .core.chunkers.hierarchical_chunker import HierarchicalChunker

# Import exporters
from .core.exporters.jsonl_exporter import JSONLExporter
from .core.exporters.json_exporter import JSONExporter
from .core.exporters.csv_exporter import CSVExporter
from .core.exporters.parquet_exporter import ParquetExporter

__all__ = [
    # Main classes
    "DocToAI",
    "DocumentLoader",
    "ExtractorPlugin", 
    "TextProcessor",
    "MetadataManager",
    
    # Data models
    "Document",
    "Chunk", 
    "RAGEntry",
    "FineTuneEntry",
    "DocumentMetadata",
    "ChunkLocation",
    "ProcessingInfo",
    
    # Chunkers
    "FixedChunker",
    "SemanticChunker", 
    "HierarchicalChunker",
    
    # Exporters
    "JSONLExporter",
    "JSONExporter",
    "CSVExporter",
    "ParquetExporter",
]