import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from core.exporters.base_exporter import BaseExporter
from core.data_models import RAGEntry, FineTuneEntry

logger = logging.getLogger(__name__)


class JSONExporter(BaseExporter):
    """Standard JSON exporter."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.indent = self.config.get('indent', 2)
        self.sort_keys = self.config.get('sort_keys', False)
    
    @property
    def file_extension(self) -> str:
        return '.json' + self._get_compression_extension()
    
    @property
    def format_name(self) -> str:
        return 'JSON'
    
    def export_rag_data(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Export RAG entries to JSON format."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        logger.info(f"Exporting {len(entries)} RAG entries to {output_path}")
        
        data = {
            'format': 'rag',
            'version': '1.0',
            'entry_count': len(entries),
            'entries': [self._rag_entry_to_dict(entry) for entry in entries]
        }
        
        with self._open_file(output_path, 'w') as f:
            json.dump(
                data, 
                f, 
                indent=self.indent, 
                sort_keys=self.sort_keys, 
                ensure_ascii=False
            )
        
        logger.info(f"Successfully exported RAG data to {output_path}")
    
    def export_finetune_data(self, entries: List[FineTuneEntry], output_path: Path) -> None:
        """Export fine-tuning entries to JSON format."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        logger.info(f"Exporting {len(entries)} fine-tuning entries to {output_path}")
        
        data = {
            'format': 'finetune',
            'version': '1.0',
            'entry_count': len(entries),
            'entries': [self._finetune_entry_to_dict(entry) for entry in entries]
        }
        
        with self._open_file(output_path, 'w') as f:
            json.dump(
                data, 
                f, 
                indent=self.indent, 
                sort_keys=self.sort_keys, 
                ensure_ascii=False
            )
        
        logger.info(f"Successfully exported fine-tuning data to {output_path}")
    
    def _rag_entry_to_dict(self, entry: RAGEntry) -> Dict[str, Any]:
        """Convert RAGEntry to dictionary."""
        record = {
            'id': entry.id,
            'text': entry.text,
            'metadata': entry.metadata
        }
        
        # Include embedding if present
        if entry.embedding is not None:
            record['embedding'] = entry.embedding
        
        return record
    
    def _finetune_entry_to_dict(self, entry: FineTuneEntry) -> Dict[str, Any]:
        """Convert FineTuneEntry to dictionary."""
        return {
            'id': entry.id,
            'conversations': [
                {
                    'role': turn.role,
                    'content': turn.content
                }
                for turn in entry.conversations
            ],
            'metadata': entry.metadata
        }
    
    def export_with_schema(self, entries: List[RAGEntry], output_path: Path, include_schema: bool = True) -> None:
        """Export with JSON schema information."""
        self._ensure_output_dir(output_path)
        
        data = {
            'format': 'rag',
            'version': '1.0',
            'entry_count': len(entries),
            'entries': [self._rag_entry_to_dict(entry) for entry in entries]
        }
        
        if include_schema:
            data['schema'] = {
                'id': {'type': 'string', 'description': 'Unique identifier for the entry'},
                'text': {'type': 'string', 'description': 'The main text content'},
                'embedding': {
                    'type': 'array', 
                    'items': {'type': 'number'}, 
                    'description': 'Optional embedding vector'
                },
                'metadata': {
                    'type': 'object',
                    'description': 'Additional metadata about the entry'
                }
            }
        
        with self._open_file(output_path, 'w') as f:
            json.dump(
                data, 
                f, 
                indent=self.indent, 
                sort_keys=self.sort_keys, 
                ensure_ascii=False
            )
        
        logger.info(f"Successfully exported data with schema to {output_path}")
    
    def export_summary(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Export a summary of the dataset."""
        self._ensure_output_dir(output_path)
        
        # Calculate statistics
        total_entries = len(entries)
        total_chars = sum(len(entry.text) for entry in entries)
        total_words = sum(len(entry.text.split()) for entry in entries)
        
        # Metadata analysis
        sources = set()
        formats = set()
        
        for entry in entries:
            metadata = entry.metadata
            if 'source' in metadata:
                source_info = metadata['source']
                if 'filename' in source_info:
                    sources.add(source_info['filename'])
                if 'format' in source_info:
                    formats.add(source_info['format'])
        
        summary = {
            'dataset_summary': {
                'total_entries': total_entries,
                'total_characters': total_chars,
                'total_words': total_words,
                'avg_chars_per_entry': total_chars / total_entries if total_entries > 0 else 0,
                'avg_words_per_entry': total_words / total_entries if total_entries > 0 else 0,
                'unique_sources': len(sources),
                'source_formats': list(formats),
                'source_files': list(sources)
            },
            'sample_entries': [
                self._rag_entry_to_dict(entry) 
                for entry in entries[:3]  # First 3 entries as samples
            ]
        }
        
        with self._open_file(output_path, 'w') as f:
            json.dump(
                summary, 
                f, 
                indent=self.indent, 
                sort_keys=self.sort_keys, 
                ensure_ascii=False
            )
        
        logger.info(f"Successfully exported dataset summary to {output_path}")