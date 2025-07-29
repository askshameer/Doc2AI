import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from core.exporters.base_exporter import BaseExporter
from core.data_models import RAGEntry, FineTuneEntry

logger = logging.getLogger(__name__)


class JSONLExporter(BaseExporter):
    """JSONL (newline-delimited JSON) exporter."""
    
    @property
    def file_extension(self) -> str:
        return '.jsonl' + self._get_compression_extension()
    
    @property
    def format_name(self) -> str:
        return 'JSONL'
    
    def export_rag_data(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Export RAG entries to JSONL format."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        logger.info(f"Exporting {len(entries)} RAG entries to {output_path}")
        
        with self._open_file(output_path, 'w') as f:
            for entry in entries:
                rag_record = self._rag_entry_to_dict(entry)
                f.write(json.dumps(rag_record, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully exported RAG data to {output_path}")
    
    def export_finetune_data(self, entries: List[Any], output_path: Path) -> None:
        """Export fine-tuning entries to JSONL format."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        logger.info(f"Exporting {len(entries)} fine-tuning entries to {output_path}")
        
        with self._open_file(output_path, 'w') as f:
            for entry in entries:
                # Handle both FineTuneEntry objects and dictionaries
                if isinstance(entry, dict):
                    ft_record = entry
                else:
                    ft_record = self._finetune_entry_to_dict(entry)
                f.write(json.dumps(ft_record, ensure_ascii=False) + '\n')
        
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
    
    def export_mixed_data(
        self, 
        rag_entries: List[RAGEntry], 
        finetune_entries: List[FineTuneEntry], 
        output_path: Path
    ) -> None:
        """Export mixed RAG and fine-tuning data to a single JSONL file."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        total_entries = len(rag_entries) + len(finetune_entries)
        logger.info(f"Exporting {total_entries} mixed entries to {output_path}")
        
        with self._open_file(output_path, 'w') as f:
            # Export RAG entries with type marker
            for entry in rag_entries:
                record = self._rag_entry_to_dict(entry)
                record['entry_type'] = 'rag'
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
            
            # Export fine-tuning entries with type marker
            for entry in finetune_entries:
                record = self._finetune_entry_to_dict(entry)
                record['entry_type'] = 'finetune'
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully exported mixed data to {output_path}")
    
    def append_entries(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Append entries to an existing JSONL file."""
        logger.info(f"Appending {len(entries)} entries to {output_path}")
        
        with self._open_file(output_path, 'a') as f:
            for entry in entries:
                record = self._rag_entry_to_dict(entry)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Successfully appended entries to {output_path}")
    
    def validate_jsonl_file(self, file_path: Path) -> List[str]:
        """Validate an existing JSONL file."""
        warnings = []
        
        if not file_path.exists():
            warnings.append(f"File does not exist: {file_path}")
            return warnings
        
        try:
            with self._open_file(file_path, 'r') as f:
                line_count = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        json.loads(line)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        warnings.append(f"Invalid JSON on line {line_num}: {e}")
                
                logger.info(f"Validated {line_count} entries in {file_path}")
                
        except Exception as e:
            warnings.append(f"Error reading file {file_path}: {e}")
        
        return warnings