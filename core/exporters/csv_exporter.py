import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from core.exporters.base_exporter import BaseExporter
from core.data_models import RAGEntry, FineTuneEntry

logger = logging.getLogger(__name__)


class CSVExporter(BaseExporter):
    """CSV exporter with proper escaping for large text content."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.flatten_metadata = self.config.get('flatten_metadata', True)
        self.max_field_size = self.config.get('max_field_size', csv.field_size_limit())
        self.include_embeddings = self.config.get('include_embeddings', False)
        
        # Set CSV field size limit
        csv.field_size_limit(self.max_field_size)
    
    @property
    def file_extension(self) -> str:
        return '.csv' + self._get_compression_extension()
    
    @property
    def format_name(self) -> str:
        return 'CSV'
    
    def export_rag_data(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Export RAG entries to CSV format."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        logger.info(f"Exporting {len(entries)} RAG entries to {output_path}")
        
        if not entries:
            logger.warning("No entries to export")
            return
        
        # Determine column names from first entry
        fieldnames = self._get_rag_fieldnames(entries[0])
        
        with self._open_file(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=fieldnames, 
                quoting=csv.QUOTE_MINIMAL,
                escapechar='\\'
            )
            writer.writeheader()
            
            for entry in entries:
                row = self._rag_entry_to_row(entry)
                writer.writerow(row)
        
        logger.info(f"Successfully exported RAG data to {output_path}")
    
    def export_finetune_data(self, entries: List[FineTuneEntry], output_path: Path) -> None:
        """Export fine-tuning entries to CSV format."""
        self._ensure_output_dir(output_path)
        
        # Add compression extension if needed
        if self.compression and not str(output_path).endswith(self._get_compression_extension()):
            output_path = output_path.with_suffix(output_path.suffix + self._get_compression_extension())
        
        logger.info(f"Exporting {len(entries)} fine-tuning entries to {output_path}")
        
        if not entries:
            logger.warning("No entries to export")
            return
        
        # Determine column names from first entry
        fieldnames = self._get_finetune_fieldnames(entries[0])
        
        with self._open_file(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(
                f, 
                fieldnames=fieldnames, 
                quoting=csv.QUOTE_MINIMAL,
                escapechar='\\'
            )
            writer.writeheader()
            
            for entry in entries:
                row = self._finetune_entry_to_row(entry)
                writer.writerow(row)
        
        logger.info(f"Successfully exported fine-tuning data to {output_path}")
    
    def _get_rag_fieldnames(self, entry: RAGEntry) -> List[str]:
        """Get CSV fieldnames for RAG entries."""
        fieldnames = ['id', 'text']
        
        if self.include_embeddings and entry.embedding is not None:
            fieldnames.append('embedding')
        
        if self.flatten_metadata:
            # Flatten metadata fields
            flattened = self._flatten_dict(entry.metadata)
            fieldnames.extend(sorted(flattened.keys()))
        else:
            fieldnames.append('metadata')
        
        return fieldnames
    
    def _get_finetune_fieldnames(self, entry: FineTuneEntry) -> List[str]:
        """Get CSV fieldnames for fine-tuning entries."""
        fieldnames = ['id']
        
        # Add conversation fields based on max conversation length
        max_turns = len(entry.conversations)
        for i in range(max_turns):
            fieldnames.extend([f'conversation_{i}_role', f'conversation_{i}_content'])
        
        if self.flatten_metadata:
            # Flatten metadata fields
            flattened = self._flatten_dict(entry.metadata)
            fieldnames.extend(sorted(flattened.keys()))
        else:
            fieldnames.append('metadata')
        
        return fieldnames
    
    def _rag_entry_to_row(self, entry: RAGEntry) -> Dict[str, str]:
        """Convert RAGEntry to CSV row."""
        row = {
            'id': entry.id,
            'text': self._escape_text(entry.text)
        }
        
        if self.include_embeddings and entry.embedding is not None:
            # Convert embedding to JSON string
            row['embedding'] = json.dumps(entry.embedding)
        
        if self.flatten_metadata:
            flattened = self._flatten_dict(entry.metadata)
            row.update(flattened)
        else:
            row['metadata'] = json.dumps(entry.metadata, ensure_ascii=False)
        
        return row
    
    def _finetune_entry_to_row(self, entry: FineTuneEntry) -> Dict[str, str]:
        """Convert FineTuneEntry to CSV row."""
        row = {'id': entry.id}
        
        # Add conversation turns
        for i, turn in enumerate(entry.conversations):
            row[f'conversation_{i}_role'] = turn.role
            row[f'conversation_{i}_content'] = self._escape_text(turn.content)
        
        if self.flatten_metadata:
            flattened = self._flatten_dict(entry.metadata)
            row.update(flattened)
        else:
            row['metadata'] = json.dumps(entry.metadata, ensure_ascii=False)
        
        return row
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, str]:
        """Flatten nested dictionary for CSV export."""
        items = []
        
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert lists to JSON strings
                items.append((new_key, json.dumps(v, ensure_ascii=False)))
            elif v is None:
                items.append((new_key, ''))
            else:
                items.append((new_key, str(v)))
        
        return dict(items)
    
    def _escape_text(self, text: str) -> str:
        """Escape text for CSV, handling newlines and special characters."""
        if not text:
            return ''
        
        # Replace problematic characters
        text = text.replace('\r\n', '\\n')
        text = text.replace('\n', '\\n')
        text = text.replace('\r', '\\n')
        text = text.replace('\t', '\\t')
        
        # Truncate if too long
        if len(text) > self.max_field_size:
            text = text[:self.max_field_size - 3] + '...'
        
        return text
    
    def export_separate_files(
        self, 
        entries: List[RAGEntry], 
        output_dir: Path, 
        entries_per_file: int = 10000
    ) -> List[Path]:
        """Export entries to multiple CSV files for large datasets."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        file_paths = []
        
        for i in range(0, len(entries), entries_per_file):
            batch = entries[i:i + entries_per_file]
            batch_num = i // entries_per_file + 1
            
            output_path = output_dir / f"dataset_part_{batch_num:04d}.csv"
            if self.compression:
                output_path = output_path.with_suffix(
                    output_path.suffix + self._get_compression_extension()
                )
            
            self.export_rag_data(batch, output_path)
            file_paths.append(output_path)
        
        logger.info(f"Exported {len(entries)} entries to {len(file_paths)} CSV files")
        return file_paths
    
    def export_with_sample(self, entries: List[RAGEntry], output_path: Path, sample_size: int = 100) -> None:
        """Export full dataset and a sample for quick inspection."""
        # Export full dataset
        self.export_rag_data(entries, output_path)
        
        # Export sample
        sample_path = output_path.with_stem(output_path.stem + '_sample')
        sample_entries = entries[:sample_size]
        self.export_rag_data(sample_entries, sample_path)
        
        logger.info(f"Exported full dataset ({len(entries)} entries) and sample ({len(sample_entries)} entries)")
    
    def validate_csv_export(self, entries: List[RAGEntry]) -> List[str]:
        """Validate entries before CSV export."""
        warnings = super().validate_entries(entries)
        
        # Check for problematic characters
        problematic_count = 0
        for entry in entries[:100]:  # Sample check
            if any(ord(c) > 65535 for c in entry.text):  # Unicode beyond BMP
                problematic_count += 1
        
        if problematic_count > 0:
            warnings.append(f"Found {problematic_count} entries with extended Unicode characters")
        
        # Check field sizes
        oversized_count = 0
        for entry in entries[:100]:  # Sample check
            if len(entry.text) > self.max_field_size:
                oversized_count += 1
        
        if oversized_count > 0:
            warnings.append(f"Found {oversized_count} entries exceeding field size limit")
        
        return warnings