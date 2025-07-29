from pathlib import Path
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

try:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False
    if TYPE_CHECKING:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    else:
        pd = None
        pa = None
        pq = None

from core.exporters.base_exporter import BaseExporter
from core.data_models import RAGEntry, FineTuneEntry

logger = logging.getLogger(__name__)


class ParquetExporter(BaseExporter):
    """Parquet exporter for efficient columnar storage."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.compression_type = self.config.get('compression_type', 'snappy')  # 'snappy', 'gzip', 'brotli', 'lz4'
        self.row_group_size = self.config.get('row_group_size', 10000)
        self.use_dictionary = self.config.get('use_dictionary', True)
        self.include_embeddings = self.config.get('include_embeddings', True)
    
    @property
    def file_extension(self) -> str:
        return '.parquet'
    
    @property
    def format_name(self) -> str:
        return 'Parquet'
    
    def export_rag_data(self, entries: List[RAGEntry], output_path: Path) -> None:
        """Export RAG entries to Parquet format."""
        if not HAS_PARQUET:
            raise RuntimeError("Parquet export requires pandas and pyarrow. Please install with: pip install pandas pyarrow")
        self._ensure_output_dir(output_path)
        
        logger.info(f"Exporting {len(entries)} RAG entries to {output_path}")
        
        if not entries:
            logger.warning("No entries to export")
            return
        
        # Convert to DataFrame
        df = self._rag_entries_to_dataframe(entries)
        
        # Define schema with appropriate types
        schema = self._create_rag_schema(entries[0])
        
        # Write to Parquet
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(
            table,
            output_path,
            compression=self.compression_type,
            row_group_size=self.row_group_size,
            use_dictionary=self.use_dictionary
        )
        
        logger.info(f"Successfully exported RAG data to {output_path}")
        
        # Log file size and compression info
        file_size = output_path.stat().st_size
        logger.info(f"Output file size: {file_size / (1024*1024):.2f} MB")
    
    def export_finetune_data(self, entries: List[FineTuneEntry], output_path: Path) -> None:
        """Export fine-tuning entries to Parquet format."""
        if not HAS_PARQUET:
            raise RuntimeError("Parquet export requires pandas and pyarrow. Please install with: pip install pandas pyarrow")
        self._ensure_output_dir(output_path)
        
        logger.info(f"Exporting {len(entries)} fine-tuning entries to {output_path}")
        
        if not entries:
            logger.warning("No entries to export")
            return
        
        # Convert to DataFrame
        df = self._finetune_entries_to_dataframe(entries)
        
        # Define schema
        schema = self._create_finetune_schema(entries[0])
        
        # Write to Parquet
        table = pa.Table.from_pandas(df, schema=schema)
        pq.write_table(
            table,
            output_path,
            compression=self.compression_type,
            row_group_size=self.row_group_size,
            use_dictionary=self.use_dictionary
        )
        
        logger.info(f"Successfully exported fine-tuning data to {output_path}")
    
    def _rag_entries_to_dataframe(self, entries: List[RAGEntry]) -> "pd.DataFrame":
        """Convert RAG entries to pandas DataFrame."""
        data = []
        
        for entry in entries:
            row = {
                'id': entry.id,
                'text': entry.text,
            }
            
            # Add embedding if present and configured to include
            if self.include_embeddings and entry.embedding is not None:
                row['embedding'] = entry.embedding
            
            # Flatten metadata
            for key, value in entry.metadata.items():
                if isinstance(value, dict):
                    # Flatten nested dictionaries
                    for subkey, subvalue in value.items():
                        # Convert complex types to strings for parquet compatibility
                        if isinstance(subvalue, (dict, list)):
                            row[f'metadata_{key}_{subkey}'] = str(subvalue)
                        else:
                            row[f'metadata_{key}_{subkey}'] = subvalue
                elif isinstance(value, list):
                    # Convert lists to strings (or handle specially)
                    row[f'metadata_{key}'] = str(value)
                else:
                    row[f'metadata_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _finetune_entries_to_dataframe(self, entries: List[FineTuneEntry]) -> "pd.DataFrame":
        """Convert fine-tuning entries to pandas DataFrame."""
        data = []
        
        for entry in entries:
            row = {'id': entry.id}
            
            # Add conversations as separate columns
            for i, turn in enumerate(entry.conversations):
                row[f'conversation_{i}_role'] = turn.role
                row[f'conversation_{i}_content'] = turn.content
            
            # Flatten metadata
            for key, value in entry.metadata.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        # Convert complex types to strings for parquet compatibility
                        if isinstance(subvalue, (dict, list)):
                            row[f'metadata_{key}_{subkey}'] = str(subvalue)
                        else:
                            row[f'metadata_{key}_{subkey}'] = subvalue
                elif isinstance(value, list):
                    row[f'metadata_{key}'] = str(value)
                else:
                    row[f'metadata_{key}'] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_rag_schema(self, sample_entry: RAGEntry) -> "pa.Schema":
        """Create PyArrow schema for RAG entries."""
        fields = [
            pa.field('id', pa.string()),
            pa.field('text', pa.string()),
        ]
        
        # Add embedding field if present
        if self.include_embeddings and sample_entry.embedding is not None:
            fields.append(pa.field('embedding', pa.list_(pa.float32())))
        
        # Add metadata fields based on sample
        for key, value in sample_entry.metadata.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    field_name = f'metadata_{key}_{subkey}'
                    # Convert complex types to strings for schema consistency
                    if isinstance(subvalue, (dict, list)):
                        field_type = pa.string()
                    else:
                        field_type = self._infer_arrow_type(subvalue)
                    fields.append(pa.field(field_name, field_type))
            else:
                field_name = f'metadata_{key}'
                field_type = self._infer_arrow_type(value)
                fields.append(pa.field(field_name, field_type))
        
        return pa.schema(fields)
    
    def _create_finetune_schema(self, sample_entry: FineTuneEntry) -> pa.Schema:
        """Create PyArrow schema for fine-tuning entries."""
        fields = [pa.field('id', pa.string())]
        
        # Add conversation fields
        for i in range(len(sample_entry.conversations)):
            fields.extend([
                pa.field(f'conversation_{i}_role', pa.string()),
                pa.field(f'conversation_{i}_content', pa.string())
            ])
        
        # Add metadata fields
        for key, value in sample_entry.metadata.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    field_name = f'metadata_{key}_{subkey}'
                    # Convert complex types to strings for schema consistency
                    if isinstance(subvalue, (dict, list)):
                        field_type = pa.string()
                    else:
                        field_type = self._infer_arrow_type(subvalue)
                    fields.append(pa.field(field_name, field_type))
            else:
                field_name = f'metadata_{key}'
                field_type = self._infer_arrow_type(value)
                fields.append(pa.field(field_name, field_type))
        
        return pa.schema(fields)
    
    def _infer_arrow_type(self, value: Any) -> pa.DataType:
        """Infer PyArrow data type from Python value."""
        if isinstance(value, bool):
            return pa.bool_()
        elif isinstance(value, int):
            return pa.int64()
        elif isinstance(value, float):
            return pa.float64()
        elif isinstance(value, str):
            return pa.string()
        elif isinstance(value, list):
            if value and isinstance(value[0], (int, float)):
                return pa.list_(pa.float32())
            else:
                return pa.list_(pa.string())
        else:
            return pa.string()  # Default to string
    
    def export_partitioned(
        self, 
        entries: List[RAGEntry], 
        output_dir: Path, 
        partition_cols: List[str] = None
    ) -> None:
        """Export to partitioned Parquet dataset."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Exporting {len(entries)} entries to partitioned dataset at {output_dir}")
        
        df = self._rag_entries_to_dataframe(entries)
        
        # Default partitioning by source filename
        if partition_cols is None:
            partition_cols = ['metadata_source_filename'] if 'metadata_source_filename' in df.columns else None
        
        # Create table
        table = pa.Table.from_pandas(df)
        
        # Write partitioned dataset
        pq.write_to_dataset(
            table,
            root_path=output_dir,
            partition_cols=partition_cols,
            compression=self.compression_type,
            use_dictionary=self.use_dictionary
        )
        
        logger.info(f"Successfully exported partitioned dataset to {output_dir}")
    
    def export_with_statistics(self, entries: List[RAGEntry], output_path: Path) -> Dict[str, Any]:
        """Export data and return statistics about the Parquet file."""
        self.export_rag_data(entries, output_path)
        
        # Read back to get statistics
        parquet_file = pq.ParquetFile(output_path)
        
        stats = {
            'num_rows': parquet_file.metadata.num_rows,
            'num_columns': parquet_file.metadata.num_columns,
            'num_row_groups': parquet_file.metadata.num_row_groups,
            'serialized_size': parquet_file.metadata.serialized_size,
            'file_size_bytes': output_path.stat().st_size,
            'compression': self.compression_type,
            'schema': str(parquet_file.schema),
            'created_by': parquet_file.metadata.created_by
        }
        
        logger.info(f"Parquet statistics: {stats}")
        return stats
    
    def read_parquet_sample(self, file_path: Path, num_rows: int = 10) -> "pd.DataFrame":
        """Read a sample of rows from Parquet file for inspection."""
        return pd.read_parquet(file_path, nrows=num_rows)
    
    def validate_parquet_file(self, file_path: Path) -> List[str]:
        """Validate a Parquet file."""
        warnings = []
        
        if not file_path.exists():
            warnings.append(f"File does not exist: {file_path}")
            return warnings
        
        try:
            parquet_file = pq.ParquetFile(file_path)
            
            # Basic validation
            if parquet_file.metadata.num_rows == 0:
                warnings.append("Parquet file contains no rows")
            
            # Check schema
            schema = parquet_file.schema
            required_fields = ['id', 'text']
            
            schema_fields = [field.name for field in schema]
            for required_field in required_fields:
                if required_field not in schema_fields:
                    warnings.append(f"Missing required field: {required_field}")
            
            logger.info(f"Validated Parquet file: {parquet_file.metadata.num_rows} rows, {parquet_file.metadata.num_columns} columns")
            
        except Exception as e:
            warnings.append(f"Error reading Parquet file: {e}")
        
        return warnings