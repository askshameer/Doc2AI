#!/usr/bin/env python3
"""
DocToAI - Document to AI Dataset Converter
Command-line interface for converting various document formats to AI-ready datasets.

Author: Shameer Mohammed
Email: mohammed.shameer@gmail.com
GitHub: https://github.com/askshameer/Doc2AI
"""

import click
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple fallback progress indicator
    class tqdm:
        def __init__(self, iterable, desc="Processing"):
            self.iterable = iterable
            self.desc = desc
            self.total = len(iterable) if hasattr(iterable, '__len__') else 0
            self.count = 0
            
        def __enter__(self):
            print(f"{self.desc}...")
            return self
            
        def __exit__(self, *args):
            print(f"Completed processing {self.count} items")
            
        def __iter__(self):
            for item in self.iterable:
                yield item
                self.count += 1
                if self.count % 10 == 0 or self.count == self.total:
                    print(f"  Processed {self.count}/{self.total}")
                    
        def set_description(self, desc):
            pass  # Simple fallback doesn't show individual file progress

# Import core components
from core.document_loader import DocumentLoader
from core.text_processor import TextProcessor
from core.chunkers.fixed_chunker import FixedChunker
from core.chunkers.semantic_chunker import SemanticChunker
from core.chunkers.hierarchical_chunker import HierarchicalChunker
from core.exporters.jsonl_exporter import JSONLExporter
from core.exporters.json_exporter import JSONExporter
from core.exporters.csv_exporter import CSVExporter
from utils.metadata_manager import MetadataManager

# Import parquet exporter conditionally
try:
    import pandas
    import pyarrow
    from core.exporters.parquet_exporter import ParquetExporter
    HAS_PARQUET_EXPORTER = True
except ImportError:
    HAS_PARQUET_EXPORTER = False
    ParquetExporter = None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Available output formats
AVAILABLE_FORMATS = ['jsonl', 'json', 'csv']
if HAS_PARQUET_EXPORTER:
    AVAILABLE_FORMATS.append('parquet')


class DocToAI:
    """Main application class for document processing."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_processor = TextProcessor(self.config.get('text_processing', {}))
        self.metadata_manager = MetadataManager(self.config.get('metadata', {}))
        
        # Initialize chunkers
        self.chunkers = {
            'fixed': FixedChunker(self.config.get('chunking', {}).get('fixed', {})),
            'semantic': SemanticChunker(self.config.get('chunking', {}).get('semantic', {})),
            'hierarchical': HierarchicalChunker(self.config.get('chunking', {}).get('hierarchical', {}))
        }
        
        # Initialize exporters
        self.exporters = {
            'jsonl': JSONLExporter(self.config.get('export', {}).get('jsonl', {})),
            'json': JSONExporter(self.config.get('export', {}).get('json', {})),
            'csv': CSVExporter(self.config.get('export', {}).get('csv', {}))
        }
        
        # Add parquet exporter if available
        if HAS_PARQUET_EXPORTER:
            self.exporters['parquet'] = ParquetExporter(self.config.get('export', {}).get('parquet', {}))
    
    def process_documents(
        self,
        input_paths: List[Path],
        output_path: Path,
        mode: str = 'rag',
        chunk_strategy: str = 'semantic',
        output_format: str = 'jsonl',
        clean_text: bool = True,
        verbose: bool = False,
        **kwargs
    ) -> None:
        """Process documents and generate dataset."""
        
        if verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        logger.info(f"Processing {len(input_paths)} input paths")
        logger.info(f"Output: {output_path}")
        logger.info(f"Mode: {mode}, Chunking: {chunk_strategy}, Format: {output_format}")
        
        # Collect all document files
        document_files = []
        for input_path in input_paths:
            if input_path.is_file():
                document_files.append(input_path)
            elif input_path.is_dir():
                # Find all supported files in directory
                for ext in self.document_loader.get_supported_formats():
                    document_files.extend(input_path.rglob(f"*{ext}"))
        
        if not document_files:
            logger.error("No supported documents found")
            return
        
        logger.info(f"Found {len(document_files)} document files")
        
        # Process documents
        all_entries = []
        
        with tqdm(document_files, desc="Processing documents") as pbar:
            for doc_path in pbar:
                pbar.set_description(f"Processing {doc_path.name}")
                
                try:
                    # Load document
                    document = self.document_loader.load_document(doc_path)
                    if not document:
                        logger.warning(f"Failed to load document: {doc_path}")
                        continue
                    
                    # Clean text if requested
                    if clean_text:
                        document.content = self.text_processor.process(document.content)
                    
                    # Chunk document
                    chunker = self.chunkers[chunk_strategy]
                    chunks = chunker.chunk(document)
                    
                    if not chunks:
                        logger.warning(f"No chunks generated for document: {doc_path}")
                        continue
                    
                    # Convert to output format
                    for chunk in chunks:
                        if mode == 'rag':
                            entry = self.metadata_manager.create_rag_entry(chunk, document)
                            all_entries.append(entry)
                        elif mode == 'finetune':
                            entry = self.metadata_manager.create_finetune_entry(
                                chunk, document, 
                                model_id=kwargs.get('model_id', 'gpt-3.5-turbo'),
                                question_type=kwargs.get('question_type', 'general')
                            )
                            all_entries.append(entry)
                    
                    logger.debug(f"Generated {len(chunks)} chunks from {doc_path}")
                    
                except Exception as e:
                    logger.error(f"Error processing {doc_path}: {e}")
                    continue
        
        if not all_entries:
            logger.error("No entries generated")
            return
        
        logger.info(f"Generated {len(all_entries)} total entries")
        
        # Export data
        exporter = self.exporters[output_format]
        
        try:
            if mode == 'rag':
                exporter.export_rag_data(all_entries, output_path)
            elif mode == 'finetune':
                exporter.export_finetune_data(all_entries, output_path)
            
            logger.info(f"Successfully exported dataset to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            sys.exit(1)


@click.group()
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config, verbose):
    """DocToAI - Document to AI Dataset Converter"""
    
    # Load configuration
    config_data = {}
    if config:
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    
    ctx.ensure_object(dict)
    ctx.obj['config'] = config_data
    ctx.obj['verbose'] = verbose


@cli.command()
@click.argument('input_paths', nargs=-1, required=True, type=click.Path(exists=True, path_type=Path))
@click.option('--output', '-o', required=True, type=click.Path(path_type=Path), help='Output file path')
@click.option('--mode', type=click.Choice(['rag', 'finetune']), default='rag', help='Dataset mode')
@click.option('--chunk-strategy', type=click.Choice(['fixed', 'semantic', 'hierarchical']), default='semantic', help='Chunking strategy')
@click.option('--chunk-size', type=int, default=512, help='Chunk size (for fixed chunking)')
@click.option('--overlap', type=int, default=50, help='Chunk overlap (for fixed chunking)')
@click.option('--output-format', type=click.Choice(AVAILABLE_FORMATS), default='jsonl', help='Output format')
@click.option('--clean-text/--no-clean-text', default=True, help='Enable text cleaning')
@click.option('--model-id', default='gpt-3.5-turbo', help='Target model for fine-tuning (e.g., gpt-3.5-turbo, claude-3-haiku, llama-2-7b)')
@click.option('--question-type', default='general', type=click.Choice(['general', 'summary', 'explanation', 'context', 'analytical']), help='Type of questions to generate')
@click.option('--system-message', help='Custom system message for fine-tuning')
@click.option('--parallel', type=int, help='Number of parallel processes')
@click.pass_context
def convert(ctx, input_paths, output, mode, chunk_strategy, chunk_size, overlap, output_format, clean_text, model_id, question_type, system_message, parallel):
    """Convert documents to AI dataset format."""
    
    config = ctx.obj['config']
    verbose = ctx.obj['verbose']
    
    # Update config with command line options
    if 'chunking' not in config:
        config['chunking'] = {}
    if 'fixed' not in config['chunking']:
        config['chunking']['fixed'] = {}
    
    config['chunking']['fixed']['chunk_size'] = chunk_size
    config['chunking']['fixed']['overlap'] = overlap
    
    # Initialize application
    app = DocToAI(config)
    
    # Process documents
    app.process_documents(
        input_paths=list(input_paths),
        output_path=output,
        mode=mode,
        chunk_strategy=chunk_strategy,
        output_format=output_format,
        clean_text=clean_text,
        verbose=verbose,
        model_id=model_id,
        question_type=question_type,
        system_message=system_message
    )


@cli.command()
@click.argument('file_path', type=click.Path(exists=True, path_type=Path))
@click.option('--format', 'file_format', type=click.Choice(['jsonl', 'json', 'csv', 'parquet']), help='File format (auto-detect if not specified)')
@click.option('--sample-size', type=int, default=5, help='Number of sample entries to show')
def inspect(file_path, file_format, sample_size):
    """Inspect a generated dataset file."""
    
    if not file_format:
        # Auto-detect format from extension
        if file_path.suffix == '.jsonl':
            file_format = 'jsonl'
        elif file_path.suffix == '.json':
            file_format = 'json'
        elif file_path.suffix == '.csv':
            file_format = 'csv'
        elif file_path.suffix == '.parquet':
            file_format = 'parquet'
        else:
            click.echo(f"Cannot auto-detect format for {file_path}")
            return
    
    click.echo(f"Inspecting {file_format.upper()} file: {file_path}")
    
    try:
        if file_format == 'jsonl':
            import json
            with open(file_path, 'r') as f:
                entries = []
                for i, line in enumerate(f):
                    if i >= sample_size:
                        break
                    entries.append(json.loads(line))
                
                click.echo(f"Sample entries (showing {len(entries)}):")
                for i, entry in enumerate(entries):
                    click.echo(f"\nEntry {i + 1}:")
                    click.echo(f"  ID: {entry.get('id', 'N/A')}")
                    text = entry.get('text', '')
                    click.echo(f"  Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                    if 'metadata' in entry:
                        click.echo(f"  Metadata keys: {list(entry['metadata'].keys())}")
        
        elif file_format == 'parquet':
            import pandas as pd
            df = pd.read_parquet(file_path, nrows=sample_size)
            click.echo(f"Dataset shape: {df.shape}")
            click.echo(f"Columns: {list(df.columns)}")
            click.echo("\nSample entries:")
            click.echo(df.head())
        
        # Add other format inspections as needed
        
    except Exception as e:
        click.echo(f"Error inspecting file: {e}")


@cli.command()
def formats():
    """List supported input and output formats."""
    
    app = DocToAI()
    
    click.echo("Supported input formats:")
    for fmt in app.document_loader.get_supported_formats():
        click.echo(f"  {fmt}")
    
    click.echo("\nSupported output formats:")
    for name, exporter in app.exporters.items():
        click.echo(f"  {name}: {exporter.format_name}")
    
    click.echo("\nSupported chunking strategies:")
    for name in app.chunkers.keys():
        click.echo(f"  {name}")


@cli.command()
def models():
    """List supported models for fine-tuning."""
    from core.model_templates import ModelTemplateManager
    
    template_manager = ModelTemplateManager()
    models = template_manager.get_available_models()
    
    click.echo("Available models for fine-tuning:")
    click.echo("=" * 50)
    
    providers = {}
    for model in models:
        provider = model['provider']
        if provider not in providers:
            providers[provider] = []
        providers[provider].append(model)
    
    for provider, provider_models in providers.items():
        click.echo(f"\n{provider}:")
        for model in provider_models:
            max_tokens = f"{model['max_tokens']:,}" if model['max_tokens'] else "N/A"
            system_support = "✅" if model['supports_system'] else "❌"
            click.echo(f"  {model['id']}")
            click.echo(f"    Name: {model['name']}")
            click.echo(f"    Max Tokens: {max_tokens}")
            click.echo(f"    System Messages: {system_support}")
    
    click.echo("\nUsage:")
    click.echo("  doctoai convert document.pdf --mode finetune --model-id gpt-3.5-turbo")
    click.echo("  doctoai convert document.pdf --mode finetune --model-id claude-3-haiku --question-type analytical")


@cli.command('generate-config')
@click.option('--output', '-o', type=click.Path(path_type=Path), default='config.yaml', help='Output configuration file')
def generate_config(output):
    """Generate a sample configuration file."""
    
    sample_config = {
        'input': {
            'formats': ['pdf', 'docx', 'epub', 'html', 'txt'],
            'recursive': True
        },
        'text_processing': {
            'remove_extra_whitespace': True,
            'fix_encoding_issues': True,
            'normalize_unicode': True,
            'remove_control_characters': True,
            'preserve_paragraph_breaks': True,
            'remove_headers_footers': False,
            'remove_page_numbers': False,
            'expand_abbreviations': False
        },
        'chunking': {
            'fixed': {
                'chunk_size': 512,
                'overlap': 50,
                'split_on': 'tokens',
                'preserve_sentences': True
            },
            'semantic': {
                'method': 'sentence_boundary',
                'min_size': 100,
                'max_size': 1000,
                'preserve_paragraphs': True,
                'target_size': 500,
                'language': 'english'
            },
            'hierarchical': {
                'respect_sections': True,
                'max_chunk_size': 2000,
                'min_chunk_size': 200,
                'section_overlap': 100,
                'preserve_headers': True
            }
        },
        'export': {
            'jsonl': {
                'compression': None
            },
            'json': {
                'indent': 2,
                'sort_keys': False
            },
            'csv': {
                'flatten_metadata': True,
                'include_embeddings': False
            },
            'parquet': {
                'compression_type': 'snappy',
                'row_group_size': 10000
            }
        },
        'metadata': {
            'include_fields': 'all',
            'custom_fields': {}
        }
    }
    
    with open(output, 'w') as f:
        yaml.dump(sample_config, f, default_flow_style=False, indent=2)
    
    click.echo(f"Generated sample configuration file: {output}")


if __name__ == '__main__':
    cli()