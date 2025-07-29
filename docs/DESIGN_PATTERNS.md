# DocToAI Design Patterns Documentation

**Author**: Shameer Mohammed (mohammed.shameer@gmail.com)  
**GitHub**: https://github.com/askshameer/Doc2AI

## Table of Contents
1. [Overview](#overview)
2. [Core Design Patterns](#core-design-patterns)
3. [Architectural Patterns](#architectural-patterns)
4. [Implementation Patterns](#implementation-patterns)
5. [Best Practices](#best-practices)

## Overview

DocToAI employs several well-established design patterns to ensure maintainability, extensibility, and robustness. This document details the design patterns used throughout the system and explains how they contribute to the overall architecture.

## Core Design Patterns

### 1. Plugin Architecture Pattern

**Purpose**: Enable extensible document format support without modifying core code.

**Implementation**: `core/base_extractor.py` and `core/document_loader.py`

```python
# Abstract Plugin Interface
class ExtractorPlugin(ABC):
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if this plugin can handle the file"""
        
    @abstractmethod
    def extract(self, file_path: Path) -> Document:
        """Extract content from the file"""
```

**Benefits**:
- New format support without core changes
- Loose coupling between formats and processing
- Easy testing and maintenance of individual extractors

**Usage in DocToAI**:
- PDF, DOCX, ePub, HTML, and TXT extractors
- Automatic format detection and routing
- Plugin registration system

### 2. Strategy Pattern

**Purpose**: Allow different chunking algorithms to be used interchangeably.

**Implementation**: `core/chunkers/` directory

```python
# Strategy Interface
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Apply chunking strategy to document"""

# Concrete Strategies
class FixedChunker(BaseChunker): ...
class SemanticChunker(BaseChunker): ...
class HierarchicalChunker(BaseChunker): ...
```

**Benefits**:
- Runtime selection of chunking strategy
- Easy addition of new chunking algorithms
- Clean separation of chunking logic

**Usage in DocToAI**:
- Fixed, semantic, and hierarchical chunking
- Configuration-driven strategy selection
- Pluggable chunking system

### 3. Template Method Pattern

**Purpose**: Define common workflow while allowing customization of specific steps.

**Implementation**: `core/model_templates.py`

```python
class BaseTemplate(ABC):
    def format_conversation(self, chunk, document, **kwargs):
        # Template method defining the workflow
        system_msg = self.create_system_message(**kwargs)
        question = self.generate_question(chunk, document, **kwargs)
        
        return self.format_entry(system_msg, question, chunk.text)
    
    @abstractmethod
    def format_entry(self, system_msg, question, answer):
        """Concrete implementations customize this step"""
```

**Benefits**:
- Consistent workflow across all model templates
- Customizable formatting for each model
- Code reuse and maintainability

**Usage in DocToAI**:
- Model-specific conversation formatting
- Standardized question generation
- Consistent metadata handling

### 4. Factory Pattern

**Purpose**: Create objects without specifying exact classes.

**Implementation**: `core/model_templates.py`

```python
class ModelTemplateManager:
    def __init__(self):
        self.templates = {
            model_id: TEMPLATE_CLASSES[config.template_type](config)
            for model_id, config in MODEL_CONFIGS.items()
        }
    
    def get_template(self, model_id: str) -> BaseTemplate:
        return self.templates.get(model_id)
```

**Benefits**:
- Centralized object creation
- Easy addition of new model types
- Configuration-driven instantiation

**Usage in DocToAI**:
- Model template creation
- Chunker instantiation
- Exporter selection

### 5. Builder Pattern

**Purpose**: Construct complex objects step by step.

**Implementation**: `utils/metadata_manager.py`

```python
class MetadataManager:
    def enrich_chunk_metadata(self, chunk: Chunk, document: Document):
        metadata = {}
        metadata['source'] = self._get_source_metadata(document)
        metadata['location'] = self._get_location_metadata(chunk)
        metadata['processing'] = self._get_processing_metadata(chunk)
        metadata['content_stats'] = self._get_content_stats(chunk)
        metadata['semantic_info'] = self._get_semantic_info(chunk)
        return metadata
```

**Benefits**:
- Step-by-step construction of complex metadata
- Flexible metadata composition
- Reusable metadata components

**Usage in DocToAI**:
- Metadata enrichment
- Entry construction for different formats
- Configuration building

## Architectural Patterns

### 1. Layered Architecture

**Implementation**: Clear separation of concerns across layers

```
┌─────────────────────────────────────┐
│        Presentation Layer          │  UI components, CLI
├─────────────────────────────────────┤
│        Application Layer           │  Main orchestration
├─────────────────────────────────────┤
│        Business Logic Layer        │  Processing pipeline
├─────────────────────────────────────┤
│        Data Access Layer           │  File I/O, export
└─────────────────────────────────────┘
```

**Benefits**:
- Clear separation of responsibilities
- Independent layer evolution
- Testability and maintainability

### 2. Pipeline Pattern

**Purpose**: Process data through a series of transformations.

**Implementation**: Document processing workflow

```python
def process_documents(self, input_paths, **kwargs):
    # Pipeline stages
    for doc_path in input_paths:
        document = self.document_loader.load_document(doc_path)    # Stage 1
        document.content = self.text_processor.process(content)    # Stage 2
        chunks = self.chunker.chunk(document)                      # Stage 3
        entries = self.metadata_manager.create_entries(chunks)     # Stage 4
        self.exporter.export(entries, output_path)               # Stage 5
```

**Benefits**:
- Clear data flow
- Composable processing stages
- Easy to add/remove processing steps

### 3. Observer Pattern

**Purpose**: Notify interested parties of processing events.

**Implementation**: Progress tracking and logging

```python
# Implicit through logging and progress bars
with tqdm(document_files, desc="Processing documents") as pbar:
    for doc_path in pbar:
        pbar.set_description(f"Processing {doc_path.name}")
        # Processing steps with logging
        logger.info(f"Processing {doc_path}")
```

**Benefits**:
- Decoupled progress reporting
- Flexible notification mechanisms
- Easy monitoring and debugging

## Implementation Patterns

### 1. Dependency Injection

**Purpose**: Provide dependencies from external sources.

**Implementation**: Configuration-driven component initialization

```python
class DocToAI:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Inject configuration into components
        self.text_processor = TextProcessor(self.config.get('text_processing', {}))
        self.chunkers = {
            'fixed': FixedChunker(self.config.get('chunking', {}).get('fixed', {})),
            'semantic': SemanticChunker(self.config.get('chunking', {}).get('semantic', {}))
        }
```

**Benefits**:
- Loose coupling between components
- Easy testing with mock dependencies
- Configuration-driven behavior

### 2. Data Transfer Object (DTO)

**Purpose**: Transfer data between layers without behavior.

**Implementation**: `core/data_models.py`

```python
@dataclass
class Document:
    document_id: str
    content: str
    metadata: DocumentMetadata
    structure: Dict[str, Any] = field(default_factory=dict)
    chunks: List[Chunk] = field(default_factory=list)

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    # ... other fields
```

**Benefits**:
- Clear data contracts
- Type safety
- Immutable data structures

### 3. Null Object Pattern

**Purpose**: Provide default behavior for missing objects.

**Implementation**: Default configurations and fallbacks

```python
class TextProcessor:
    def __init__(self, config: Dict[str, Any]):
        # Merge with defaults to avoid None checks
        self.settings = {**self.default_config, **config}
    
    @property
    def default_config(self):
        return {
            'remove_extra_whitespace': True,
            'fix_encoding_issues': True,
            'normalize_unicode': True
        }
```

**Benefits**:
- Eliminates null checks
- Provides sensible defaults
- Reduces conditional logic

### 4. Command Pattern

**Purpose**: Encapsulate requests as objects.

**Implementation**: CLI commands

```python
@cli.command()
@click.argument('input_paths', nargs=-1, required=True)
@click.option('--output', '-o', required=True)
# ... other options
def convert(ctx, input_paths, output, **kwargs):
    """Convert documents to AI dataset format."""
    app = DocToAI(ctx.obj['config'])
    app.process_documents(
        input_paths=list(input_paths),
        output_path=output,
        **kwargs
    )
```

**Benefits**:
- Encapsulated operations
- Easy to add new commands
- Consistent parameter handling

### 5. Adapter Pattern

**Purpose**: Make incompatible interfaces work together.

**Implementation**: Format-specific extractors adapting to common interface

```python
class PDFExtractor(ExtractorPlugin):
    def extract(self, file_path: Path) -> Document:
        # Adapt PyPDF2/pdfplumber API to Document interface
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            # Extract content and adapt to Document format
            return Document(...)
```

**Benefits**:
- Integration with external libraries
- Consistent internal interfaces
- Isolation of third-party dependencies

## Best Practices

### 1. Single Responsibility Principle (SRP)

Each class has one reason to change:
- `TextProcessor`: Only handles text cleaning
- `MetadataManager`: Only handles metadata enrichment
- `DocumentLoader`: Only handles format detection and routing

### 2. Open/Closed Principle (OCP)

Open for extension, closed for modification:
- New extractors can be added without changing `DocumentLoader`
- New chunking strategies without changing the pipeline
- New model templates without changing the template system

### 3. Dependency Inversion Principle (DIP)

Depend on abstractions, not concretions:
- `DocumentLoader` depends on `ExtractorPlugin` interface
- Main application depends on abstract chunker interface
- Template system depends on abstract template interface

### 4. Configuration Over Code

Behavior driven by configuration:
```python
# Configuration drives behavior
config = {
    'chunking': {
        'semantic': {
            'method': 'sentence_boundary',
            'min_size': 100,
            'max_size': 1000
        }
    }
}
```

### 5. Fail-Safe Defaults

Always provide sensible defaults:
```python
def __init__(self, config: Optional[Dict[str, Any]] = None):
    self.config = config or {}  # Never None
    self.settings = {**self.default_config, **self.config}
```

### 6. Explicit Error Handling

Clear error handling at each layer:
```python
try:
    document = self.document_loader.load_document(doc_path)
    if not document:
        logger.warning(f"Failed to load document: {doc_path}")
        continue
except Exception as e:
    logger.error(f"Error processing {doc_path}: {e}")
    continue
```

### 7. Immutable Data Structures

Use dataclasses for immutable data:
```python
@dataclass(frozen=True)
class Location:
    page: Optional[int] = None
    section: Optional[str] = None
    char_start: int = 0
    char_end: int = 0
```

### 8. Type Safety

Comprehensive type hints throughout:
```python
def process_documents(
    self,
    input_paths: List[Path],
    output_path: Path,
    mode: str = 'rag',
    **kwargs
) -> None:
```

These design patterns work together to create a maintainable, extensible, and robust document processing system that can evolve with changing requirements while maintaining code quality and reliability.