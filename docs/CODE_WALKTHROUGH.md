# DocToAI Code Walkthrough

**Author**: Shameer Mohammed (mohammed.shameer@gmail.com)  
**GitHub**: https://github.com/askshameer/Doc2AI

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Core Components Deep Dive](#core-components-deep-dive)
4. [Data Flow Analysis](#data-flow-analysis)
5. [Key Implementation Details](#key-implementation-details)
6. [Integration Points](#integration-points)
7. [Testing Strategy](#testing-strategy)

## Overview

This document provides a comprehensive walkthrough of the DocToAI codebase, explaining how components interact, key implementation decisions, and the reasoning behind the architecture. It serves as a guide for developers who need to understand, modify, or extend the system.

## Project Structure

```
DocToAI/
├── core/                          # Core business logic
│   ├── data_models.py            # Data structures and types
│   ├── base_extractor.py         # Abstract extractor interface
│   ├── document_loader.py        # Format detection and routing
│   ├── text_processor.py         # Text cleaning pipeline
│   ├── model_templates.py        # Model-specific formatting
│   ├── chunkers/                 # Chunking strategies
│   │   ├── base_chunker.py      # Abstract chunker interface
│   │   ├── fixed_chunker.py     # Fixed-size chunking
│   │   ├── semantic_chunker.py  # Boundary-aware chunking
│   │   └── hierarchical_chunker.py # Structure-aware chunking
│   ├── extractors/              # Format-specific extractors
│   │   ├── pdf_extractor.py     # PDF processing
│   │   ├── docx_extractor.py    # Word document processing
│   │   ├── epub_extractor.py    # ePub processing
│   │   ├── html_extractor.py    # HTML processing
│   │   └── text_extractor.py    # Plain text processing
│   └── exporters/               # Output format handlers
│       ├── base_exporter.py     # Abstract exporter interface
│       ├── jsonl_exporter.py    # JSONL output
│       ├── json_exporter.py     # JSON output
│       ├── csv_exporter.py      # CSV output
│       └── parquet_exporter.py  # Parquet output
├── utils/                       # Utility components
│   └── metadata_manager.py     # Metadata enrichment
├── ui/                         # User interface components
│   ├── enhanced_app.py         # Streamlit web interface
│   └── components/             # UI components
├── docs/                       # Documentation
├── cli.py                      # Command-line interface
├── run_ui.py                   # UI launcher
└── requirements.txt            # Dependencies
```

## Core Components Deep Dive

### 1. Data Models (`core/data_models.py`)

**Purpose**: Define the core data structures used throughout the system.

```python
@dataclass
class Document:
    document_id: str           # Unique identifier
    content: str              # Raw extracted text
    metadata: DocumentMetadata # Source information
    structure: Dict[str, Any] # Document structure (headings, etc.)
    chunks: List[Chunk]       # Generated chunks
```

**Key Design Decisions**:
- **Immutable where possible**: Using `@dataclass(frozen=True)` for data integrity
- **Rich metadata**: Comprehensive information for traceability
- **Type safety**: Full type annotations for better development experience

**Usage Pattern**:
```python
# Document creation in extractors
document = Document(
    document_id=f"doc_{file_hash[:8]}",
    content=extracted_text,
    metadata=DocumentMetadata(...),
    structure=document_structure
)
```

### 2. Document Loader (`core/document_loader.py`)

**Purpose**: Central orchestrator for format detection and extractor routing.

```python
class DocumentLoader:
    def __init__(self):
        self.extractors = [
            PDFExtractor(),
            EPubExtractor(),
            DocxExtractor(),
            HTMLExtractor(),
            TextExtractor()
        ]
    
    def load_document(self, file_path: Path) -> Optional[Document]:
        extractor = self._find_extractor(file_path)
        if extractor:
            return extractor.extract(file_path)
        return None
```

**Key Features**:
- **Automatic format detection**: Based on file extension and content
- **Fallback mechanism**: Multiple detection strategies
- **Plugin architecture**: Easy to add new extractors

**Error Handling Strategy**:
```python
def _find_extractor(self, file_path: Path) -> Optional[ExtractorPlugin]:
    # Try extension-based matching first
    for extractor in self.extractors:
        if file_path.suffix.lower() in extractor.supported_extensions:
            if extractor.can_handle(file_path):
                return extractor
    
    # Fallback to content-based detection
    for extractor in self.extractors:
        try:
            if extractor.can_handle(file_path):
                return extractor
        except Exception:
            continue  # Try next extractor
    
    return None
```

### 3. Extractors (`core/extractors/`)

Each extractor implements the `ExtractorPlugin` interface:

#### PDF Extractor (`core/extractors/pdf_extractor.py`)

```python
class PDFExtractor(ExtractorPlugin):
    def can_handle(self, file_path: Path) -> bool:
        if file_path.suffix.lower() != '.pdf':
            return False
        
        try:
            with open(file_path, 'rb') as file:
                # Quick PDF header check
                header = file.read(4)
                return header == b'%PDF'
        except Exception:
            return False
    
    def extract(self, file_path: Path) -> Document:
        # Multi-strategy extraction
        try:
            return self._extract_with_pdfplumber(file_path)
        except Exception:
            try:
                return self._extract_with_pypdf2(file_path)
            except Exception:
                return self._extract_with_ocr(file_path)
```

**Extraction Strategy**:
1. **Primary**: pdfplumber for better layout preservation
2. **Fallback**: PyPDF2 for basic text extraction
3. **Last resort**: OCR for scanned documents

#### DOCX Extractor (`core/extractors/docx_extractor.py`)

```python
def _extract_content_and_structure(self, doc) -> Tuple[str, Dict]:
    content_parts = []
    structure = {
        'headings': [],
        'paragraphs': [],
        'tables': []
    }
    
    for paragraph in doc.paragraphs:
        if paragraph.style.name.startswith('Heading'):
            # Track heading structure
            structure['headings'].append({
                'text': paragraph.text,
                'level': int(paragraph.style.name[-1]),
                'position': len(content_parts)
            })
        
        content_parts.append(paragraph.text)
    
    return '\n'.join(content_parts), structure
```

**Key Features**:
- **Structure preservation**: Maintains document hierarchy
- **Style detection**: Identifies headings, lists, tables
- **Metadata extraction**: Title, author, creation date

### 4. Text Processor (`core/text_processor.py`)

**Purpose**: Configurable text cleaning pipeline.

```python
class TextProcessor:
    def process(self, text: str) -> str:
        if self.settings['normalize_unicode']:
            text = self._normalize_unicode(text)
        
        if self.settings['fix_encoding_issues']:
            text = self._fix_encoding(text)
        
        if self.settings['remove_extra_whitespace']:
            text = self._clean_whitespace(text)
        
        # Additional processing steps...
        return text
```

**Processing Pipeline**:
1. **Unicode normalization**: Convert to NFC form
2. **Encoding fixes**: Handle common encoding issues
3. **Whitespace cleaning**: Remove excessive whitespace
4. **Format-specific fixes**: Hyphenation, quotes, etc.

### 5. Chunking System (`core/chunkers/`)

#### Base Chunker (`core/chunkers/base_chunker.py`)

```python
class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks"""
        pass
    
    def _create_chunk(self, document: Document, text: str, 
                     index: int, location: Location) -> Chunk:
        """Standardized chunk creation with metadata"""
        return Chunk(
            chunk_id=f"{document.document_id}_chunk_{index:03d}",
            document_id=document.document_id,
            text=text.strip(),
            chunk_index=index,
            location=location,
            semantic_tags=self._extract_semantic_tags(text),
            processing=ProcessingMetadata(...)
        )
```

#### Semantic Chunker (`core/chunkers/semantic_chunker.py`)

**Algorithm**:
```python
def _create_semantic_chunks(self, sentences: List[str]) -> List[str]:
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # Check if adding sentence would exceed max size
        if (current_size + sentence_size > self.max_size and 
            current_chunk and 
            current_size >= self.min_size):
            # Finalize current chunk
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size
    
    # Handle remaining sentences
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

**Benefits**:
- **Boundary awareness**: Respects sentence boundaries
- **Size optimization**: Targets optimal chunk size
- **Language support**: Uses NLTK for proper sentence detection

### 6. Model Templates (`core/model_templates.py`)

**Purpose**: Format chunks for specific LLM fine-tuning formats.

#### Template System Architecture:

```python
class ModelTemplateManager:
    def format_for_model(self, model_id: str, chunk: Chunk, 
                        document: Document, **kwargs) -> Dict[str, Any]:
        template = self.get_template(model_id)
        return template.format_conversation(chunk, document, **kwargs)

class OpenAITemplate(BaseTemplate):
    def format_conversation(self, chunk, document, **kwargs):
        system_message = kwargs.get('system_message') or self.get_default_system_message()
        question = self.generate_question(chunk, document, kwargs.get('question_type', 'general'))
        
        return {
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": question},
                {"role": "assistant", "content": chunk.text}
            ]
        }
```

**Model Support**:
- **OpenAI**: GPT-3.5, GPT-4 conversation format
- **Anthropic**: Claude prompt format with system messages
- **Llama**: Llama-2 conversation format
- **Mistral**: Mistral-specific formatting
- **HuggingFace**: Generic format for custom models

### 7. Metadata Manager (`utils/metadata_manager.py`)

**Purpose**: Enrich chunks with comprehensive metadata.

```python
def create_finetune_entry(self, chunk: Chunk, document: Document, 
                         model_id: str, **kwargs) -> Dict[str, Any]:
    # Use model template for formatting
    template_manager = ModelTemplateManager()
    entry = template_manager.format_for_model(model_id, chunk, document, **kwargs)
    
    # Add DocToAI metadata
    doctoai_metadata = self.enrich_chunk_metadata(chunk, document)
    doctoai_metadata.update({
        'model_id': model_id,
        'question_type': kwargs.get('question_type', 'general'),
        'template_version': '2.0'
    })
    
    # Validate and add warnings if needed
    warnings = template_manager.validate_for_model(model_id, entry)
    if warnings:
        entry['metadata']['validation_warnings'] = warnings
    
    return entry
```

**Metadata Categories**:
- **Source**: File information, hashes, dates
- **Location**: Page, section, character positions
- **Processing**: Methods used, quality scores
- **Content**: Statistics, reading level, semantic tags
- **Document**: Title, author, structure information

## Data Flow Analysis

### 1. Input Processing Flow

```
File Input → Format Detection → Extractor Selection → Content Extraction
     ↓
Document Object Creation → Text Processing → Structure Analysis
     ↓
Chunking Strategy Selection → Chunk Generation → Metadata Enrichment
     ↓
Model Template Application → Entry Creation → Export Format Selection
     ↓
Output File Generation
```

### 2. Error Handling Flow

```
Error Occurs → Log Error Details → Attempt Recovery → Continue Processing
     ↓
Fallback Strategy → Alternative Method → Success/Failure
     ↓
Graceful Degradation → Warning Generation → Process Next Item
```

### 3. Configuration Flow

```
CLI Arguments → Configuration File → Default Values → Component Initialization
     ↓
Runtime Configuration → Dynamic Behavior → Processing Customization
```

## Key Implementation Details

### 1. Import Resolution Strategy

**Problem**: Circular imports between core modules.

**Solution**: Created `core/base_extractor.py` to house abstract interfaces:

```python
# Before: circular dependency
# core/document_loader.py imports from core/extractors/
# core/extractors/ imports from core/document_loader.py

# After: clean hierarchy
# core/base_extractor.py (abstract interfaces)
#   ↓
# core/extractors/ (implementations)
#   ↓
# core/document_loader.py (orchestration)
```

### 2. Dependency Management

**Strategy**: Graceful degradation for optional dependencies:

```python
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("PyMuPDF not available, falling back to PyPDF2")

def extract_pdf(self, file_path: Path) -> Document:
    if HAS_PYMUPDF:
        return self._extract_with_pymupdf(file_path)
    else:
        return self._extract_with_pypdf2(file_path)
```

### 3. Memory Management

**Large File Handling**:
```python
def process_large_document(self, file_path: Path) -> Iterator[Chunk]:
    """Stream processing for large documents"""
    with open(file_path, 'r', encoding='utf-8') as file:
        buffer = ""
        for line in file:
            buffer += line
            if len(buffer) > self.chunk_size:
                yield self._create_chunk(buffer)
                buffer = ""
        
        if buffer:
            yield self._create_chunk(buffer)
```

### 4. Token Counting and Validation

**Model-Specific Validation**:
```python
def validate_for_model(self, model_id: str, entry: Dict[str, Any]) -> List[str]:
    warnings = []
    config = self.get_model_config(model_id)
    
    # Count tokens in the conversation
    total_tokens = self._count_tokens(entry, model_id)
    
    if total_tokens > config.max_tokens:
        warnings.append(f"Token count ({total_tokens}) exceeds model limit ({config.max_tokens})")
    
    # Check for required fields
    if config.requires_system_message and 'system' not in entry:
        warnings.append("Model requires system message but none provided")
    
    return warnings
```

## Integration Points

### 1. CLI Integration (`cli.py`)

**Design**: Click-based command system with configuration injection:

```python
@cli.command()
@click.option('--model-id', default='gpt-3.5-turbo')
@click.option('--question-type', type=click.Choice(['general', 'summary', ...]))
def convert(ctx, **kwargs):
    config = ctx.obj['config']
    app = DocToAI(config)
    app.process_documents(**kwargs)
```

### 2. Web UI Integration (`ui/enhanced_app.py`)

**Architecture**: Streamlit-based interface with session state management:

```python
def main():
    st.set_page_config(page_title="DocToAI", layout="wide")
    
    # Initialize session state
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    
    # Configuration sidebar
    config = build_config_from_ui()
    
    # File upload and processing
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)
    
    if st.button("Process Documents"):
        with st.spinner("Processing..."):
            results = process_with_progress(uploaded_files, config)
            st.session_state.processed_data = results
```

### 3. API Integration

**Future Extension Point**: RESTful API interface:

```python
from fastapi import FastAPI, UploadFile
from typing import List

app = FastAPI()

@app.post("/convert")
async def convert_documents(
    files: List[UploadFile],
    mode: str = "rag",
    chunk_strategy: str = "semantic"
):
    doctoai = DocToAI()
    results = doctoai.process_files(files, mode=mode, chunk_strategy=chunk_strategy)
    return {"status": "success", "entries": len(results)}
```

## Testing Strategy

### 1. Unit Testing

**Pattern**: Test each component in isolation:

```python
class TestSemanticChunker(unittest.TestCase):
    def setUp(self):
        self.chunker = SemanticChunker({
            'min_size': 100,
            'max_size': 500,
            'target_size': 300
        })
    
    def test_respects_sentence_boundaries(self):
        document = create_test_document(
            "First sentence. Second sentence. Third sentence."
        )
        chunks = self.chunker.chunk(document)
        
        for chunk in chunks:
            # Ensure no chunks split sentences
            self.assertNotIn('. ', chunk.text[:-1])
```

### 2. Integration Testing

**Pattern**: Test component interactions:

```python
def test_full_pipeline():
    """Test complete document processing pipeline"""
    app = DocToAI(test_config)
    
    # Process test document
    results = app.process_documents(
        input_paths=[TEST_PDF_PATH],
        output_path=temp_output_path,
        mode='rag'
    )
    
    # Verify output
    assert temp_output_path.exists()
    with open(temp_output_path) as f:
        entries = [json.loads(line) for line in f]
    
    assert len(entries) > 0
    assert all('id' in entry for entry in entries)
```

### 3. End-to-End Testing

**Pattern**: Test user workflows:

```python
def test_cli_convert_command():
    """Test CLI convert command end-to-end"""
    result = runner.invoke(cli, [
        'convert',
        'test_docs/sample.pdf',
        '--output', 'output.jsonl',
        '--mode', 'finetune',
        '--model-id', 'gpt-3.5-turbo'
    ])
    
    assert result.exit_code == 0
    assert Path('output.jsonl').exists()
```

This code walkthrough demonstrates how DocToAI's modular architecture, clear separation of concerns, and extensible design patterns work together to create a maintainable and robust document processing system. The implementation prioritizes error handling, configurability, and extensibility while maintaining clean interfaces between components.