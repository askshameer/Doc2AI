# DocToAI Architecture Documentation

**Author**: Shameer Mohammed (mohammed.shameer@gmail.com)  
**GitHub**: https://github.com/askshameer/Doc2AI

## Table of Contents
1. [Solution Overview](#solution-overview)
2. [System Architecture](#system-architecture)
3. [Design Principles](#design-principles)
4. [Core Components](#core-components)
5. [Data Flow](#data-flow)
6. [Module Dependencies](#module-dependencies)
7. [Extensibility Points](#extensibility-points)

## Solution Overview

DocToAI is a comprehensive document-to-AI dataset conversion system designed to transform various document formats into structured datasets optimized for AI model training and retrieval-augmented generation (RAG) systems.

### Key Features
- **Multi-format Support**: PDF, DOCX, ePub, HTML, TXT, Markdown
- **Intelligent Processing**: OCR, text cleaning, metadata extraction
- **Flexible Chunking**: Fixed, semantic, and hierarchical strategies
- **Model-Specific Output**: Templates for OpenAI, Anthropic, Llama, Mistral
- **Multiple Formats**: JSONL, JSON, CSV, Parquet
- **Web Interface**: User-friendly Streamlit application
- **CLI Interface**: Command-line tool for automation

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DocToAI System                          │
├─────────────────────────────────────────────────────────────────┤
│                     User Interfaces                            │
├──────────────────────┬──────────────────────┬──────────────────┤
│   Web UI (Streamlit) │   CLI Interface     │   Python API    │
│   - Enhanced UI      │   - Click-based     │   - Direct       │
│   - Simple UI        │   - Model selection │     Import       │
│   - Components       │   - Batch processing│                  │
└──────────────────────┴──────────────────────┴──────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                      Core Engine                               │
├─────────────────────────────────────────────────────────────────┤
│                   Document Loader                              │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Format Detection & Routing                    ││
│  └─────────────────────────────────────────────────────────────┘│
│                               │                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Extractors                                ││
│  │  ┌───────┬───────┬───────┬───────┬───────┬─────────────────┐││
│  │  │  PDF  │ DOCX  │ ePub  │ HTML  │  TXT  │  Plugin System  │││
│  │  │       │       │       │       │       │                 │││
│  │  └───────┴───────┴───────┴───────┴───────┴─────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                  Processing Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                Text Processor                              ││
│  │  • Unicode normalization    • Encoding fixes              ││
│  │  • Whitespace cleaning      • Header/footer removal       ││
│  │  • Hyphenation fixes        • Quote standardization       ││
│  └─────────────────────────────────────────────────────────────┘│
│                               │                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                   Chunkers                                 ││
│  │  ┌─────────────┬─────────────┬─────────────────────────────┐││
│  │  │    Fixed    │  Semantic   │       Hierarchical          │││
│  │  │  • Size     │ • Sentence  │ • Document structure        │││
│  │  │  • Overlap  │ • Paragraph │ • Section awareness         │││
│  │  │  • Tokens   │ • Boundary  │ • Chapter preservation      │││
│  │  └─────────────┴─────────────┴─────────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                   Model Templates                              │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              Template Manager                              ││
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────────────┐││
│  │  │ OpenAI  │ Claude  │  Llama  │ Mistral │  HuggingFace    │││
│  │  │ • GPT   │ • Haiku │ • 2-7B  │ • 7B    │ • Generic       │││
│  │  │ • 3.5/4 │ • Sonnet│ • 2-13B │         │ • Custom        │││
│  │  └─────────┴─────────┴─────────┴─────────┴─────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                               │
┌─────────────────────────────────────────────────────────────────┐
│                    Output Layer                                │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────┐│
│  │               Metadata Manager                             ││
│  │  • Source information      • Processing metadata           ││
│  │  • Location data           • Content statistics            ││
│  │  • Quality scoring         • Semantic tagging              ││
│  └─────────────────────────────────────────────────────────────┘│
│                               │                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                  Exporters                                 ││
│  │  ┌─────────┬─────────┬─────────┬─────────────────────────┐││
│  │  │  JSONL  │  JSON   │   CSV   │       Parquet           │││
│  │  │ • Stream│ • Struct│ • Flat  │ • Columnar              │││
│  │  │ • Comp  │ • Pretty│ • Excel │ • Compressed            │││
│  │  └─────────┴─────────┴─────────┴─────────────────────────┘││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Design Principles

### 1. **Modularity**
- Each component has a single responsibility
- Clear interfaces between modules
- Pluggable architecture for extensibility

### 2. **Extensibility**
- Plugin system for new document formats
- Template system for new AI models
- Configurable processing pipeline

### 3. **Robustness**
- Graceful error handling and recovery
- Input validation at each stage
- Comprehensive logging and monitoring

### 4. **Performance**
- Streaming processing for large files
- Memory-efficient chunking strategies
- Parallel processing capabilities

### 5. **Usability**
- Multiple interfaces (CLI, Web UI, API)
- Sensible defaults with customization options
- Clear documentation and examples

## Core Components

### Document Processing Layer

#### **Document Loader** (`core/document_loader.py`)
```python
class DocumentLoader:
    """Main orchestrator for document processing"""
    
    def __init__(self):
        self.extractors = [
            PDFExtractor(),
            EPubExtractor(), 
            DocxExtractor(),
            HTMLExtractor(),
            TextExtractor()
        ]
    
    def load_document(self, file_path: Path) -> Document:
        """Route file to appropriate extractor"""
        extractor = self._find_extractor(file_path)
        return extractor.extract(file_path)
```

**Responsibilities:**
- File format detection
- Extractor selection and routing
- Error handling and fallbacks
- Plugin registration

#### **Base Extractor** (`core/base_extractor.py`)
```python
class ExtractorPlugin(ABC):
    """Abstract interface for all extractors"""
    
    @abstractmethod
    def can_handle(self, file_path: Path) -> bool:
        """Check if extractor supports this file"""
        
    @abstractmethod 
    def extract(self, file_path: Path) -> Document:
        """Extract content and metadata"""
        
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List supported file extensions"""
```

**Key Features:**
- Standardized extractor interface
- Plugin architecture
- Type safety and validation

### Text Processing Layer

#### **Text Processor** (`core/text_processor.py`)
```python
class TextProcessor:
    """Configurable text cleaning pipeline"""
    
    def __init__(self, config: Dict[str, Any]):
        self.settings = {**self.default_config, **config}
        self._compile_patterns()
    
    def process(self, text: str) -> str:
        """Apply cleaning pipeline"""
        # Unicode normalization
        # Encoding fixes
        # Whitespace cleaning
        # Format-specific processing
        return processed_text
```

**Processing Steps:**
1. **Encoding Detection & Normalization**
2. **Content Cleaning** (whitespace, control chars)
3. **Format Fixes** (hyphenation, quotes)
4. **Optional Processing** (headers, abbreviations)

### Chunking Layer

#### **Base Chunker** (`core/chunkers/base_chunker.py`)
```python
class BaseChunker(ABC):
    """Abstract base for chunking strategies"""
    
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        """Split document into chunks"""
        
    def _create_chunk(self, document, text, index, location):
        """Standardized chunk creation with metadata"""
```

#### **Chunking Strategies**

**Fixed Chunker** (`core/chunkers/fixed_chunker.py`)
- Equal-sized chunks with configurable overlap
- Token, character, or word-based splitting
- Sentence boundary preservation option

**Semantic Chunker** (`core/chunkers/semantic_chunker.py`)
- Respects sentence and paragraph boundaries
- Size constraints with target optimization
- Language-aware tokenization

**Hierarchical Chunker** (`core/chunkers/hierarchical_chunker.py`)
- Document structure preservation
- Section and chapter awareness
- Header-based segmentation

### Model Template Layer

#### **Template Manager** (`core/model_templates.py`)
```python
class ModelTemplateManager:
    """Central coordinator for model-specific formatting"""
    
    def __init__(self):
        self.templates = {
            model_id: TEMPLATE_CLASSES[config.template_type](config)
            for model_id, config in MODEL_CONFIGS.items()
        }
    
    def format_for_model(self, model_id, chunk, document, **kwargs):
        """Format chunk for specific model"""
        template = self.get_template(model_id)
        return template.format_conversation(chunk, document, **kwargs)
```

#### **Model Templates**

**OpenAI Template**
```python
def format_conversation(self, chunk, document, **kwargs):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question},
            {"role": "assistant", "content": chunk.text}
        ]
    }
```

**Anthropic Template**
```python
def format_conversation(self, chunk, document, **kwargs):
    return {
        "prompt": f"Human: {question}\n\nAssistant: {chunk.text}",
        "system": system_message
    }
```

### Output Layer

#### **Metadata Manager** (`utils/metadata_manager.py`)
```python
class MetadataManager:
    """Enriches data with comprehensive metadata"""
    
    def create_rag_entry(self, chunk, document):
        """Generate RAG-optimized entry"""
        
    def create_finetune_entry(self, chunk, document, model_id, **kwargs):
        """Generate model-specific fine-tuning entry"""
```

#### **Export System**

**Base Exporter** (`core/exporters/base_exporter.py`)
```python
class BaseExporter(ABC):
    """Abstract interface for output formats"""
    
    @abstractmethod
    def export_rag_data(self, entries, output_path):
        """Export RAG entries"""
        
    @abstractmethod 
    def export_finetune_data(self, entries, output_path):
        """Export fine-tuning entries"""
```

## Data Flow

### 1. **Input Stage**
```
Document File → Format Detection → Extractor Selection
```

### 2. **Extraction Stage**
```
Raw Document → Content Extraction → Metadata Extraction → Document Object
```

### 3. **Processing Stage**
```
Document → Text Cleaning → Structure Analysis → Processed Document
```

### 4. **Chunking Stage**
```
Processed Document → Strategy Selection → Chunk Generation → Chunk Objects
```

### 5. **Template Stage** (Fine-tuning only)
```
Chunks → Model Selection → Template Application → Formatted Entries
```

### 6. **Output Stage**
```
Entries → Metadata Enrichment → Format Export → Output Files
```

## Module Dependencies

### Import Hierarchy
```
Data Models (core/data_models.py)
    ↓
Base Classes (core/base_extractor.py, core/chunkers/base_chunker.py)
    ↓
Implementation Classes (extractors/, chunkers/, exporters/)
    ↓
Coordination Classes (document_loader.py, model_templates.py)
    ↓
Utilities (metadata_manager.py, text_processor.py)
    ↓
Main Application (cli.py)
    ↓
User Interfaces (ui/, run_ui.py)
```

### Key Dependencies
- **No Circular Dependencies**: Clean hierarchy maintained
- **Absolute Imports**: All imports use absolute paths
- **Optional Dependencies**: Graceful degradation for missing packages
- **Interface Segregation**: Minimal required imports

## Extensibility Points

### 1. **New Document Formats**
```python
class CustomExtractor(ExtractorPlugin):
    def can_handle(self, file_path: Path) -> bool:
        return file_path.suffix == '.custom'
    
    def extract(self, file_path: Path) -> Document:
        # Custom extraction logic
        pass
    
    @property
    def supported_extensions(self) -> List[str]:
        return ['.custom']

# Register with loader
loader.register_extractor(CustomExtractor())
```

### 2. **New AI Models**
```python
class CustomModelTemplate(BaseTemplate):
    def format_conversation(self, chunk, document, **kwargs):
        # Custom format logic
        return formatted_entry

# Add to template manager
MODEL_CONFIGS['custom-model'] = ModelConfig(...)
TEMPLATE_CLASSES['custom'] = CustomModelTemplate
```

### 3. **New Chunking Strategies**
```python
class CustomChunker(BaseChunker):
    def chunk(self, document: Document) -> List[Chunk]:
        # Custom chunking logic
        return chunks

# Register with application
app.chunkers['custom'] = CustomChunker(config)
```

### 4. **New Output Formats**
```python
class CustomExporter(BaseExporter):
    def export_rag_data(self, entries, output_path):
        # Custom export logic
        pass

# Register with application  
app.exporters['custom'] = CustomExporter(config)
```

This architecture ensures DocToAI is maintainable, extensible, and robust while providing clear separation of concerns and well-defined interfaces between components.