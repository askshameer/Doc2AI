# DocToAI - Document to AI Dataset Converter

**Author**: Shameer Mohammed (mohammed.shameer@gmail.com)

DocToAI is a comprehensive command-line tool designed to convert various document formats into structured datasets optimized for AI model fine-tuning and Retrieval-Augmented Generation (RAG) systems.

## Features

### 📄 Input Format Support
- **PDF** (.pdf) - Including scanned documents with OCR support
- **ePub** (.epub) - Electronic publication format with chapter structure preservation
- **Microsoft Word** (.docx, .doc) - Paragraphs, tables, and formatting
- **HTML** (.html, .htm) - Semantic structure preservation
- **Plain Text** (.txt, .md) - Markdown and plain text files

### 🧩 Intelligent Chunking Strategies
- **Fixed Size**: Configurable chunk size with overlap
- **Semantic**: Sentence and paragraph boundary-aware chunking
- **Hierarchical**: Document structure-aware chunking (chapters, sections)

### 📊 Output Formats
- **JSONL** - Newline-delimited JSON for streaming processing
- **JSON** - Standard JSON array format
- **CSV** - Comma-separated values with escaped content
- **Parquet** - Columnar storage for large datasets

### 🧹 Text Processing
- Unicode normalization and encoding fixes
- Whitespace cleaning and formatting
- Header/footer removal
- Hyphenation fixing
- Configurable cleaning pipeline

### 📋 Rich Metadata
- Source document information
- Processing metadata
- Content statistics
- Semantic tags
- Location information (page, section, etc.)

## Installation

```bash
# Install from source
git clone https://github.com/askshameer/Doc2AI.git
cd Doc2AI
pip install -e .

# Install with OCR support
pip install -e .[ocr]

# Install with all optional dependencies (includes UI)
pip install -e .[all]

# For UI only (includes Streamlit and visualization)
pip install -r requirements.txt
```

## Web UI (User-Friendly Interface)

DocToAI now includes a beautiful web interface built with Streamlit for users who prefer a graphical interface over the command line.

### Quick Start with UI

```bash
# Setup the UI (one-time)
python setup_ui.py

# Launch the web interface (recommended)
python launch_ui_safe.py

# Or use the standard launcher
python run_ui.py

# Or run directly with Streamlit
streamlit run ui/enhanced_app.py
```

The web UI will open at `http://localhost:8501` and provides:

### 🌟 UI Features
- **Drag & Drop File Upload**: Easy document upload with format detection
- **Interactive Configuration**: Visual controls for all processing options
- **Real-time Progress**: Live progress tracking during processing
- **Data Preview**: Interactive preview of generated datasets
- **Multiple Download Options**: Full dataset, samples, and format conversion
- **Statistics Dashboard**: Comprehensive analytics and visualizations
- **Preset Configurations**: Quick setup for common use cases

### 📱 UI Screenshots
- **Upload Interface**: Drag-and-drop with file validation
- **Configuration Panel**: Intuitive controls for chunking and processing
- **Results Dashboard**: Rich visualizations and data preview
- **Download Center**: Multiple export options and formats

## Quick Start

### Basic Usage

```bash
# Convert a PDF to JSONL for RAG
doctoai convert document.pdf --output dataset.jsonl --mode rag

# Process multiple documents with custom chunking
doctoai convert *.pdf --chunk-strategy semantic --chunk-size 1000 --output dataset.jsonl

# Fine-tuning dataset generation for specific models
doctoai convert docs/ --mode finetune --model-id gpt-3.5-turbo --output training.jsonl
doctoai convert docs/ --mode finetune --model-id claude-3-haiku --question-type analytical --output training.jsonl
```

### Advanced Usage

```bash
# Full pipeline with all options
doctoai convert \
    documents/*.pdf \
    --output output/dataset.jsonl \
    --mode rag \
    --chunk-strategy semantic \
    --chunk-size 512 \
    --overlap 50 \
    --output-format jsonl \
    --clean-text \
    --verbose
```

### Python API

```python
from pathlib import Path
from doctoai import DocToAI

# Initialize with configuration
app = DocToAI()

# Process documents
app.process_documents(
    input_paths=[Path("document.pdf")],
    output_path=Path("dataset.jsonl"),
    mode="rag",
    chunk_strategy="semantic",
    output_format="jsonl"
)
```

## Configuration

Generate a sample configuration file:

```bash
doctoai generate-config --output config.yaml
```

Example configuration:

```yaml
text_processing:
  remove_extra_whitespace: true
  fix_encoding_issues: true
  normalize_unicode: true
  preserve_paragraph_breaks: true

chunking:
  semantic:
    method: sentence_boundary
    min_size: 100
    max_size: 1000
    target_size: 500
  
  fixed:
    chunk_size: 512
    overlap: 50
    preserve_sentences: true

export:
  jsonl:
    compression: gzip
  
  parquet:
    compression_type: snappy
    row_group_size: 10000
```

## Output Formats

### RAG Format (JSONL)

```json
{
  "id": "doc_001_chunk_042",
  "text": "The microservices architecture consists of multiple independent services...",
  "metadata": {
    "source": {
      "filename": "technical_manual.pdf",
      "file_hash": "sha256:abcd1234...",
      "format": "pdf"
    },
    "location": {
      "page": 15,
      "section": "Chapter 3: Implementation",
      "char_start": 4521,
      "char_end": 5033
    },
    "processing": {
      "extraction_method": "pdfplumber",
      "chunking_strategy": "semantic",
      "processing_timestamp": "2024-07-28T10:30:00Z"
    },
    "content_stats": {
      "char_count": 512,
      "word_count": 89,
      "sentence_count": 4
    }
  }
}
```

### Fine-tuning Format (JSONL)

```json
{
  "id": "ft_sample_001",
  "conversations": [
    {
      "role": "system",
      "content": "You are a helpful assistant trained on technical documentation."
    },
    {
      "role": "user", 
      "content": "Explain the architecture described in section 3.2"
    },
    {
      "role": "assistant",
      "content": "The microservices architecture consists of multiple independent services..."
    }
  ],
  "metadata": {
    "source_document": "technical_manual.pdf",
    "section": "3.2 Architecture",
    "template": "qa"
  }
}
```

## Commands

### Convert Documents
```bash
doctoai convert [INPUT_PATHS...] --output OUTPUT_PATH [OPTIONS]
```

Options:
- `--mode`: Dataset mode (`rag`, `finetune`)
- `--chunk-strategy`: Chunking strategy (`fixed`, `semantic`, `hierarchical`) 
- `--chunk-size`: Chunk size for fixed chunking
- `--overlap`: Overlap size for fixed chunking
- `--output-format`: Output format (`jsonl`, `json`, `csv`, `parquet`)
- `--model-id`: Target model for fine-tuning (e.g., `gpt-3.5-turbo`, `claude-3-haiku`, `llama-2-7b`)
- `--question-type`: Question generation type (`general`, `summary`, `explanation`, `context`, `analytical`)
- `--system-message`: Custom system message for fine-tuning
- `--clean-text`: Enable text cleaning
- `--config`: Configuration file path

### List Available Models
```bash
doctoai models
```

### Inspect Dataset
```bash
doctoai inspect DATASET_FILE [OPTIONS]
```

### List Formats
```bash
doctoai formats
```

### Generate Configuration
```bash
doctoai generate-config --output config.yaml
```

## Chunking Strategies

### Fixed Size Chunking
- Splits text into fixed-size chunks with configurable overlap
- Supports splitting by characters, words, or approximate tokens
- Option to preserve sentence boundaries

### Semantic Chunking  
- Respects sentence and paragraph boundaries
- Configurable min/max sizes with target size optimization
- Language-aware sentence detection

### Hierarchical Chunking
- Preserves document structure (chapters, sections, headings)
- Respects document hierarchy from source format
- Automatic header detection for unstructured documents

## Text Processing Pipeline

1. **Encoding Detection & Normalization**
   - Automatic encoding detection
   - Unicode normalization (NFC)
   - Encoding issue fixes

2. **Content Cleaning**
   - Extra whitespace removal
   - Control character removal
   - Line ending normalization
   - Hyphenation fixes

3. **Optional Processing**
   - Header/footer removal
   - Page number removal
   - Abbreviation expansion
   - Quote standardization

## Metadata Schema

Each chunk includes comprehensive metadata:

- **Source Information**: Filename, hash, format, dates
- **Location Data**: Page, section, character positions
- **Processing Info**: Methods used, timestamps, quality scores
- **Content Statistics**: Word count, reading level, semantic tags
- **Document Metadata**: Title, author, language, structure info

## Requirements

- Python 3.8+
- Core dependencies: pandas, PyPDF2, python-docx, beautifulsoup4, nltk
- Optional: pytesseract (OCR), PyMuPDF (advanced PDF processing)

## Development

```bash
# Install development dependencies
pip install -e .[dev]

# Run tests
pytest

# Format code
black .

# Type checking
mypy .
```

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- 📖 [Documentation](https://github.com/askshameer/Doc2AI/tree/main/docs)
- 🐛 [Issue Tracker](https://github.com/askshameer/Doc2AI/issues)
- 💬 [Discussions](https://github.com/askshameer/Doc2AI/discussions)
- 📧 [Contact](mailto:mohammed.shameer@gmail.com)

## Citation

If you use DocToAI in your research, please cite:

```bibtex
@software{doctoai2025,
  title={DocToAI: Document to AI Dataset Converter},
  author={Shameer Mohammed},
  year={2025},
  url={https://github.com/askshameer/Doc2AI},
  email={mohammed.shameer@gmail.com}
}
```