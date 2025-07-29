# DocToAI Web UI

A user-friendly web interface for DocToAI built with Streamlit.

## Quick Start

### 1. Setup Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# Or install specific UI dependencies
pip install streamlit plotly pandas
```

### 2. Launch the UI

```bash
# From the main DocToAI directory
python run_ui.py

# Or run directly
streamlit run ui/enhanced_app.py

# Or use the simple fallback version
streamlit run ui/simple_app.py
```

### 3. Access the Interface

Open your browser to `http://localhost:8501`

## UI Components

### 📁 File Upload (`ui/app.py`)
- Basic Streamlit interface
- File upload and processing
- Configuration options
- Results display

### 🎨 Enhanced UI (`ui/enhanced_app.py`)
- Advanced styling and animations
- Interactive charts and visualizations
- Progress tracking
- Multiple download options
- Preset configurations

### 🔧 Simple UI (`ui/simple_app.py`)
- Fallback version for missing dependencies
- Dependency checking and installation instructions
- Basic functionality preview

### 🧩 Components (`ui/components.py`)
- Reusable UI components
- Metric cards, status badges
- Charts and visualizations
- Help sections

## Features

### Core Functionality
- **Drag & Drop Upload**: Easy file upload with validation
- **Real-time Processing**: Live progress tracking
- **Interactive Configuration**: Visual controls for all options
- **Data Preview**: Inspect generated datasets
- **Multiple Downloads**: Various formats and sample sizes

### Advanced Features
- **Preset Configurations**: Quick setup for common use cases
- **Statistics Dashboard**: Comprehensive analytics
- **Format Conversion**: Convert between output formats
- **Error Handling**: Graceful error display and recovery
- **Responsive Design**: Works on desktop and tablet

### Visual Elements
- **Modern Styling**: Gradient themes and smooth animations
- **Color-coded Status**: Visual feedback for all operations
- **Interactive Charts**: Plotly-based visualizations
- **Card-based Layout**: Clean, organized interface

## Configuration

### Presets
- **RAG Optimized**: Semantic chunking, 500-char targets
- **Fine-tuning Ready**: Conversation format, optimized for training
- **Large Documents**: Hierarchical chunking, efficient processing

### Chunking Options
- **Fixed Size**: Equal chunks with configurable overlap
- **Semantic**: Sentence/paragraph boundary-aware
- **Hierarchical**: Document structure preservation

### Output Formats
- **JSONL**: Streaming-friendly, one entry per line
- **JSON**: Standard format with full structure
- **CSV**: Spreadsheet-compatible with flattened metadata
- **Parquet**: Columnar format for big data applications

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Install missing dependencies
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Use different port
   python run_ui.py --port 8502
   ```

3. **Browser Doesn't Open**
   ```bash
   # Disable auto-open
   python run_ui.py --no-browser
   # Then manually navigate to http://localhost:8501
   ```

### Dependency Issues

If you encounter import errors, the simple UI will automatically display setup instructions and show which dependencies are missing.

### Performance

For large document collections:
- Use the Parquet output format
- Enable compression
- Process in smaller batches
- Use the hierarchical chunking strategy

## Development

### File Structure
```
ui/
├── app.py              # Basic Streamlit app
├── enhanced_app.py     # Full-featured UI with styling
├── simple_app.py       # Fallback UI for missing deps
├── components.py       # Reusable UI components
└── README.md          # This file
```

### Adding New Features

1. Add reusable components to `components.py`
2. Update the main UI in `enhanced_app.py`
3. Test with the simple fallback in `simple_app.py`
4. Update configuration options as needed

### Styling

The UI uses custom CSS for:
- Gradient backgrounds and themes
- Card-based layouts
- Smooth animations and transitions
- Responsive design elements

Colors follow the DocToAI theme:
- Primary: `#667eea`
- Secondary: `#764ba2`
- Success: `#28a745`
- Warning: `#ffc107`
- Error: `#dc3545`

## API Integration

The UI integrates with DocToAI core components:
- `DocumentLoader` for file processing
- `TextProcessor` for cleaning
- `MetadataManager` for enrichment
- Various `Chunkers` for text segmentation
- Multiple `Exporters` for output generation

## License

Same as DocToAI main project - MIT License.