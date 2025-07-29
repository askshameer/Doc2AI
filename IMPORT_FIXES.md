# Import Fixes Documentation

This document outlines the import issues that were resolved in DocToAI and the solutions implemented.

## Issues Identified

### 1. Circular Import Problem
**Error**: `ImportError: cannot import name 'ExtractorPlugin' from partially initialized module 'core.document_loader'`

**Root Cause**: 
- `ExtractorPlugin` was defined in `document_loader.py`
- Extractor modules imported `ExtractorPlugin` from `document_loader.py`
- `document_loader.py` imported the extractor modules
- This created a circular dependency

**Solution**: 
- Created `core/base_extractor.py` to house the `ExtractorPlugin` abstract base class
- Updated all imports to use the new base module
- Eliminated circular dependency

### 2. Relative Import Issues
**Error**: `ImportError: attempted relative import beyond top-level package`

**Root Cause**: 
- Modules used relative imports (e.g., `from ..data_models import Document`)
- When running scripts directly, Python doesn't treat the directory as a package
- Relative imports failed when not running as a package

**Solution**: 
- Converted all relative imports to absolute imports
- Changed `from ..data_models` to `from core.data_models`
- Maintained module structure while enabling direct script execution

## Files Modified

### New Files Created
- `core/base_extractor.py` - Base class for all extractors

### Files Updated with Import Changes

#### Core Modules
- `core/document_loader.py` - Updated to import from base_extractor
- `core/base_extractor.py` - New base class module
- `core/chunkers/base_chunker.py` - Fixed imports
- `core/chunkers/fixed_chunker.py` - Fixed imports
- `core/chunkers/semantic_chunker.py` - Fixed imports  
- `core/chunkers/hierarchical_chunker.py` - Fixed imports

#### Extractors
- `core/extractors/pdf_extractor.py` - Fixed imports
- `core/extractors/epub_extractor.py` - Fixed imports
- `core/extractors/docx_extractor.py` - Fixed imports
- `core/extractors/html_extractor.py` - Fixed imports
- `core/extractors/txt_extractor.py` - Fixed imports

#### Exporters
- `core/exporters/base_exporter.py` - Fixed imports
- `core/exporters/jsonl_exporter.py` - Fixed imports
- `core/exporters/json_exporter.py` - Fixed imports
- `core/exporters/csv_exporter.py` - Fixed imports
- `core/exporters/parquet_exporter.py` - Fixed imports

#### Utilities
- `utils/metadata_manager.py` - Fixed imports

#### Package Exports
- `__init__.py` - Updated to include new base_extractor

## Import Structure (After Fixes)

```
core/
├── data_models.py          # Core data structures (no dependencies)
├── base_extractor.py       # Abstract base class (depends on data_models)
├── document_loader.py      # Main loader (imports extractors and base)
├── text_processor.py       # Text processing (independent)
├── extractors/
│   ├── pdf_extractor.py    # Imports base_extractor and data_models
│   ├── epub_extractor.py   # Imports base_extractor and data_models
│   └── ...                 # Other extractors
├── chunkers/
│   ├── base_chunker.py     # Base chunker (imports data_models)
│   └── ...                 # Specific chunkers
└── exporters/
    ├── base_exporter.py    # Base exporter (imports data_models)
    └── ...                 # Specific exporters
```

## Verification

The fixes were verified by:

1. **Core Import Test**: Verified that all core modules can be imported without external dependencies
2. **Full Import Test**: Confirmed that the full application can be imported (dependencies permitting)
3. **UI Launch Test**: Ensured the web UI can start without import errors

## Safe Launch System

Created a robust launch system to handle various scenarios:

### `launch_ui_safe.py`
- Checks Python version compatibility
- Verifies core imports work
- Automatically selects appropriate UI version based on available dependencies
- Provides helpful setup instructions if issues are found
- Sets PYTHONPATH correctly for imports

### UI Fallback Hierarchy
1. **Enhanced UI** (`ui/enhanced_app.py`) - Full features with all dependencies
2. **Basic UI** (`ui/app.py`) - Standard interface with core dependencies
3. **Simple UI** (`ui/simple_app.py`) - Minimal interface with dependency checking

## Best Practices Implemented

1. **No Circular Dependencies**: Each module has a clear dependency hierarchy
2. **Absolute Imports**: All imports use absolute paths from the project root
3. **Graceful Degradation**: UI versions adapt to available dependencies
4. **Clear Error Messages**: Import errors provide actionable guidance
5. **Dependency Isolation**: Core functionality doesn't require heavy external libraries

## Testing

To verify imports work correctly:

```bash
# Test core imports (no external dependencies required)
python -c "from core.data_models import Document; print('✅ Core imports work')"

# Test full imports (requires dependencies)
python -c "from cli import DocToAI; print('✅ Full imports work')"

# Launch UI with automatic dependency handling
python launch_ui_safe.py
```

## Migration Guide

If you encounter import errors after updating:

1. **Update your imports**: Change relative imports to absolute imports
2. **Check directory**: Ensure you're running from the DocToAI root directory
3. **Install dependencies**: Run `pip install -r requirements.txt`
4. **Use safe launcher**: Use `python launch_ui_safe.py` for automatic handling

## Future Considerations

- All new modules should use absolute imports
- External dependencies should be optional with graceful fallbacks
- The base_extractor pattern can be extended for other plugin types
- Consider using proper Python package installation for production deployments