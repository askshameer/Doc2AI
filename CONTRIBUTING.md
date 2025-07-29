# Contributing to DocToAI

Thank you for your interest in contributing to DocToAI! This document provides guidelines for contributing to the project.

## Author

**Shameer Mohammed** (mohammed.shameer@gmail.com)  
GitHub: https://github.com/askshameer/Doc2AI

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/Doc2AI.git
   cd Doc2AI
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Setup

### Prerequisites

- Python 3.8+
- Git

### Optional Dependencies

For full functionality, install optional dependencies:

```bash
# PDF processing
pip install pdfplumber PyPDF2 pytesseract PyMuPDF

# Document formats
pip install python-docx ebooklib beautifulsoup4

# Export formats
pip install pandas pyarrow

# Text processing
pip install nltk chardet

# UI components
pip install tqdm
```

## Making Contributions

### Types of Contributions

1. **Bug Reports**: Report bugs using GitHub issues
2. **Feature Requests**: Suggest new features
3. **Code Contributions**: Submit pull requests
4. **Documentation**: Improve documentation
5. **Testing**: Add tests for existing or new functionality

### Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and ensure they follow the coding standards

3. Add tests for new functionality

4. Update documentation if needed

5. Commit your changes with clear messages:
   ```bash
   git commit -m "Add: description of your changes"
   ```

6. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

7. Submit a pull request

### Coding Standards

- Follow PEP 8 Python style guide
- Use type hints where appropriate
- Add docstrings to functions and classes
- Keep functions focused and small
- Handle errors gracefully
- Write comprehensive tests

### Code Style

```python
def example_function(param: str) -> Dict[str, Any]:
    """
    Example function with proper documentation.
    
    Args:
        param: Description of parameter
        
    Returns:
        Dictionary containing results
    """
    # Implementation here
    pass
```

## Testing

Run tests before submitting:

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=core --cov=utils

# Run specific test file
python -m pytest tests/test_specific.py
```

## Documentation

### Adding Documentation

- Update relevant documentation files in `docs/`
- Add docstrings to new functions/classes
- Update README.md if adding new features
- Include examples where helpful

### Documentation Structure

```
docs/
├── README.md              # Documentation overview
├── ARCHITECTURE.md        # System architecture
├── DESIGN_PATTERNS.md     # Design patterns used
└── CODE_WALKTHROUGH.md    # Code explanation
```

## Reporting Issues

When reporting bugs, please include:

1. **Environment**: Python version, OS, dependencies
2. **Steps to Reproduce**: Clear steps to reproduce the issue
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Error Messages**: Full error messages if any
6. **Sample Files**: If relevant, provide sample input files

## Feature Requests

For feature requests, please include:

1. **Use Case**: Why is this feature needed?
2. **Proposed Solution**: How should it work?
3. **Alternatives**: Any alternative approaches considered?
4. **Examples**: Usage examples if applicable

## Code of Conduct

### Our Standards

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Collaborate openly and transparently

### Unacceptable Behavior

- Harassment or discrimination
- Offensive comments or personal attacks
- Publishing private information
- Spam or off-topic discussions

## Getting Help

- 📧 **Email**: mohammed.shameer@gmail.com
- 🐛 **Issues**: [GitHub Issues](https://github.com/askshameer/Doc2AI/issues)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/askshameer/Doc2AI/discussions)

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Documentation credits

Thank you for contributing to DocToAI! 🚀