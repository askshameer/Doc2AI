from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    with open(requirements_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="doctoai",
    version="1.0.0",
    author="Shameer Mohammed",
    author_email="mohammed.shameer@gmail.com",
    description="Document to AI Dataset Converter - Convert various document formats to AI-ready datasets for fine-tuning and RAG systems",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/askshameer/Doc2AI",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
        ],
        'ocr': [
            'pytesseract>=0.3.10',
            'PyMuPDF>=1.23.0',
        ],
        'all': [
            'pytesseract>=0.3.10',
            'PyMuPDF>=1.23.0',
        ]
    },
    entry_points={
        'console_scripts': [
            'doctoai=cli:cli',
        ],
    },
    include_package_data=True,
    keywords=[
        'document-processing',
        'ai-dataset',
        'machine-learning',
        'nlp',
        'text-processing',
        'pdf',
        'epub',
        'docx',
        'html',
        'chunking',
        'rag',
        'fine-tuning'
    ],
    project_urls={
        "Bug Reports": "https://github.com/askshameer/Doc2AI/issues",
        "Source": "https://github.com/askshameer/Doc2AI",
        "Documentation": "https://github.com/askshameer/Doc2AI/tree/main/docs",
        "Author": "mailto:mohammed.shameer@gmail.com",
    },
)