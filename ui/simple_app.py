#!/usr/bin/env python3
"""
DocToAI Simple Web UI - Streamlit Application
A simplified version that gracefully handles missing dependencies.
"""

import streamlit as st
import sys
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Configure page
st.set_page_config(
    page_title="DocToAI - Document to AI Dataset Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def check_dependencies():
    """Check and report on available dependencies."""
    missing_deps = []
    available_features = []
    
    # Core dependencies
    try:
        import pandas
        available_features.append("✅ Data processing (pandas)")
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import plotly
        available_features.append("✅ Visualizations (plotly)")
    except ImportError:
        missing_deps.append("plotly")
    
    # Document processing dependencies
    try:
        import pdfplumber
        available_features.append("✅ PDF processing (pdfplumber)")
    except ImportError:
        missing_deps.append("pdfplumber")
    
    try:
        from docx import Document
        available_features.append("✅ DOCX processing (python-docx)")
    except ImportError:
        missing_deps.append("python-docx")
    
    try:
        import ebooklib
        available_features.append("✅ ePub processing (ebooklib)")
    except ImportError:
        missing_deps.append("ebooklib")
    
    try:
        from bs4 import BeautifulSoup
        available_features.append("✅ HTML processing (beautifulsoup4)")
    except ImportError:
        missing_deps.append("beautifulsoup4")
    
    try:
        import nltk
        available_features.append("✅ Text processing (nltk)")
    except ImportError:
        missing_deps.append("nltk")
    
    return missing_deps, available_features

def render_setup_instructions():
    """Render setup instructions for missing dependencies."""
    st.error("🚨 **Setup Required**")
    
    st.markdown("""
    To use DocToAI, you need to install the required dependencies first.
    
    **Quick Setup:**
    ```bash
    # Navigate to the DocToAI directory
    cd /path/to/Doc2dataset
    
    # Install all dependencies
    pip install -r requirements.txt
    
    # Or install manually
    pip install pdfplumber PyPDF2 python-docx ebooklib beautifulsoup4 nltk pandas plotly streamlit
    ```
    
    **After installation:**
    1. Restart this application
    2. Start processing your documents!
    """)

def render_main_ui():
    """Render the main UI when dependencies are available."""
    try:
        # Try to import core modules first
        from core.data_models import Document
        from utils.metadata_manager import MetadataManager
        
        # Then try to import the full app
        from cli import DocToAI
        
        st.success("✅ **All dependencies are available!**")
        st.info("🔄 **Loading full DocToAI interface...**")
        
        # Simple file upload for testing
        st.header("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['pdf', 'docx', 'doc', 'epub', 'html', 'htm', 'txt', 'md'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} files uploaded!")
            
            with st.expander("File Details"):
                for file in uploaded_files:
                    st.write(f"📄 **{file.name}** ({file.size / 1024:.1f} KB)")
            
            st.info("💡 **Full processing capabilities will be available once you install all dependencies.**")
        
        # Configuration preview
        st.header("⚙️ Configuration Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.selectbox("Processing Mode", ["rag", "finetune"])
            chunk_strategy = st.selectbox("Chunking Strategy", ["semantic", "fixed", "hierarchical"])
        
        with col2:
            output_format = st.selectbox("Output Format", ["jsonl", "json", "csv", "parquet"])
            clean_text = st.checkbox("Enable Text Cleaning", value=True)
        
        if st.button("🚀 Process Documents (Demo)", type="primary"):
            st.warning("⚠️ **Demo Mode**: Install dependencies for full processing capabilities.")
            st.info("Your configuration is ready! Install requirements.txt to enable processing.")
        
    except ImportError as e:
        st.error(f"❌ **Import Error**: {e}")
        st.info("Some dependencies are still missing. Please install them to continue.")

def main():
    """Main application entry point."""
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; margin-bottom: 2rem;">
        <h1>📄 DocToAI</h1>
        <p>Document to AI Dataset Converter</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check dependencies
    missing_deps, available_features = check_dependencies()
    
    # Sidebar with status
    st.sidebar.title("🔧 System Status")
    
    if available_features:
        st.sidebar.subheader("✅ Available Features")
        for feature in available_features:
            st.sidebar.write(feature)
    
    if missing_deps:
        st.sidebar.subheader("❌ Missing Dependencies")
        for dep in missing_deps:
            st.sidebar.write(f"• {dep}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📦 Installation")
        st.sidebar.code("pip install -r requirements.txt")
    
    # Main content
    if missing_deps:
        # Show setup instructions
        tab1, tab2 = st.tabs(["🚨 Setup Required", "ℹ️ About"])
        
        with tab1:
            render_setup_instructions()
        
        with tab2:
            st.markdown("""
            ## About DocToAI
            
            DocToAI converts various document formats into AI-ready datasets.
            
            ### Supported Formats:
            - **PDF** (.pdf) - Including OCR for scanned documents
            - **DOCX** (.docx, .doc) - Microsoft Word documents  
            - **ePub** (.epub) - Electronic books
            - **HTML** (.html, .htm) - Web pages
            - **Text** (.txt, .md) - Plain text and Markdown
            
            ### Output Formats:
            - **JSONL** - Streaming-friendly format
            - **JSON** - Standard JSON format
            - **CSV** - Spreadsheet-compatible
            - **Parquet** - Columnar format for big data
            
            ### Use Cases:
            - **RAG Systems** - Retrieval-Augmented Generation
            - **Fine-tuning** - Training data for language models
            - **Content Analysis** - Document processing at scale
            
            **Install the dependencies to get started!**
            """)
    else:
        # Dependencies available, show main UI
        render_main_ui()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        DocToAI v1.0.0 | <a href="https://github.com/doctoai/doctoai">GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()