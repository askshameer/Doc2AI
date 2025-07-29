#!/usr/bin/env python3
"""
DocToAI Web UI - Streamlit Application
User-friendly web interface for document to AI dataset conversion.
"""

import streamlit as st
import pandas as pd
import json
import tempfile
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

# Import DocToAI components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from cli import DocToAI
from core.data_models import RAGEntry, FineTuneEntry

# Configure page
st.set_page_config(
    page_title="DocToAI - Document to AI Dataset Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    
    .success-message {
        background: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
    }
    
    .error-message {
        background: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #f5c6cb;
    }
    
    .stProgress .st-bo {
        background-color: #667eea;
    }
</style>
""", unsafe_allow_html=True)


class DocToAIUI:
    """Main UI application class."""
    
    def __init__(self):
        self.setup_logging()
        self.initialize_session_state()
    
    def setup_logging(self):
        """Configure logging for the UI."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables."""
        if 'processing_status' not in st.session_state:
            st.session_state.processing_status = 'idle'
        if 'processed_files' not in st.session_state:
            st.session_state.processed_files = []
        if 'dataset_entries' not in st.session_state:
            st.session_state.dataset_entries = []
        if 'processing_stats' not in st.session_state:
            st.session_state.processing_stats = {}
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            <h1>📄 DocToAI</h1>
            <p>Convert documents to AI-ready datasets with ease</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self) -> Dict[str, Any]:
        """Render sidebar with configuration options."""
        st.sidebar.title("⚙️ Configuration")
        
        config = {}
        
        # Processing Mode
        st.sidebar.subheader("🎯 Processing Mode")
        config['mode'] = st.sidebar.selectbox(
            "Dataset Type",
            ['rag', 'finetune'],
            help="RAG: For retrieval-augmented generation. Fine-tune: For model training."
        )
        
        # Chunking Strategy
        st.sidebar.subheader("🧩 Chunking Strategy")
        config['chunk_strategy'] = st.sidebar.selectbox(
            "Strategy",
            ['semantic', 'fixed', 'hierarchical'],
            help="Semantic: Respects sentence boundaries. Fixed: Equal-sized chunks. Hierarchical: Document structure-aware."
        )
        
        # Chunking Parameters
        if config['chunk_strategy'] == 'fixed':
            config['chunk_size'] = st.sidebar.slider(
                "Chunk Size",
                min_value=100,
                max_value=2000,
                value=512,
                help="Number of characters/tokens per chunk"
            )
            config['overlap'] = st.sidebar.slider(
                "Overlap",
                min_value=0,
                max_value=200,
                value=50,
                help="Character overlap between chunks"
            )
            config['split_on'] = st.sidebar.selectbox(
                "Split On",
                ['tokens', 'characters', 'words'],
                help="Unit for chunking"
            )
        
        elif config['chunk_strategy'] == 'semantic':
            config['min_size'] = st.sidebar.slider(
                "Min Size",
                min_value=50,
                max_value=500,
                value=100,
                help="Minimum chunk size"
            )
            config['max_size'] = st.sidebar.slider(
                "Max Size",
                min_value=500,
                max_value=3000,
                value=1000,
                help="Maximum chunk size"
            )
            config['target_size'] = st.sidebar.slider(
                "Target Size",
                min_value=200,
                max_value=1500,
                value=500,
                help="Target chunk size"
            )
        
        # Text Processing
        st.sidebar.subheader("🧹 Text Processing")
        config['clean_text'] = st.sidebar.checkbox(
            "Enable Text Cleaning",
            value=True,
            help="Apply text normalization and cleaning"
        )
        
        if config['clean_text']:
            config['remove_headers_footers'] = st.sidebar.checkbox(
                "Remove Headers/Footers",
                value=False
            )
            config['remove_page_numbers'] = st.sidebar.checkbox(
                "Remove Page Numbers",
                value=False
            )
            config['expand_abbreviations'] = st.sidebar.checkbox(
                "Expand Abbreviations",
                value=False
            )
        
        # Output Format
        st.sidebar.subheader("📊 Output Format")
        config['output_format'] = st.sidebar.selectbox(
            "Format",
            ['jsonl', 'json', 'csv', 'parquet'],
            help="Output file format"
        )
        
        if config['output_format'] in ['jsonl', 'json']:
            config['compression'] = st.sidebar.selectbox(
                "Compression",
                [None, 'gzip', 'bz2'],
                help="Optional compression"
            )
        
        return config
    
    def render_file_upload(self) -> List[Path]:
        """Render file upload interface."""
        st.header("📁 Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['pdf', 'docx', 'doc', 'epub', 'html', 'htm', 'txt', 'md'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, ePub, HTML, TXT, Markdown"
        )
        
        if uploaded_files:
            st.success(f"✅ {len(uploaded_files)} file(s) uploaded successfully!")
            
            # Display file info
            with st.expander("📋 File Details", expanded=True):
                file_data = []
                temp_paths = []
                
                for file in uploaded_files:
                    # Save to temporary location
                    temp_dir = Path(tempfile.gettempdir()) / "doctoai_uploads"
                    temp_dir.mkdir(exist_ok=True)
                    temp_path = temp_dir / file.name
                    
                    with open(temp_path, 'wb') as f:
                        f.write(file.getbuffer())
                    
                    temp_paths.append(temp_path)
                    
                    file_data.append({
                        'Name': file.name,
                        'Size': f"{file.size / 1024:.1f} KB",
                        'Type': file.type or 'Unknown'
                    })
                
                df = pd.DataFrame(file_data)
                st.dataframe(df, use_container_width=True)
                
                return temp_paths
        
        return []
    
    def process_documents(self, file_paths: List[Path], config: Dict[str, Any]):
        """Process documents with progress tracking."""
        if not file_paths:
            st.warning("⚠️ Please upload some documents first!")
            return
        
        st.header("🔄 Processing Documents")
        
        # Initialize DocToAI with config
        doctoai_config = self.build_doctoai_config(config)
        app = DocToAI(doctoai_config)
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Processing statistics
        stats_container = st.container()
        
        try:
            st.session_state.processing_status = 'running'
            total_files = len(file_paths)
            processed_entries = []
            
            for i, file_path in enumerate(file_paths):
                status_text.text(f"Processing {file_path.name}...")
                progress_bar.progress((i + 1) / total_files)
                
                # Load and process document
                document = app.document_loader.load_document(file_path)
                if not document:
                    st.warning(f"⚠️ Failed to process {file_path.name}")
                    continue
                
                # Clean text if enabled
                if config.get('clean_text', True):
                    document.content = app.text_processor.process(
                        document.content,
                        custom_config=self.get_text_processing_config(config)
                    )
                
                # Chunk document
                chunker = app.chunkers[config['chunk_strategy']]
                chunks = chunker.chunk(document)
                
                # Convert to entries
                for chunk in chunks:
                    if config['mode'] == 'rag':
                        entry = app.metadata_manager.create_rag_entry(chunk, document)
                    else:  # finetune
                        entry = app.metadata_manager.create_finetune_entry(
                            chunk, document, template='qa'
                        )
                    processed_entries.append(entry)
                
                # Update session state
                st.session_state.processed_files.append({
                    'name': file_path.name,
                    'chunks': len(chunks),
                    'status': 'success'
                })
            
            # Store results
            st.session_state.dataset_entries = processed_entries
            st.session_state.processing_stats = {
                'total_files': total_files,
                'total_entries': len(processed_entries),
                'processing_time': time.time(),
                'mode': config['mode'],
                'chunk_strategy': config['chunk_strategy']
            }
            
            status_text.text("✅ Processing completed!")
            st.session_state.processing_status = 'completed'
            
            # Show success message
            st.markdown(f"""
            <div class="success-message">
                <h4>🎉 Processing Completed Successfully!</h4>
                <p>Generated <strong>{len(processed_entries)}</strong> entries from <strong>{total_files}</strong> documents.</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.session_state.processing_status = 'error'
            st.markdown(f"""
            <div class="error-message">
                <h4>❌ Processing Error</h4>
                <p>Error: {str(e)}</p>
            </div>
            """, unsafe_allow_html=True)
            self.logger.error(f"Processing error: {e}")
    
    def render_results(self, config: Dict[str, Any]):
        """Render processing results and download options."""
        if st.session_state.processing_status != 'completed':
            return
        
        st.header("📊 Results")
        
        entries = st.session_state.dataset_entries
        stats = st.session_state.processing_stats
        
        # Statistics Dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📄 Files</h3>
                <h2>{}</h2>
            </div>
            """.format(stats['total_files']), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🧩 Chunks</h3>
                <h2>{}</h2>
            </div>
            """.format(stats['total_entries']), unsafe_allow_html=True)
        
        with col3:
            avg_size = sum(len(e.text) for e in entries) / len(entries) if entries else 0
            st.markdown("""
            <div class="metric-card">
                <h3>📏 Avg Size</h3>
                <h2>{:.0f}</h2>
                <p>characters</p>
            </div>
            """.format(avg_size), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Mode</h3>
                <h2>{}</h2>
                <p>{}</p>
            </div>
            """.format(stats['mode'].upper(), stats['chunk_strategy']), unsafe_allow_html=True)
        
        # Data Preview
        st.subheader("🔍 Data Preview")
        
        if entries:
            # Show sample entries
            preview_count = min(5, len(entries))
            
            for i in range(preview_count):
                with st.expander(f"Entry {i+1}: {entries[i].id}"):
                    if config['mode'] == 'rag':
                        entry = entries[i]
                        st.write("**Text:**")
                        st.write(entry.text[:500] + "..." if len(entry.text) > 500 else entry.text)
                        
                        st.write("**Metadata:**")
                        st.json(entry.metadata, expanded=False)
                    
                    else:  # finetune
                        entry = entries[i]
                        st.write("**Conversations:**")
                        for turn in entry.conversations:
                            st.write(f"**{turn.role.title()}:** {turn.content}")
            
            # Statistics Charts
            st.subheader("📈 Dataset Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Text length distribution
                text_lengths = [len(e.text) for e in entries]
                fig = px.histogram(
                    x=text_lengths,
                    title="Text Length Distribution",
                    labels={'x': 'Characters', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # File distribution
                file_stats = {}
                for file_info in st.session_state.processed_files:
                    file_stats[file_info['name']] = file_info['chunks']
                
                fig = px.bar(
                    x=list(file_stats.keys()),
                    y=list(file_stats.values()),
                    title="Chunks per File",
                    labels={'x': 'File', 'y': 'Chunks'}
                )
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        # Download Section
        st.subheader("⬇️ Download Dataset")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Generate dataset file
            if st.button("🔄 Generate Dataset File", type="primary"):
                self.generate_dataset_file(entries, config)
        
        with col2:
            # Download sample
            if st.button("📋 Download Sample (First 10 entries)"):
                sample_entries = entries[:10]
                self.generate_dataset_file(sample_entries, config, filename_suffix="_sample")
    
    def generate_dataset_file(self, entries: List, config: Dict[str, Any], filename_suffix: str = ""):
        """Generate and offer dataset file for download."""
        try:
            # Create temporary file
            temp_dir = Path(tempfile.gettempdir()) / "doctoai_output"
            temp_dir.mkdir(exist_ok=True)
            
            output_format = config['output_format']
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"doctoai_dataset{filename_suffix}_{timestamp}.{output_format}"
            output_path = temp_dir / filename
            
            # Initialize DocToAI and exporter
            doctoai_config = self.build_doctoai_config(config)
            app = DocToAI(doctoai_config)
            exporter = app.exporters[output_format]
            
            # Export data
            if config['mode'] == 'rag':
                exporter.export_rag_data(entries, output_path)
            else:
                exporter.export_finetune_data(entries, output_path)
            
            # Read file for download
            with open(output_path, 'rb') as f:
                file_data = f.read()
            
            # Offer download
            st.download_button(
                label=f"📥 Download {filename}",
                data=file_data,
                file_name=filename,
                mime=self.get_mime_type(output_format)
            )
            
            st.success(f"✅ Dataset file generated: {filename}")
            
        except Exception as e:
            st.error(f"❌ Error generating dataset file: {str(e)}")
    
    def build_doctoai_config(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build DocToAI configuration from UI settings."""
        config = {
            'chunking': {
                ui_config['chunk_strategy']: {}
            },
            'text_processing': self.get_text_processing_config(ui_config),
            'export': {
                ui_config['output_format']: {}
            }
        }
        
        # Add chunking parameters
        strategy_config = config['chunking'][ui_config['chunk_strategy']]
        
        if ui_config['chunk_strategy'] == 'fixed':
            strategy_config.update({
                'chunk_size': ui_config.get('chunk_size', 512),
                'overlap': ui_config.get('overlap', 50),
                'split_on': ui_config.get('split_on', 'tokens')
            })
        elif ui_config['chunk_strategy'] == 'semantic':
            strategy_config.update({
                'min_size': ui_config.get('min_size', 100),
                'max_size': ui_config.get('max_size', 1000),
                'target_size': ui_config.get('target_size', 500)
            })
        
        # Add export parameters
        export_config = config['export'][ui_config['output_format']]
        if 'compression' in ui_config:
            export_config['compression'] = ui_config['compression']
        
        return config
    
    def get_text_processing_config(self, ui_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text processing configuration."""
        return {
            'remove_headers_footers': ui_config.get('remove_headers_footers', False),
            'remove_page_numbers': ui_config.get('remove_page_numbers', False),
            'expand_abbreviations': ui_config.get('expand_abbreviations', False)
        }
    
    def get_mime_type(self, format: str) -> str:
        """Get MIME type for file format."""
        mime_types = {
            'jsonl': 'application/json',
            'json': 'application/json',
            'csv': 'text/csv',
            'parquet': 'application/octet-stream'
        }
        return mime_types.get(format, 'application/octet-stream')
    
    def run(self):
        """Main application runner."""
        self.render_header()
        
        # Get configuration from sidebar
        config = self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3 = st.tabs(["📁 Upload & Process", "📊 Results", "ℹ️ About"])
        
        with tab1:
            # File upload
            uploaded_files = self.render_file_upload()
            
            # Processing button
            if uploaded_files:
                st.markdown("---")
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write("Ready to process documents with the selected configuration.")
                
                with col2:
                    if st.button("🚀 Start Processing", type="primary"):
                        self.process_documents(uploaded_files, config)
        
        with tab2:
            self.render_results(config)
        
        with tab3:
            st.markdown("""
            ## About DocToAI
            
            DocToAI is a comprehensive tool for converting various document formats into 
            structured datasets optimized for AI applications.
            
            ### Supported Features:
            - **Input Formats**: PDF, DOCX, ePub, HTML, TXT, Markdown
            - **Chunking Strategies**: Fixed, Semantic, Hierarchical
            - **Output Formats**: JSONL, JSON, CSV, Parquet
            - **Use Cases**: RAG systems, Fine-tuning datasets
            
            ### Getting Started:
            1. Upload your documents using the file uploader
            2. Configure processing settings in the sidebar
            3. Click "Start Processing" to convert your documents
            4. Download the generated dataset from the Results tab
            
            ### Tips:
            - **Semantic chunking** works best for most documents
            - **RAG mode** is ideal for knowledge retrieval systems
            - **Fine-tune mode** creates conversation datasets for training
            - Use **text cleaning** for better quality outputs
            
            For more information, visit the [DocToAI documentation](https://github.com/doctoai/doctoai).
            """)


def main():
    """Main entry point for the Streamlit app."""
    app = DocToAIUI()
    app.run()


if __name__ == "__main__":
    main()