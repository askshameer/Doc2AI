#!/usr/bin/env python3
"""
DocToAI Enhanced Web UI - Streamlit Application
Enhanced user-friendly web interface with improved components and styling.

Author: Shameer Mohammed
Email: mohammed.shameer@gmail.com
GitHub: https://github.com/askshameer/Doc2AI
"""

import streamlit as st
import pandas as pd
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import time
from datetime import datetime
import plotly.express as px

# Import DocToAI components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from cli import DocToAI
from core.data_models import RAGEntry, FineTuneEntry
from ui.components import *

# Configure page
st.set_page_config(
    page_title="DocToAI - Document to AI Dataset Converter",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #667eea;
        --secondary-color: #764ba2;
        --success-color: #28a745;
        --warning-color: #ffc107;
        --error-color: #dc3545;
        --info-color: #17a2b8;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.2rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #e9ecef;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin: 0.5rem 0;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Status messages */
    .status-message {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid;
        font-weight: 500;
    }
    
    .status-success {
        background: #d4edda;
        color: #155724;
        border-left-color: var(--success-color);
    }
    
    .status-error {
        background: #f8d7da;
        color: #721c24;
        border-left-color: var(--error-color);
    }
    
    .status-warning {
        background: #fff3cd;
        color: #856404;
        border-left-color: var(--warning-color);
    }
    
    .status-info {
        background: #d1ecf1;
        color: #0c5460;
        border-left-color: var(--info-color);
    }
    
    /* Upload area styling */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: 15px;
        padding: 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9ff 0%, #ffffff 100%);
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        border-radius: 25px;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--primary-color) 0%, var(--secondary-color) 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        color: #666;
        padding: 2rem;
        border-top: 1px solid #e9ecef;
        margin-top: 3rem;
    }
    
    /* Animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-in-out;
    }
</style>
""", unsafe_allow_html=True)


class EnhancedDocToAIUI:
    """Enhanced UI application class with improved components."""
    
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
        defaults = {
            'processing_status': 'idle',
            'processed_files': [],
            'dataset_entries': [],
            'processing_stats': {},
            'current_config': {},
            'upload_key': 0
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render the enhanced main header."""
        st.markdown("""
        <div class="main-header fade-in">
            <h1>📄 DocToAI</h1>
            <p>Transform documents into AI-ready datasets with intelligence</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_enhanced_sidebar(self) -> Dict[str, Any]:
        """Render enhanced sidebar with better organization."""
        st.sidebar.title("⚙️ Configuration")
        
        config = {}
        
        # Quick presets
        st.sidebar.subheader("🚀 Quick Presets")
        preset = st.sidebar.selectbox(
            "Choose a preset",
            ["Custom", "RAG Optimized", "Fine-tuning Ready", "Large Documents"],
            help="Pre-configured settings for common use cases"
        )
        
        if preset != "Custom":
            config.update(self.get_preset_config(preset))
        
        st.sidebar.markdown("---")
        
        # Processing Mode
        st.sidebar.subheader("🎯 Processing Mode")
        config['mode'] = st.sidebar.radio(
            "Dataset Type",
            ['rag', 'finetune'],
            help="RAG: For retrieval systems. Fine-tune: For model training.",
            horizontal=True
        )
        
        # Model Selection (only for fine-tuning)
        if config['mode'] == 'finetune':
            st.sidebar.subheader("🤖 Target Model")
            
            # Load available models
            try:
                from core.model_templates import ModelTemplateManager
                template_manager = ModelTemplateManager()
                available_models = template_manager.get_available_models()
                
                # Group by provider
                providers = {}
                for model in available_models:
                    provider = model['provider']
                    if provider not in providers:
                        providers[provider] = []
                    providers[provider].append(model)
                
                # Model selection
                provider_names = list(providers.keys())
                selected_provider = st.sidebar.selectbox(
                    "Model Provider",
                    provider_names,
                    help="Choose the provider of your target model"
                )
                
                provider_models = providers[selected_provider]
                model_options = {f"{m['id']} ({m['name']})": m['id'] for m in provider_models}
                
                selected_model_display = st.sidebar.selectbox(
                    "Specific Model",
                    list(model_options.keys()),
                    help="Choose the specific model you want to fine-tune"
                )
                config['model_id'] = model_options[selected_model_display]
                
                # Show model info
                selected_model_info = next(m for m in provider_models if m['id'] == config['model_id'])
                
                with st.sidebar.expander("📋 Model Information", expanded=False):
                    st.write(f"**Max Tokens:** {selected_model_info['max_tokens']:,}")
                    st.write(f"**System Messages:** {'✅ Supported' if selected_model_info['supports_system'] else '❌ Not supported'}")
                    st.write(f"**Provider:** {selected_model_info['provider']}")
                
                # Question type
                config['question_type'] = st.sidebar.selectbox(
                    "Question Type",
                    ['general', 'summary', 'explanation', 'context', 'analytical'],
                    help="Type of questions to generate for training"
                )
                
                # System message
                if selected_model_info['supports_system']:
                    config['system_message'] = st.sidebar.text_area(
                        "System Message (Optional)",
                        placeholder="You are a helpful assistant...",
                        help="Custom system message for the model"
                    )
                
            except ImportError:
                st.sidebar.warning("⚠️ Model templates not available. Using default settings.")
                config['model_id'] = 'gpt-3.5-turbo'
                config['question_type'] = 'general'
        
        # Chunking Strategy with visual guide
        st.sidebar.subheader("🧩 Chunking Strategy")
        
        strategy_help = {
            'semantic': "🧠 Respects sentence and paragraph boundaries",
            'fixed': "📏 Creates equal-sized chunks with overlap", 
            'hierarchical': "🏗️ Preserves document structure (sections, chapters)"
        }
        
        config['chunk_strategy'] = st.sidebar.selectbox(
            "Strategy",
            ['semantic', 'fixed', 'hierarchical'],
            format_func=lambda x: strategy_help[x]
        )
        
        # Strategy-specific parameters with better UI
        with st.sidebar.expander("🔧 Strategy Parameters", expanded=True):
            if config['chunk_strategy'] == 'fixed':
                config['chunk_size'] = st.slider(
                    "Chunk Size", 100, 2000, 512,
                    help="Number of characters/tokens per chunk"
                )
                config['overlap'] = st.slider(
                    "Overlap", 0, 200, 50,
                    help="Character overlap between chunks"
                )
                config['split_on'] = st.selectbox(
                    "Split On", ['tokens', 'characters', 'words']
                )
            
            elif config['chunk_strategy'] == 'semantic':
                col1, col2 = st.columns(2)
                with col1:
                    config['min_size'] = st.number_input(
                        "Min Size", 50, 500, 100, step=50
                    )
                with col2:
                    config['max_size'] = st.number_input(
                        "Max Size", 500, 3000, 1000, step=100
                    )
                config['target_size'] = st.slider(
                    "Target Size", 200, 1500, 500
                )
        
        st.sidebar.markdown("---")
        
        # Text Processing with toggles
        st.sidebar.subheader("🧹 Text Processing")
        config['clean_text'] = st.sidebar.toggle(
            "Enable Text Cleaning",
            value=True,
            help="Apply comprehensive text normalization"
        )
        
        if config['clean_text']:
            with st.sidebar.expander("🔍 Cleaning Options"):
                config['remove_headers_footers'] = st.checkbox(
                    "Remove Headers/Footers", value=False
                )
                config['remove_page_numbers'] = st.checkbox(
                    "Remove Page Numbers", value=False
                )
                config['expand_abbreviations'] = st.checkbox(
                    "Expand Abbreviations", value=False
                )
                config['fix_hyphenation'] = st.checkbox(
                    "Fix Hyphenation", value=True
                )
        
        st.sidebar.markdown("---")
        
        # Output Format with descriptions
        st.sidebar.subheader("📊 Output Format")
        
        format_descriptions = {
            'jsonl': "📄 JSONL - Streaming-friendly, one entry per line",
            'json': "🗂️ JSON - Standard format with full structure", 
            'csv': "📊 CSV - Spreadsheet compatible",
            'parquet': "🗃️ Parquet - Columnar, efficient for big data"
        }
        
        config['output_format'] = st.sidebar.selectbox(
            "Format",
            ['jsonl', 'json', 'csv', 'parquet'],
            format_func=lambda x: format_descriptions[x]
        )
        
        # Format-specific options
        with st.sidebar.expander("📤 Export Options"):
            if config['output_format'] in ['jsonl', 'json']:
                config['compression'] = st.selectbox(
                    "Compression", [None, 'gzip', 'bz2']
                )
            
            if config['output_format'] == 'csv':
                config['flatten_metadata'] = st.checkbox(
                    "Flatten Metadata", value=True
                )
            
            config['include_sample'] = st.checkbox(
                "Include Sample File", value=True,
                help="Generate a separate sample file with first 10 entries"
            )
        
        # Save current config
        st.session_state.current_config = config
        
        return config
    
    def get_preset_config(self, preset: str) -> Dict[str, Any]:
        """Get configuration for selected preset."""
        presets = {
            "RAG Optimized": {
                'mode': 'rag',
                'chunk_strategy': 'semantic',
                'min_size': 200,
                'max_size': 800,
                'target_size': 500,
                'clean_text': True,
                'output_format': 'jsonl'
            },
            "Fine-tuning Ready": {
                'mode': 'finetune',
                'chunk_strategy': 'semantic',
                'min_size': 100,
                'max_size': 1200,
                'target_size': 600,
                'clean_text': True,
                'output_format': 'jsonl'
            },
            "Large Documents": {
                'mode': 'rag',
                'chunk_strategy': 'hierarchical',
                'max_chunk_size': 1500,
                'clean_text': True,
                'output_format': 'parquet'
            }
        }
        return presets.get(preset, {})
    
    def render_enhanced_upload(self) -> List[Path]:
        """Render enhanced file upload with drag-and-drop styling."""
        st.markdown("""
        <div class="upload-area">
            <h3>📁 Upload Your Documents</h3>
            <p>Drag and drop files here or click to browse</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose document files",
            type=['pdf', 'docx', 'doc', 'epub', 'html', 'htm', 'txt', 'md'],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.upload_key}",
            label_visibility="hidden"
        )
        
        if uploaded_files:
            # Success message
            st.markdown(f"""
            <div class="status-message status-success">
                ✅ <strong>{len(uploaded_files)} file(s) uploaded successfully!</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # File details with enhanced table
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
                    
                    # Determine file type icon
                    ext = Path(file.name).suffix.lower()
                    icon = {
                        '.pdf': '📕', '.docx': '📘', '.doc': '📘',
                        '.epub': '📗', '.html': '🌐', '.htm': '🌐',
                        '.txt': '📄', '.md': '📝'
                    }.get(ext, '📄')
                    
                    file_data.append({
                        'Icon': icon,
                        'Name': file.name,
                        'Size': f"{file.size / 1024:.1f} KB",
                        'Type': file.type or 'Unknown'
                    })
                
                # Display as styled dataframe
                df = pd.DataFrame(file_data)
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Quick stats
                total_size = sum(f.size for f in uploaded_files) / 1024 / 1024
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    render_metric_card("Files", str(len(uploaded_files)), "documents")
                with col2:
                    render_metric_card("Total Size", f"{total_size:.1f} MB", "all files")
                with col3:
                    formats = set(Path(f.name).suffix.lower() for f in uploaded_files)
                    render_metric_card("Formats", str(len(formats)), "different types")
                
                return temp_paths
        
        return []
    
    def render_enhanced_processing(self, file_paths: List[Path], config: Dict[str, Any]):
        """Enhanced processing with better progress tracking."""
        if not file_paths:
            st.markdown("""
            <div class="status-message status-warning">
                ⚠️ <strong>Please upload some documents first!</strong>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Configuration summary
        render_configuration_summary(config)
        
        st.markdown("---")
        
        # Processing controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.write("Ready to process documents with the selected configuration.")
        
        with col2:
            if st.button("🚀 Start Processing", type="primary", use_container_width=True):
                self.process_documents_enhanced(file_paths, config)
        
        with col3:
            if st.button("🔄 Reset", use_container_width=True):
                self.reset_session()
    
    def process_documents_enhanced(self, file_paths: List[Path], config: Dict[str, Any]):
        """Enhanced document processing with better UX."""
        st.markdown("""
        <div class="status-message status-info">
            🔄 <strong>Processing started...</strong>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize DocToAI
        doctoai_config = self.build_doctoai_config(config)
        app = DocToAI(doctoai_config)
        
        # Progress tracking containers
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Processing metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                files_metric = st.empty()
            with col2:
                chunks_metric = st.empty()
            with col3:
                time_metric = st.empty()
        
        start_time = time.time()
        
        try:
            st.session_state.processing_status = 'running'
            total_files = len(file_paths)
            processed_entries = []
            processed_files = []
            
            for i, file_path in enumerate(file_paths):
                current_progress = i / total_files
                progress_bar.progress(current_progress)
                status_text.text(f"📄 Processing: {file_path.name}")
                
                # Update metrics
                files_metric.metric("Files Processed", f"{i}/{total_files}")
                chunks_metric.metric("Chunks Generated", len(processed_entries))
                elapsed = time.time() - start_time
                time_metric.metric("Elapsed Time", f"{elapsed:.1f}s")
                
                try:
                    # Load document
                    document = app.document_loader.load_document(file_path)
                    if not document:
                        processed_files.append({
                            'name': file_path.name,
                            'chunks': 0,
                            'status': 'failed',
                            'error': 'Failed to load document'
                        })
                        continue
                    
                    # Clean text
                    if config.get('clean_text', True):
                        document.content = app.text_processor.process(
                            document.content,
                            custom_config=self.get_text_processing_config(config)
                        )
                    
                    # Chunk document
                    chunker = app.chunkers[config['chunk_strategy']]
                    chunks = chunker.chunk(document)
                    
                    # Convert to entries
                    file_entries = []
                    for chunk in chunks:
                        if config['mode'] == 'rag':
                            entry = app.metadata_manager.create_rag_entry(chunk, document)
                        else:
                            entry = app.metadata_manager.create_finetune_entry(
                                chunk, document, 
                                model_id=config.get('model_id', 'gpt-3.5-turbo'),
                                question_type=config.get('question_type', 'general'),
                                system_message=config.get('system_message')
                            )
                        file_entries.append(entry)
                        processed_entries.append(entry)
                    
                    # Track file processing
                    processed_files.append({
                        'name': file_path.name,
                        'chunks': len(chunks),
                        'status': 'success',
                        'size': len(document.content),
                        'avg_chunk_size': len(document.content) / len(chunks) if chunks else 0
                    })
                    
                except Exception as e:
                    processed_files.append({
                        'name': file_path.name,
                        'chunks': 0,
                        'status': 'error',
                        'error': str(e)
                    })
                    self.logger.error(f"Error processing {file_path.name}: {e}")
            
            # Final progress update
            progress_bar.progress(1.0)
            status_text.text("✅ Processing completed!")
            
            # Update session state
            st.session_state.dataset_entries = processed_entries
            st.session_state.processed_files = processed_files
            st.session_state.processing_stats = {
                'total_files': total_files,
                'successful_files': len([f for f in processed_files if f['status'] == 'success']),
                'total_entries': len(processed_entries),
                'processing_time': time.time() - start_time,
                'mode': config['mode'],
                'chunk_strategy': config['chunk_strategy'],
                'avg_entries_per_file': len(processed_entries) / total_files if total_files > 0 else 0
            }
            
            st.session_state.processing_status = 'completed'
            
            # Success message with statistics
            stats = st.session_state.processing_stats
            st.markdown(f"""
            <div class="status-message status-success">
                <h4>🎉 Processing Completed Successfully!</h4>
                <ul>
                    <li><strong>{stats['total_entries']}</strong> entries generated from <strong>{stats['successful_files']}/{stats['total_files']}</strong> documents</li>
                    <li>Processing took <strong>{stats['processing_time']:.1f} seconds</strong></li>
                    <li>Average <strong>{stats['avg_entries_per_file']:.1f}</strong> entries per file</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.session_state.processing_status = 'error'
            st.markdown(f"""
            <div class="status-message status-error">
                <h4>❌ Processing Error</h4>
                <p><strong>Error:</strong> {str(e)}</p>
                <p>Please check your files and configuration, then try again.</p>
            </div>
            """, unsafe_allow_html=True)
            self.logger.error(f"Processing error: {e}")
    
    def render_enhanced_results(self, config: Dict[str, Any]):
        """Render enhanced results with better visualization."""
        if st.session_state.processing_status != 'completed':
            st.info("🔄 No results to display yet. Please process some documents first.")
            return
        
        entries = st.session_state.dataset_entries
        stats = st.session_state.processing_stats
        processed_files = st.session_state.processed_files
        
        # Results header
        st.markdown("""
        <div style="text-align: center; margin: 2rem 0;">
            <h2>📊 Processing Results</h2>
            <p>Your documents have been successfully converted to an AI-ready dataset!</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Enhanced statistics dashboard
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            render_metric_card(
                "📄 Files", 
                f"{stats['successful_files']}/{stats['total_files']}", 
                "processed/total"
            )
        
        with col2:
            render_metric_card(
                "🧩 Entries", 
                str(stats['total_entries']), 
                f"{config['mode']} format"
            )
        
        with col3:
            def get_entry_text_length(entry):
                """Get text length from either RAG entry or fine-tuning dictionary."""
                if hasattr(entry, 'text'):
                    return len(entry.text)
                elif isinstance(entry, dict):
                    if 'text' in entry:
                        return len(entry['text'])
                    elif 'messages' in entry:
                        # For fine-tuning entries, sum up all message content
                        total_length = 0
                        for msg in entry['messages']:
                            content = msg.get('content') or ''
                            total_length += len(str(content))
                        return total_length
                    elif 'conversations' in entry:
                        # Alternative format
                        total_length = 0
                        for turn in entry['conversations']:
                            content = turn.get('content') or ''
                            total_length += len(str(content))
                        return total_length
                return 0
            
            avg_size = sum(get_entry_text_length(e) for e in entries) / len(entries) if entries else 0
            render_metric_card(
                "📏 Avg Size", 
                f"{avg_size:.0f}", 
                "characters"
            )
        
        with col4:
            render_metric_card(
                "⏱️ Time", 
                f"{stats['processing_time']:.1f}s", 
                "processing"
            )
        
        with col5:
            render_metric_card(
                "🎯 Strategy", 
                stats['chunk_strategy'].title(), 
                "chunking"
            )
        
        st.markdown("---")
        
        # Tabbed results view
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Data Preview", 
            "📈 Statistics", 
            "📁 File Details", 
            "⬇️ Download"
        ])
        
        with tab1:
            render_dataset_preview(entries, config['mode'], max_entries=5)
        
        with tab2:
            self.render_statistics_tab(entries, processed_files)
        
        with tab3:
            self.render_file_details_tab(processed_files)
        
        with tab4:
            self.render_download_tab(entries, config)
    
    def render_statistics_tab(self, entries: List, processed_files: List[Dict]):
        """Render statistics tab with visualizations."""
        if not entries:
            st.info("No data to display")
            return
        
        # Text statistics
        render_text_statistics_chart(entries)
        
        st.markdown("---")
        
        # File processing statistics
        col1, col2 = st.columns(2)
        
        with col1:
            # Success rate pie chart
            success_count = len([f for f in processed_files if f['status'] == 'success'])
            failed_count = len(processed_files) - success_count
            
            if failed_count > 0:
                fig = px.pie(
                    values=[success_count, failed_count],
                    names=['Success', 'Failed'],
                    title="File Processing Success Rate",
                    color_discrete_map={'Success': '#28a745', 'Failed': '#dc3545'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Chunks per file
            file_chunks = [(f['name'], f['chunks']) for f in processed_files if f['status'] == 'success']
            if file_chunks:
                df = pd.DataFrame(file_chunks, columns=['File', 'Chunks'])
                fig = px.bar(
                    df, x='File', y='Chunks',
                    title="Chunks Generated per File",
                    color='Chunks',
                    color_continuous_scale='viridis'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    def render_file_details_tab(self, processed_files: List[Dict]):
        """Render file details tab."""
        if not processed_files:
            st.info("No file details available")
            return
        
        # Summary cards
        successful_files = [f for f in processed_files if f['status'] == 'success']
        failed_files = [f for f in processed_files if f['status'] != 'success']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            render_metric_card("✅ Successful", str(len(successful_files)), "files processed")
        
        with col2:
            render_metric_card("❌ Failed", str(len(failed_files)), "files skipped")
        
        with col3:
            total_chunks = sum(f.get('chunks', 0) for f in successful_files)
            render_metric_card("🧩 Total Chunks", str(total_chunks), "generated")
        
        # Detailed file table
        st.subheader("📄 File Processing Details")
        
        # Prepare data for display
        display_data = []
        for file_info in processed_files:
            status_icon = {
                'success': '✅',
                'failed': '❌', 
                'error': '🚫'
            }.get(file_info['status'], '❓')
            
            display_data.append({
                'Status': status_icon,
                'File Name': file_info['name'],
                'Chunks': file_info.get('chunks', 0),
                'Size (chars)': file_info.get('size', 'N/A'),
                'Avg Chunk Size': f"{file_info.get('avg_chunk_size', 0):.0f}" if file_info.get('avg_chunk_size') else 'N/A',
                'Notes': file_info.get('error', 'Successfully processed')
            })
        
        df = pd.DataFrame(display_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Failed files details
        if failed_files:
            with st.expander("❌ Failed Files Details", expanded=False):
                for file_info in failed_files:
                    st.error(f"**{file_info['name']}**: {file_info.get('error', 'Unknown error')}")
    
    def render_download_tab(self, entries: List, config: Dict[str, Any]):
        """Render download tab with multiple options."""
        if not entries:
            st.info("No data to download")
            return
        
        st.subheader("⬇️ Download Your Dataset")
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📦 Full Dataset")
            st.write(f"Complete dataset with {len(entries)} entries")
            
            if st.button("🔄 Generate Full Dataset", type="primary", use_container_width=True):
                self.generate_enhanced_dataset_file(entries, config)
        
        with col2:
            st.markdown("### 📋 Sample Dataset")
            sample_size = min(10, len(entries))
            st.write(f"Sample dataset with first {sample_size} entries for testing")
            
            if st.button("📋 Generate Sample", use_container_width=True):
                sample_entries = entries[:sample_size]
                self.generate_enhanced_dataset_file(sample_entries, config, filename_suffix="_sample")
        
        # Additional options
        with st.expander("🔧 Advanced Download Options", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                custom_sample_size = st.number_input(
                    "Custom Sample Size",
                    min_value=1,
                    max_value=len(entries),
                    value=min(50, len(entries))
                )
                
                if st.button("📊 Generate Custom Sample"):
                    custom_entries = entries[:custom_sample_size]
                    self.generate_enhanced_dataset_file(
                        custom_entries, 
                        config, 
                        filename_suffix=f"_custom_{custom_sample_size}"
                    )
            
            with col2:
                # Format conversion
                st.write("**Convert to Different Format:**")
                new_format = st.selectbox(
                    "Target Format",
                    ['jsonl', 'json', 'csv', 'parquet'],
                    index=['jsonl', 'json', 'csv', 'parquet'].index(config['output_format'])
                )
                
                if new_format != config['output_format']:
                    if st.button(f"🔄 Convert to {new_format.upper()}"):
                        new_config = config.copy()
                        new_config['output_format'] = new_format
                        self.generate_enhanced_dataset_file(
                            entries, 
                            new_config, 
                            filename_suffix=f"_{new_format}"
                        )
    
    def generate_enhanced_dataset_file(self, entries: List, config: Dict[str, Any], filename_suffix: str = ""):
        """Generate dataset file with enhanced feedback."""
        with st.spinner("🔄 Generating dataset file..."):
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
                
                # File statistics
                file_size = len(file_data)
                
                # Success message with file info
                st.markdown(f"""
                <div class="status-message status-success">
                    <h4>✅ Dataset Generated Successfully!</h4>
                    <ul>
                        <li><strong>File:</strong> {filename}</li>
                        <li><strong>Format:</strong> {output_format.upper()}</li>
                        <li><strong>Entries:</strong> {len(entries)}</li>
                        <li><strong>Size:</strong> {file_size / 1024:.1f} KB</li>
                    </ul>
                </div>
                """, unsafe_allow_html=True)
                
                # Download button
                st.download_button(
                    label=f"📥 Download {filename}",
                    data=file_data,
                    file_name=filename,
                    mime=self.get_mime_type(output_format),
                    use_container_width=True
                )
                
            except Exception as e:
                st.markdown(f"""
                <div class="status-message status-error">
                    <h4>❌ Error Generating Dataset</h4>
                    <p><strong>Error:</strong> {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def reset_session(self):
        """Reset the session state."""
        st.session_state.processing_status = 'idle'
        st.session_state.processed_files = []
        st.session_state.dataset_entries = []
        st.session_state.processing_stats = {}
        st.session_state.upload_key += 1
        st.rerun()
    
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
            'expand_abbreviations': ui_config.get('expand_abbreviations', False),
            'fix_hyphenation': ui_config.get('fix_hyphenation', True)
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
        config = self.render_enhanced_sidebar()
        
        # Main content area with enhanced tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📁 Upload & Process", 
            "📊 Results", 
            "❓ Help & Tips",
            "ℹ️ About"
        ])
        
        with tab1:
            # File upload
            uploaded_files = self.render_enhanced_upload()
            
            if uploaded_files:
                st.markdown("---")
                self.render_enhanced_processing(uploaded_files, config)
        
        with tab2:
            self.render_enhanced_results(config)
        
        with tab3:
            render_help_section()
            
            # Format Preview Section
            if config['mode'] == 'finetune':
                st.markdown("---")
                st.subheader("🔍 Format Preview")
                
                try:
                    from core.model_templates import ModelTemplateManager
                    from core.data_models import Chunk, ChunkLocation, ProcessingInfo, Document, DocumentMetadata
                    from datetime import datetime
                    
                    # Create sample data for preview
                    sample_metadata = DocumentMetadata(
                        filename="sample.pdf",
                        file_hash="abc123",
                        file_size=1024,
                        format="pdf"
                    )
                    
                    sample_document = Document(
                        document_id="sample_doc",
                        content="This is a sample document content for preview purposes.",
                        metadata=sample_metadata
                    )
                    
                    sample_location = ChunkLocation(page=1, section="Introduction")
                    sample_processing = ProcessingInfo(
                        extraction_method="sample",
                        chunking_strategy="preview",
                        processing_timestamp=datetime.now()
                    )
                    
                    sample_chunk = Chunk(
                        chunk_id="sample_chunk_001",
                        document_id="sample_doc",
                        text="Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models that enable computers to improve their performance on a specific task through experience.",
                        location=sample_location,
                        processing=sample_processing
                    )
                    
                    # Generate preview
                    template_manager = ModelTemplateManager()
                    preview_entry = template_manager.format_for_model(
                        model_id=config.get('model_id', 'gpt-3.5-turbo'),
                        chunk=sample_chunk,
                        document=sample_document,
                        question_type=config.get('question_type', 'general'),
                        system_message=config.get('system_message')
                    )
                    
                    st.write("**Sample output for your selected model:**")
                    st.json(preview_entry, expanded=True)
                    
                    # Token estimation
                    template = template_manager.get_template(config.get('model_id', 'gpt-3.5-turbo'))
                    estimated_tokens = template.estimate_total_tokens(preview_entry)
                    max_tokens = template.config.max_tokens
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Estimated Tokens", estimated_tokens)
                    with col2:
                        st.metric("Model Max Tokens", f"{max_tokens:,}")
                    
                    if estimated_tokens > max_tokens:
                        st.error(f"⚠️ Sample exceeds model limit! Consider shorter chunks.")
                    else:
                        st.success(f"✅ Sample fits within model limits ({estimated_tokens}/{max_tokens} tokens)")
                
                except Exception as e:
                    st.warning(f"⚠️ Preview not available: {e}")
            
            # Additional tips
            st.markdown("---")
            st.subheader("💡 Pro Tips")
            
            tips = [
                "🎯 **Choose the right mode**: Use RAG for knowledge retrieval, Fine-tune for model training",
                "🧩 **Semantic chunking works best** for most documents as it respects natural boundaries",
                "📏 **Adjust chunk sizes** based on your model's context window (e.g., 512 for older models, 2048+ for newer ones)",
                "🧹 **Enable text cleaning** for better quality, especially with OCR'd documents",
                "💾 **Use Parquet format** for large datasets to save space and enable faster loading",
                "📋 **Try the sample first** to verify your configuration before processing large batches",
                "🔄 **Use presets** for quick configuration based on common use cases"
            ]
            
            for tip in tips:
                st.markdown(f"- {tip}")
        
        with tab4:
            st.markdown("""
            ## 📄 About DocToAI
            
            DocToAI is a comprehensive tool for converting various document formats into 
            structured datasets optimized for AI applications.
            
            ### 🌟 Key Features:
            - **Multi-format Support**: PDF, DOCX, ePub, HTML, TXT, Markdown
            - **Intelligent Chunking**: Fixed, Semantic, and Hierarchical strategies
            - **Rich Metadata**: Comprehensive metadata preservation and enrichment
            - **Multiple Outputs**: JSONL, JSON, CSV, Parquet formats
            - **AI-Ready**: Optimized for RAG systems and fine-tuning workflows
            
            ### 🚀 Quick Start:
            1. **Upload** your documents using the file uploader
            2. **Configure** processing settings or use a preset
            3. **Process** your documents with one click
            4. **Download** the generated dataset
            
            ### 📊 Use Cases:
            - **Knowledge Retrieval**: Create RAG datasets for Q&A systems
            - **Model Training**: Generate fine-tuning datasets for language models
            - **Content Analysis**: Extract and analyze document content at scale
            - **Data Migration**: Convert legacy documents to modern formats
            
            ### 🔧 Technical Details:
            - **Built with**: Python, Streamlit, pandas, PyPDF2, python-docx
            - **Extensible**: Plugin architecture for custom extractors
            - **Scalable**: Efficient processing of large document collections
            - **Configurable**: Highly customizable processing pipeline
            
            ### 📚 Resources:
            - [Documentation](https://doctoai.readthedocs.io/)
            - [GitHub Repository](https://github.com/doctoai/doctoai)
            - [Issue Tracker](https://github.com/doctoai/doctoai/issues)
            
            ---
            
            **Version:** 1.0.0 | **License:** MIT | **Made with ❤️ for the AI community**
            """)
        
        # Footer
        render_footer()


def main():
    """Main entry point for the enhanced Streamlit app."""
    app = EnhancedDocToAIUI()
    app.run()


if __name__ == "__main__":
    main()