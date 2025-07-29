"""
UI Components for DocToAI Web Interface
Reusable components for the Streamlit application.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
import json

def render_metric_card(title: str, value: str, subtitle: str = "", color: str = "#667eea"):
    """Render a styled metric card."""
    st.markdown(f"""
    <div style="
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid {color};
        margin: 0.5rem 0;
        text-align: center;
    ">
        <h3 style="color: {color}; margin: 0;">{title}</h3>
        <h2 style="margin: 0.5rem 0;">{value}</h2>
        <p style="color: #666; margin: 0; font-size: 0.9rem;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def render_status_badge(status: str, message: str = ""):
    """Render a status badge with color coding."""
    colors = {
        'success': '#28a745',
        'warning': '#ffc107', 
        'error': '#dc3545',
        'info': '#17a2b8',
        'processing': '#6f42c1'
    }
    
    color = colors.get(status, '#6c757d')
    
    st.markdown(f"""
    <div style="
        background: {color}20;
        color: {color};
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid {color}40;
        display: inline-block;
        margin: 0.25rem;
        font-size: 0.9rem;
        font-weight: 500;
    ">
        {status.upper()}: {message}
    </div>
    """, unsafe_allow_html=True)

def render_progress_section(current: int, total: int, message: str = ""):
    """Render an enhanced progress section."""
    percentage = (current / total * 100) if total > 0 else 0
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.progress(current / total if total > 0 else 0)
        if message:
            st.caption(message)
    
    with col2:
        st.metric("Progress", f"{current}/{total}", f"{percentage:.1f}%")

def render_file_info_table(files: List[Dict[str, Any]]):
    """Render a styled file information table."""
    if not files:
        st.info("No files uploaded yet.")
        return
    
    # Convert to DataFrame for better display
    df = pd.DataFrame(files)
    
    # Style the dataframe
    styled_df = df.style.format({
        'Size': lambda x: f"{x}" if isinstance(x, str) else f"{x:.1f} KB"
    }).background_gradient(subset=['Size'] if 'Size' in df.columns else [])
    
    st.dataframe(styled_df, use_container_width=True)

def render_text_statistics_chart(entries: List[Any]):
    """Render text statistics visualization."""
    if not entries:
        st.info("No data to display")
        return
    
    def get_text_length(entry):
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
    
    # Extract text lengths
    text_lengths = [get_text_length(entry) for entry in entries]
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram
        fig_hist = px.histogram(
            x=text_lengths,
            title="Text Length Distribution",
            labels={'x': 'Characters', 'y': 'Count'},
            color_discrete_sequence=['#667eea']
        )
        fig_hist.update_layout(showlegend=False)
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Box plot
        fig_box = go.Figure()
        fig_box.add_trace(go.Box(
            y=text_lengths,
            name="Text Length",
            marker_color='#667eea'
        ))
        fig_box.update_layout(
            title="Text Length Statistics",
            yaxis_title="Characters"
        )
        st.plotly_chart(fig_box, use_container_width=True)

def render_dataset_preview(entries: List[Any], mode: str = 'rag', max_entries: int = 3):
    """Render a preview of dataset entries."""
    if not entries:
        st.info("No entries to preview")
        return
    
    def get_entry_id(entry):
        """Get entry ID from either object or dictionary."""
        if hasattr(entry, 'id'):
            return entry.id
        elif isinstance(entry, dict):
            return entry.get('id', f"entry_{hash(str(entry))}")
        return f"entry_{hash(str(entry))}"
    
    def get_entry_text(entry):
        """Get text content from either RAG entry or fine-tuning dictionary."""
        if hasattr(entry, 'text'):
            return entry.text
        elif isinstance(entry, dict):
            if 'text' in entry:
                return entry['text']
            elif 'messages' in entry:
                # For fine-tuning entries, combine all message content
                content_parts = []
                for msg in entry['messages']:
                    content = msg.get('content') or ''
                    if content:
                        content_parts.append(f"{msg.get('role', 'unknown')}: {content}")
                return "\n".join(content_parts)
        return "No text content available"
    
    def get_entry_metadata(entry):
        """Get metadata from either object or dictionary."""
        if hasattr(entry, 'metadata'):
            return entry.metadata
        elif isinstance(entry, dict):
            return entry.get('metadata', {})
        return {}
    
    st.subheader(f"📋 Dataset Preview ({mode.upper()} mode)")
    
    preview_count = min(max_entries, len(entries))
    
    for i in range(preview_count):
        entry = entries[i]
        entry_id = get_entry_id(entry)
        
        with st.expander(f"Entry {i+1}: {entry_id}", expanded=i == 0):
            if mode == 'rag' or (isinstance(entry, dict) and 'text' in entry):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("**Text Content:**")
                    text_content = get_entry_text(entry)
                    preview_text = text_content[:300] + "..." if len(text_content) > 300 else text_content
                    st.write(preview_text)
                
                with col2:
                    st.write("**Statistics:**")
                    st.metric("Characters", len(text_content))
                    st.metric("Words", len(text_content.split()))
                
                st.write("**Metadata:**")
                with st.expander("View metadata", expanded=False):
                    st.json(get_entry_metadata(entry))
            
            elif mode == 'finetune' or (isinstance(entry, dict) and 'messages' in entry):
                st.write("**Conversation:**")
                
                # Handle both object format and dictionary format
                if hasattr(entry, 'conversations'):
                    conversations = entry.conversations
                elif isinstance(entry, dict) and 'messages' in entry:
                    # Convert messages format to conversations format for display
                    conversations = []
                    for msg in entry['messages']:
                        conversations.append({
                            'role': msg.get('role', 'unknown'),
                            'content': msg.get('content', '')
                        })
                elif isinstance(entry, dict) and 'conversations' in entry:
                    conversations = entry['conversations']
                else:
                    conversations = []
                
                for j, turn in enumerate(conversations):
                    role = turn.get('role') if isinstance(turn, dict) else getattr(turn, 'role', 'unknown')
                    content = turn.get('content') if isinstance(turn, dict) else getattr(turn, 'content', '')
                    
                    role_color = {
                        'system': '#28a745',
                        'user': '#007bff', 
                        'assistant': '#6f42c1'
                    }.get(role, '#6c757d')
                    
                    if content:  # Only show non-empty content
                        st.markdown(f"""
                        <div style="
                            border-left: 3px solid {role_color};
                            padding-left: 1rem;
                            margin: 0.5rem 0;
                        ">
                            <strong style="color: {role_color};">{role.title()}:</strong><br>
                            {content}
                        </div>
                        """, unsafe_allow_html=True)

def render_configuration_summary(config: Dict[str, Any]):
    """Render a summary of the current configuration."""
    st.subheader("⚙️ Configuration Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Processing Mode:**")
        st.code(config.get('mode', 'Not set'))
        
        st.write("**Chunking Strategy:**")
        st.code(config.get('chunk_strategy', 'Not set'))
    
    with col2:
        st.write("**Output Format:**")
        st.code(config.get('output_format', 'Not set'))
        
        st.write("**Text Cleaning:**")
        st.code("Enabled" if config.get('clean_text', False) else "Disabled")
    
    with col3:
        # Strategy-specific parameters
        strategy = config.get('chunk_strategy')
        st.write("**Strategy Parameters:**")
        
        if strategy == 'fixed':
            params = f"Size: {config.get('chunk_size', 'N/A')}\nOverlap: {config.get('overlap', 'N/A')}"
        elif strategy == 'semantic':
            params = f"Min: {config.get('min_size', 'N/A')}\nMax: {config.get('max_size', 'N/A')}\nTarget: {config.get('target_size', 'N/A')}"
        else:
            params = "Default settings"
        
        st.code(params)

def render_help_section():
    """Render a help section with tips and guidance."""
    with st.expander("❓ Need Help?", expanded=False):
        st.markdown("""
        ### Quick Start Guide:
        
        1. **Upload Documents**: Use the file uploader to select your documents
        2. **Configure Settings**: Adjust processing options in the sidebar
        3. **Start Processing**: Click the "Start Processing" button
        4. **Download Results**: Get your dataset from the Results tab
        
        ### Chunking Strategies:
        - **Semantic**: Best for most documents, respects sentence boundaries
        - **Fixed**: Creates equal-sized chunks, good for consistency
        - **Hierarchical**: Preserves document structure (chapters, sections)
        
        ### Output Formats:
        - **JSONL**: Best for large datasets, streaming-friendly
        - **JSON**: Standard format, good for small to medium datasets
        - **CSV**: Spreadsheet-compatible, good for analysis
        - **Parquet**: Columnar format, efficient for big data
        
        ### Tips:
        - Enable text cleaning for better quality
        - Use RAG mode for knowledge retrieval systems
        - Use Fine-tune mode for training conversational models
        - Preview your results before downloading large datasets
        """)

def render_advanced_options():
    """Render advanced configuration options."""
    with st.expander("🔧 Advanced Options", expanded=False):
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Text Processing")
            
            normalize_unicode = st.checkbox(
                "Unicode Normalization",
                value=True,
                help="Normalize Unicode characters"
            )
            
            fix_encoding = st.checkbox(
                "Fix Encoding Issues", 
                value=True,
                help="Automatically fix common encoding problems"
            )
            
            standardize_quotes = st.checkbox(
                "Standardize Quotes",
                value=False,
                help="Convert smart quotes to standard quotes"
            )
        
        with col2:
            st.subheader("Export Options")
            
            include_embeddings = st.checkbox(
                "Include Embeddings Placeholder",
                value=False,
                help="Add empty embedding fields for future use"
            )
            
            flatten_metadata = st.checkbox(
                "Flatten Metadata (CSV only)",
                value=True,
                help="Flatten nested metadata for CSV export"
            )
            
            compression_level = st.selectbox(
                "Compression Level",
                ["None", "Fast", "Best"],
                help="Compression level for supported formats"
            )
        
        return {
            'normalize_unicode': normalize_unicode,
            'fix_encoding': fix_encoding,
            'standardize_quotes': standardize_quotes,
            'include_embeddings': include_embeddings,
            'flatten_metadata': flatten_metadata,
            'compression_level': compression_level
        }

def render_footer():
    """Render application footer."""
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>📄 DocToAI - Document to AI Dataset Converter</p>
        <p>Created by <a href="https://www.shameermohammed.com" target="_blank" style="color: #667eea; text-decoration: none;"><strong>Shameer Mohammed</strong></a> | Built with ❤️ using Streamlit</p>
        <p>
        <a href="https://github.com/askshameer/Doc2AI" target="_blank">GitHub</a> | 
        <a href="https://github.com/askshameer/Doc2AI/tree/main/docs" target="_blank">Documentation</a> |
        <a href="mailto:mohammed.shameer@gmail.com">Contact</a>
        </p>
    </div>
    """, unsafe_allow_html=True)