#!/usr/bin/env python3
"""
Safe DocToAI UI Launcher
Handles import issues gracefully and provides helpful guidance.
"""

import subprocess
import sys
from pathlib import Path
import os

def check_python_version():
    """Check Python version compatibility."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def check_streamlit():
    """Check if Streamlit is available."""
    try:
        import streamlit
        print(f"✅ Streamlit available: {streamlit.__version__}")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        print("Install with: pip install streamlit")
        return False

def check_core_imports():
    """Check if core DocToAI modules can be imported."""
    try:
        # Test core imports without external dependencies
        from core.data_models import Document
        from core.base_extractor import ExtractorPlugin
        from utils.metadata_manager import MetadataManager
        print("✅ Core DocToAI modules available")
        return True
    except ImportError as e:
        print(f"❌ Core import issue: {e}")
        print("Make sure you're running from the DocToAI root directory")
        return False

def choose_ui_version():
    """Choose the appropriate UI version based on available dependencies."""
    ui_dir = Path(__file__).parent / "ui"
    
    # Check available UI files
    enhanced_ui = ui_dir / "enhanced_app.py"
    basic_ui = ui_dir / "app.py"
    simple_ui = ui_dir / "simple_app.py"
    
    # Try to determine which dependencies are available
    has_full_deps = False
    has_basic_deps = False
    
    try:
        import pdfplumber
        import pandas
        import plotly
        has_full_deps = True
    except ImportError:
        pass
    
    try:
        import streamlit
        has_basic_deps = True
    except ImportError:
        pass
    
    if has_full_deps and enhanced_ui.exists():
        print("🎨 Using enhanced UI with full features")
        return enhanced_ui
    elif has_basic_deps and basic_ui.exists():
        print("📄 Using basic UI")
        return basic_ui
    elif simple_ui.exists():
        print("🔧 Using simple UI with dependency checking")
        return simple_ui
    else:
        print("❌ No UI files found")
        return None

def launch_streamlit(ui_file, port=8501):
    """Launch Streamlit with the specified UI file."""
    if not ui_file or not ui_file.exists():
        print(f"❌ UI file not found: {ui_file}")
        return False
    
    print(f"🚀 Starting DocToAI UI: {ui_file.name}")
    print(f"🌐 Will be available at: http://localhost:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        # Set PYTHONPATH to include current directory
        env = os.environ.copy()
        env['PYTHONPATH'] = str(Path(__file__).parent) + os.pathsep + env.get('PYTHONPATH', '')
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(ui_file),
            "--server.port", str(port),
            "--server.headless", "false",
            "--browser.gatherUsageStats", "false"
        ], env=env)
        return True
    except KeyboardInterrupt:
        print("\n👋 DocToAI UI stopped")
        return True
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        return False

def provide_setup_help():
    """Provide setup instructions."""
    print("\n📋 Setup Instructions:")
    print("=" * 40)
    
    print("\n1. Install basic dependencies:")
    print("   pip install streamlit plotly pandas")
    
    print("\n2. Install full DocToAI dependencies:")
    print("   pip install -r requirements.txt")
    
    print("\n3. Launch the UI:")
    print("   python launch_ui_safe.py")
    
    print("\n4. If you get import errors:")
    print("   - Make sure you're in the DocToAI root directory")
    print("   - Check that all files are present")
    print("   - Try: python -c 'from core.data_models import Document'")
    
    print("\n💡 For minimal setup, just install streamlit to see the dependency checker UI")

def main():
    """Main launcher function."""
    print("🚀 DocToAI Safe UI Launcher")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Python version too old. Please upgrade to Python 3.8+")
        return
    
    # Check if we're in the right directory
    if not (Path.cwd() / "core" / "data_models.py").exists():
        print("❌ Not in DocToAI directory or files missing")
        print("Please run this from the DocToAI root directory")
        return
    
    # Check Streamlit
    if not check_streamlit():
        print("\n📦 Installing Streamlit...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"], check=True)
            print("✅ Streamlit installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install Streamlit")
            provide_setup_help()
            return
    
    # Check core imports
    if not check_core_imports():
        print("\n❌ Core imports failed")
        provide_setup_help()
        return
    
    # Choose UI version
    ui_file = choose_ui_version()
    if not ui_file:
        print("\n❌ No suitable UI found")
        return
    
    # Launch UI
    success = launch_streamlit(ui_file)
    if not success:
        print("\n❌ Failed to launch UI")
        provide_setup_help()

if __name__ == "__main__":
    main()