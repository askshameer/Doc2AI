#!/usr/bin/env python3
"""
DocToAI Web UI Launcher
Simple launcher for the Streamlit web interface.
"""

import subprocess
import sys
from pathlib import Path
import webbrowser
import time
import threading

def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import streamlit
        import plotly
        print("✅ Dependencies verified")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def launch_ui(port=8501, auto_open=True):
    """Launch the Streamlit UI."""
    # Try enhanced app first, fallback to simple app
    enhanced_ui_path = Path(__file__).parent / "ui" / "enhanced_app.py"
    simple_ui_path = Path(__file__).parent / "ui" / "simple_app.py"
    
    # Choose the UI based on available dependencies
    ui_path = enhanced_ui_path if enhanced_ui_path.exists() else simple_ui_path
    
    if not ui_path.exists():
        print(f"❌ UI file not found: {ui_path}")
        return
    
    print("🚀 Starting DocToAI Web UI...")
    print(f"📍 UI will be available at: http://localhost:{port}")
    
    # Auto-open browser after a delay
    if auto_open:
        def open_browser():
            time.sleep(2)  # Wait for server to start
            try:
                webbrowser.open(f"http://localhost:{port}")
                print("🌐 Opened browser automatically")
            except Exception as e:
                print(f"⚠️ Could not open browser automatically: {e}")
        
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(ui_path), 
            "--server.port", str(port),
            "--server.headless", "false" if auto_open else "true",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 DocToAI UI stopped")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")

def main():
    """Main launcher function."""
    print("📄 DocToAI - Document to AI Dataset Converter")
    print("=" * 50)
    
    if not check_dependencies():
        return
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Launch DocToAI Web UI")
    parser.add_argument("--port", type=int, default=8501, help="Port to run the UI on")
    parser.add_argument("--no-browser", action="store_true", help="Don't auto-open browser")
    
    args = parser.parse_args()
    
    launch_ui(port=args.port, auto_open=not args.no_browser)

if __name__ == "__main__":
    main()