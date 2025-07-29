#!/usr/bin/env python3
"""
DocToAI UI Setup Script
Checks dependencies and provides setup instructions for the web UI.
"""

import sys
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version.split()[0]}")
    return True

def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing requirements...")
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ], check=True)
        print("✅ Requirements installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_streamlit():
    """Check if Streamlit is properly installed."""
    try:
        import streamlit
        print(f"✅ Streamlit version: {streamlit.__version__}")
        return True
    except ImportError:
        print("❌ Streamlit not found")
        return False

def check_doctoai_components():
    """Check if DocToAI components are available."""
    try:
        # Check core components
        from core.document_loader import DocumentLoader
        from core.text_processor import TextProcessor
        from cli import DocToAI
        print("✅ DocToAI components available")
        return True
    except ImportError as e:
        print(f"❌ DocToAI components not found: {e}")
        return False

def create_launcher_script():
    """Create a simple launcher script."""
    launcher_content = '''#!/usr/bin/env python3
"""DocToAI UI Launcher"""
import subprocess
import sys
from pathlib import Path

ui_path = Path(__file__).parent / "ui" / "enhanced_app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])
'''
    
    launcher_path = Path(__file__).parent / "launch_ui.py"
    with open(launcher_path, 'w') as f:
        f.write(launcher_content)
    
    # Make executable on Unix-like systems
    try:
        launcher_path.chmod(0o755)
    except:
        pass
    
    print(f"✅ Created launcher script: {launcher_path}")

def main():
    """Main setup function."""
    print("🚀 DocToAI UI Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Check Streamlit
    if not check_streamlit():
        return False
    
    # Check DocToAI components
    if not check_doctoai_components():
        print("\n💡 Tip: Make sure you're running this from the DocToAI root directory")
        return False
    
    # Create launcher script
    create_launcher_script()
    
    print("\n🎉 Setup completed successfully!")
    print("\n🚀 How to start the UI:")
    print("1. Run: python run_ui.py")
    print("2. Or: python launch_ui.py") 
    print("3. Or: streamlit run ui/enhanced_app.py")
    print("\n🌐 The UI will open at: http://localhost:8501")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)