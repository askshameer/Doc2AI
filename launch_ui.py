#!/usr/bin/env python3
"""DocToAI UI Launcher"""
import subprocess
import sys
from pathlib import Path

ui_path = Path(__file__).parent / "ui" / "enhanced_app.py"
subprocess.run([sys.executable, "-m", "streamlit", "run", str(ui_path)])
