# config.py
import os
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent

# Relative paths to the Python interpreters in each virtual environment
OCR_PYTHON = str(BASE_DIR / "env" / "OCRenv" / "Scripts" / "python.exe")
HV_PYTHON = str(BASE_DIR / "env" / "hvenv" / "Scripts" / "python.exe")
