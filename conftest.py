"""
Pytest configuration for the DeepFake Detection FYP.

Adds src/ to sys.path so that 'from deepfake import ...' works during tests
without requiring an installed package.
"""
import sys
from pathlib import Path

# Allow importing from src/ without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))
