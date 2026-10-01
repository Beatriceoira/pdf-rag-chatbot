"""Pytest configuration."""

pytest_plugins = []

# Add project root to Python path
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
