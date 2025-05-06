"""
Common test fixtures and utilities for the CasaLingua test suite.
"""

import os
import sys
import pytest
from pathlib import Path

# Add project root to Python path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

@pytest.fixture
def project_root():
    """Return the project root directory as a Path object."""
    return ROOT_DIR

@pytest.fixture
def app_dir():
    """Return the app directory as a Path object."""
    return ROOT_DIR / "app"

@pytest.fixture
def data_dir():
    """Return the data directory as a Path object."""
    return ROOT_DIR / "data"

@pytest.fixture
def config_dir():
    """Return the config directory as a Path object."""
    return ROOT_DIR / "config"

@pytest.fixture
def models_dir():
    """Return the models directory as a Path object."""
    return ROOT_DIR / "models"

# Add more fixtures as needed