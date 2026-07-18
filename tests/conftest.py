import logging
import os
import sys

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch

# Disable MLflow tracing during tests to prevent warnings and background communication
import mlflow
import pytest

mlflow.tracing.disable()
# Suppress informational messages about trace logging queues shutting down
logging.getLogger("mlflow.tracing.export.async_export_queue").setLevel(logging.WARNING)


@pytest.fixture
def mock_workspace(tmp_path):
    """Provide a Workspace instance operating in an isolated temporary directory."""
    from utils import Workspace

    return Workspace(str(tmp_path))


@pytest.fixture
def mock_genai_client():
    """Provide a GenAIClient with the Google GenAI SDK completely mocked out."""
    from utils import GenAIClient

    with patch("utils.genai.Client") as MockClient:
        # Prevent actual API calls during tests
        client = GenAIClient(debug_mode=False)
        yield client, MockClient


@pytest.fixture
def mock_mlflow():
    """Mock MLflow tracking to prevent polluting the tracking server during tests."""
    with patch("utils.mlflow") as mock_mlf:
        yield mock_mlf
