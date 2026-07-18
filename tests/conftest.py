import logging
import os
import shutil
import sys
import time
from unittest.mock import patch

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Disable MLflow tracing during tests to prevent warnings and background communication
import mlflow
import pytest

from tddrobo import config

mlflow.tracing.disable()
# Suppress informational messages about trace logging queues shutting down
logging.getLogger("mlflow.tracing.export.async_export_queue").setLevel(logging.WARNING)


@pytest.fixture(autouse=True)
def disable_mlflow_initialization():
    """Globally disable MLflow initialization in E2E/CLI tests to prevent database locks."""
    with patch("cli.initialize_mlflow"):
        yield


@pytest.fixture(autouse=True)
def mock_artifacts_dir(tmp_path):
    """
    Automatically redirect all artifact creations during testing
    to a temporary directory to avoid polluting the workspace.
    """
    test_artifacts_dir = str(tmp_path / "artifacts")
    test_session_id = f"test_session_{time.strftime('%Y%m%d_%H%M%S')}"
    test_session_dir = os.path.join(test_artifacts_dir, test_session_id)

    with (
        patch("tddrobo.config.TDD_BASE_ARTIFACTS_DIR", test_artifacts_dir),
        patch("tddrobo.config.SESSION_ID", test_session_id),
        patch("tddrobo.config.ARTIFACTS_DIR", test_session_dir),
    ):
        old_base = config.TDD_BASE_ARTIFACTS_DIR
        old_session = config.SESSION_ID
        old_artifacts = config.ARTIFACTS_DIR

        config.TDD_BASE_ARTIFACTS_DIR = test_artifacts_dir
        config.SESSION_ID = test_session_id
        config.ARTIFACTS_DIR = test_session_dir

        try:
            yield test_session_dir
        finally:
            config.TDD_BASE_ARTIFACTS_DIR = old_base
            config.SESSION_ID = old_session
            config.ARTIFACTS_DIR = old_artifacts
            if os.path.exists(test_artifacts_dir):
                try:
                    shutil.rmtree(test_artifacts_dir)
                except Exception:
                    pass


@pytest.fixture
def mock_workspace(tmp_path):
    """Provide a Workspace instance operating in an isolated temporary directory."""
    from tddrobo.utils import Workspace

    return Workspace(str(tmp_path))


@pytest.fixture
def mock_genai_client():
    """Provide a GenAIClient with the Google GenAI SDK completely mocked out."""
    from tddrobo.utils import GenAIClient

    with patch("tddrobo.utils.genai.Client") as MockClient:
        # Prevent actual API calls during tests
        client = GenAIClient(debug_mode=False)
        yield client, MockClient


@pytest.fixture
def mock_mlflow():
    """Mock MLflow tracking to prevent polluting the tracking server during tests."""
    with patch("tddrobo.utils.mlflow") as mock_mlf:
        yield mock_mlf
