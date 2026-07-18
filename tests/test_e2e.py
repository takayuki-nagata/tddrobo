# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import os
from unittest.mock import MagicMock, patch

import pytest

from cli import main
from tddrobo.schema import (
    BugReport,
    DesignDocument,
    DesignReviewReport,
    FilePlan,
    RefactorDecision,
    RequirementsList,
    TestPlan,
    TestPlanReviewReport,
    TestReviewReport,
)


@pytest.fixture
def e2e_env(tmp_path):
    """
    Fixture to redirect all artifact creations to a temporary directory
    and mock MLflow to prevent tracking during E2E tests.
    """
    with (
        patch("tddrobo.config.ARTIFACTS_DIR", str(tmp_path)),
        patch("tddrobo.config.TDD_BASE_ARTIFACTS_DIR", str(tmp_path)),
        patch("mlflow.start_run"),
        patch("mlflow.log_param"),
        patch("mlflow.log_metric"),
        patch("mlflow.log_text"),
    ):
        yield str(tmp_path)


def test_e2e_happy_path(e2e_env):
    """
    Simulates a perfect scenario where the LLM correctly generates code
    and all tests pass on the first iteration.
    """

    def mock_llm_reasoning(prompt, response_schema=None, **kwargs):
        if response_schema == DesignDocument:
            return (
                "```json\n"
                '{"module_responsibilities": "calc", "architecture_and_components": "none", '
                '"interface_definitions": "def add(a, b):", "data_structures": "none", '
                '"logic_and_algorithms": "a+b", "edge_cases_and_limitations": "none", '
                '"error_handling": "none", "command_line_interface": "none"}\n'
                "```"
            )
        if response_schema == DesignReviewReport:
            return '```json\n{"estimated_quality": 100, "comments": "No gaps detected."}\n```'
        if response_schema == TestPlan:
            return '```json\n{"test_cases": [{"action": "add numbers", "expected_outcome": "returns sum"}]}\n```'
        if response_schema == TestPlanReviewReport:
            return '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
        if response_schema == TestReviewReport:
            return '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
        if response_schema == RefactorDecision:
            return '```json\n{"chain_of_thought": "good", "refactor_needed": false, "reasons": []}\n```'

        if "Test Generation & Review (Current Phase)" in prompt:
            return "```python\ndef test_add():\n    from app import add\n    assert add(1, 2) == 3\n```"
        return "```python\ndef add(a, b):\n    return a + b\n```"

    def mock_llm_standard(prompt, response_schema=None, **kwargs):
        if response_schema == FilePlan:
            return '```json\n{"impl_filename": "app.py", "test_filename": "test_app.py"}\n```'
        if response_schema == TestPlan:
            return '```json\n{"test_cases": [{"action": "add numbers", "expected_outcome": "returns sum"}]}\n```'
        if response_schema == RequirementsList:
            return '```json\n{"requirements": [{"id": "REQ001", "description": "Add functionality"}]}\n```'
        return "# Mocked README\nThis is a mocked project."

    with (
        patch("tddrobo.utils._default_llm.generate_with_reasoning", side_effect=mock_llm_reasoning),
        patch("tddrobo.utils._default_llm.generate_standard", side_effect=mock_llm_standard),
        patch("tddrobo.agent.requests.get") as mock_get,
        patch("tddrobo.agent.subprocess.run") as mock_run,
        patch("tddrobo.agent.subprocess.Popen") as mock_popen,
    ):
        # Mock requests.get to return a dummy specification
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Specification: Build a simple calculator."
        mock_response.headers = {"Content-Type": "text/plain"}
        mock_get.return_value = mock_response

        # Mock subprocess.run to simulate successful syntax check (flake8)
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        # Mock subprocess.Popen to simulate successful test execution (pytest)
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = ("Success", "")
        mock_popen.return_value = mock_process

        # Execute the main workflow
        args = ["--goal", "Build a calculator", "--spec-url", "http://example.com/spec"]
        final_state = main(args)

        # Verify that the state reached END and succeeded
        assert final_state is not None
        assert final_state.get("success") is True
        assert final_state.get("iterations", 0) == 0

        # Verify that expected artifacts were correctly saved to the session directory
        from tddrobo import config

        session_dir = config.ARTIFACTS_DIR
        assert os.path.exists(os.path.join(session_dir, "app.py"))
        assert os.path.exists(os.path.join(session_dir, "test_app_req001_unit.py"))
        assert os.path.exists(os.path.join(session_dir, "test_app_req001_integration.py"))
        assert os.path.exists(os.path.join(session_dir, "README.md"))
        assert os.path.exists(os.path.join(session_dir, "specification.txt"))


def test_e2e_recovery_path(e2e_env):
    """
    Simulates a scenario where the first test execution fails, triggering a BugReport,
    and then the agent successfully fixes the implementation on the second attempt.
    """

    def mock_llm_reasoning(prompt, response_schema=None, **kwargs):
        if response_schema == DesignDocument:
            return (
                "```json\n"
                '{"module_responsibilities": "calc", "architecture_and_components": "none", '
                '"interface_definitions": "def add(a, b):", "data_structures": "none", '
                '"logic_and_algorithms": "a+b", "edge_cases_and_limitations": "none", '
                '"error_handling": "none", "command_line_interface": "none"}\n'
                "```"
            )
        if response_schema == DesignReviewReport:
            return '```json\n{"estimated_quality": 100, "comments": "No gaps detected."}\n```'
        if response_schema == TestPlan:
            return '```json\n{"test_cases": [{"action": "add numbers", "expected_outcome": "returns sum"}]}\n```'
        if response_schema == TestPlanReviewReport:
            return '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
        if response_schema == TestReviewReport:
            return '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
        if response_schema == BugReport:
            return (
                "```json\n"
                '{"failed_test_cases": ["test_add"], "expected_vs_actual": "Expected 3, got 0", '
                '"fix_instructions": "Implement addition properly", "target_to_fix": "implement_logic"}\n'
                "```"
            )
        if response_schema == RefactorDecision:
            return '```json\n{"chain_of_thought": "good", "refactor_needed": false, "reasons": []}\n```'

        if "Test Generation & Review (Current Phase)" in prompt:
            return "```python\ndef test_add():\n    from app import add\n    assert add(1, 2) == 3\n```"

        # Implementation logic: return broken code first, then correct code after reading the Bug Report
        if "Bug Report" in prompt:
            return "```python\ndef add(a, b):\n    return a + b\n```"
        return "```python\ndef add(a, b):\n    return 0\n```"

    def mock_llm_standard(prompt, response_schema=None, **kwargs):
        if response_schema == FilePlan:
            return '```json\n{"impl_filename": "app.py", "test_filename": "test_app.py"}\n```'
        if response_schema == TestPlan:
            return '```json\n{"test_cases": [{"action": "add numbers", "expected_outcome": "returns sum"}]}\n```'
        if response_schema == RequirementsList:
            return '```json\n{"requirements": [{"id": "REQ001", "description": "Add functionality"}]}\n```'
        return "# Mocked README\nThis is a mocked project."

    with (
        patch("tddrobo.utils._default_llm.generate_with_reasoning", side_effect=mock_llm_reasoning),
        patch("tddrobo.utils._default_llm.generate_standard", side_effect=mock_llm_standard),
        patch("tddrobo.agent.requests.get") as mock_get,
        patch("tddrobo.agent.subprocess.run") as mock_run,
        patch("tddrobo.agent.subprocess.Popen") as mock_popen,
    ):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = "Specification: Build a simple calculator."
        mock_get.return_value = mock_response

        # Mock subprocess.run to simulate successful syntax check (flake8)
        mock_run.return_value = MagicMock(returncode=0, stdout="Success", stderr="")

        pytest_called = False

        def popen_side_effect(args, **kwargs):
            nonlocal pytest_called
            mock_proc = MagicMock()
            if "pytest" in args:
                if not pytest_called:
                    pytest_called = True
                    mock_proc.returncode = 1
                    mock_proc.communicate.return_value = ("Test Failed", "")
                else:
                    mock_proc.returncode = 0
                    mock_proc.communicate.return_value = ("Test Passed", "")
            else:
                mock_proc.returncode = 0
                mock_proc.communicate.return_value = ("Success", "")
            return mock_proc

        mock_popen.side_effect = popen_side_effect
        final_state = main(["--goal", "Build a calculator", "--spec-url", "http://example.com/spec"])
        assert final_state.get("iterations", 0) == 0


def test_cli_draw_graph(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_graph = MagicMock()
    mock_graph.draw_mermaid.return_value = "mermaid definition"
    mock_graph.draw_mermaid_png.return_value = b"dummy bytes"
    with patch("cli.TDDAgent") as mock_agent_class:
        mock_agent_class.return_value.get_graph.return_value = mock_graph
        res = main(["--draw-graph"])
        assert res is None

        mmd_file = tmp_path / "docs" / "workflow_graph.mmd"
        assert mmd_file.exists()
        assert mmd_file.read_text(encoding="utf-8") == "mermaid definition"

        graph_file = tmp_path / "docs" / "workflow_graph.png"
        assert graph_file.exists()
        assert graph_file.read_bytes() == b"dummy bytes"


def test_cli_draw_graph_failure(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    mock_graph = MagicMock()
    mock_graph.draw_mermaid.return_value = "mermaid definition"
    mock_graph.draw_mermaid_png.side_effect = Exception("draw error")
    with (
        patch("cli.TDDAgent") as mock_agent_class,
        pytest.raises(SystemExit) as exc_info,
    ):
        mock_agent_class.return_value.get_graph.return_value = mock_graph
        main(["--draw-graph"])
    assert exc_info.value.code == 1
