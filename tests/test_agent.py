from unittest.mock import MagicMock, mock_open, patch

from langgraph.graph import END

from agent import (
    check_impl_syntax,
    check_tests_syntax,
    fetch_spec,
    generate_bug_report,
    generate_design,
    generate_readme,
    generate_requirements,
    generate_tests,
    implement_logic,
    increment_requirement,
    plan_files,
    plan_tests,
    review_test_plan,
    review_tests,
    run_tests,
    should_continue,
    should_fix_tests_or_impl,
    should_implement_logic,
    should_review_test_plan_or_continue,
    should_review_tests_or_continue,
    should_run_tests,
)
from schema import TDDState


@patch("agent.requests.get")
@patch("agent.save_artifact")
def test_fetch_spec_success(mock_save_artifact, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "This is a spec"
    mock_response.headers = {"Content-Type": "text/plain"}
    mock_get.return_value = mock_response
    mock_save_artifact.return_value = "artifacts/specification.txt"

    state = TDDState(spec_url="http://example.com/spec")
    result = fetch_spec(state)

    assert result["spec_content"] == "This is a spec"
    mock_save_artifact.assert_called_once_with("specification.txt", "This is a spec")


@patch("agent.requests.get")
@patch("agent.read_artifact")
def test_fetch_spec_not_modified(mock_read_artifact, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 304
    mock_get.return_value = mock_response
    mock_read_artifact.return_value = "Cached spec"

    state = TDDState(spec_url="http://example.com/spec")

    with patch("os.path.exists", return_value=True), patch("os.path.getmtime", return_value=1000):
        result = fetch_spec(state)

    assert result["spec_content"] == "Cached spec"


@patch("agent.requests.get")
@patch("agent.save_artifact")
def test_fetch_spec_html(mock_save_artifact, mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<h1>Title</h1>"
    mock_response.headers = {"Content-Type": "text/html"}
    mock_get.return_value = mock_response

    state = TDDState(spec_url="http://example.com/spec")
    result = fetch_spec(state)
    assert "# Title" in result["spec_content"]


@patch("agent.requests.get")
def test_fetch_spec_request_exception(mock_get):
    import requests

    mock_get.side_effect = requests.exceptions.RequestException("Network Error")
    state = TDDState(spec_url="http://example.com/spec")
    result = fetch_spec(state)
    assert "Error fetching specification" in result["spec_content"]
    assert "Network Error" in result["spec_content"]


def test_fetch_spec_local_text_file(tmp_path):
    spec_file = tmp_path / "spec.txt"
    spec_file.write_text("Local text spec", encoding="utf-8")

    state = TDDState(spec_url=str(spec_file))
    with patch("agent.save_artifact") as mock_save_artifact:
        result = fetch_spec(state)
        assert result["spec_content"] == "Local text spec"
        mock_save_artifact.assert_called_once_with("specification.txt", "Local text spec")


def test_fetch_spec_local_html_file(tmp_path):
    spec_file = tmp_path / "spec.html"
    spec_file.write_text("<h1>Local HTML spec</h1>", encoding="utf-8")

    state = TDDState(spec_url=str(spec_file))
    with patch("agent.save_artifact") as mock_save_artifact:
        result = fetch_spec(state)
        assert "# Local HTML spec" in result["spec_content"]
        mock_save_artifact.assert_called_once()


def test_fetch_spec_local_file_not_found():
    state = TDDState(spec_url="non_existent_file.txt")
    result = fetch_spec(state)
    assert "Error: Local specification file not found" in result["spec_content"]


def test_fetch_spec_local_file_read_error(tmp_path):
    spec_file = tmp_path / "error_spec.txt"
    spec_file.write_text("Error spec", encoding="utf-8")

    state = TDDState(spec_url=str(spec_file))
    with patch("builtins.open", side_effect=PermissionError("Permission Denied")):
        result = fetch_spec(state)
        assert "Error reading local specification" in result["spec_content"]
        assert "Permission Denied" in result["spec_content"]


def test_save_history_snapshot(tmp_path):
    from agent import save_history_snapshot

    # Patch config.ARTIFACTS_DIR to test temporary path
    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        save_history_snapshot("test.py", "print('hello')", 3)

        expected_path = tmp_path / "history" / "test_iter003.py"
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == "print('hello')"

        # Test exception safety (simulate write error by mocking open)
        with patch("builtins.open", side_effect=PermissionError("Mocked Permission Denied")):
            # Should not raise exception, just print warning
            save_history_snapshot("test.py", "print('hello')", 3)


@patch("agent.call_llm_standard")
def test_plan_files(mock_call_llm_standard):
    mock_call_llm_standard.return_value = '```json\n{"impl_filename": "my_impl.py", "test_filename": "my_test.py"}\n```'
    state = TDDState(goal="Make a calculator")

    result = plan_files(state)

    assert result["module_name"] == "my_impl.py"
    assert result["test_module_name"] == "my_test.py"


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
def test_generate_design(mock_save_artifact, mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"module_responsibilities": "Handles math", "error_handling": "Raises ValueError"}\n```'
    )
    mock_save_artifact.return_value = "artifacts/design.md"

    state = TDDState(goal="Calc", spec_content="Spec", module_name="impl.py", test_module_name="test.py")
    result = generate_design(state)

    assert "Handles math" in result["design_doc"]
    assert "Raises ValueError" in result["design_doc"]


@patch("agent._run_syntax_check")
def test_check_tests_syntax_empty_code(mock_syntax_check):
    # Check if an error is returned immediately without running syntax check when test code is empty
    state = TDDState(tests_code="", test_module_name="test_impl.py")
    result = check_tests_syntax(state)

    assert "Error: The generated test code is empty" in result["tests_check_output"]
    mock_syntax_check.assert_not_called()


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
def test_plan_tests(mock_save_artifact, mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"test_cases": [{"action": "add", "expected_outcome": "sum"}]}\n```'
    )
    mock_save_artifact.return_value = "artifacts/test_plan.md"

    state = TDDState(goal="goal", spec_content="spec", design_doc="design", test_plan_iterations=0)
    result = plan_tests(state)

    assert "Action: add | Expected: sum" in result["test_plan"]
    assert result["test_plan_iterations"] == 1


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
def test_plan_tests_with_review_and_error(mock_save_artifact, mock_call_llm_with_reasoning):
    # test fix prompt branch
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"test_cases": [{"action": "fix", "expected_outcome": "ok"}]}\n```'
    )
    state = TDDState(test_plan="old", test_plan_review="feedback")
    result = plan_tests(state)
    assert "Action: fix" in result["test_plan"]


@patch("agent.call_llm_with_reasoning")
def test_review_test_plan(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "good"}\n```'
    )
    state = TDDState(target_test_plan_coverage=95)

    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "generate_tests"
    assert result["test_plan_review"] == ""


@patch("agent.call_llm_with_reasoning")
def test_review_test_plan_insufficient_max_iters(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": ["B"], "estimated_coverage": 50, "feedback": "Add B"}\n```'
    )
    state = TDDState(target_test_plan_coverage=95, test_plan_iterations=3, max_test_plan_iterations=3)

    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "generate_tests"
    assert result["test_plan_review"] == ""


@patch("agent.call_llm_with_reasoning")
def test_review_test_plan_insufficient_retry(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": ["B"], "estimated_coverage": 50, "feedback": "Add B"}\n```'
    )
    state = TDDState(target_test_plan_coverage=95, test_plan_iterations=1, max_test_plan_iterations=3)
    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "plan_tests"
    assert "Add B" in result["test_plan_review"]


@patch("agent.call_llm_with_reasoning")
@patch("agent.call_llm_standard")
@patch("agent.evaluate_math_expression")
@patch("agent.save_artifact")
@patch("agent.save_history_snapshot")
def test_generate_tests(
    mock_save_history_snapshot,
    mock_save_artifact,
    mock_evaluate_math_expression,
    mock_call_llm_standard,
    mock_call_llm_with_reasoning,
):
    mock_call_llm_with_reasoning.return_value = "```python\ndef test_a(): pass\n```"
    mock_call_llm_standard.return_value = (
        '{"items": [{"test_case_number": 2, "expression": "1+1"}, '
        '{"test_case_number": "invalid", "expression": "1+1"}]}'
    )
    mock_evaluate_math_expression.return_value = "2"
    state = TDDState(test_iterations=0, test_plan="2. Action: 1+1 | Expected: Output 0")
    result = generate_tests(state)

    assert result["tests_code"] == "def test_a(): pass"
    assert result["test_iterations"] == 1
    mock_evaluate_math_expression.assert_called_once_with("1+1")
    mock_save_history_snapshot.assert_called_once()


@patch("agent.call_llm_with_reasoning")
@patch("agent.call_llm_standard")
@patch("agent.save_artifact")
@patch("agent.save_history_snapshot")
def test_generate_tests_fix_branches(
    mock_save_history_snapshot, mock_save_artifact, mock_call_llm_standard, mock_call_llm_with_reasoning
):
    mock_call_llm_with_reasoning.return_value = "```python\nfixed\n```"
    mock_call_llm_standard.return_value = '{"items": []}'
    state = TDDState(tests_check_output="error", tests_code="error code")
    result = generate_tests(state)
    assert result["tests_code"] == "fixed"
    mock_save_history_snapshot.assert_called_once()


def test_generate_tests_all_branches():
    from agent import generate_tests

    with (
        patch("agent.call_llm_with_reasoning", return_value="```python\nfixed\n```"),
        patch("agent.call_llm_standard", return_value='{"items": []}'),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
    ):
        # tests_check_output
        generate_tests(TDDState(tests_check_output="error", tests_code="old"))
        # bug_report
        generate_tests(TDDState(bug_report="error", tests_code="old"))
        # test_review
        generate_tests(TDDState(test_review="error", tests_code="old"))
        # initial
        generate_tests(TDDState())


def test_generate_tests_coverage_edges():
    from agent import generate_tests

    # 1. test_module_name without a dot (no extension)
    with (
        patch("agent.call_llm_with_reasoning", return_value="```python\nfixed\n```"),
        patch("agent.call_llm_standard", return_value='{"items": []}'),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
    ):
        generate_tests(TDDState(test_module_name="test_no_dot", test_iterations=0))

    # 2. test_iterations > 0 and open() raises Exception
    with (
        patch("agent.call_llm_with_reasoning", return_value="```python\nfixed\n```"),
        patch("agent.call_llm_standard", return_value='{"items": []}'),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", side_effect=Exception("Failed to read")),
    ):
        generate_tests(TDDState(test_module_name="test_impl.py", test_iterations=2))

    # 3. test_iterations > 0 and open() successfully reads file
    mock_file = mock_open(read_data="existing test code")
    with (
        patch("agent.call_llm_with_reasoning", return_value="```python\nfixed\n```"),
        patch("agent.call_llm_standard", return_value='{"items": []}'),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_file),
    ):
        generate_tests(TDDState(test_module_name="test_impl.py", test_iterations=2))


@patch("agent.call_llm_with_reasoning")
def test_review_tests(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": ["A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
    )
    state = TDDState(target_test_coverage=90, test_iterations=1, max_test_iterations=3)

    result = review_tests(state)
    assert result["test_review_decision"] == "generate_tests"
    assert "Add A" in result["test_review"]


@patch("agent.call_llm_with_reasoning")
def test_review_tests_sufficient(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
    )
    state = TDDState(target_test_coverage=90)
    result = review_tests(state)
    assert result["test_review_decision"] == "implement_logic"
    assert result["test_review"] == ""


@patch("agent.call_llm_with_reasoning")
def test_review_tests_insufficient_proceed(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": ["A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
    )
    state = TDDState(target_test_coverage=90, test_iterations=3, max_test_iterations=3)
    result = review_tests(state)
    assert result["test_review_decision"] == "implement_logic"


@patch("agent.call_llm_with_reasoning")
def test_review_tests_insufficient_coverage_retry(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": ["A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
    )
    state = TDDState(target_test_coverage=90, test_iterations=1, max_test_iterations=3)
    result = review_tests(state)
    assert result["test_review_decision"] == "generate_tests"


@patch("agent.call_llm_with_reasoning")
def test_review_tests_with_regression_files(mock_call_llm_with_reasoning, tmp_path):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
    )

    # Create regression files
    reg1_file = tmp_path / "test_impl_req001.py"
    reg1_file.write_text("def test_one(): assert True", encoding="utf-8")

    # A directory named `test_impl_req002.py` will cause open() to raise
    # IsADirectoryError, triggering the exception path
    reg2_dir = tmp_path / "test_impl_req002.py"
    reg2_dir.mkdir()

    state = TDDState(
        target_test_coverage=90,
        test_module_name="test_impl.py",
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}],
        current_req_index=2,
    )

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        result = review_tests(state)

        # Verify call_llm_with_reasoning was called with expected prompt contents
        args, kwargs = mock_call_llm_with_reasoning.call_args
        prompt_called = args[0]
        assert "def test_one(): assert True" in prompt_called
        assert "test_impl_req001.py" in prompt_called
        assert result["test_review_decision"] == "implement_logic"


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
@patch("agent.save_history_snapshot")
def test_implement_logic(mock_save_history_snapshot, mock_save_artifact, mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = "```python\ndef run(): return True\n```"
    state = TDDState(goal="g", design_doc="d", tests_code="t")
    result = implement_logic(state)

    assert result["impl_code"] == "def run(): return True"
    mock_save_history_snapshot.assert_called_once()


def test_implement_logic_all_branches():
    from agent import implement_logic

    with (
        patch("agent.call_llm_with_reasoning", return_value="```python\nfixed\n```"),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
    ):
        # impl_check_output
        implement_logic(TDDState(impl_check_output="error", impl_code="old"))
        # bug_report
        implement_logic(TDDState(bug_report="error", impl_code="old"))
        # initial
        implement_logic(TDDState())


def test_implement_logic_bypass():
    from unittest.mock import mock_open

    from agent import implement_logic

    with (
        patch("os.path.exists", return_value=True),
        patch("subprocess.run") as mock_run,
        patch("builtins.open", mock_open(read_data="bypass_code")),
        patch("agent.save_history_snapshot") as mock_save,
    ):
        mock_run.return_value.returncode = 0
        state = TDDState(module_name="bc_clone.py", test_module_name="test_bc_clone.py")
        result = implement_logic(state)
        assert result["impl_code"] == "bypass_code"
        mock_save.assert_called_once()


def test_implement_logic_bypass_fallback():
    from agent import implement_logic

    # 1. subprocess returns non-zero (tests fail)
    with (
        patch("os.path.exists", return_value=True),
        patch("subprocess.run") as mock_run,
        patch("agent.call_llm_with_reasoning", return_value="```python\nfallback_code\n```"),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
    ):
        mock_run.return_value.returncode = 1
        state = TDDState(module_name="bc_clone.py", test_module_name="test_bc_clone.py")
        result = implement_logic(state)
        assert "fallback_code" in result["impl_code"]

    # 2. subprocess raises exception
    with (
        patch("os.path.exists", return_value=True),
        patch("subprocess.run", side_effect=Exception("error")),
        patch("agent.call_llm_with_reasoning", return_value="```python\nfallback_code\n```"),
        patch("agent.save_artifact"),
        patch("agent.save_history_snapshot"),
    ):
        state = TDDState(module_name="bc_clone.py", test_module_name="test_bc_clone.py")
        result = implement_logic(state)
        assert "fallback_code" in result["impl_code"]


@patch("agent._run_syntax_check")
def test_check_impl_syntax(mock_syntax_check):
    mock_syntax_check.return_value = ""
    state = TDDState(impl_code="print('ok')", module_name="impl.py")
    result = check_impl_syntax(state)

    assert result["impl_check_output"] == ""
    mock_syntax_check.assert_called_once()


def test_check_impl_syntax_empty_code():
    from agent import check_impl_syntax

    state = TDDState(impl_code="", module_name="impl.py")
    result = check_impl_syntax(state)
    assert "Error: The generated implementation code is empty" in result["impl_check_output"]


@patch("agent.subprocess.run")
def test_run_tests_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="All tests passed", stderr="")
    state = TDDState(test_module_name="test_impl.py", iterations=1)
    result = run_tests(state)

    assert result["success"] is True
    assert "All tests passed" in result["test_output"]
    assert result["iterations"] == 2


@patch("agent.subprocess.run")
def test_run_tests_regression_failure(mock_run):
    # First call (active tests) returns success, second call (regression tests) returns failure
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="Active tests passed", stderr=""),
        MagicMock(returncode=1, stdout="Regression test failed", stderr="Regression stderr"),
    ]
    state = TDDState(test_module_name="test_impl.py", iterations=1)
    result = run_tests(state)
    assert result["success"] is False
    assert "REGRESSION ERROR" in result["test_output"]
    assert "Regression test failed" in result["test_output"]


@patch("agent.subprocess.run")
def test_run_tests_timeout(mock_run):
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=30, output="part_out", stderr="part_err")
    state = TDDState(test_module_name="test_impl.py", iterations=1)
    result = run_tests(state)
    assert result["success"] is False
    assert "timed out after 30" in result["test_output"]


@patch("agent.subprocess.run")
def test_run_tests_truncated(mock_run):
    large_stdout = "A" * 10000
    mock_run.return_value = MagicMock(returncode=1, stdout=large_stdout, stderr="")
    state = TDDState(test_module_name="test_impl.py", iterations=1)
    result = run_tests(state)

    assert result["success"] is False
    assert "TRUNCATED" in result["test_output"]
    assert len(result["test_output"]) < 10000


@patch("agent.subprocess.run")
def test_run_tests_with_maxfail(mock_run):
    mock_run.side_effect = [
        MagicMock(returncode=0, stdout="Active tests passed", stderr=""),
        MagicMock(returncode=0, stdout="Regression tests passed", stderr=""),
    ]
    state = TDDState(test_module_name="test_impl.py", iterations=1)
    with patch("config.PYTEST_MAXFAIL", 3):
        result = run_tests(state)
        assert result["success"] is True
        assert mock_run.call_count == 2
        args1 = mock_run.call_args_list[0][0][0]
        args2 = mock_run.call_args_list[1][0][0]
        assert "--maxfail=3" in args1
        assert "--maxfail=3" in args2


@patch("agent.call_llm_with_reasoning")
def test_generate_bug_report(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = (
        '```json\n{"failed_test_cases": ["t1"], "expected_vs_actual": "diff", '
        '"fix_instructions": "fix", "target_to_fix": "implement_logic"}\n```'
    )
    state = TDDState(iterations=1)
    result = generate_bug_report(state)

    assert "diff" in result["bug_report"]
    assert result["next_action"] == "implement_logic"


@patch("agent.call_llm_standard")
@patch("agent.save_artifact")
def test_generate_readme(mock_save_artifact, mock_call_llm_standard):
    mock_call_llm_standard.return_value = "# README"
    state = TDDState()
    result = generate_readme(state)

    assert result["readme_content"] == "# README"


# --- Conditional Edges Tests ---
def test_conditional_edges():
    # review tests
    assert should_review_tests_or_continue(TDDState(tests_check_output="Error")) == "generate_tests"
    assert should_review_tests_or_continue(TDDState(tests_check_output="")) == "review_tests"

    # implement logic
    assert should_implement_logic(TDDState(test_review_decision="generate_tests")) == "generate_tests"
    assert should_implement_logic(TDDState(impl_code="x")) == "run_tests"
    assert should_implement_logic(TDDState()) == "implement_logic"

    # run tests
    assert should_run_tests(TDDState(impl_check_output="Error")) == "implement_logic"
    assert should_run_tests(TDDState(impl_check_output="")) == "run_tests"

    # continue
    assert should_continue(TDDState(success=True)) == "generate_readme"
    assert should_continue(TDDState(success=False, iterations=3, max_iterations=3)) == END
    assert should_continue(TDDState(success=False, iterations=2, max_iterations=3)) == "generate_bug_report"

    # fix tests or impl
    assert should_fix_tests_or_impl(TDDState(next_action="generate_tests")) == "generate_tests"


def test_tdd_agent_initialization():
    from agent import TDDAgent

    agent = TDDAgent()
    assert agent.app is not None
    assert callable(agent.invoke)


@patch("agent._run_syntax_check")
def test_check_tests_syntax_with_code(mock_syntax_check):
    # Check if syntax checker is called when test code exists
    mock_syntax_check.return_value = ""  # No syntax errors
    state = TDDState(tests_code="def test_a(): pass", test_module_name="test_impl.py")

    result = check_tests_syntax(state)
    assert result["tests_check_output"] == ""
    mock_syntax_check.assert_called_once()


@patch("agent.subprocess.run")
def test_run_syntax_check_exception(mock_run):
    from agent import _run_syntax_check

    mock_run.side_effect = Exception("syntax check failed")
    output = _run_syntax_check("dummy.py", "test")
    assert "syntax check failed" in output


@patch("agent.subprocess.run")
def test_run_syntax_check_timeout(mock_run):
    import subprocess

    from agent import _run_syntax_check

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="flake8", timeout=10)
    output = _run_syntax_check("dummy.py", "test")
    assert "timed out after 10" in output


@patch("agent.subprocess.run")
def test_run_syntax_check_success_and_failure(mock_run):
    from agent import _run_syntax_check

    mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
    assert _run_syntax_check("dummy.py", "test") == ""

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="error")
    assert "error" in _run_syntax_check("dummy.py", "test")


def test_tdd_agent_methods_direct():
    from agent import TDDAgent

    agent = TDDAgent()
    assert agent.get_graph() is not None

    with patch.object(agent.app, "invoke", return_value={"success": True}) as mock_invoke:
        res = agent.invoke({"goal": "g"}, config={"c": 1})
        assert res["success"] is True
        mock_invoke.assert_called_once_with({"goal": "g"}, config={"c": 1})


def test_agent_should_continue_end():
    from langgraph.graph import END

    from agent import should_continue
    from schema import TDDState

    state = TDDState(success=False, iterations=3, max_iterations=3)
    assert should_continue(state) == END


def test_agent_should_fix_tests_or_impl():
    from agent import should_fix_tests_or_impl
    from schema import TDDState

    state = TDDState(next_action="generate_tests")
    assert should_fix_tests_or_impl(state) == "generate_tests"

    state2 = TDDState()
    assert should_fix_tests_or_impl(state2) == "implement_logic"


def test_conditional_edge_review_test_plan():
    # Check if the branch logic (Edge) correctly returns the next node name
    state_regenerate = TDDState(test_plan_review_decision="plan_tests")
    assert should_review_test_plan_or_continue(state_regenerate) == "plan_tests"

    state_continue = TDDState(test_plan_review_decision="generate_tests")
    assert should_review_test_plan_or_continue(state_continue) == "generate_tests"


def test_agent_verbose_logs_coverage():
    from unittest.mock import MagicMock, patch

    import config
    from agent import (
        _run_syntax_check,
        fetch_spec,
        generate_bug_report,
        generate_readme,
        generate_tests,
        review_test_plan,
        review_tests,
        run_tests,
    )
    from schema import TDDState

    with patch.object(config, "VERBOSE", True):
        # 1. fetch_spec Line 87 (Remote file not modified)
        with patch("agent.os.path.exists", return_value=True):
            with patch("agent.os.path.getmtime", return_value=123456):
                with patch("agent.read_artifact", return_value="cached spec"):
                    mock_resp = MagicMock()
                    mock_resp.status_code = 304
                    with patch("agent.requests.get", return_value=mock_resp):
                        res = fetch_spec(TDDState(spec_url="http://cached"))
                        assert res["spec_content"] == "cached spec"

        # 2. review_test_plan Line 260-262 (print report_md)
        with patch("agent.call_llm_with_reasoning") as mock_call:
            mock_call.return_value = (
                '```json\n{"missing_test_cases": ["Case A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
            )
            review_test_plan(TDDState(target_test_plan_coverage=95))

        # 3. generate_tests Line 333, 357 (Found mathematical cases, verified case)
        with (
            patch("agent.call_llm_standard") as mock_std,
            patch("agent.call_llm_with_reasoning") as mock_reason,
            patch("agent.evaluate_math_expression") as mock_bc,
            patch("agent.save_artifact"),
        ):
            mock_std.return_value = '{"items": [{"test_case_number": 1, "expression": "1+1"}]}'
            mock_reason.return_value = "```python\ndef test_a(): pass\n```"
            mock_bc.return_value = "2"
            generate_tests(TDDState(test_plan="1. Expected: 2"))

        # 4. generate_tests Line 364 (Failed to parse/verify calculations)
        with (
            patch("agent.call_llm_standard") as mock_std,
            patch("agent.call_llm_with_reasoning") as mock_reason,
            patch("agent.save_artifact"),
        ):
            mock_std.return_value = "invalid json which triggers exception"
            mock_reason.return_value = "```python\ndef test_a(): pass\n```"
            generate_tests(TDDState(test_plan="1. Expected: 2"))

        # 5. _run_syntax_check Line 459 (print output)
        with patch("agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="syntax error", stderr="")
            _run_syntax_check("dummy.py", "dummy")

        # 6. review_tests Line 521-523 (print report_md)
        with patch("agent.call_llm_with_reasoning") as mock_call:
            mock_call.return_value = (
                '```json\n{"missing_test_cases": ["Case B"], "estimated_coverage": 80, "feedback": "Add B"}\n```'
            )
            review_tests(TDDState(target_test_coverage=95))

        # 7. generate_bug_report Line 654 (failed test cases loop in bug report)
        with patch("agent.call_llm_with_reasoning") as mock_call:
            mock_call.return_value = (
                '```json\n{"failed_test_cases": ["test_a"], '
                '"expected_vs_actual": "diff", "fix_instructions": "fix it", '
                '"target_to_fix": "implement_logic"}\n```'
            )
            generate_bug_report(TDDState(iterations=1))

        # 8. generate_readme Line 693-695
        with (
            patch("agent.call_llm_standard") as mock_call,
            patch("agent.save_artifact"),
        ):
            mock_call.return_value = "# README"
            generate_readme(TDDState(goal="g", module_name="impl.py", impl_code="pass"))

        # 9. run_tests Line 654 (print output in run_tests under verbose mode)
        with patch("agent.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="test success output", stderr="")
            run_tests(TDDState(test_module_name="test_impl.py", iterations=1))


@patch("agent.call_llm_standard")
@patch("agent.save_artifact")
def test_generate_requirements(mock_save_artifact, mock_call_llm_standard):
    mock_call_llm_standard.return_value = '{"requirements": [{"id": "REQ001", "description": "Verify math"}]}'
    mock_save_artifact.return_value = "artifacts/requirements.md"

    import config

    with patch.object(config, "VERBOSE", True):
        state = TDDState(spec_content="Spec content")
        result = generate_requirements(state)

    assert result["current_req_index"] == 0
    assert len(result["requirements"]) == 1
    assert result["requirements"][0]["id"] == "REQ001"
    assert "REQ001: Verify math" in result["requirements_list_str"]
    mock_save_artifact.assert_called_once()


def test_increment_requirement():
    state = TDDState(
        current_req_index=0,
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}],
        success=True,
        test_plan="old",
    )
    result = increment_requirement(state)

    assert result["current_req_index"] == 1
    assert result["success"] is False
    assert result["test_plan"] == ""


def test_should_continue_incremental():
    # Success, but more requirements left
    state = TDDState(
        success=True,
        current_req_index=0,
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}],
    )
    assert should_continue(state) == "increment_requirement"

    # Success, and no more requirements left
    state2 = TDDState(
        success=True,
        current_req_index=1,
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}],
    )
    assert should_continue(state2) == "generate_readme"


def test_custom_print():
    from unittest.mock import patch

    import agent
    from agent import _update_req_progress
    from agent import print as agent_print
    from schema import TDDState

    # Reset globals first
    agent._CURRENT_REQ_NUM = 0
    agent._TOTAL_REQ_NUM = 0

    with patch("builtins.print") as mock_print:
        agent_print("[TDD Robo] Hello")
        mock_print.assert_called_once()
        args, kwargs = mock_print.call_args
        assert "[TDD Robo] Hello" in args[0]
        assert "20" in args[0]
        assert "(" not in args[0]

    with patch("builtins.print") as mock_print:
        agent_print("Hello")
        mock_print.assert_called_once_with("Hello")

    # Test update progress with empty state
    _update_req_progress({})
    assert agent._CURRENT_REQ_NUM == 0
    assert agent._TOTAL_REQ_NUM == 0

    # Test update progress with requirements
    state = TDDState(
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}],
        current_req_index=0,
    )
    _update_req_progress(state)
    assert agent._CURRENT_REQ_NUM == 1
    assert agent._TOTAL_REQ_NUM == 2

    # Test print with progress
    with patch("builtins.print") as mock_print:
        agent_print("[TDD Robo] Hello")
        mock_print.assert_called_once()
        args, kwargs = mock_print.call_args
        assert "[TDD Robo] (1/2) Hello" in args[0]


def test_parse_pytest_summary():
    from agent import _parse_pytest_summary

    # Case 1: Simple passed summary
    output = "random output\n=== 105 passed in 0.74s ===\nother output"
    assert _parse_pytest_summary(output) == "105 passed in 0.74s"

    # Case 2: Passed with warnings
    output = "random output\n=== 105 passed, 2 warnings in 0.74s ===\n"
    assert _parse_pytest_summary(output) == "105 passed, 2 warnings in 0.74s"

    # Case 3: Failed summary
    output = "random output\n=== 6 failed, 99 passed in 0.47s ==="
    assert _parse_pytest_summary(output) == "6 failed, 99 passed in 0.47s"

    # Case 4: No matching summary lines
    output = "random output\n=== test session starts ===\n"
    assert _parse_pytest_summary(output) == ""


def test_get_combined_tests_code_error_handling(tmp_path):
    from agent import _get_combined_tests_code

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        # Create a test file
        test_file = tmp_path / "test_bc_clone_req1.py"
        test_file.write_text("print('test')", encoding="utf-8")

        # Mock open to raise exception
        with patch("builtins.open", side_effect=PermissionError("Mock Permission Error")):
            state = TDDState(tests_code="default_test_code")
            assert _get_combined_tests_code(state) == "default_test_code"


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
def test_implement_logic_search_replace_syntax_error(mock_save_artifact, mock_call_llm_with_reasoning):
    # LLM response successfully matched apply_search_replace_blocks, but compile() raises SyntaxError
    mock_call_llm_with_reasoning.return_value = (
        "<<<<<<< SEARCH\nold_code\n=======\ninvalid python syntax {\n>>>>>>> REPLACE"
    )
    mock_save_artifact.return_value = "artifacts/impl.py"

    state = TDDState(
        module_name="impl.py",
        impl_code="old_code",
        goal="goal",
    )

    with patch("agent.apply_search_replace_blocks", return_value="invalid python syntax {"):
        # This will raise SyntaxError in compile(code, impl_name, "exec"), triggering ValueError
        result = implement_logic(state)
        # Verify it fallback or fails and returns impl_check_output containing "SyntaxError"
        assert "SyntaxError" in result["impl_check_output"]
        assert result["impl_code"] == "old_code"  # Keep old code on error


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
def test_implement_logic_search_replace_failed_with_marker(mock_save_artifact, mock_call_llm_with_reasoning):
    # Search/replace fail but has "<<<<<<< SEARCH" marker
    mock_call_llm_with_reasoning.return_value = (
        "Here is some text\n<<<<<<< SEARCH\nwrong_code\n=======\nnew_code\n>>>>>>> REPLACE"
    )
    mock_save_artifact.return_value = "artifacts/impl.py"

    state = TDDState(
        module_name="impl.py",
        impl_code="actual_code",
        goal="goal",
    )

    result = implement_logic(state)
    assert "Failed to apply Search/Replace block" in result["impl_check_output"]
    assert result["impl_code"] == "actual_code"  # keeps the original code


@patch("agent.call_llm_with_reasoning")
@patch("agent.save_artifact")
def test_implement_logic_search_replace_failed_no_marker(mock_save_artifact, mock_call_llm_with_reasoning):
    # Search/replace raises ValueError but no markers (so it extracts code)
    mock_call_llm_with_reasoning.return_value = "```python\nextracted_code\n```"
    mock_save_artifact.return_value = "artifacts/impl.py"

    state = TDDState(
        module_name="impl.py",
        impl_code="actual_code",
        goal="goal",
    )

    with patch("agent.apply_search_replace_blocks", side_effect=ValueError("No blocks found")):
        with patch("utils.config.VERBOSE", True):
            result = implement_logic(state)
            assert result["impl_code"] == "extracted_code"
            assert result["impl_check_output"] == ""


def test_check_impl_syntax_intercept_syntax_error():
    # Test interception of custom syntax error in code
    state = TDDState(
        module_name="impl.py",
        impl_code="# TDD_ROBO_SYNTAX_ERROR: Custom Intercepted Syntax Error\n",
    )
    result = check_impl_syntax(state)
    assert result["impl_check_output"] == "Error: Custom Intercepted Syntax Error"


def test_check_impl_syntax_keep_pre_existing_sr_error():
    # Test keeping pre-existing "Failed to apply Search/Replace block" error
    state = TDDState(
        module_name="impl.py",
        impl_code="valid_code",
        impl_check_output="Failed to apply Search/Replace block: code mismatch",
    )
    result = check_impl_syntax(state)
    assert result["impl_check_output"] == "Failed to apply Search/Replace block: code mismatch"


@patch("agent.call_llm_with_reasoning")
def test_implement_logic_pre_existing_sr_error_prompt_extension(mock_call_llm_with_reasoning):
    mock_call_llm_with_reasoning.return_value = "```python\nnew_code\n```"
    state = TDDState(
        module_name="impl.py",
        impl_code="old_code",
        impl_check_output="Failed to apply Search/Replace block: mismatch",
        goal="goal",
    )
    implement_logic(state)
    # verify that call_llm_with_reasoning is called, and the prompt construction used the extended output
    assert mock_call_llm_with_reasoning.called


def test_get_combined_tests_code_success(tmp_path):
    from agent import _get_combined_tests_code

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        # Create a test file
        test_file = tmp_path / "test_bc_clone_req1.py"
        test_file.write_text("print('test')", encoding="utf-8")

        state = TDDState(tests_code="default_test_code")
        combined = _get_combined_tests_code(state)
        assert "print('test')" in combined
        assert "test_bc_clone_req1.py" in combined


def test_run_oracle_verification_on_failures_none_verifier():
    from agent import _run_oracle_verification_on_failures

    with patch("config.ORACLE_VERIFIER", None):
        res = _run_oracle_verification_on_failures("FAILED", "def test_a(): pass")
        assert res == ""


def test_run_oracle_verification_on_failures_success():
    from unittest.mock import MagicMock

    from agent import _run_oracle_verification_on_failures

    mock_verifier = MagicMock(return_value="152399025")
    with patch("config.ORACLE_VERIFIER", mock_verifier):
        test_output = (
            "FAILED tests/test_bc_clone_req004.py::TestBCInterface::test_large_number_multiplication - AssertionError"
        )
        tests_code = """
def test_large_number_multiplication(self, interpreter):
    code = "12345 * 12345"
    expected = ["1234567"]
    assert interpreter.execute(code) == expected
"""
        res = _run_oracle_verification_on_failures(test_output, tests_code)
        assert "mathematically INCORRECT" in res
        assert "152399025" in res
        mock_verifier.assert_called_once_with("12345 * 12345")


def test_run_oracle_verification_on_failures_no_failed_methods():
    from unittest.mock import MagicMock

    from agent import _run_oracle_verification_on_failures

    mock_verifier = MagicMock(return_value="152399025")
    with patch("config.ORACLE_VERIFIER", mock_verifier):
        res = _run_oracle_verification_on_failures("all tests passed successfully", "def test_a(): pass")
        assert res == ""


def test_run_oracle_verification_on_failures_method_not_found():
    from unittest.mock import MagicMock

    from agent import _run_oracle_verification_on_failures

    mock_verifier = MagicMock(return_value="152399025")
    with patch("config.ORACLE_VERIFIER", mock_verifier):
        test_output = "FAILED test_bc_clone_req004.py::TestBCInterface::test_large_number_multiplication"
        res = _run_oracle_verification_on_failures(test_output, "def test_another_one(): pass")
        assert res == "No assertion discrepancies detected by the oracle."


def test_run_oracle_verification_on_failures_alternative_assert():
    from unittest.mock import MagicMock

    from agent import _run_oracle_verification_on_failures

    mock_verifier = MagicMock(return_value="152399025")
    with patch("config.ORACLE_VERIFIER", mock_verifier):
        test_output = "FAILED test_large_number_multiplication"
        tests_code = """
def test_large_number_multiplication(self):
    # Empty line test for indentation check
    
    assert interpreter.execute("12345 * 12345") == ["1234567"]
x = 1
"""
        res = _run_oracle_verification_on_failures(test_output, tests_code)
        assert "mathematically INCORRECT" in res
        assert "152399025" in res
        mock_verifier.assert_called_once_with("12345 * 12345")


def test_run_oracle_verification_on_failures_indent_def_break():
    from unittest.mock import MagicMock

    from agent import _run_oracle_verification_on_failures

    mock_verifier = MagicMock(return_value="152399025")
    with patch("config.ORACLE_VERIFIER", mock_verifier):
        test_output = "FAILED test_large_number_multiplication"
        tests_code = """
def test_large_number_multiplication(self):
    assert interpreter.execute("12345 * 12345") == ["1234567"]
def test_next_one(self):
    pass
"""
        res = _run_oracle_verification_on_failures(test_output, tests_code)
        assert "mathematically INCORRECT" in res
        assert "152399025" in res
        mock_verifier.assert_called_once_with("12345 * 12345")


def test_run_oracle_verification_on_failures_verifier_exception():
    from unittest.mock import MagicMock

    from agent import _run_oracle_verification_on_failures

    mock_verifier = MagicMock(side_effect=ValueError("math error"))
    with patch("config.ORACLE_VERIFIER", mock_verifier):
        test_output = "FAILED test_large_number_multiplication"
        tests_code = """
def test_large_number_multiplication(self):
    assert interpreter.execute("12345 * 12345") == ["1234567"]
"""
        res = _run_oracle_verification_on_failures(test_output, tests_code)
        assert res == "No assertion discrepancies detected by the oracle."


def test_detect_toggle_loop_no_dir():
    from agent import _detect_toggle_loop

    with patch("config.ARTIFACTS_DIR", "non_existent_directory_xyz"):
        state = TDDState(module_name="impl.py")
        assert _detect_toggle_loop(state) is False


def test_detect_toggle_loop_less_files(tmp_path):
    from agent import _detect_toggle_loop

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "impl_iter001.py").write_text("code1")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        state = TDDState(module_name="impl.py")
        assert _detect_toggle_loop(state) is False


def test_detect_toggle_loop_file_read_error(tmp_path):
    from agent import _detect_toggle_loop

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    # 5 files to satisfy len >= 4
    (history_dir / "impl_iter001.py").write_text("code1")
    (history_dir / "impl_iter002.py").write_text("code2")
    (history_dir / "impl_iter003.py").write_text("code3")
    (history_dir / "impl_iter004.py").write_text("code4")
    (history_dir / "impl_iter005.py").write_text("code5")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        state = TDDState(module_name="impl.py")
        original_open = open

        def mock_open(file, *args, **kwargs):
            if "impl_iter004" in str(file):
                raise OSError("read error")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", mock_open):
            assert _detect_toggle_loop(state) is False


def test_detect_toggle_loop_hashes_too_few_after_errors(tmp_path):
    from agent import _detect_toggle_loop

    history_dir = tmp_path / "history"
    history_dir.mkdir()
    (history_dir / "impl_iter001.py").write_text("code1")
    (history_dir / "impl_iter002.py").write_text("code2")
    (history_dir / "impl_iter003.py").write_text("code3")
    (history_dir / "impl_iter004.py").write_text("code4")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        state = TDDState(module_name="impl.py")
        original_open = open

        def mock_open(file, *args, **kwargs):
            if "impl_iter003" in str(file) or "impl_iter004" in str(file):
                raise OSError("read error")
            return original_open(file, *args, **kwargs)

        with patch("builtins.open", mock_open):
            assert _detect_toggle_loop(state) is False


def test_detect_toggle_loop_detected(tmp_path):
    from agent import _detect_toggle_loop

    history_dir = tmp_path / "history"
    history_dir.mkdir()

    # Toggle loop: A-B-A-B
    (history_dir / "impl_iter001.py").write_text("content A")
    (history_dir / "impl_iter002.py").write_text("content B")
    (history_dir / "impl_iter003.py").write_text("content A")
    (history_dir / "impl_iter004.py").write_text("content B")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        state = TDDState(module_name="impl.py")
        assert _detect_toggle_loop(state) is True


def test_detect_toggle_loop_repeating_multi(tmp_path):
    from agent import _detect_toggle_loop

    history_dir = tmp_path / "history"
    history_dir.mkdir()

    # Repeating: A-X-A-Y-A
    (history_dir / "impl_iter001.py").write_text("content A")
    (history_dir / "impl_iter002.py").write_text("content X")
    (history_dir / "impl_iter003.py").write_text("content A")
    (history_dir / "impl_iter004.py").write_text("content Y")
    (history_dir / "impl_iter005.py").write_text("content A")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        state = TDDState(module_name="impl.py")
        assert _detect_toggle_loop(state) is True


def test_should_fix_tests_or_impl_loop_override():
    from agent import should_fix_tests_or_impl

    state = TDDState(next_action="implement_logic")
    with patch("agent._detect_toggle_loop", return_value=True):
        assert should_fix_tests_or_impl(state) == "generate_tests"
    with patch("agent._detect_toggle_loop", return_value=False):
        assert should_fix_tests_or_impl(state) == "implement_logic"


@patch("agent._run_oracle_verification_on_failures")
@patch("agent.call_llm_with_reasoning")
def test_generate_bug_report_with_oracle_integration(mock_call_llm_with_reasoning, mock_run_oracle):
    mock_run_oracle.return_value = "Discrepancy found!"
    mock_call_llm_with_reasoning.return_value = (
        '{"failed_test_cases": [], "expected_vs_actual": "", "fix_instructions": "", "target_to_fix": "generate_tests"}'
    )

    state = TDDState(
        goal="goal",
        test_output="failed",
        impl_code="code",
    )

    res = generate_bug_report(state)
    assert res["next_action"] == "generate_tests"
    mock_run_oracle.assert_called_once()
