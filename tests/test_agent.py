from unittest.mock import MagicMock, patch

from langgraph.graph import END

from agent import (
    check_impl_syntax,
    check_tests_syntax,
    fetch_spec,
    generate_bug_report,
    generate_design,
    generate_readme,
    generate_tests,
    implement_logic,
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


@patch("agent.llm_gendoc")
def test_plan_files(mock_llm_gendoc):
    mock_llm_gendoc.return_value = '```json\n{"impl_filename": "my_impl.py", "test_filename": "my_test.py"}\n```'
    state = TDDState(goal="Make a calculator")

    result = plan_files(state)

    assert result["module_name"] == "my_impl.py"
    assert result["test_module_name"] == "my_test.py"


@patch("agent.llm_gendoc")
def test_plan_files_decode_error(mock_llm_gendoc):
    mock_llm_gendoc.return_value = "invalid json format"
    state = TDDState(goal="Make a calculator")
    result = plan_files(state)
    assert result["module_name"] == "impl.py"  # default fallback
    assert result["test_module_name"] == "test_impl.py"


@patch("agent.llm_gencode")
@patch("agent.save_artifact")
def test_generate_design(mock_save_artifact, mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"module_responsibilities": "Handles math", "error_handling": "Raises ValueError"}\n```'
    )
    mock_save_artifact.return_value = "artifacts/design.md"

    state = TDDState(goal="Calc", spec_content="Spec", module_name="impl.py", test_module_name="test.py")
    result = generate_design(state)

    assert "Handles math" in result["design_doc"]
    assert "Raises ValueError" in result["design_doc"]


@patch("agent.llm_gencode")
@patch("agent.save_artifact")
def test_generate_design_decode_error(mock_save_artifact, mock_llm_gencode):
    mock_llm_gencode.return_value = "not json"
    state = TDDState(goal="Calc")
    result = generate_design(state)
    assert "not json" in result["design_doc"]


@patch("agent._run_syntax_check")
def test_check_tests_syntax_empty_code(mock_syntax_check):
    # Check if an error is returned immediately without running syntax check when test code is empty
    state = TDDState(tests_code="", test_module_name="test_impl.py")
    result = check_tests_syntax(state)

    assert "Error: The generated test code is empty" in result["tests_check_output"]
    mock_syntax_check.assert_not_called()


@patch("agent.llm_gendoc")
@patch("agent.save_artifact")
def test_plan_tests(mock_save_artifact, mock_llm_gendoc):
    mock_llm_gendoc.return_value = '```json\n{"test_cases": [{"action": "add", "expected_outcome": "sum"}]}\n```'
    mock_save_artifact.return_value = "artifacts/test_plan.md"

    state = TDDState(goal="goal", spec_content="spec", design_doc="design", test_plan_iterations=0)
    result = plan_tests(state)

    assert "Action: add | Expected: sum" in result["test_plan"]
    assert result["test_plan_iterations"] == 1


@patch("agent.llm_gendoc")
@patch("agent.save_artifact")
def test_plan_tests_with_review_and_error(mock_save_artifact, mock_llm_gendoc):
    # test fix prompt branch and decode error fallback
    mock_llm_gendoc.return_value = "invalid json"
    state = TDDState(test_plan="old", test_plan_review="feedback")
    result = plan_tests(state)
    assert "invalid json" in result["test_plan"]


@patch("agent.llm_gencode")
def test_review_test_plan_exception(mock_llm_gencode):
    # test exception fallback
    mock_llm_gencode.return_value = "invalid json"
    state = TDDState(target_test_plan_coverage=95)
    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "generate_tests"
    assert result["test_plan_review"] == ""


@patch("agent.llm_gencode")
def test_review_test_plan(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "good"}\n```'
    )
    state = TDDState(target_test_plan_coverage=95)

    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "generate_tests"
    assert result["test_plan_review"] == ""


@patch("agent.llm_gencode")
def test_review_test_plan_insufficient_max_iters(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": ["B"], "estimated_coverage": 50, "feedback": "Add B"}\n```'
    )
    state = TDDState(target_test_plan_coverage=95, test_plan_iterations=3, max_test_plan_iterations=3)

    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "generate_tests"
    assert result["test_plan_review"] == ""


@patch("agent.llm_gencode")
def test_review_test_plan_insufficient_retry(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": ["B"], "estimated_coverage": 50, "feedback": "Add B"}\n```'
    )
    state = TDDState(target_test_plan_coverage=95, test_plan_iterations=1, max_test_plan_iterations=3)
    result = review_test_plan(state)
    assert result["test_plan_review_decision"] == "plan_tests"
    assert "Add B" in result["test_plan_review"]


@patch("agent.llm_gencode")
@patch("agent.save_artifact")
def test_generate_tests(mock_save_artifact, mock_llm_gencode):
    mock_llm_gencode.return_value = "```python\ndef test_a(): pass\n```"
    state = TDDState(test_iterations=0)
    result = generate_tests(state)

    assert result["tests_code"] == "def test_a(): pass"
    assert result["test_iterations"] == 1


@patch("agent.llm_gencode")
@patch("agent.save_artifact")
def test_generate_tests_fix_branches(mock_save_artifact, mock_llm_gencode):
    mock_llm_gencode.return_value = "```python\nfixed\n```"
    state = TDDState(tests_check_output="error", tests_code="error code")
    result = generate_tests(state)
    assert result["tests_code"] == "fixed"


def test_generate_tests_all_branches():
    from agent import generate_tests

    with patch("agent.llm_gencode", return_value="```python\nfixed\n```"), patch("agent.save_artifact"):
        # tests_check_output
        generate_tests(TDDState(tests_check_output="error", tests_code="old"))
        # bug_report
        generate_tests(TDDState(bug_report="error", tests_code="old"))
        # test_review
        generate_tests(TDDState(test_review="error", tests_code="old"))
        # initial
        generate_tests(TDDState())


@patch("agent.llm_gencode")
def test_review_tests(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": ["A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
    )
    state = TDDState(target_test_coverage=90, test_iterations=1, max_test_iterations=3)

    result = review_tests(state)
    assert result["test_review_decision"] == "generate_tests"
    assert "Add A" in result["test_review"]


@patch("agent.llm_gencode")
def test_review_tests_sufficient(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": [], "estimated_coverage": 100, "feedback": "Good"}\n```'
    )
    state = TDDState(target_test_coverage=90)
    result = review_tests(state)
    assert result["test_review_decision"] == "implement_logic"
    assert result["test_review"] == ""


@patch("agent.llm_gencode")
def test_review_tests_insufficient_proceed(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": ["A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
    )
    state = TDDState(target_test_coverage=90, test_iterations=3, max_test_iterations=3)
    result = review_tests(state)
    assert result["test_review_decision"] == "implement_logic"


@patch("agent.llm_gencode")
def test_review_tests_insufficient_coverage_retry(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"missing_test_cases": ["A"], "estimated_coverage": 80, "feedback": "Add A"}\n```'
    )
    state = TDDState(target_test_coverage=90, test_iterations=1, max_test_iterations=3)
    result = review_tests(state)
    assert result["test_review_decision"] == "generate_tests"


@patch("agent.llm_gencode")
def test_review_tests_exception(mock_llm_gencode):
    mock_llm_gencode.return_value = "invalid"
    state = TDDState(target_test_coverage=90)
    result = review_tests(state)
    assert result["test_review_decision"] == "implement_logic"


@patch("agent.llm_gencode")
@patch("agent.save_artifact")
def test_implement_logic(mock_save_artifact, mock_llm_gencode):
    mock_llm_gencode.return_value = "```python\ndef run(): return True\n```"
    state = TDDState(goal="g", design_doc="d", tests_code="t")
    result = implement_logic(state)

    assert result["impl_code"] == "def run(): return True"


def test_implement_logic_all_branches():
    from agent import implement_logic

    with patch("agent.llm_gencode", return_value="```python\nfixed\n```"), patch("agent.save_artifact"):
        # impl_check_output
        implement_logic(TDDState(impl_check_output="error", impl_code="old"))
        # bug_report
        implement_logic(TDDState(bug_report="error", impl_code="old"))
        # initial
        implement_logic(TDDState())


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
def test_run_tests_timeout(mock_run):
    import subprocess

    mock_run.side_effect = subprocess.TimeoutExpired(cmd="pytest", timeout=30, output="part_out", stderr="part_err")
    state = TDDState(test_module_name="test_impl.py", iterations=1)
    result = run_tests(state)
    assert result["success"] is False
    assert "timed out after 30" in result["test_output"]


@patch("agent.llm_gencode")
def test_generate_bug_report(mock_llm_gencode):
    mock_llm_gencode.return_value = (
        '```json\n{"failed_test_cases": ["t1"], "expected_vs_actual": "diff", '
        '"fix_instructions": "fix", "target_to_fix": "implement_logic"}\n```'
    )
    state = TDDState(iterations=1)
    result = generate_bug_report(state)

    assert "diff" in result["bug_report"]
    assert result["next_action"] == "implement_logic"


@patch("agent.llm_gencode")
def test_generate_bug_report_decode_error(mock_llm_gencode):
    mock_llm_gencode.return_value = "invalid"
    state = TDDState()
    result = generate_bug_report(state)
    assert result["next_action"] == "implement_logic"


@patch("agent.llm_gendoc")
@patch("agent.save_artifact")
def test_generate_readme(mock_save_artifact, mock_llm_gendoc):
    mock_llm_gendoc.return_value = "# README"
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


def test_agent_plan_tests_decode_error():
    from unittest.mock import patch

    from agent import plan_tests
    from schema import TDDState

    with patch("agent.llm_gendoc", return_value="invalid json"), patch("agent.save_artifact"):
        state = TDDState()
        result = plan_tests(state)
        assert "invalid json" in result["test_plan"]


def test_agent_generate_bug_report_decode_error():
    from unittest.mock import patch

    from agent import generate_bug_report
    from schema import TDDState

    with patch("agent.llm_gencode", return_value="invalid json"):
        state = TDDState()
        result = generate_bug_report(state)
        assert result["next_action"] == "implement_logic"


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
