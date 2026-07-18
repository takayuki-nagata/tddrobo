from unittest.mock import MagicMock, patch

from langgraph.graph import END

from agent import (
    TDDAgent,
    check_initial_impl_syntax,
    check_integration_impl_syntax,
    check_integration_tests_syntax,
    check_refactored_impl_syntax,
    check_regression_impl_syntax,
    check_unit_tests_syntax,
    decide_refactor,
    fetch_spec,
    generate_design_initial,
    generate_integration_bug_report,
    generate_integration_tests,
    generate_readme,
    generate_refactor_bug_report,
    generate_regression_bug_report,
    generate_unit_bug_report,
    generate_unit_tests,
    implement_initial_logic,
    implement_integration_logic,
    implement_regression_logic,
    increment_requirement,
    plan_files,
    plan_integration_tests,
    plan_unit_tests,
    refactor_logic,
    review_integration_test_plan,
    review_unit_test_plan,
    run_integration_tests,
    run_regression_tests,
    run_unit_tests,
    should_continue_integration,
    should_continue_regression,
    should_continue_unit,
    should_continue_workflow,
    should_fix_integration_tests_or_impl,
    should_fix_refactor_or_continue,
    should_fix_regression_tests_or_impl,
    should_fix_unit_tests_or_impl,
    should_refactor,
    should_review_integration_tests_or_continue,
    should_review_test_plan_or_continue,
    should_review_unit_tests_or_continue,
    should_route_from_audit,
    should_run_integration_tests,
    should_run_regression_after_refactor,
    should_run_regression_tests,
    should_run_unit_tests,
    update_design_for_req,
)
from schema import BugReport, DesignDocument, RefactorDecision, TDDState, TestCase, TestPlan, TestPlanReviewReport


def test_agent_print_override(capsys):
    from agent import print as agent_print

    agent_print("Hello", end="", flush=True)
    captured = capsys.readouterr()
    assert captured.out == "Hello"


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

    with (
        patch("os.path.exists", return_value=True),
        patch("os.path.getmtime", return_value=1000),
        patch("config.VERBOSE", True),
    ):
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

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        save_history_snapshot("test.py", "print('hello')", 3)

        expected_path = tmp_path / "history" / "test_iter003.py"
        assert expected_path.exists()
        assert expected_path.read_text(encoding="utf-8") == "print('hello')"

        state = TDDState(design_iterations=2)
        save_history_snapshot("test_impl.py", "print('hello')", 4, state=state)
        expected_path_d = tmp_path / "history" / "test_impl_d002_iter004.py"
        assert expected_path_d.exists()

        state_impl = TDDState(
            requirements=[{"id": "REQ001", "description": "desc"}],
            current_req_index=0,
            test_iterations=1,
            design_iterations=5,
        )
        save_history_snapshot("impl.py", "impl_code", 3, state=state_impl)
        expected_path_impl = tmp_path / "history" / "impl_req001_d005_test_iter001_impl_iter003.py"
        assert expected_path_impl.exists()

        with patch("builtins.open", side_effect=PermissionError("Mocked Permission Denied")):
            save_history_snapshot("test.py", "print('hello')", 3)


@patch("agent.call_llm_standard")
def test_plan_files(mock_call_llm_standard):
    mock_call_llm_standard.return_value = '```json\n{"impl_filename": "my_impl.py", "test_filename": "my_test.py"}\n```'
    state = TDDState(goal="Make a calculator")

    result = plan_files(state)

    assert result["module_name"] == "my_impl.py"
    assert result["test_module_name"] == "my_test.py"


@patch("agent._call_llm_structured")
@patch("agent.save_artifact")
def test_generate_design_initial(mock_save_artifact, mock_llm):
    mock_doc = DesignDocument(
        module_responsibilities="responsibilities",
        architecture_and_components="architecture",
        interface_definitions="interfaces",
        data_structures="data",
        logic_and_algorithms="logic",
        edge_cases_and_limitations="edge",
        error_handling="errors",
        command_line_interface="cli",
    )
    mock_llm.return_value = mock_doc
    state = TDDState(goal="calculator")

    res = generate_design_initial(state)
    assert res["design_updated"] is True
    assert "responsibilities" in res["design_doc"]


@patch("agent._call_llm_structured")
@patch("agent.save_artifact")
def test_update_design_for_req(mock_save_artifact, mock_llm):
    mock_doc = DesignDocument(
        module_responsibilities="updated",
        architecture_and_components="architecture",
        interface_definitions="interfaces",
        data_structures="data",
        logic_and_algorithms="logic",
        edge_cases_and_limitations="edge",
        error_handling="errors",
        command_line_interface="cli",
    )
    mock_llm.return_value = mock_doc
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "some requirement"}],
        current_req_index=0,
        design_updated=False,
    )

    res = update_design_for_req(state)
    assert res["design_updated"] is True
    assert "updated" in res["design_doc"]
    assert res["unit_test_plan"] is None
    assert res["integration_test_plan"] is None
    assert res["test_plan"] is None
    assert res["test_plan_review"] is None
    assert res["test_plan_review_decision"] is None


@patch("agent._call_llm_structured")
@patch("agent.save_artifact")
def test_update_design_for_req_with_failure_context(mock_save_artifact, mock_llm):
    mock_doc = DesignDocument(
        module_responsibilities="updated",
        architecture_and_components="architecture",
        interface_definitions="interfaces",
        data_structures="data",
        logic_and_algorithms="logic",
        edge_cases_and_limitations="edge",
        error_handling="errors",
        command_line_interface="cli",
    )
    mock_llm.return_value = mock_doc
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "some requirement"}],
        current_req_index=0,
        design_updated=False,
        loop_detected=True,
        bug_report="Expected '3.33333', got '3'.",
        test_output="test failed: division scale precision error\n" * 10,
    )

    res = update_design_for_req(state)
    assert res["design_updated"] is True
    assert res["loop_detected"] is False

    called_prompt = mock_llm.call_args[0][0]
    assert "Previous Implementation Failure Context" in called_prompt
    assert "Expected '3.33333', got '3'." in called_prompt
    assert "test failed: division scale precision error" in called_prompt
    assert "CRITICAL REFRACTOR DIRECTIVE" in called_prompt


@patch("agent.save_artifact")
@patch("agent._call_llm_structured")
def test_update_design_for_req_with_last_green(mock_llm, mock_save_artifact):
    mock_doc = DesignDocument(
        module_responsibilities="updated",
        architecture_and_components="architecture",
        interface_definitions="interfaces",
        data_structures="data",
        logic_and_algorithms="logic",
        edge_cases_and_limitations="edge",
        error_handling="errors",
        command_line_interface="cli",
    )
    mock_llm.return_value = mock_doc
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "some requirement"}],
        current_req_index=0,
        design_updated=False,
        last_green_impl_code="def last_green(): pass",
        impl_code="def broken(): pass",
    )

    res = update_design_for_req(state)
    assert res["design_updated"] is True

    called_prompt = mock_llm.call_args[0][0]
    assert "def last_green(): pass" in called_prompt
    assert "def broken(): pass" not in called_prompt


@patch("agent._call_llm_structured")
def test_plan_unit_tests(mock_llm):
    mock_plan = TestPlan(test_cases=[TestCase(action="test 1", expected_outcome="outcome 1")])
    mock_llm.return_value = mock_plan
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        unit_test_plan="",
    )

    res = plan_unit_tests(state)
    assert "test 1" in res["test_plan"]


@patch("agent._call_llm_structured")
def test_review_unit_test_plan(mock_llm):
    mock_review = TestPlanReviewReport(missing_test_cases=[], estimated_coverage=95, feedback="good")
    mock_llm.return_value = mock_review
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan="plan",
    )

    res = review_unit_test_plan(state)
    assert res["test_plan_review_decision"] == "continue"


@patch("agent._call_llm_structured")
def test_plan_integration_tests(mock_llm):
    mock_plan = TestPlan(test_cases=[TestCase(action="test 2", expected_outcome="outcome 2")])
    mock_llm.return_value = mock_plan
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
    )

    res = plan_integration_tests(state)
    assert "test 2" in res["test_plan"]


@patch("agent._call_llm_structured")
def test_review_integration_test_plan(mock_llm):
    mock_review = TestPlanReviewReport(missing_test_cases=["missing 1"], estimated_coverage=60, feedback="bad")
    mock_llm.return_value = mock_review
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan="plan",
    )

    res = review_integration_test_plan(state)
    assert res["test_plan_review_decision"] == "review_test_plan"


@patch("agent._call_llm_text")
@patch("agent.save_artifact")
def test_generate_unit_tests(mock_save, mock_llm):
    mock_llm.return_value = "def test_unit(): pass"
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
    )

    res = generate_unit_tests(state)
    assert res["unit_tests_code"] == "def test_unit(): pass"
    assert "unit.py" in res["test_module_name"]


@patch("agent._syntax_check_helper")
def test_check_unit_tests_syntax(mock_helper):
    mock_helper.return_value = {"tests_check_output": ""}
    state = TDDState(test_module_name="test_unit.py")
    res = check_unit_tests_syntax(state)
    assert res["tests_check_output"] == ""


@patch("agent._call_llm_text")
@patch("agent.save_artifact")
def test_generate_integration_tests(mock_save, mock_llm):
    mock_llm.return_value = "def test_integ(): pass"
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
    )

    res = generate_integration_tests(state)
    assert res["integration_tests_code"] == "def test_integ(): pass"
    assert "integration.py" in res["test_module_name"]


@patch("agent._syntax_check_helper")
def test_check_integration_tests_syntax(mock_helper):
    mock_helper.return_value = {"tests_check_output": ""}
    state = TDDState(test_module_name="test_integration.py")
    res = check_integration_tests_syntax(state)
    assert res["tests_check_output"] == ""


@patch("agent._implement_logic_helper")
def test_implement_logic_wrappers(mock_helper):
    mock_helper.return_value = {"impl_code": "print(1)"}
    state = TDDState()

    res1 = implement_initial_logic(state)
    assert res1["impl_code"] == "print(1)"
    mock_helper.assert_called_with(state, "unit")

    res2 = implement_integration_logic(state)
    assert res2["impl_code"] == "print(1)"
    mock_helper.assert_called_with(state, "integration")

    res3 = implement_regression_logic(state)
    assert res3["impl_code"] == "print(1)"
    mock_helper.assert_called_with(state, "regression")


@patch("agent._syntax_check_helper")
def test_check_impl_syntax_wrappers(mock_helper):
    mock_helper.return_value = {"impl_check_output": ""}
    state = TDDState(module_name="impl.py")

    assert check_initial_impl_syntax(state) == {"impl_check_output": "", "impl_updated": False}
    assert check_integration_impl_syntax(state) == {"impl_check_output": "", "impl_updated": False}
    assert check_regression_impl_syntax(state) == {"impl_check_output": "", "impl_updated": False}
    assert check_refactored_impl_syntax(state) == {"impl_check_output": "", "impl_updated": False}


@patch("agent._syntax_check_helper")
def test_check_impl_syntax_wrappers_skipping(mock_helper):
    state = TDDState(module_name="impl.py", impl_updated=False)
    expected = {"syntax_error_iterations": 0, "impl_check_output": "", "impl_updated": False}

    assert check_initial_impl_syntax(state) == expected
    assert check_integration_impl_syntax(state) == expected
    assert check_regression_impl_syntax(state) == expected
    assert check_refactored_impl_syntax(state) == expected
    mock_helper.assert_not_called()


@patch("agent._execute_tests_helper")
def test_run_tests_wrappers(mock_helper):
    mock_helper.return_value = {"success": True, "test_output": "passed"}
    state = TDDState(test_module_name="test_unit.py")

    res1 = run_unit_tests(state)
    assert res1["success"] is True
    assert res1["unit_test_iterations"] == 1

    res2 = run_integration_tests(state)
    assert res2["success"] is True
    assert res2["integration_test_iterations"] == 1

    res3 = run_regression_tests(state)
    assert res3["success"] is True
    assert res3["regression_test_iterations"] == 1


@patch("agent._call_llm_structured")
def test_generate_bug_reports(mock_llm):
    mock_bug = BugReport(
        failed_test_cases=["case 1"],
        expected_vs_actual="diff",
        fix_instructions="fix it",
        target_to_fix="implement_logic",
    )
    mock_llm.return_value = mock_bug
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
    )

    res1 = generate_unit_bug_report(state)
    assert "case 1" in res1["bug_report"]

    res2 = generate_integration_bug_report(state)
    assert "case 1" in res2["bug_report"]

    res3 = generate_regression_bug_report(state)
    assert "case 1" in res3["bug_report"]

    # Test regression bug report with last_green_impl_code diff calculation
    state_with_green = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        last_green_impl_code="def old():\n    return 1\n",
        impl_code="def new():\n    return 2\n",
    )
    res_green = generate_regression_bug_report(state_with_green)
    assert "case 1" in res_green["bug_report"]

    res4 = generate_refactor_bug_report(state)
    assert "case 1" in res4["bug_report"]


@patch("agent._call_llm_structured")
def test_decide_refactor(mock_llm):
    mock_dec = RefactorDecision(chain_of_thought="cot", refactor_needed=True, reasons=["duplication"])
    mock_llm.return_value = mock_dec
    state = TDDState(impl_code="def green(): pass")

    res = decide_refactor(state)
    assert res["refactor_decision"] == "refactor"
    assert "duplication" in res["reasons"]
    assert res["last_green_impl_code"] == "def green(): pass"


@patch("agent.save_artifact")
@patch("agent.save_history_snapshot")
def test_generate_refactor_bug_report_circuit_breaker(mock_snapshot, mock_save):
    state = TDDState(
        refactor_iterations=5,
        last_green_impl_code="def green(): pass",
        module_name="bc_clone.py",
    )
    with patch("config.MAX_REFACTOR_ITERATIONS", 5):
        res = generate_refactor_bug_report(state)
        assert res["next_action"] == "rollback_continue"
        assert res["impl_code"] == "def green(): pass"
        assert res["impl_updated"] is True
        mock_save.assert_called_once_with("bc_clone.py", "def green(): pass")
        mock_snapshot.assert_called_once()


@patch("agent._call_llm_text")
@patch("agent.save_artifact")
def test_refactor_logic(mock_save, mock_llm):
    mock_llm.return_value = "def clean(): pass"
    state = TDDState(reasons=["duplication"])

    res = refactor_logic(state)
    assert res["impl_code"] == "def clean(): pass"


def test_increment_requirement():
    state = TDDState(
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}],
        current_req_index=0,
    )
    res = increment_requirement(state)
    assert res["current_req_index"] == 1
    assert res["design_updated"] is False


@patch("agent.call_llm_standard")
@patch("agent.save_artifact")
def test_generate_readme(mock_save, mock_llm):
    mock_llm.return_value = "README content"
    state = TDDState(goal="goal")
    res = generate_readme(state)
    assert res["readme_content"] == "README content"


def test_conditional_routing_edges():
    # should_review_test_plan_or_continue
    assert should_review_test_plan_or_continue({"test_plan_review_decision": "review_test_plan"}) == "plan_tests"
    assert should_review_test_plan_or_continue({"test_plan_review_decision": "continue"}) == "generate_tests"

    # should_review_unit_tests_or_continue
    assert should_review_unit_tests_or_continue({"tests_check_output": "error"}) == "generate_unit_tests"
    assert (
        should_review_unit_tests_or_continue({"tests_check_output": "error", "test_syntax_error_iterations": 3})
        == "generate_unit_bug_report"
    )
    assert should_review_unit_tests_or_continue({"tests_check_output": ""}) == "implement_initial_logic"

    # should_review_integration_tests_or_continue
    assert should_review_integration_tests_or_continue({"tests_check_output": "error"}) == "generate_integration_tests"
    assert (
        should_review_integration_tests_or_continue({"tests_check_output": "error", "test_syntax_error_iterations": 3})
        == "generate_integration_bug_report"
    )
    assert should_review_integration_tests_or_continue({"tests_check_output": ""}) == "implement_integration_logic"

    # should_run_unit_tests
    assert should_run_unit_tests({"impl_check_output": "error"}) == "implement_initial_logic"
    assert (
        should_run_unit_tests({"impl_check_output": "error", "syntax_error_iterations": 3}) == "update_design_for_req"
    )
    assert should_run_unit_tests({"impl_check_output": ""}) == "run_unit_tests"

    # should_run_integration_tests
    assert should_run_integration_tests({"impl_check_output": "error"}) == "implement_integration_logic"
    assert (
        should_run_integration_tests({"impl_check_output": "error", "syntax_error_iterations": 3})
        == "update_design_for_req"
    )
    assert should_run_integration_tests({"impl_check_output": ""}) == "run_integration_tests"

    # should_run_regression_tests
    assert should_run_regression_tests({"impl_check_output": "error"}) == "implement_regression_logic"
    assert (
        should_run_regression_tests({"impl_check_output": "error", "syntax_error_iterations": 3})
        == "update_design_for_req"
    )
    assert should_run_regression_tests({"impl_check_output": ""}) == "run_regression_tests"

    # should_run_regression_after_refactor
    assert should_run_regression_after_refactor({"impl_check_output": "error"}) == "refactor_logic"
    assert (
        should_run_regression_after_refactor({"impl_check_output": "error", "syntax_error_iterations": 3})
        == "generate_refactor_bug_report"
    )
    assert should_run_regression_after_refactor({"impl_check_output": ""}) == "run_regression_tests"

    # should_continue_unit
    assert should_continue_unit({"success": True}) == "plan_integration_tests"
    assert should_continue_unit({"success": False, "iterations": 5, "max_iterations": 5}) == END
    assert should_continue_unit({"success": False, "iterations": 2, "max_iterations": 5}) == "generate_unit_bug_report"

    # should_continue_integration
    assert should_continue_integration({"success": True}) == "run_regression_tests"
    assert should_continue_integration({"success": False, "iterations": 5, "max_iterations": 5}) == END
    assert (
        should_continue_integration({"success": False, "iterations": 2, "max_iterations": 5})
        == "generate_integration_bug_report"
    )

    # should_continue_regression
    assert should_continue_regression({"success": True, "refactor_decision": "refactor"}) == "increment_requirement"
    assert should_continue_regression({"success": True, "refactor_decision": "continue"}) == "decide_refactor"
    assert (
        should_continue_regression({"success": False, "refactor_decision": "refactor"})
        == "generate_refactor_bug_report"
    )
    assert (
        should_continue_regression(
            {"success": False, "refactor_decision": "continue", "iterations": 5, "max_iterations": 5}
        )
        == END
    )
    assert (
        should_continue_regression(
            {"success": False, "refactor_decision": "continue", "iterations": 2, "max_iterations": 5}
        )
        == "generate_regression_bug_report"
    )

    # should_route_from_audit
    assert should_route_from_audit({"next_action": "implement_initial_logic"}) == "implement_initial_logic"
    assert should_route_from_audit({"next_action": "update_design_for_req"}) == "update_design_for_req"
    assert should_route_from_audit({}) == "update_design_for_req"

    # should_fix_unit_tests_or_impl
    with patch("agent._detect_toggle_loop", return_value=True):
        assert should_fix_unit_tests_or_impl({"next_action": "implement_initial_logic"}) == "analyze_architecture"
    with patch("agent._detect_toggle_loop", return_value=False):
        assert should_fix_unit_tests_or_impl({"next_action": "implement_initial_logic"}) == "implement_initial_logic"
    assert should_fix_unit_tests_or_impl({"next_action": "generate_tests"}) == "generate_unit_tests"

    # should_fix_integration_tests_or_impl
    with patch("agent._detect_toggle_loop", return_value=True):
        assert (
            should_fix_integration_tests_or_impl({"next_action": "implement_integration_logic"})
            == "analyze_architecture"
        )
        assert should_fix_integration_tests_or_impl({"next_action": "implement_logic"}) == "analyze_architecture"
    with patch("agent._detect_toggle_loop", return_value=False):
        assert (
            should_fix_integration_tests_or_impl({"next_action": "implement_integration_logic"})
            == "implement_integration_logic"
        )
        assert should_fix_integration_tests_or_impl({"next_action": "implement_logic"}) == "implement_integration_logic"
    assert should_fix_integration_tests_or_impl({"next_action": "generate_tests"}) == "generate_integration_tests"

    # should_fix_regression_tests_or_impl
    with patch("agent._detect_toggle_loop", return_value=True):
        assert (
            should_fix_regression_tests_or_impl({"next_action": "implement_regression_logic"}) == "analyze_architecture"
        )
        assert should_fix_regression_tests_or_impl({"next_action": "implement_logic"}) == "analyze_architecture"
    with patch("agent._detect_toggle_loop", return_value=False):
        assert (
            should_fix_regression_tests_or_impl({"next_action": "implement_regression_logic"})
            == "implement_regression_logic"
        )
        assert should_fix_regression_tests_or_impl({"next_action": "implement_logic"}) == "implement_regression_logic"
    assert should_fix_regression_tests_or_impl({"next_action": "generate_design"}) == "update_design_for_req"
    # Test should_fix_regression_tests_or_impl when next_action is generate_tests
    # Case 1: active requirement unit test failed
    state_unit = TDDState(
        next_action="generate_tests",
        requirements=[{"id": "REQ004"}],
        current_req_index=0,
        test_output="FAILED test_bc_clone_req004_unit.py::test_set_get_scale",
    )
    assert should_fix_regression_tests_or_impl(state_unit) == "generate_unit_tests"

    # Case 2: active requirement integration test failed
    state_integration = TDDState(
        next_action="generate_tests",
        requirements=[{"id": "REQ004"}],
        current_req_index=0,
        test_output="FAILED test_bc_clone_req004_integration.py::test_scale_precision",
    )
    assert should_fix_regression_tests_or_impl(state_integration) == "generate_integration_tests"

    # Case 3: fallback to unit test when "_unit" in test_output
    state_fallback_unit = TDDState(
        next_action="generate_tests",
        test_output="some random failure in a _unit test file",
    )
    assert should_fix_regression_tests_or_impl(state_fallback_unit) == "generate_unit_tests"

    # Case 4: fallback to integration tests (none of the above matched)
    state_fallback_integration = TDDState(
        next_action="generate_tests",
        test_output="some random failure",
    )
    assert should_fix_regression_tests_or_impl(state_fallback_integration) == "generate_integration_tests"

    # Case 5: previous requirement unit test failed (Circuit Breaker)
    state_circuit_breaker = TDDState(
        next_action="generate_tests",
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}, {"id": "REQ004"}],
        current_req_index=3,
        test_output="FAILED test_bc_clone_req001_unit.py::test_evaluate_ast_scale_zero_addition",
        regression_failure_policy="halt",
    )
    assert should_fix_regression_tests_or_impl(state_circuit_breaker) == "halt_regression_test_failure"

    # should_fix_refactor_or_continue
    assert should_fix_refactor_or_continue({}) == "refactor_logic"
    assert should_fix_refactor_or_continue({"next_action": "rollback_continue"}) == "increment_requirement"

    # should_refactor
    assert should_refactor({"refactor_decision": "refactor"}) == "refactor_logic"
    assert should_refactor({"refactor_decision": "continue"}) == "increment_requirement"

    # should_continue_workflow
    assert (
        should_continue_workflow({"requirements": [{"id": "REQ001"}, {"id": "REQ002"}], "current_req_index": 1})
        == "update_design_for_req"
    )
    assert (
        should_continue_workflow({"requirements": [{"id": "REQ001"}, {"id": "REQ002"}], "current_req_index": 2})
        == "generate_readme"
    )


def test_agent_graph_compile():
    agent = TDDAgent()
    graph = agent.get_graph()
    assert graph is not None


def test_agent_additional_coverage(tmp_path, monkeypatch):
    import os
    import subprocess
    from unittest.mock import MagicMock, patch

    import agent
    import config
    from schema import OracleAssertionTarget, TDDState, TestPlan, TestPlanReviewReport

    # 1. save_history_snapshot req_id fallback (line 113) and config.VERBOSE (line 133)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "VERBOSE", True)

    state_no_req_id = TDDState(current_req_index=0)  # no requirements list
    agent.save_history_snapshot("app.py", "print(1)", 1, state=state_no_req_id)
    assert os.path.exists(tmp_path / "history" / "app_req001_test_iter001_impl_iter001.py")

    # save_history_snapshot exception (line 135)
    with patch("builtins.open", side_effect=PermissionError("Mock Permission Error")):
        # Should catch exception and not crash
        agent.save_history_snapshot("app.py", "print(1)", 1, state=state_no_req_id)

    # 2. _call_llm_structured fallback/data validations (lines 157-161)
    # schema has no model_validate (returns dict)
    with patch("utils.call_llm_standard", return_value='{"key": "val"}'):
        res = agent._call_llm_structured("prompt", dict, model_name="secondary")
        assert res == {"key": "val"}

    # exception parsing LLM structured response
    with patch("utils.call_llm_standard", return_value="invalid json"):
        import pytest

        with pytest.raises(Exception):
            agent._call_llm_structured("prompt", TestPlan, model_name="secondary")

    # response is not string
    with patch("utils.call_llm_standard", return_value={"mock_key": "mock_val"}):
        assert agent._call_llm_structured("prompt", TestPlan, model_name="secondary") is not None

    # 3. _call_llm_text (line 172)
    with patch("utils.call_llm_standard", return_value="```python\n# code\n```"):
        assert agent._call_llm_text("prompt", model_name="secondary") == "# code"

    # 4. _execute_tests_helper maxfail, timeout, output truncation, config.VERBOSE
    # maxfail > 0
    monkeypatch.setattr(config, "PYTEST_MAXFAIL", 2)
    monkeypatch.setattr(config, "VERBOSE", True)

    import signal

    # simulate pytest timeout
    mock_process_timeout = MagicMock()
    mock_process_timeout.pid = 12345
    mock_process_timeout.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="pytest", timeout=5),
        ("partial stdout", "partial stderr"),
    ]

    with patch("subprocess.Popen", return_value=mock_process_timeout), patch("os.killpg") as mock_killpg:
        res = agent._execute_tests_helper("test_app.py", state_no_req_id)
        assert "timed out" in res["test_output"]
        assert "partial stdout" in res["test_output"]
        assert res["success"] is False
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    # simulate pytest timeout with OSError on killpg
    mock_process_timeout_err = MagicMock()
    mock_process_timeout_err.pid = 12345
    mock_process_timeout_err.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="pytest", timeout=5),
        ("partial stdout", "partial stderr"),
    ]

    with (
        patch("subprocess.Popen", return_value=mock_process_timeout_err),
        patch("os.killpg", side_effect=OSError("No such process")),
    ):
        res = agent._execute_tests_helper("test_app.py", state_no_req_id)
        assert "timed out" in res["test_output"]
        assert res["success"] is False

    # simulate pytest timeout with no killpg (Windows/non-posix compatibility path)
    mock_process_timeout_no_killpg = MagicMock()
    mock_process_timeout_no_killpg.pid = 12345
    mock_process_timeout_no_killpg.communicate.side_effect = [
        subprocess.TimeoutExpired(cmd="pytest", timeout=5),
        ("partial stdout", "partial stderr"),
    ]

    import os as real_os

    if hasattr(real_os, "killpg"):
        orig_killpg = real_os.killpg
        del real_os.killpg
    else:
        orig_killpg = None

    try:
        with patch("subprocess.Popen", return_value=mock_process_timeout_no_killpg):
            res = agent._execute_tests_helper("test_app.py", state_no_req_id)
            assert "timed out" in res["test_output"]
            assert res["success"] is False
            mock_process_timeout_no_killpg.kill.assert_called_once()
    finally:
        if orig_killpg is not None:
            real_os.killpg = orig_killpg

    # output truncation (> 8000 chars)
    long_output = "A" * 9000
    mock_process_normal = MagicMock()
    mock_process_normal.returncode = 0
    mock_process_normal.communicate.return_value = (long_output, "")

    with patch("subprocess.Popen", return_value=mock_process_normal) as mock_popen:
        res = agent._execute_tests_helper("test_app.py", state_no_req_id)
        assert len(res["test_output"]) < 9000
        assert "TRUNCATED" in res["test_output"]
        assert res["success"] is True
        assert res["audit_loop_count"] == 0
        mock_popen.assert_called_once()
        passed_env = mock_popen.call_args[1].get("env", {})
        assert passed_env.get("TDD_ROBO_DEBUG") == config.TDD_ROBO_DEBUG

    # general exception in Popen
    with patch("subprocess.Popen", side_effect=ValueError("General Popen error")):
        res = agent._execute_tests_helper("test_app.py", state_no_req_id)
        assert "Failed to execute tests" in res["test_output"]
        assert "General Popen error" in res["test_output"]
        assert res["success"] is False

    # 5. _syntax_check_helper with syntax error (lines 254-256)
    with patch("agent._run_syntax_check", return_value="syntax error details"):
        res = agent._syntax_check_helper("app.py", "syntax_error_iterations", state_no_req_id)
        assert res["syntax_error_iterations"] == 1
        assert res["impl_check_output"] == "syntax error details"

    # 6. _run_syntax_check exceptions & verbose output
    # timeout
    def mock_run_syntax_timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=5)

    with patch("subprocess.run", side_effect=mock_run_syntax_timeout):
        out = agent._run_syntax_check("app.py", "app.py")
        assert "timed out" in out

    # generic exception
    with patch("subprocess.run", side_effect=ValueError("General Value Error")):
        out = agent._run_syntax_check("app.py", "app.py")
        assert "General Value Error" in out

    # 7. _get_combined_tests_code (lines 825-843)
    # read from files
    test_dir = tmp_path
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(test_dir))
    with open(test_dir / "test_unit.py", "w", encoding="utf-8") as f:
        f.write("def test_one(): pass")
    with open(test_dir / "test_integ.py", "w", encoding="utf-8") as f:
        f.write("def test_two(): pass")

    orig_open = open

    def mock_open(file, *args, **kwargs):
        if "test_integ.py" in str(file):
            raise PermissionError("Unreadable file")
        return orig_open(file, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open):
        combined = agent._get_combined_tests_code(state_no_req_id)
        assert "test_one" in combined
        assert "test_two" not in combined

    # fallback to state code keys if no files exist
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path / "non_existent_directory_123"))
    state_with_keys = TDDState(unit_tests_code="unit_code", integration_tests_code="integ_code")
    combined_fallback = agent._get_combined_tests_code(state_with_keys)
    assert "unit_code" in combined_fallback
    assert "integ_code" in combined_fallback

    # test _get_existing_tests_context
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(test_dir))
    state_for_existing = TDDState(test_module_name="test_unit.py")
    # test_integ.py will raise PermissionError, so write test_other.py
    with open(test_dir / "test_other.py", "w", encoding="utf-8") as f:
        f.write("def test_three(): pass")
    with patch("builtins.open", side_effect=mock_open):
        existing_tests_context = agent._get_existing_tests_context(state_for_existing)
        assert "test_three" in existing_tests_context
        assert "test_one" not in existing_tests_context  # Excluded since test_unit.py is active

    # Clean up test_other.py so it doesn't pollute subsequent tests
    try:
        os.remove(test_dir / "test_other.py")
    except Exception:
        pass

    # test empty/non-existent directory fallback for _get_existing_tests_context
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path / "non_existent_directory_456"))
    assert agent._get_existing_tests_context(state_for_existing) == ""

    # 8. Out of bounds current_req_index in various nodes (returns {})
    state_out_of_bounds = TDDState(requirements=[], current_req_index=0)
    assert agent.update_design_for_req(state_out_of_bounds) == {}
    assert agent.plan_unit_tests(state_out_of_bounds) == {}
    assert agent.review_unit_test_plan(state_out_of_bounds) == {}
    assert agent.plan_integration_tests(state_out_of_bounds) == {}
    assert agent.review_integration_test_plan(state_out_of_bounds) == {}
    assert agent.generate_unit_tests(state_out_of_bounds) == {}
    assert agent.generate_integration_tests(state_out_of_bounds) == {}
    assert agent.generate_unit_bug_report(state_out_of_bounds) == {}
    assert agent.generate_integration_bug_report(state_out_of_bounds) == {}
    assert agent.generate_regression_bug_report(state_out_of_bounds) == {}

    # 9. plan_unit_tests and plan_integration_tests PLAN_PROMPT_FIX paths (lines 536 and 626)
    state_plan_fix = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_review="some feedback",
        unit_test_plan="{}",
        integration_test_plan="{}",
        design_updated=False,
    )
    with patch("agent._call_llm_structured", return_value=TestPlan(test_cases=[])):
        res_unit = agent.plan_unit_tests(state_plan_fix)
        res_integ = agent.plan_integration_tests(state_plan_fix)
        assert "unit_test_plan" in res_unit
        assert "integration_test_plan" in res_integ

    # 9b. plan_unit_tests and plan_integration_tests exception and merge paths
    state_plan_fix_exception = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_review="some feedback",
        unit_test_plan="invalid json",
        integration_test_plan="invalid json",
        design_updated=False,
    )
    with patch("agent._call_llm_structured", return_value=TestPlan(test_cases=[])):
        agent.plan_unit_tests(state_plan_fix_exception)
        agent.plan_integration_tests(state_plan_fix_exception)

    state_plan_fix_merge = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_review="some feedback",
        unit_test_plan='{"test_cases": [{"action": "Existing Case", "expected_outcome": "Outcome"}]}',
        integration_test_plan='{"test_cases": [{"action": "Existing Case", "expected_outcome": "Outcome"}]}',
        design_updated=False,
    )
    with patch(
        "agent._call_llm_structured",
        return_value=TestPlan(
            test_cases=[
                TestCase(action="Existing Case", expected_outcome="Outcome"),
                TestCase(action="New Case", expected_outcome="New Outcome"),
            ]
        ),
    ):
        res_unit = agent.plan_unit_tests(state_plan_fix_merge)
        res_integ = agent.plan_integration_tests(state_plan_fix_merge)
        assert "New Case" in res_unit["test_plan"]
        assert "New Case" in res_integ["test_plan"]

    # 9c. plan_unit_tests/plan_integration_tests with design_updated=True (is_fix should be True)
    state_plan_design_updated = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_review="some feedback",
        unit_test_plan="{}",
        integration_test_plan="{}",
        design_updated=True,
    )
    captured_prompts = []

    def mock_call_llm(prompt, schema, model_name=None):
        captured_prompts.append(prompt)
        return TestPlan(test_cases=[])

    with patch("agent._call_llm_structured", side_effect=mock_call_llm):
        agent.plan_unit_tests(state_plan_design_updated)
        agent.plan_integration_tests(state_plan_design_updated)

    assert len(captured_prompts) == 2
    assert "# Previous Test Plan" in captured_prompts[0]
    assert "# Previous Test Plan" in captured_prompts[1]

    # 10. review test plan looping branch with coverage < target and iters >= max
    state_review_loop = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_iterations=3,
        max_test_plan_iterations=3,
        target_test_plan_coverage=95,
    )
    with patch(
        "agent._call_llm_structured",
        return_value=TestPlanReviewReport(missing_test_cases=[], estimated_coverage=50, feedback="poor"),
    ):
        res_review_unit = agent.review_unit_test_plan(state_review_loop)
        assert res_review_unit["test_plan_review_decision"] == "continue"

        res_review_integ = agent.review_integration_test_plan(state_review_loop)
        assert res_review_integ["test_plan_review_decision"] == "continue"

    # review test plan looping branch with coverage < target and iters < max
    state_review_loop_low = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_iterations=1,
        max_test_plan_iterations=3,
        target_test_plan_coverage=95,
    )
    with patch(
        "agent._call_llm_structured",
        return_value=TestPlanReviewReport(missing_test_cases=[], estimated_coverage=50, feedback="poor"),
    ):
        res_review_unit = agent.review_unit_test_plan(state_review_loop_low)
        assert res_review_unit["test_plan_review_decision"] == "review_test_plan"

        res_review_integ = agent.review_integration_test_plan(state_review_loop_low)
        assert res_review_integ["test_plan_review_decision"] == "review_test_plan"

    # 11. _implement_logic_helper Search/Replace matching multiple times context hints (lines 911-1016)
    error_msg = "Failed to apply Search/Replace block\nmatches multiple times. Matches found at line numbers: [2, 5]"
    state_sr_mult = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        impl_code="line1\nline2\nline3\nline4\nline5\nline6",
        impl_check_output=error_msg,
        module_name="app.py",
        test_module_name="test_app.py",
    )
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    (tmp_path / "app.py").write_text("line1\nline2\nline3\nline4\nline5\nline6")
    (tmp_path / "test_app.py").write_text("def test_app(): pass")

    with (
        patch("subprocess.run", return_value=MagicMock(returncode=1)),
        patch("agent.call_llm_with_reasoning", return_value="```python\ndef add(a,b): return a+b\n```"),
    ):
        res = agent._implement_logic_helper(state_sr_mult, "unit")
        assert res is not None

    # Test Search/Replace failed block extraction and display in feedback
    error_msg_failed_block = (
        "Failed to apply Search/Replace block: Target SEARCH block that failed to match:\n"
        "def add(a, b): return a + b\n"
        "[TDD Robo] info context"
    )
    state_sr_failed_block = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        impl_code="def add(a, b): return a + b\n",
        impl_check_output=error_msg_failed_block,
        module_name="app.py",
        test_module_name="test_app.py",
    )
    with (
        patch("subprocess.run", return_value=MagicMock(returncode=1)),
        patch("agent.call_llm_with_reasoning", return_value="```python\ndef add(a,b): return a+b\n```"),
    ):
        res_sr_fb = agent._implement_logic_helper(state_sr_failed_block, "unit")
        assert res_sr_fb is not None

    # 12. _parse_pytest_summary (lines 1169-1170, 1172)
    assert agent._parse_pytest_summary("=== 1 passed ===") == "1 passed"
    assert agent._parse_pytest_summary("=== 1 failed ===") == "1 failed"
    assert agent._parse_pytest_summary("=== no summary ===") == ""

    # 13. _detect_toggle_loop detailed branches (lines 1629-1634, 1639, 1659, 1668, 1678-1706)
    # iterations >= 8
    state_loop_8 = TDDState(iterations=8)
    assert agent._detect_toggle_loop(state_loop_8) is True

    # history_dir doesn't exist
    monkeypatch.setattr(config, "ARTIFACTS_DIR", "/non_existent_dir_loop_test")
    state_no_hist = TDDState(module_name="app.py")
    assert agent._detect_toggle_loop(state_no_hist) is False

    # req_id None fallback & pattern A_iter*.py
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    history_dir = tmp_path / "history"
    os.makedirs(history_dir, exist_ok=True)
    # create files that match A_iter*.py
    (history_dir / "app_iter001.py").write_text("content A")
    (history_dir / "app_iter002.py").write_text("content B")
    (history_dir / "app_iter003.py").write_text("content A")
    (history_dir / "app_iter004.py").write_text("content B")
    state_fallback_loop = TDDState(module_name="app.py")
    if "current_req_index" in state_fallback_loop:
        del state_fallback_loop["current_req_index"]
    if "requirements" in state_fallback_loop:
        del state_fallback_loop["requirements"]
    assert agent._detect_toggle_loop(state_fallback_loop) is True

    # repeat state A-X-A-Y-A loop
    (history_dir / "app_iter001.py").write_text("content A")
    (history_dir / "app_iter002.py").write_text("content X")
    (history_dir / "app_iter003.py").write_text("content A")
    (history_dir / "app_iter004.py").write_text("content Y")
    (history_dir / "app_iter005.py").write_text("content A")
    assert agent._detect_toggle_loop(state_fallback_loop) is True

    # 14. _run_oracle_verification_on_failures (lines 1216-1381)
    def mock_extract_oracle_target(prompt, schema, model_name=None):
        import re

        failing_line_match = re.search(r"<failing_line>\s*(.*?)\s*</failing_line>", prompt, re.DOTALL)
        failing_line = failing_line_match.group(1).strip() if failing_line_match else ""
        expr = "1 + 2"
        expected = "3"
        preceding = []

        match_direct = re.search(r"calc\.evaluate\('([^']+)'\)\s*==\s*\[?'([^']+)'\]?", failing_line)
        if not match_direct:
            match_direct = re.search(r"interpreter\.execute\('([^']+)'\)\s*==\s*'([^']+)'", failing_line)
        if not match_direct:
            match_direct = re.search(r"run_bc_clone\('([^']+)'\)\s*==\s*'([^']+)'", failing_line)

        if match_direct:
            expr = match_direct.group(1)
            expected = match_direct.group(2)
        else:
            method_body_match = re.search(r"<method_body>\s*(.*?)\s*</method_body>", prompt, re.DOTALL)
            method_body = method_body_match.group(1) if method_body_match else ""
            found_body = False

            match_body = re.search(r"calc\.evaluate\('([^']+)'\)\s*==\s*\[?'([^']+)'\]?", method_body)
            if match_body:
                expr = match_body.group(1)
                expected = match_body.group(2)
                found_body = True
            else:
                match_body = re.search(r"interpreter\.execute\('([^']+)'\)\s*==\s*'([^']+)'", method_body)
                if match_body:
                    expr = match_body.group(1)
                    expected = match_body.group(2)
                    found_body = True
                else:
                    match_body = re.search(r"run_bc_clone\('([^']+)'\)\s*==\s*'([^']+)'", method_body)
                    if match_body:
                        expr = match_body.group(1)
                        expected = match_body.group(2)
                        found_body = True
                    else:
                        code_match = re.search(r"code\s*=\s*'([^']+)'", method_body)
                        expected_match = re.search(r"expected\s*=\s*\[?'([^']+)'\]?", method_body)
                        if code_match and expected_match:
                            expr = code_match.group(1)
                            expected = expected_match.group(1)
                            found_body = True

            if found_body:
                pass
            else:
                if "expected_sum" in failing_line:
                    expr = "123456789012345678901234567890 + 123456789012345678901234567890"
                    expected = "1111111110111111111111111111100"
                elif "stdout.strip() == expected" in failing_line or failing_line.strip().endswith("== expected"):
                    expr = "1234 + 5678"
                    expected = "1111"
                if "unresolved_var" in failing_line:
                    expr = "1 + 2"
                    expected = "unresolved_var"
                elif "stdout.strip() == '15'" in failing_line:
                    expr = "10 + 5"
                    expected = "15"
                elif "result == '5'" in failing_line:
                    if "ibase=2" in method_body:
                        expr = "101"
                        expected = "5"
                        preceding = ["ibase=2"]
                    else:
                        expr = "101"
                        expected = "101"

        return OracleAssertionTarget(expression=expr, expected=expected, preceding=preceding)

    with patch("agent._call_llm_structured", side_effect=mock_extract_oracle_target):
        # verifier is None
        monkeypatch.setattr(config, "ORACLE_VERIFIER", None)
        assert agent._run_oracle_verification_on_failures("output", "code") == ""

        # verifier exists
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "4")
        # test case body and traceback simulation
        test_output_sim = (
            "tests/test_app.py::test_add_two\n"
            "___________ test_add_two ___________\n"
            "> assert calc.evaluate('1 + 2') == ['3']\n"
            "E AssertionError\n"
            "=== short test summary info ===\n"
        )
        tests_code_sim = "def test_add_two():\n    calc.execute('1 + 1')\n    assert calc.evaluate('1 + 2') == ['3']\n"
        feedback = agent._run_oracle_verification_on_failures(test_output_sim, tests_code_sim)
        assert "ORACLE VERIFICATION FEEDBACK" in feedback
        assert "mathematically INCORRECT" in feedback

        # verifier with exception path (line 1367)
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: exec('raise ValueError("Verifier Error")'))
        feedback_err = agent._run_oracle_verification_on_failures(test_output_sim, tests_code_sim)
        assert "mathematically INCORRECT" not in feedback_err

    # 15. Coverage for line 161 (res not string in _call_llm_structured)
    # We need this to return a dict for coverage of fallback, so we patch it temporarily here
    with patch("utils.call_llm_standard", return_value={"some_dict": 1}):
        res_dict = agent._call_llm_structured("prompt", TestPlan, model_name="secondary")
        assert res_dict == {"some_dict": 1}

    # 16. Coverage for line 305 (fetch_spec cache print in verbose mode)
    # 17. Coverage for line 356 (generate_requirements verbose print)
    state_reqs = TDDState(spec_content="Some spec", spec_url="http://example.com/spec")
    with patch(
        "agent.call_llm_standard", return_value='{"requirements": [{"id": "REQ001", "description": "req desc"}]}'
    ):
        monkeypatch.setattr(config, "VERBOSE", True)
        res_reqs = agent.generate_requirements(state_reqs)
        assert len(res_reqs["requirements"]) == 1

    # 18. Coverage for lines 876-877 (_implement_logic_helper skip check exception)
    impl_name = "app.py"
    test_name = "test_app.py"
    (tmp_path / impl_name).write_text("print(1)")
    (tmp_path / test_name).write_text("def test_app(): pass")
    state_skip_exc = TDDState(
        module_name=impl_name,
        test_module_name=test_name,
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
    )
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    with patch("subprocess.run", side_effect=RuntimeError("Pytest execution failed")):
        with patch("agent.call_llm_with_reasoning", return_value="```python\n# dummy code\n```"):
            res = agent._implement_logic_helper(state_skip_exc, "unit")
            assert res is not None

    # 19. Coverage for line 886 (No active target requirement description)
    state_no_active_req = TDDState(
        module_name=impl_name,
        test_module_name=test_name,
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=5,
        bug_report="some bug",
    )
    with patch("agent.call_llm_with_reasoning", return_value="```python\n# dummy code\n```"):
        res = agent._implement_logic_helper(state_no_active_req, "unit")
        assert res is not None

    # Clean up any existing test_*.py files in tmp_path so that existing_tests is empty for integration phase
    import glob

    for p in glob.glob(os.path.join(str(tmp_path), "test_*.py")):
        try:
            os.remove(p)
        except Exception:
            pass

    # 20. Coverage for lines 899-902 (phase integration and regression paths, design_updated param check)
    state_integ = TDDState(
        module_name=impl_name,
        test_module_name=test_name,
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        integration_tests_code="integration test code",
        design_updated=True,
        bug_report="some bug",
    )
    with patch("agent.call_llm_with_reasoning", return_value="```python\n# dummy code\n```"):
        res_integ = agent._implement_logic_helper(state_integ, "integration")
        assert res_integ is not None

        # Coverage for integration phase when existing_tests is NOT empty (line 1184)
        with open(tmp_path / "test_other_integ.py", "w", encoding="utf-8") as f:
            f.write("def test_dummy(): pass")
        res_integ_non_empty = agent._implement_logic_helper(state_integ, "integration")
        assert res_integ_non_empty is not None
        try:
            os.remove(tmp_path / "test_other_integ.py")
        except Exception:
            pass

        res_regr = agent._implement_logic_helper(state_integ, "regression")
        assert res_regr is not None

    # 21. Coverage for lines 944-945, 950, 979, 991
    # (Search/Replace matching multiple times, block_len check, idx < 0, and pre/post lines offset)
    error_msg_mult = (
        "Failed to apply Search/Replace block\nmatches multiple times. Matches found at line numbers: [0, 4]"
    )
    state_sr_mult_detailed = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        impl_code="X\nX\nX\nA\nX\nX\nX",
        impl_check_output=error_msg_mult,
        module_name="app.py",
        test_module_name="test_app.py",
        bug_report="some bug",
    )
    (tmp_path / "app.py").write_text("X\nX\nX\nA\nX\nX\nX")
    with patch("agent.call_llm_with_reasoning", return_value="```python\n# dummy\n```"):
        res_sr = agent._implement_logic_helper(state_sr_mult_detailed, "unit")
        assert res_sr is not None

    # 22. Coverage for lines 1063-1068 (loop warning prompt modification in _implement_logic_helper)
    state_loop_warning = TDDState(
        module_name=impl_name,
        test_module_name=test_name,
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        loop_detected=True,
        bug_report="some bug",
    )
    with patch("agent.call_llm_with_reasoning", return_value="```python\n# dummy code\n```"):
        res_loop_w = agent._implement_logic_helper(state_loop_warning, "unit")
        assert res_loop_w is not None

    # 23. Coverage for lines 1086-1090 and 1093-1098 (search-replace syntax error & fallback with raw error)
    state_syntax_err = TDDState(
        module_name="app.py",
        test_module_name="test_app.py",
        impl_code="def add(a, b): return a + b\n",
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        bug_report="some bug",
    )
    (tmp_path / "app.py").write_text("def add(a, b): return a + b\n")
    (tmp_path / "test_app.py").write_text("def test_app(): pass")
    bad_sr_response = "<<<<<<< SEARCH\ndef add(a, b): return a + b\n=======\ndef add(a, b): return a +\n>>>>>>> REPLACE"
    with patch("agent.call_llm_with_reasoning", return_value=bad_sr_response):
        res = agent._implement_logic_helper(state_syntax_err, "unit")
        assert "SyntaxError" in res["impl_check_output"]
        assert res["impl_code"] == "def add(a, b): return a + b"

    # 24. Coverage for lines 1232, 1241, 1253-1254, 1256, 1261, 1274, 1314-1315,
    # 1327-1333, 1340, 1346-1348 (_run_oracle_verification_on_failures paths)
    with patch("agent._call_llm_structured", side_effect=mock_extract_oracle_target):
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3")

        # Empty failed method name
        assert agent._run_oracle_verification_on_failures("no test failed", "code") == ""

        # Missing method in code
        assert (
            agent._run_oracle_verification_on_failures("tests/test_app.py::test_missing", "code")
            == "No assertion discrepancies detected by the oracle."
        )

        # Multiple tracebacks splitter, indentation bounds, empty lines, class/def breaks early
        test_output_mult = (
            "___________ test_math ___________\n"
            "> assert calc.evaluate('1 + 2') == ['4']\n"
            "E AssertionError\n"
            "___________ test_other ___________\n"
            "> assert False\n"
            "E AssertionError\n"
            "___________ test_end ___________\n"
        )
        tests_code_mult = (
            "def test_math():\n"
            "    assert calc.evaluate('1 + 2') == ['4']\n"
            "\n"
            "def test_other():\n"
            "    calc.evaluate('1 + 2') == ['4']\n"
            "x = 1\n"
        )
        feedback_mult = agent._run_oracle_verification_on_failures(test_output_mult, tests_code_mult)
        assert "ORACLE VERIFICATION" in feedback_mult

        # Parameterized code match & preceding expression cleaning & empty/cleaned preceding expressions
        test_output_param = (
            "___________ test_param ___________\n> assert calc.evaluate('1 + 2') == ['4']\nE AssertionError\n"
        )
        tests_code_param = (
            "def test_param():\n"
            "    code = '1 + 2'\n"
            "    expected = ['4']\n"
            "    calc.evaluate('1 + 2')\n"
            "    assert calc.evaluate(code) == expected\n"
        )
        feedback_param = agent._run_oracle_verification_on_failures(test_output_param, tests_code_param)
        assert "ORACLE VERIFICATION" in feedback_param

        # Fallback to method body level matching
        test_output_body = "___________ test_body ___________\n>     assert False\nE AssertionError\n"
        tests_code_body = "def test_body():\n    calc.evaluate('1 + 2') == ['4']\n    assert False\n"
        feedback_body = agent._run_oracle_verification_on_failures(test_output_body, tests_code_body)
        assert "ORACLE VERIFICATION" in feedback_body

        # CLI/Integration test pattern matching
        test_output_cli = (
            "___________ test_large_integer_addition ___________\n"
            ">     assert stdout.strip() == expected\n"
            "E AssertionError\n"
        )
        tests_code_cli = (
            "def test_large_integer_addition():\n"
            "    val1 = '1234'\n"
            "    val2 = '5678'\n"
            "    expected = '1111'\n"
            "    stdout, stderr = run_bc_clone(f'{val1} + {val2}\\n')\n"
            "    assert stdout.strip() == expected\n"
        )
        # Verifier mock returns different result, so it finds mismatch
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "6912")
        feedback_cli = agent._run_oracle_verification_on_failures(test_output_cli, tests_code_cli)
        assert "ORACLE VERIFICATION" in feedback_cli
        assert "mathematically INCORRECT" in feedback_cli

        # CLI/Integration test with direct string assertion
        test_output_cli_direct = (
            "___________ test_addition_direct ___________\n>     assert stdout.strip() == '15'\nE AssertionError\n"
        )
        tests_code_cli_direct = (
            "def test_addition_direct():\n"
            "    stdout, stderr = run_bc_clone('10 + 5')\n"
            "    assert stdout.strip() == '15'\n"
        )
        feedback_cli_direct = agent._run_oracle_verification_on_failures(test_output_cli_direct, tests_code_cli_direct)
        assert "ORACLE VERIFICATION" in feedback_cli_direct
        assert "mathematically INCORRECT" in feedback_cli_direct

        # Test path for assert_match where preceding_exprs exists
        test_output_assert_match_preceding = (
            "___________ test_assert_match_preceding ___________\n>     assert result == '5'\nE AssertionError\n"
        )
        tests_code_assert_match_preceding = (
            "def test_assert_match_preceding():\n"
            "    interpreter.execute('ibase=2')\n"
            "    result = interpreter.execute('101')\n"
            "    assert result == '5'\n"
        )
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "101")
        feedback_amp = agent._run_oracle_verification_on_failures(
            test_output_assert_match_preceding, tests_code_assert_match_preceding
        )
        assert "ORACLE VERIFICATION" in feedback_amp
        assert "mathematically INCORRECT" in feedback_amp

        # Test path for assert_match where preceding_exprs is empty
        test_output_assert_match_empty = (
            "___________ test_assert_match_empty ___________\n>     assert result == '5'\nE AssertionError\n"
        )
        tests_code_assert_match_empty = "def test_assert_match_empty():\n    assert result == '5'\n"
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "101")
        feedback_ame = agent._run_oracle_verification_on_failures(
            test_output_assert_match_empty, tests_code_assert_match_empty
        )
        # Since preceding_exprs is empty and expr is None, it should not find mismatch
        assert feedback_ame == "No assertion discrepancies detected by the oracle."

        # Test path for assert_direct: assert interpreter.execute("...") == "..."
        test_output_assert_direct = (
            "___________ test_assert_direct ___________\n"
            ">     assert interpreter.execute('10/3') == '3'\n"
            "E AssertionError\n"
        )
        tests_code_assert_direct = "def test_assert_direct():\n    assert interpreter.execute('10/3') == '3'\n"
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3.33333")
        feedback_ad = agent._run_oracle_verification_on_failures(test_output_assert_direct, tests_code_assert_direct)
        assert "ORACLE VERIFICATION" in feedback_ad
        assert "mathematically INCORRECT" in feedback_ad

        # Test path for variable-based f-string assertion: assert interpreter.execute(f'{v1} + {v2}') == expected_sum
        test_output_new = (
            "___________ test_large_addition ___________\n"
            ">     assert interpreter.execute(f'{v1} + {v2}') == expected_sum\n"
            "E AssertionError\n"
        )
        tests_code_new = (
            "def test_large_addition():\n"
            "    v1 = '123456789012345678901234567890'\n"
            "    v2 = f'{v1}'\n"
            "    var_list = [5]\n"
            "    var_num = 123\n"
            "    expected_sum = '1111111110111111111111111111100'\n"
            "    assert interpreter.execute(f'{v1} + {v2}') == expected_sum\n"
        )
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "1111111110111111111011111111100")
        feedback_new = agent._run_oracle_verification_on_failures(test_output_new, tests_code_new)
        assert "ORACLE VERIFICATION" in feedback_new
        assert "mathematically INCORRECT" in feedback_new
        assert "Expected value hardcoded in test: `1111111110111111111111111111100`" in feedback_new
        assert "Actual correct oracle value: `1111111110111111111011111111100`" in feedback_new

        # Test path for resolving plain value fallthrough in resolve_val
        test_output_resolve_fallthrough = (
            "___________ test_resolve_fallthrough ___________\n"
            ">     assert result == unresolved_var\n"
            "E AssertionError\n"
        )
        tests_code_resolve_fallthrough = (
            "def test_resolve_fallthrough():\n"
            "    interpreter.execute('1 + 2')\n"
            "    result = '3'\n"
            "    assert result == unresolved_var\n"
        )
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "4")
        feedback_rf = agent._run_oracle_verification_on_failures(
            test_output_resolve_fallthrough, tests_code_resolve_fallthrough
        )
        assert "ORACLE VERIFICATION" in feedback_rf

        # Test path for code_match/expected_match fallback in parameterised test body (lines 1658-1659)
        test_output_pb = "___________ test_param_body ___________\n>     assert False\nE AssertionError\n"
        tests_code_pb = "def test_param_body():\n    code = '1 + 2'\n    expected = ['4']\n    assert False\n"
        monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3")
        feedback_pb = agent._run_oracle_verification_on_failures(test_output_pb, tests_code_pb)
        assert "ORACLE VERIFICATION" in feedback_pb

    # 25. Coverage for line 1582 (increment_requirement pending status)
    state_inc = TDDState(
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}],
        current_req_index=0,
    )
    res_inc = agent.increment_requirement(state_inc)
    assert res_inc["current_req_index"] == 1

    # 26. Coverage for line 1657 (_detect_toggle_loop fallback req_id)
    state_dt = TDDState(
        module_name="app.py",
        current_req_index=0,
    )
    if "requirements" in state_dt:
        del state_dt["requirements"]
    assert agent._detect_toggle_loop(state_dt) is False

    # 27. Coverage for lines 1686-1687 and 1690 (IsADirectoryError skip and len < 4)
    history_dir = tmp_path / "history"
    os.makedirs(history_dir, exist_ok=True)
    import glob

    for hist_file in glob.glob(str(history_dir / "*")):
        if os.path.isdir(hist_file):
            os.rmdir(hist_file)
        else:
            os.remove(hist_file)
    state_err_dt = TDDState(
        module_name="app.py",
        requirements=[{"id": "REQ001"}],
        current_req_index=0,
        test_iterations=1,
    )
    (history_dir / "app_req001_test_iter001_impl_iter001.py").write_text("content A")
    (history_dir / "app_req001_test_iter001_impl_iter002.py").write_text("content B")
    (history_dir / "app_req001_test_iter001_impl_iter003.py").write_text("content C")
    os.makedirs(history_dir / "app_req001_test_iter001_impl_iter004.py", exist_ok=True)
    assert agent._detect_toggle_loop(state_err_dt) is False

    # 28. Coverage for line 1704 (not matching toggle loop patterns)
    os.rmdir(history_dir / "app_req001_test_iter001_impl_iter004.py")
    (history_dir / "app_req001_test_iter001_impl_iter004.py").write_text("content D")
    assert agent._detect_toggle_loop(state_err_dt) is False

    # 29. Coverage for _syntax_check_helper with Search/Replace application failure
    state_sr_err = TDDState(impl_check_output="Failed to apply Search/Replace block: some error")
    res_sr_err = agent._syntax_check_helper("app.py", "syntax_error_iterations", state_sr_err)
    assert res_sr_err["syntax_error_iterations"] == 1
    assert "Failed to apply Search/Replace block" in res_sr_err["impl_check_output"]

    # 30. Coverage for generate_unit_tests with bug_report (reading from file path)
    (tmp_path / "test_app_req001_unit.py").write_text("def test_old(): pass")
    state_unit_bug = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        bug_report="Some test bug",
        test_module_name="test_app_req001_unit.py",
        module_name="app.py",
    )
    with patch("agent._call_llm_text", return_value="def test_new(): pass") as mock_call_llm:
        res_unit_bug = agent.generate_unit_tests(state_unit_bug)
        assert res_unit_bug["unit_tests_code"] == "def test_new(): pass"
        called_prompt = mock_call_llm.call_args[0][0]
        assert "def test_old(): pass" in called_prompt
        assert "Some test bug" in called_prompt

    # 31. Coverage for generate_integration_tests with bug_report (reading from file path)
    (tmp_path / "test_app_req001_integration.py").write_text("def test_old_integ(): pass")
    state_integ_bug = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        bug_report="Some integ bug",
        test_module_name="test_app_req001_integration.py",
        module_name="app.py",
    )
    with patch("agent._call_llm_text", return_value="def test_new(): pass") as mock_call_llm:
        res_integ_bug = agent.generate_integration_tests(state_integ_bug)
        assert res_integ_bug["integration_tests_code"] == "def test_new(): pass"
        called_prompt = mock_call_llm.call_args[0][0]
        assert "def test_old_integ(): pass" in called_prompt
        assert "Some integ bug" in called_prompt

    # 32. Coverage for refactor_logic with bug_report
    state_refactor_bug = TDDState(
        reasons=["cleanup"],
        bug_report="Some refactor bug",
    )
    with patch("agent._call_llm_text", return_value="def clean(): pass") as mock_call_llm:
        res_refactor_bug = agent.refactor_logic(state_refactor_bug)
        assert res_refactor_bug["impl_code"] == "def clean(): pass"
        called_prompt = mock_call_llm.call_args[0][0]
        assert "Some refactor bug" in called_prompt

    # 33. Coverage for generate_unit_tests & generate_integration_tests read from file exception
    state_unit_err = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        bug_report="Some test bug",
        test_module_name="test_app_req001_unit.py",
        module_name="app.py",
    )
    state_integ_err = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        bug_report="Some integ bug",
        test_module_name="test_app_req001_integration.py",
        module_name="app.py",
    )
    real_open = open

    def mock_open_impl(file, mode="r", *args, **kwargs):
        if mode == "r" and "test_app_req001" in str(file):
            raise PermissionError("mock open error")
        return real_open(file, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_impl):
        with patch("agent._call_llm_text", return_value="def test_new(): pass"):
            res_unit = agent.generate_unit_tests(state_unit_err)
            assert res_unit is not None
            res_integ = agent.generate_integration_tests(state_integ_err)
            assert res_integ is not None


def test_get_regression_test_code_context():
    import agent

    # We mock glob.glob and builtins.open
    test_files = [
        "artifacts/test_error.py",
        "artifacts/test_foo.py",
        "artifacts/test_history.py",
    ]

    def mock_open_impl(file, mode="r", *args, **kwargs):
        if "test_error" in str(file):
            raise IOError("mock open error")
        mock_file = MagicMock()
        mock_file.__enter__.return_value = mock_file
        mock_file.read.return_value = "def test_foo(): pass"
        return mock_file

    with patch("glob.glob", return_value=test_files):
        with patch("builtins.open", side_effect=mock_open_impl):
            res = agent._get_regression_test_code_context()
            assert "test_foo.py" in res
            assert "def test_foo(): pass" in res
            assert "test_history.py" not in res
            assert "test_error.py" not in res


def test_refactor_logic_bug_fix_search_replace():
    from unittest.mock import mock_open, patch

    import agent
    from schema import TDDState

    # Case 1: is_bug_fix = True (bug_report + impl_code exist),
    # Search/Replace block successfully applied
    state_bug_fix = TDDState(
        reasons=["cleanup"],
        bug_report="Some refactor bug",
        impl_code="def old_func():\n    return 42\n",
        module_name="app.py",
        python_tips="Ensure Python 3.14 compatibility.",
    )
    llm_response = (
        "<<<<<<< SEARCH\ndef old_func():\n    return 42\n=======\ndef old_func():\n    return 43\n>>>>>>> REPLACE"
    )

    with patch("agent._call_llm_text", return_value=llm_response) as m_llm:
        with patch("agent.save_artifact", return_value="artifacts/app.py"):
            with patch("agent.save_history_snapshot"):
                res = agent.refactor_logic(state_bug_fix)
                assert res["impl_code"] == "def old_func():\n    return 43\n"
                m_llm.assert_called_once()
                assert "Ensure Python 3.14" in m_llm.call_args[0][0]

    # Case 2: is_bug_fix = True, but impl_code is empty and loaded from file
    state_file_load = TDDState(
        reasons=["cleanup"],
        bug_report="Some refactor bug",
        impl_code="",
        module_name="app.py",
    )
    mock_file_data = "def file_func():\n    pass\n"
    llm_sr_response = (
        "<<<<<<< SEARCH\ndef file_func():\n    pass\n=======\ndef file_func():\n    return 1\n>>>>>>> REPLACE"
    )
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=mock_file_data)) as m_file,
        patch("agent._call_llm_text", return_value=llm_sr_response),
        patch("agent.save_artifact", return_value="artifacts/app.py"),
        patch("agent.save_history_snapshot"),
    ):
        res = agent.refactor_logic(state_file_load)
        assert res["impl_code"] == "def file_func():\n    return 1\n"
        m_file.assert_called_once()

    # Subcase 2b: open raises exception (covers 1680-1681)
    with (
        patch("os.path.exists", return_value=True),
        patch("builtins.open", side_effect=IOError("read error")),
        patch("agent._call_llm_text", return_value="def fallback(): pass"),
        patch("agent.save_artifact", return_value="artifacts/app.py"),
        patch("agent.save_history_snapshot"),
    ):
        res = agent.refactor_logic(state_file_load)
        assert res["impl_code"] == "def fallback(): pass"

    # Case 3: is_bug_fix = True, Search/Replace block application failed
    state_fail = TDDState(
        reasons=["cleanup"],
        bug_report="Some refactor bug",
        impl_code="def func(): pass",
        module_name="app.py",
        refactor_iterations=1,
    )
    # LLM response has SEARCH but the search target doesn't match original
    llm_response_fail = (
        "<<<<<<< SEARCH\ndef nonexistent():\n    pass\n=======\ndef nonexistent():\n    return 1\n>>>>>>> REPLACE"
    )
    # S/R fails, so it falls back to full code generation (calling LLM again)
    side_effects = [llm_response_fail, "def fallback_func():\n    pass"]
    with (
        patch("agent._call_llm_text", side_effect=side_effects),
        patch("agent.save_artifact", return_value="artifacts/app.py"),
        patch("agent.save_history_snapshot"),
    ):
        res = agent.refactor_logic(state_fail)
        assert res["impl_code"] == "def fallback_func():\n    pass"
        assert res["impl_updated"] is True
        assert res["refactor_iterations"] == 2

    # Case 4: is_bug_fix = True, Search/Replace block resulting in SyntaxError
    llm_response_syntax = (
        "<<<<<<< SEARCH\ndef func(): pass\n=======\ndef func():\n    invalid syntax [\n>>>>>>> REPLACE"
    )
    # S/R fails, so it falls back to full code generation (calling LLM again)
    syntax_side_effects = [llm_response_syntax, "def syntax_fallback():\n    pass"]
    with (
        patch("agent._call_llm_text", side_effect=syntax_side_effects),
        patch("agent.save_artifact", return_value="artifacts/app.py"),
        patch("agent.save_history_snapshot"),
    ):
        res = agent.refactor_logic(state_fail)
        assert res["impl_code"] == "def syntax_fallback():\n    pass"
        assert res["impl_updated"] is True

    # Case 5: is_bug_fix = True, but LLM returned full code block (fallback)
    llm_response_full = "```python\ndef new_func():\n    return 100\n```"
    with (
        patch("agent._call_llm_text", return_value=llm_response_full),
        patch("agent.save_artifact", return_value="artifacts/app.py"),
        patch("agent.save_history_snapshot"),
    ):
        res = agent.refactor_logic(state_fail)
        assert res["impl_code"] == "def new_func():\n    return 100"


def test_stagnant_iterations_increment_and_reset():
    from unittest.mock import MagicMock, patch

    import agent
    from schema import TDDState

    state = TDDState(
        last_test_summary="",
        stagnant_iterations=0,
    )

    # Mock subprocess.Popen for a failed test (e.g. 2 failed, 1 error)
    mock_process = MagicMock()
    mock_process.communicate.return_value = ("=== 2 failed, 1 error, 1 passed ===", "")
    mock_process.returncode = 1

    with patch("subprocess.Popen", return_value=mock_process):
        # 1st execution: last_test_summary is empty, so it should treat it as progress (stagnant=0)
        res = agent._execute_tests_helper("test_app.py", state)
        assert res["stagnant_iterations"] == 0
        assert res["last_test_summary"] == "2 failed, 1 error, 1 passed"

        # 2nd execution: last_test_summary matches current exactly
        state_2 = TDDState(
            last_test_summary="2 failed, 1 error, 1 passed",
            stagnant_iterations=0,
        )
        res_2 = agent._execute_tests_helper("test_app.py", state_2)
        assert res_2["stagnant_iterations"] == 1

        # 3rd execution: last_test_summary was worse (2 failed, 1 error), and new is better (1 failed, 0 error)
        # progress, so stagnant should reset to 0
        state_3 = TDDState(
            last_test_summary="2 failed, 1 error",
            stagnant_iterations=1,
        )
        mock_process.communicate.return_value = ("=== 1 failed, 2 passed ===", "")
        res_3 = agent._execute_tests_helper("test_app.py", state_3)
        assert res_3["stagnant_iterations"] == 0

        # 4th execution: success (0 failed, 0 error)
        # progress, stagnant resets to 0
        state_4 = TDDState(
            last_test_summary="1 failed",
            stagnant_iterations=2,
        )
        mock_process.communicate.return_value = ("=== 3 passed ===", "")
        mock_process.returncode = 0
        res_4 = agent._execute_tests_helper("test_app.py", state_4)
        assert res_4["stagnant_iterations"] == 0


def test_detect_toggle_loop_stagnant_limit(tmp_path):
    import os
    from unittest.mock import patch

    import agent
    from schema import TDDState

    # Test stagnant limit
    state_stagnant = TDDState(
        iterations=2,
        stagnant_iterations=3,
    )
    # Default MAX_STAGNANT_ITERATIONS is 3, so stagnant=3 should trigger rollback
    assert agent._detect_toggle_loop(state_stagnant) is True
    assert state_stagnant["loop_detected"] is True

    # Test custom stagnant limit override
    state_stagnant_custom = TDDState(
        iterations=2,
        stagnant_iterations=2,
    )
    with patch("config.MAX_STAGNANT_ITERATIONS", 2):
        assert agent._detect_toggle_loop(state_stagnant_custom) is True

    # Test LOOP_DETECTION_THRESHOLD override
    state_iter = TDDState(
        iterations=5,
        stagnant_iterations=0,
    )
    with patch("config.LOOP_DETECTION_THRESHOLD", 5):
        assert agent._detect_toggle_loop(state_iter) is True
        assert state_iter["loop_detected"] is True

    # Test coverage for latest_test_mtime logic (Lines 2105 and 2128)
    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        history_dir = tmp_path / "history"
        history_dir.mkdir(parents=True, exist_ok=True)

        # 1. Create a test snapshot
        test_snapshot = history_dir / "test_bc_clone_req001_d002_integration_iter001.py"
        test_snapshot.write_text("def test_ok(): pass")
        os.utime(test_snapshot, (100, 100))

        # 2. Create implementation snapshots: one older (t=90) and one newer (t=110)
        impl_old = history_dir / "bc_clone_req001_d002_test_iter001_integration_impl_iter001.py"
        impl_old.write_text("def f(): pass")
        os.utime(impl_old, (90, 90))

        impl_new = history_dir / "bc_clone_req001_d002_test_iter001_integration_impl_iter002.py"
        impl_new.write_text("def f(): pass # new")
        os.utime(impl_new, (110, 110))

        # Test state
        state_toggle = TDDState(
            iterations=2,
            stagnant_iterations=0,
            module_name="bc_clone.py",
            test_module_name="test_bc_clone_req001_integration.py",
            test_iterations=1,
            requirements=[{"id": "REQ001"}],
            current_req_index=0,
        )
        assert agent._detect_toggle_loop(state_toggle) is False


def test_generate_refactor_bug_report_stagnant_limit():
    from unittest.mock import patch

    import agent
    from schema import TDDState

    state = TDDState(
        refactor_iterations=1,
        stagnant_iterations=3,
        last_green_impl_code="def green(): pass",
        module_name="app.py",
    )
    with patch("agent.save_artifact") as mock_save:
        with patch("agent.save_history_snapshot"):
            res = agent.generate_refactor_bug_report(state)
            assert res["next_action"] == "rollback_continue"
            assert res["impl_code"] == "def green(): pass"
            assert res["impl_updated"] is True
            mock_save.assert_called_once_with("app.py", "def green(): pass")


def test_syntax_error_overrides():
    from unittest.mock import patch

    import agent
    from schema import TDDState

    # Test should_review_unit_tests_or_continue with custom MAX_SYNTAX_ERROR_ITERATIONS
    state = TDDState(
        tests_check_output="syntax error",
        test_syntax_error_iterations=2,
    )
    # Default limit is 3, so it should try again (return generate_unit_tests)
    assert agent.should_review_unit_tests_or_continue(state) == "generate_unit_tests"

    # With override to 2, it should force transition to generate_unit_bug_report
    with patch("config.MAX_SYNTAX_ERROR_ITERATIONS", 2):
        assert agent.should_review_unit_tests_or_continue(state) == "generate_unit_bug_report"

    # Test should_run_unit_tests override
    state_impl = TDDState(
        impl_check_output="syntax error",
        syntax_error_iterations=2,
    )
    assert agent.should_run_unit_tests(state_impl) == "implement_initial_logic"
    with patch("config.MAX_SYNTAX_ERROR_ITERATIONS", 2):
        assert agent.should_run_unit_tests(state_impl) == "update_design_for_req"


def test_iterations_reset_on_test_success():
    from unittest.mock import MagicMock, patch

    import agent
    from schema import TDDState

    state = TDDState(
        iterations=5,
        stagnant_iterations=2,
        last_test_summary="2 failed",
    )

    mock_process = MagicMock()
    mock_process.communicate.return_value = ("=== 5 passed ===", "")
    mock_process.returncode = 0

    with patch("subprocess.Popen", return_value=mock_process):
        res = agent._execute_tests_helper("test_app.py", state)
        assert res["success"] is True
        assert res["iterations"] == 0
        assert res["stagnant_iterations"] == 0
        assert res["last_test_summary"] == ""


def test_hierarchical_naming_logic(tmp_path):
    from unittest.mock import patch

    import agent
    from schema import TDDState

    state = TDDState(
        design_iterations=2,
        test_iterations=3,
        requirements=[{"id": "REQ001"}],
        current_req_index=0,
    )

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        # Test Case 1: Integration Test file snapshot
        agent.save_history_snapshot(
            "test_bc_clone_req001_integration.py",
            "test code",
            iteration=4,
            state=state,
        )
        expected_test_snapshot = tmp_path / "history" / "test_bc_clone_req001_d002_integration_iter004.py"
        assert expected_test_snapshot.exists()
        assert expected_test_snapshot.read_text() == "test code"

        # Test Case 2: Unit Test file snapshot
        agent.save_history_snapshot(
            "test_bc_clone_req001_unit.py",
            "unit test code",
            iteration=1,
            state=state,
        )
        expected_unit_snapshot = tmp_path / "history" / "test_bc_clone_req001_d002_unit_iter001.py"
        assert expected_unit_snapshot.exists()
        assert expected_unit_snapshot.read_text() == "unit test code"

        # Test Case 3: Implementation file snapshot
        agent.save_history_snapshot(
            "bc_clone.py",
            "impl code",
            iteration=5,
            state=state,
            phase="integration",
        )
        expected_impl_snapshot = tmp_path / "history" / "bc_clone_req001_d002_test_iter003_integration_impl_iter005.py"
        assert expected_impl_snapshot.exists()
        assert expected_impl_snapshot.read_text() == "impl code"


def test_get_balanced_test_output_context():
    from agent import _get_balanced_test_output_context, _truncate_test_output_smart

    # Case 1: Empty or short text (no truncation)
    assert _get_balanced_test_output_context("", 100) == ""
    assert _get_balanced_test_output_context("short text", 100) == "short text"
    assert _truncate_test_output_smart(None, 100) == ""
    assert _truncate_test_output_smart("short text", 100) == "short text"

    # Case 2: Small max_chars (simple slice fallback)
    assert len(_get_balanced_test_output_context("A" * 500, max_chars=100)) == 100

    # Case 3: Long text (truncation happens)
    long_text = "A" * 500
    res = _get_balanced_test_output_context(long_text, max_chars=300)
    assert "[TRUNCATED" in res
    assert len(res) <= 400

    # Case 4: Long text with FAILURES block (failures priority)
    long_failures_text = (
        "Some startup logs\n=== FAILURES ===\n" + "F" * 2000 + "\n=== short test summary ===\nsome warning logs\n"
    )
    res3 = _get_balanced_test_output_context(long_failures_text, max_chars=800)
    assert "FAILURES" in res3
    assert "PRESERVED" in res3
    assert len(res3) <= 800

    # Case 5: FAILURES block is too large itself, forcing internal truncation
    huge_failures_text = "Some startup logs\n=== FAILURES ===\n" + "F" * 5000 + "\n=== summary ===\n"
    res4 = _get_balanced_test_output_context(huge_failures_text, max_chars=1000)
    assert "FAILURES" in res4
    assert "[TRUNCATED MIDDLE OF FAILURES]" in res4
    assert len(res4) <= 1000

    # Case 6: Edge case max_chars very small but above 150
    res5 = _get_balanced_test_output_context("A" * 1000, max_chars=160)
    assert len(res5) <= 160


def test_get_existing_tests_context(tmp_path, monkeypatch):
    import agent
    import config

    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))

    active_test = tmp_path / "test_active.py"
    active_test.write_text("def test_active(): pass")

    other_test = tmp_path / "test_other.py"
    other_test.write_text("def test_other(): pass")

    state: TDDState = {"test_module_name": str(active_test), "tests_code": "legacy code"}

    res = agent._get_existing_tests_context(state)
    assert "def test_other" in res
    assert "def test_active" not in res

    # Nonexistent dir fallback
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path / "nonexistent"))
    res_none = agent._get_existing_tests_context(state)
    assert res_none == ""


def test_run_oracle_verification_on_failures_popen_fallback(monkeypatch):
    from unittest.mock import MagicMock, patch

    import agent
    import config
    from schema import OracleAssertionTarget

    mock_verifier = MagicMock(return_value="different_val")
    monkeypatch.setattr(config, "ORACLE_VERIFIER", mock_verifier)

    test_output = "___________ test_popen_case ___________\n>     assert stdout.strip() == expected\nE AssertionError\n"
    tests_code = """
def test_popen_case():
    expected = "correct_val"
    proc = Popen("3 * 4")
    assert stdout.strip() == expected
"""

    def mock_extract_oracle_target(prompt, schema, model_name=None):
        return OracleAssertionTarget(expression="3 * 4", expected="correct_val", preceding=[])

    with patch("agent._call_llm_structured", side_effect=mock_extract_oracle_target):
        res = agent._run_oracle_verification_on_failures(test_output, tests_code)
    assert "mathematically INCORRECT" in res
    mock_verifier.assert_called_once_with("3 * 4", expected="correct_val")


def test_run_oracle_verification_on_failures_popen_direct_assert(monkeypatch):
    from unittest.mock import MagicMock, patch

    import agent
    import config
    from schema import OracleAssertionTarget

    mock_verifier = MagicMock(return_value="different_val")
    monkeypatch.setattr(config, "ORACLE_VERIFIER", mock_verifier)

    test_output = (
        "___________ test_popen_direct ___________\n>     assert stdout.strip() == 'correct_val'\nE AssertionError\n"
    )
    tests_code = """
def test_popen_direct():
    proc = Popen("3 * 4")
    assert stdout.strip() == 'correct_val'
"""

    def mock_extract_oracle_target(prompt, schema, model_name=None):
        return OracleAssertionTarget(expression="3 * 4", expected="correct_val", preceding=[])

    with patch("agent._call_llm_structured", side_effect=mock_extract_oracle_target):
        res = agent._run_oracle_verification_on_failures(test_output, tests_code)
    assert "mathematically INCORRECT" in res
    mock_verifier.assert_called_once_with("3 * 4", expected="correct_val")


def test_generate_unit_bug_report_oracle_override(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    from schema import BugReport, TDDState

    mock_run_oracle = MagicMock(return_value="ORACLE VERIFICATION FEEDBACK: discrepancy")
    monkeypatch.setattr(agent, "_run_oracle_verification_on_failures", mock_run_oracle)

    mock_bug_report = BugReport(
        failed_test_cases=["test_a"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix logic",
        target_to_fix="implement_logic",
    )
    mock_call_llm = MagicMock(return_value=mock_bug_report)
    monkeypatch.setattr(agent, "_call_llm_structured", mock_call_llm)

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        unit_tests_code="def test_a(): pass",
        test_output="failed",
    )

    res = agent.generate_unit_bug_report(state)
    assert res["next_action"] == "generate_tests"


def test_generate_integration_bug_report_oracle_override(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    from schema import BugReport, TDDState

    mock_run_oracle = MagicMock(return_value="ORACLE VERIFICATION FEEDBACK: discrepancy")
    monkeypatch.setattr(agent, "_run_oracle_verification_on_failures", mock_run_oracle)

    mock_bug_report = BugReport(
        failed_test_cases=["test_a"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix logic",
        target_to_fix="implement_logic",
    )
    mock_call_llm = MagicMock(return_value=mock_bug_report)
    monkeypatch.setattr(agent, "_call_llm_structured", mock_call_llm)

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        integration_tests_code="def test_a(): pass",
        test_output="failed",
    )

    res = agent.generate_integration_bug_report(state)
    assert res["next_action"] == "generate_tests"


def test_generate_regression_bug_report_oracle_override(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    from schema import BugReport, TDDState

    mock_run_oracle = MagicMock(return_value="ORACLE VERIFICATION FEEDBACK: discrepancy")
    monkeypatch.setattr(agent, "_run_oracle_verification_on_failures", mock_run_oracle)

    mock_bug_report = BugReport(
        failed_test_cases=["test_a"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix logic",
        target_to_fix="implement_logic",
    )
    mock_call_llm = MagicMock(return_value=mock_bug_report)
    monkeypatch.setattr(agent, "_call_llm_structured", mock_call_llm)

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_output="failed",
    )

    res = agent.generate_regression_bug_report(state)
    assert res["next_action"] == "generate_tests"


def test_backup_project_before_rollback_exception(monkeypatch):
    import os

    import agent
    from schema import TDDState

    def mock_makedirs(path, exist_ok=False):
        raise OSError("Permission denied")

    monkeypatch.setattr(os, "makedirs", mock_makedirs)

    state = TDDState(test_output="failed", bug_report="bug")
    # Should catch exception internally and print warning
    agent._backup_project_before_rollback(state, 0, 0)


def test_generate_regression_bug_report_triage_preserve_implement_logic(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    from schema import BugReport, TDDState

    mock_run_oracle = MagicMock(return_value="ORACLE VERIFICATION FEEDBACK: discrepancy")
    monkeypatch.setattr(agent, "_run_oracle_verification_on_failures", mock_run_oracle)

    mock_bug_report = BugReport(
        failed_test_cases=["test_py_bc_req001_unit.py::test_case"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix logic",
        target_to_fix="implement_logic",
    )
    mock_call_llm = MagicMock(return_value=mock_bug_report)
    monkeypatch.setattr(agent, "_call_llm_structured", mock_call_llm)

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "req1"}, {"id": "REQ002", "description": "req2"}],
        current_req_index=1,
        test_output="FAILED test_py_bc_req001_unit.py",
    )

    res = agent.generate_regression_bug_report(state)
    assert res["next_action"] == "implement_logic"


def test_generate_regression_bug_report_triage_override_to_generate_tests(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    from schema import BugReport, TDDState

    mock_run_oracle = MagicMock(return_value="ORACLE VERIFICATION FEEDBACK: discrepancy")
    monkeypatch.setattr(agent, "_run_oracle_verification_on_failures", mock_run_oracle)

    mock_bug_report = BugReport(
        failed_test_cases=["test_py_bc_req001_unit.py::test_case"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix logic",
        target_to_fix="generate_tests",
    )
    mock_call_llm = MagicMock(return_value=mock_bug_report)
    monkeypatch.setattr(agent, "_call_llm_structured", mock_call_llm)

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "req1"}, {"id": "REQ002", "description": "req2"}],
        current_req_index=1,
        test_output="FAILED test_py_bc_req001_unit.py",
    )

    mock_backup = MagicMock()
    monkeypatch.setattr(agent, "_backup_project_before_rollback", mock_backup)

    res = agent.generate_regression_bug_report(state)
    assert res["next_action"] == "generate_design"
    assert res["current_req_index"] == 0
    assert mock_backup.called


def test_generate_regression_bug_report_triage_pinpoint_rollback(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    from schema import BugReport, TDDState

    mock_bug_report = BugReport(
        failed_test_cases=["test_py_bc_req001_unit.py::test_a", "test_py_bc_req004_unit.py::test_b"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix tests",
        target_to_fix="generate_tests",
        target_req="REQ004",
    )
    mock_call_llm = MagicMock(return_value=mock_bug_report)
    monkeypatch.setattr(agent, "_call_llm_structured", mock_call_llm)

    state = TDDState(
        requirements=[
            {"id": "REQ001", "description": "req1"},
            {"id": "REQ002", "description": "req2"},
            {"id": "REQ003", "description": "req3"},
            {"id": "REQ004", "description": "req4"},
            {"id": "REQ005", "description": "req5"},
        ],
        current_req_index=4,
        test_output="FAILED test_py_bc_req001_unit.py\nFAILED test_py_bc_req004_unit.py",
    )

    mock_backup = MagicMock()
    monkeypatch.setattr(agent, "_backup_project_before_rollback", mock_backup)

    res = agent.generate_regression_bug_report(state)

    assert res["current_req_index"] == 3
    assert res["next_action"] == "generate_design"
    assert mock_backup.called
    mock_backup.assert_called_once_with(state, 4, 3)


def test_build_uniqueness_advice_additional_coverage():
    from agent import _build_uniqueness_advice

    # 1. Test when no Search/Replace failure message is present (covers return "")
    res1 = _build_uniqueness_advice("Success output", "def dummy(): pass")
    assert res1 == ""

    # 2. Test when failed but matches multiple times without matching line numbers (covers else clause)
    res2 = _build_uniqueness_advice(
        "Failed to apply Search/Replace block: matches multiple times.", "def dummy(): pass"
    )
    assert "UNIQUENESS ERROR" in res2
    assert "Your SEARCH block matched multiple times" in res2


def test_oracle_helper_functions_coverage():
    from agent import (
        _extract_failing_line,
        _extract_method_body,
        _find_failed_methods,
    )

    # 1. _find_failed_methods empty and direct regex matching
    assert _find_failed_methods("some random output") == set()
    assert _find_failed_methods("tests/test_app.py::test_foo FAILED") == {"test_foo"}
    assert _find_failed_methods("FAILED test_bar") == {"test_bar"}

    # 2. _extract_method_body not found
    assert _extract_method_body("test_non_existent", "def test_other():\n    pass") == ""

    # 3. _extract_failing_line no header match or summary info fallback
    assert _extract_failing_line("test_foo", "random text") is None

    test_out_summary = "____ test_foo ____\nsome traceback\n=== short test summary info ===\nFAIL test_foo"
    assert _extract_failing_line("test_foo", test_out_summary) is None

    test_out_no_failing = "____ test_foo ____\nsome traceback without starting with >\n____ test_bar ____"
    assert _extract_failing_line("test_foo", test_out_no_failing) is None


def test_cleanup_history_on_rollback(tmp_path):
    import os
    from unittest.mock import patch

    from agent import _cleanup_history_on_rollback
    from schema import TDDState

    # 1. Happy path: matching requirement files are renamed, others are not
    history_dir = tmp_path / "history"
    history_dir.mkdir(parents=True, exist_ok=True)

    f1 = history_dir / "impl_req001_test_iter001_impl_iter001.py"
    f1.write_text("print('bad')", encoding="utf-8")

    f2 = history_dir / "impl_req002_test_iter001_impl_iter001.py"
    f2.write_text("print('other')", encoding="utf-8")

    state = TDDState(
        module_name="impl.py",
        requirements=[{"id": "REQ001", "description": "req1"}],
        current_req_index=0,
        test_iterations=1,
        loop_detected=True,
    )

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        _cleanup_history_on_rollback(state)

    assert not f1.exists()
    assert os.path.exists(str(f1) + ".bak")
    assert f2.exists()
    assert not os.path.exists(str(f2) + ".bak")

    # 2. Coverage: history directory does not exist (returns early)
    non_existent_path = tmp_path / "non_existent_dir"
    state_no_dir = TDDState(
        module_name="impl.py",
        requirements=[{"id": "REQ001", "description": "req1"}],
        current_req_index=0,
        test_iterations=1,
    )
    with patch("config.ARTIFACTS_DIR", str(non_existent_path)):
        _cleanup_history_on_rollback(state_no_dir)

    # 3. Coverage: req_id fallback where requirements list is empty
    state_no_reqs = TDDState(
        module_name="impl.py",
        current_req_index=0,
        test_iterations=1,
    )
    # This should construct req_id = "req001" and lookup "impl_req001..." in history
    # Let's recreate f1 and run to verify it triggers rename
    f1_recreate = history_dir / "impl_req001_test_iter001_impl_iter001.py"
    f1_recreate.write_text("print('recreated')", encoding="utf-8")
    if os.path.exists(str(f1_recreate) + ".bak"):
        os.remove(str(f1_recreate) + ".bak")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        _cleanup_history_on_rollback(state_no_reqs)
    assert not f1_recreate.exists()
    assert os.path.exists(str(f1_recreate) + ".bak")

    # 4. Coverage: os.rename raises exception (should print warning but continue safely)
    f1_recreate_2 = history_dir / "impl_req001_test_iter001_impl_iter001.py"
    f1_recreate_2.write_text("print('recreated 2')", encoding="utf-8")
    if os.path.exists(str(f1_recreate_2) + ".bak"):
        os.remove(str(f1_recreate_2) + ".bak")

    with patch("config.ARTIFACTS_DIR", str(tmp_path)):
        with patch("os.rename", side_effect=OSError("Permission denied")):
            _cleanup_history_on_rollback(state_no_reqs)


def test_increment_requirement_last_green():
    import agent
    from schema import TDDState

    state = TDDState(
        current_req_index=0,
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}],
        impl_code="print('green')",
    )

    res = agent.increment_requirement(state)
    assert res.get("last_green_impl_code") == "print('green')"


def test_generate_tests_syntax_fix_context(tmp_path):
    from unittest.mock import patch

    import agent
    from schema import TDDState

    state = TDDState(
        module_name="impl.py",
        requirements=[{"id": "REQ001", "description": "req1"}],
        current_req_index=0,
        tests_check_output="flake8 syntax error at line 5",
        unit_tests_code="def test_broken_syntax():",
    )

    with (
        patch("agent._call_llm_text", return_value="def test_rebuilt():\n    pass") as mock_call,
        patch("agent.save_artifact", return_value=str(tmp_path / "test_impl_req001_unit.py")),
        patch("agent.save_history_snapshot"),
    ):
        agent.generate_unit_tests(state)
        called_prompt = mock_call.call_args[0][0]
        assert "Test Syntax Fix Context" in called_prompt
        assert "flake8 syntax error at line 5" in called_prompt
        assert "def test_broken_syntax():" in called_prompt

    # Integration version
    state_integ = TDDState(
        module_name="impl.py",
        requirements=[{"id": "REQ001", "description": "req1"}],
        current_req_index=0,
        tests_check_output="integration syntax error",
        integration_tests_code="def test_broken_integration_syntax():",
    )

    with (
        patch("agent._call_llm_text", return_value="def test_rebuilt_integ():\n    pass") as mock_call_integ,
        patch("agent.save_artifact", return_value=str(tmp_path / "test_impl_req001_integration.py")),
        patch("agent.save_history_snapshot"),
    ):
        agent.generate_integration_tests(state_integ)
        called_prompt_integ = mock_call_integ.call_args[0][0]
        assert "Test Syntax Fix Context" in called_prompt_integ
        assert "integration syntax error" in called_prompt_integ
        assert "def test_broken_integration_syntax():" in called_prompt_integ


def test_generate_tests_syntax_fix_context_load_file(tmp_path):
    import os
    from unittest.mock import patch

    import agent
    from schema import TDDState

    # 1. Unit test path: successful read from file
    state = TDDState(
        module_name="impl.py",
        test_module_name="test_impl_req001_unit.py",
        requirements=[{"id": "REQ001", "description": "req1"}],
        current_req_index=0,
        tests_check_output="flake8 syntax error at line 5",
        unit_tests_code="",
    )

    test_file_path = os.path.join(tmp_path, "test_impl_req001_unit.py")
    with open(test_file_path, "w", encoding="utf-8") as f:
        f.write("def test_loaded_from_disk_unit(): pass")

    with (
        patch("config.ARTIFACTS_DIR", str(tmp_path)),
        patch("agent._call_llm_text", return_value="def test_rebuilt():\n    pass") as mock_call,
        patch("agent.save_artifact", return_value=str(test_file_path)),
        patch("agent.save_history_snapshot"),
    ):
        agent.generate_unit_tests(state)
        called_prompt = mock_call.call_args[0][0]
        assert "Test Syntax Fix Context" in called_prompt
        assert "def test_loaded_from_disk_unit(): pass" in called_prompt

    # 2. Unit test path: OSError when reading from file
    with (
        patch("config.ARTIFACTS_DIR", str(tmp_path)),
        patch("agent._call_llm_text", return_value="def test_rebuilt():\n    pass") as mock_call,
        patch("agent.save_artifact", return_value=str(test_file_path)),
        patch("agent.save_history_snapshot"),
        patch("builtins.open", side_effect=OSError("Read error")),
    ):
        agent.generate_unit_tests(state)
        called_prompt = mock_call.call_args[0][0]
        assert "Test Syntax Fix Context" in called_prompt
        assert "<previous_test_code>\n\n</previous_test_code>" in called_prompt

    # 3. Integration test path: successful read from file
    state_integ = TDDState(
        module_name="impl.py",
        test_module_name="test_impl_req001_integration.py",
        requirements=[{"id": "REQ001", "description": "req1"}],
        current_req_index=0,
        tests_check_output="integration syntax error",
        integration_tests_code="",
    )

    test_integ_file_path = os.path.join(tmp_path, "test_impl_req001_integration.py")
    with open(test_integ_file_path, "w", encoding="utf-8") as f:
        f.write("def test_loaded_from_disk_integ(): pass")

    with (
        patch("config.ARTIFACTS_DIR", str(tmp_path)),
        patch("agent._call_llm_text", return_value="def test_rebuilt_integ():\n    pass") as mock_call_integ,
        patch("agent.save_artifact", return_value=str(test_integ_file_path)),
        patch("agent.save_history_snapshot"),
    ):
        agent.generate_integration_tests(state_integ)
        called_prompt_integ = mock_call_integ.call_args[0][0]
        assert "Test Syntax Fix Context" in called_prompt_integ
        assert "def test_loaded_from_disk_integ(): pass" in called_prompt_integ

    # 4. Integration test path: OSError when reading from file
    with (
        patch("config.ARTIFACTS_DIR", str(tmp_path)),
        patch("agent._call_llm_text", return_value="def test_rebuilt_integ():\n    pass") as mock_call_integ,
        patch("agent.save_artifact", return_value=str(test_integ_file_path)),
        patch("agent.save_history_snapshot"),
        patch("builtins.open", side_effect=OSError("Read error")),
    ):
        agent.generate_integration_tests(state_integ)
        called_prompt_integ = mock_call_integ.call_args[0][0]
        assert "Test Syntax Fix Context" in called_prompt_integ
        assert "<previous_test_code>\n\n</previous_test_code>" in called_prompt_integ


def test_agent_junitxml_coverage_additions(tmp_path, monkeypatch):
    from typing import cast
    from unittest.mock import MagicMock, patch

    import agent
    import config
    from schema import TDDState

    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(config, "VERBOSE", True)

    # 1. XML parsing path (failures in XML)
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
<testsuite name="pytest" errors="1" failures="1" skipped="0" tests="2" time="0.1">
    <testcase classname="test_foo" name="test_one" file="test_foo.py" line="5">
        <failure message="assertion error">E assert 1 == 2</failure>
    </testcase>
    <testcase classname="test_foo" name="test_two" file="test_foo.py" line="10">
        <error message="runtime error">E RuntimeError</error>
    </testcase>
</testsuite>
"""
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate.return_value = ("=== 1 failed, 1 error ===", "")

    with open(tmp_path / "report.xml", "w", encoding="utf-8") as f:
        f.write(xml_content)

    state = TDDState(stagnant_iterations=1, last_test_summary="2 failed, 1 error")
    with patch("subprocess.Popen", return_value=mock_process):
        res = agent._execute_tests_helper("test_app.py", state)
        assert res["success"] is False
        assert "test_one" in res["failed_methods"]
        assert "test_two" in res["failed_methods"]
        assert "test_foo.py" in res["failed_files"]
        assert "test_one" in res["failed_tests_detail"]
        assert "test_two" in res["failed_tests_detail"]
        assert res["stagnant_iterations"] == 0

    # 1b. XML parse exception path
    with open(tmp_path / "report.xml", "w", encoding="utf-8") as f:
        f.write("<corrupt_xml>")
    with patch("subprocess.Popen", return_value=mock_process):
        res_corrupt = agent._execute_tests_helper("test_app.py", state)
        assert res_corrupt["success"] is False

    # 1c. os.remove exception path
    with open(tmp_path / "report.xml", "w", encoding="utf-8") as f:
        f.write(xml_content)
    with (
        patch("subprocess.Popen", return_value=mock_process),
        patch("os.remove", side_effect=OSError("Permission error")),
    ):
        res_remove_err = agent._execute_tests_helper("test_app.py", state)
        assert res_remove_err["success"] is False

    # 2. stagnant_iterations no progress path (identical summary)
    state_no_progress = TDDState(stagnant_iterations=1, last_test_summary="1 failed")
    mock_process_no_prog = MagicMock()
    mock_process_no_prog.returncode = 1
    mock_process_no_prog.communicate.return_value = ("=== 1 failed ===", "")
    with patch("subprocess.Popen", return_value=mock_process_no_prog):
        res_no_prog = agent._execute_tests_helper("test_app.py", state_no_progress)
        assert res_no_prog["stagnant_iterations"] == 2

    # 2b. stagnant_iterations summary changed path (resets to 0)
    state_changed = TDDState(stagnant_iterations=1, last_test_summary="1 failed")
    mock_process_changed = MagicMock()
    mock_process_changed.returncode = 1
    mock_process_changed.communicate.return_value = ("=== 2 failed ===", "")
    with patch("subprocess.Popen", return_value=mock_process_changed):
        res_changed = agent._execute_tests_helper("test_app.py", state_changed)
        assert res_changed["stagnant_iterations"] == 0

    # 3. _find_failed_methods state path
    state_fm = cast(TDDState, {"failed_methods": ["test_hello"]})
    res_fm = agent._find_failed_methods("any output", state_fm)
    assert res_fm == {"test_hello"}

    # 4. _find_failed_methods PASSED filter
    output_fm = "test_foo.py::test_bar PASSED\ntest_foo.py::test_baz FAILED"
    res_fm_filter = agent._find_failed_methods(output_fm, None)
    assert "test_baz" in res_fm_filter
    assert "test_bar" not in res_fm_filter

    # 5. _extract_failing_line state path & fallback 'E ' prefix
    state_efl = cast(TDDState, {"failed_tests_detail": {"test_hello": "E       assert 1 == 2"}})
    res_efl = agent._extract_failing_line("test_hello", "any output", state_efl)
    assert res_efl == "assert 1 == 2"

    state_efl_fallback = cast(
        TDDState, {"failed_tests_detail": {"test_hello": "E       assert 5 == 10\nE       AssertionError"}}
    )
    res_efl_fb = agent._extract_failing_line("test_hello", "any output", state_efl_fallback)
    assert res_efl_fb == "assert 5 == 10"

    # 6. should_fix_regression_tests_or_impl failed_files path
    state_cb = cast(
        TDDState,
        {
            "next_action": "generate_tests",
            "current_req_index": 1,
            "requirements": [{"id": "REQ001"}, {"id": "REQ002"}],
            "failed_files": ["test_app_req001_integration.py"],
            "regression_failure_policy": "halt",
        },
    )
    res_cb = agent.should_fix_regression_tests_or_impl(state_cb)
    assert res_cb == "halt_regression_test_failure"


@patch("agent._call_llm_structured")
@patch("agent.save_artifact")
def test_design_review_and_feedback(mock_save_artifact, mock_llm):
    from agent import (
        generate_design_initial,
        review_design_incremental,
        review_design_initial,
        should_review_design_incremental_or_continue,
        should_review_design_initial_or_continue,
        update_design_for_req,
    )
    from schema import DesignDocument, DesignReviewReport

    # 1. Test generate_design_initial and update_design_for_req with feedback
    # generate_design_initial with feedback
    mock_doc = DesignDocument(
        module_responsibilities="resp",
        architecture_and_components="arch",
        interface_definitions="interfaces",
        data_structures="data",
        logic_and_algorithms="logic",
        edge_cases_and_limitations="edge",
        error_handling="errors",
        command_line_interface="cli",
    )
    mock_llm.return_value = mock_doc
    state_initial_feedback = TDDState(goal="calculator", design_review_feedback="Please add variable initializations.")
    res_initial = generate_design_initial(state_initial_feedback)
    assert res_initial["design_updated"] is True

    # update_design_for_req with feedback
    state_update_feedback = TDDState(
        requirements=[{"id": "REQ001", "description": "some requirement"}],
        current_req_index=0,
        design_updated=False,
        design_review_feedback="Please fix error boundaries.",
    )
    res_update = update_design_for_req(state_update_feedback)
    assert res_update["design_updated"] is True

    # update_design_for_req returning {}
    state_update_no_op = TDDState(
        requirements=[{"id": "REQ001", "description": "some requirement"}],
        current_req_index=0,
        design_updated=True,
    )
    res_update_no_op = update_design_for_req(state_update_no_op)
    assert res_update_no_op == {}

    # 2. Test review_design_initial and review_design_incremental
    # Quality passed (default threshold 98)
    mock_review_passed = DesignReviewReport(estimated_quality=99, comments="No gaps detected.")
    mock_llm.return_value = mock_review_passed
    state_passed = TDDState(goal="calculator", spec_content="spec", design_doc="design")
    res_passed = review_design_initial(state_passed)
    assert res_passed["design_review_feedback"] == ""
    assert res_passed["design_review_iterations"] == 0

    # Quality passed with custom threshold (e.g. 90, quality 95)
    mock_review_passed_custom = DesignReviewReport(estimated_quality=95, comments="No gaps detected.")
    mock_llm.return_value = mock_review_passed_custom
    state_passed_custom = TDDState(
        goal="calculator", spec_content="spec", design_doc="design", target_design_quality=90
    )
    res_passed_custom = review_design_initial(state_passed_custom)
    assert res_passed_custom["design_review_feedback"] == ""
    assert res_passed_custom["design_review_iterations"] == 0

    # Quality failed
    mock_review_failed = DesignReviewReport(estimated_quality=80, comments="Missing edge case.")
    mock_llm.return_value = mock_review_failed
    state_failed = TDDState(goal="calculator", spec_content="spec", design_doc="design", design_review_iterations=1)
    res_failed = review_design_incremental(state_failed)
    assert res_failed["design_review_feedback"] == "Missing edge case."
    assert res_failed["design_review_iterations"] == 2
    assert res_failed["design_updated"] is False

    # Max iterations reached
    state_max_iters = TDDState(goal="calculator", spec_content="spec", design_doc="design", design_review_iterations=3)
    res_max = review_design_initial(state_max_iters)
    assert res_max["design_review_feedback"] == ""
    assert res_max["design_review_iterations"] == 0

    # 3. Test conditional routing helpers
    # should_review_design_initial_or_continue
    assert (
        should_review_design_initial_or_continue(TDDState(design_review_feedback="error")) == "generate_design_initial"
    )
    assert should_review_design_initial_or_continue(TDDState(design_review_feedback="")) == "plan_unit_tests"

    # should_review_design_incremental_or_continue
    assert (
        should_review_design_incremental_or_continue(TDDState(design_review_feedback="error"))
        == "update_design_for_req"
    )
    assert should_review_design_incremental_or_continue(TDDState(design_review_feedback="")) == "plan_unit_tests"


def test_early_oracle_verification_discrepancy(monkeypatch):
    import config
    from agent import (
        review_integration_test_plan,
        review_unit_test_plan,
        should_review_test_plan_or_continue,
    )
    from schema import OracleDiscrepancyJudgment, TestPlanReviewReport

    # Mock oracle verifier
    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3" if "1 + 2" in expr else "4")

    # 1. Test should_review_test_plan_or_continue with rollback decision
    assert (
        should_review_test_plan_or_continue(TDDState(test_plan_review_decision="update_design_for_req"))
        == "update_design_for_req"
    )
    assert should_review_test_plan_or_continue(TDDState(test_plan_review_decision="review_test_plan")) == "plan_tests"
    assert should_review_test_plan_or_continue(TDDState(test_plan_review_decision="continue")) == "generate_tests"

    # 2. Test review_unit_test_plan with a discrepancy
    state_discrepancy = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan='{"test_cases": [{"action": "execute 1 + 2", "expected_outcome": "4"}]}',
        test_plan="1. Action: execute 1 + 2 | Expected: 4",
    )

    mock_report = TestPlanReviewReport(
        missing_test_cases=[],
        estimated_coverage=100,
        feedback="Test cases look good",
    )
    mock_judgment = OracleDiscrepancyJudgment(
        is_design_error=True, reason="Mocked design error description", corrected_expected=None
    )

    with patch("agent._call_llm_structured") as mock_call:
        mock_call.side_effect = [mock_report, mock_judgment]
        res_unit = review_unit_test_plan(state_discrepancy)
        assert res_unit["test_plan_review_decision"] == "update_design_for_req"
        assert "discrepancies" in res_unit["design_review_feedback"]
        assert res_unit["design_updated"] is False

    # 3. Test review_integration_test_plan with no discrepancy
    state_no_discrepancy = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        integration_test_plan='{"test_cases": [{"action": "execute 1 + 2", "expected_outcome": "3"}]}',
        test_plan="1. Action: execute 1 + 2 | Expected: 3",
    )

    with patch("agent._call_llm_structured", return_value=mock_report):
        res_integ = review_integration_test_plan(state_no_discrepancy)
        assert res_integ["test_plan_review_decision"] == "continue"
        assert "design_review_feedback" not in res_integ

    # 3b. Test review_integration_test_plan with a discrepancy
    state_integ_discrepancy = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        integration_test_plan='{"test_cases": [{"action": "execute 1 + 2", "expected_outcome": "4"}]}',
        test_plan="1. Action: execute 1 + 2 | Expected: 4",
    )
    with patch("agent._call_llm_structured") as mock_call:
        mock_call.side_effect = [mock_report, mock_judgment]
        res_integ_disc = review_integration_test_plan(state_integ_discrepancy)
        assert res_integ_disc["test_plan_review_decision"] == "update_design_for_req"
        assert "discrepancies" in res_integ_disc["design_review_feedback"]
        assert res_integ_disc["design_updated"] is False

    # 4. Test with evaluate placeholder discrepancy
    state_evaluate_disc = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan=(
            '{"test_cases": [{"action": "execute scale=2; 10/3", '
            '"expected_outcome": "outcome 4 [Evaluate: scale=2; 10/3]"}]}'
        ),
        test_plan="1. Action: execute scale=2; 10/3 | Expected: outcome 4 [Evaluate: scale=2; 10/3]",
    )
    # Mock verifier to return 3.33 for scale=2; 10/3
    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3.33" if "10/3" in expr else "4")

    with patch("agent._call_llm_structured") as mock_call:
        mock_call.side_effect = [mock_report, mock_judgment]
        res_evaluate = review_unit_test_plan(state_evaluate_disc)
        assert res_evaluate["test_plan_review_decision"] == "update_design_for_req"

    # 4b. Test with quotes matching and isdigit check
    state_quotes_digit = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan=(
            '{"test_cases": ['
            '{"action": "execute \'1 + 2\'", "expected_outcome": "3"},'
            '{"action": "123", "expected_outcome": "123"}'
            "]}"
        ),
        test_plan="1. Action: execute '1 + 2' | Expected: 3\n2. Action: 123 | Expected: 123",
    )
    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3" if "1 + 2" in expr else "123")
    with patch("agent._call_llm_structured", return_value=mock_report):
        res_qd = review_unit_test_plan(state_quotes_digit)
        assert res_qd["test_plan_review_decision"] == "continue"

    # 4c. Test with verifier raising exception
    state_except = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan='{"test_cases": [{"action": "execute 1 + 2", "expected_outcome": "3"}]}',
        test_plan="1. Action: execute 1 + 2 | Expected: 3",
    )

    def raise_exc(expr):
        raise ValueError("Verifier Error")

    monkeypatch.setattr(config, "ORACLE_VERIFIER", raise_exc)
    with patch("agent._call_llm_structured", return_value=mock_report):
        res_exc = review_unit_test_plan(state_except)
        assert res_exc["test_plan_review_decision"] == "continue"

    # 5. Test with ORACLE_VERIFIER = None
    monkeypatch.setattr(config, "ORACLE_VERIFIER", None)
    with patch("agent._call_llm_structured", return_value=mock_report):
        res_none = review_unit_test_plan(state_discrepancy)
        assert res_none["test_plan_review_decision"] == "continue"

    # 6. Test with JSON load error
    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "3")
    state_invalid_json = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan="{invalid json}",
        test_plan="1. Action: execute 1 + 2 | Expected: 4",
    )
    with patch("agent._call_llm_structured", return_value=mock_report):
        res_invalid = review_unit_test_plan(state_invalid_json)
        assert res_invalid["test_plan_review_decision"] == "continue"

    # 7. Test standalone relational comparison (Expected '1' but oracle returns POSIX comparison Error)
    state_rel_err = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan=(
            '{"test_cases": [{"action": "execute 10 > 5", "expected_outcome": "1", '
            '"oracle_expression": "10 > 5", "oracle_expected": "1"}]}'
        ),
        test_plan="1. Action: execute 10 > 5 | Expected: 1",
    )
    monkeypatch.setattr(
        config, "ORACLE_VERIFIER", lambda expr, expected=None: "Error: (standard_in) 2: Error: comparison in expression"
    )
    with patch("agent._call_llm_structured") as mock_call:
        mock_call.side_effect = [mock_report, mock_judgment]
        res_rel = review_unit_test_plan(state_rel_err)
        assert res_rel["test_plan_review_decision"] == "update_design_for_req"
        assert "discrepancies" in res_rel["design_review_feedback"]

    # 8. Test error expected but oracle returns normal value
    state_err_expected = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan=(
            '{"test_cases": [{"action": "execute 10 > 5", "expected_outcome": "SyntaxError", '
            '"oracle_expression": "10 > 5", "oracle_expected": "SyntaxError"}]}'
        ),
        test_plan="1. Action: execute 10 > 5 | Expected: SyntaxError",
    )
    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr, expected=None: "1")
    with patch("agent._call_llm_structured") as mock_call:
        mock_call.side_effect = [mock_report, mock_judgment]
        res_err_exp = review_unit_test_plan(state_err_expected)
        assert res_err_exp["test_plan_review_decision"] == "update_design_for_req"
        assert "discrepancies" in res_err_exp["design_review_feedback"]

    # 9. Test error expected and oracle returns error (valid negative test case)
    state_err_valid = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan=(
            '{"test_cases": [{"action": "execute 10 > 5", "expected_outcome": "SyntaxError", '
            '"oracle_expression": "10 > 5", "oracle_expected": "SyntaxError"}]}'
        ),
        test_plan="1. Action: execute 10 > 5 | Expected: SyntaxError",
    )
    monkeypatch.setattr(
        config, "ORACLE_VERIFIER", lambda expr, expected=None: "Error: (standard_in) 2: Error: comparison in expression"
    )
    with patch("agent._call_llm_structured", return_value=mock_report):
        res_err_val = review_unit_test_plan(state_err_valid)
        assert res_err_val["test_plan_review_decision"] == "continue"

    # 10. Test normal expected with runtime context error - should pass via safety valve
    state_runtime_err = TDDState(
        goal="bc math calculator",
        spec_content="Calculate simple math",
        requirements=[{"id": "REQ001", "description": "Addition"}],
        current_req_index=0,
        unit_test_plan=(
            '{"test_cases": [{"action": "execute f(5)", "expected_outcome": "1", '
            '"oracle_expression": "1 + 2", "oracle_expected": "1"}]}'
        ),
        test_plan="1. Action: execute f(5) | Expected: 1",
    )
    monkeypatch.setattr(
        config, "ORACLE_VERIFIER", lambda expr, expected=None: "Error: runtime error: undefined function: f"
    )
    with patch("agent._call_llm_structured", return_value=mock_report):
        res_runtime = review_unit_test_plan(state_runtime_err)
        assert res_runtime["test_plan_review_decision"] == "continue"


def test_regression_failure_policy_logic():
    from agent import (
        generate_regression_bug_report,
        should_fix_regression_tests_or_impl,
    )
    from schema import BugReport

    mock_bug_report_design = BugReport(
        failed_test_cases=["test_add_req1_unit"],
        expected_vs_actual="Expected 3, got 4",
        fix_instructions="Fix addition",
        target_to_fix="generate_design",
    )

    mock_bug_report_impl = BugReport(
        failed_test_cases=["test_add_req1_unit"],
        expected_vs_actual="Expected 3, got 4",
        fix_instructions="Fix addition",
        target_to_fix="implement_logic",
    )

    from typing import cast

    # State with regression on REQ001 (index 0) while current index is 1
    state_reg = cast(
        TDDState,
        {
            "goal": "calculator",
            "requirements": [{"id": "REQ001"}, {"id": "REQ002"}],
            "current_req_index": 1,
            "failed_files": [],
            "test_output": "FAILED test_calc_req001_unit.py",
            "regression_failure_policy": "rollback",
            "rollback_counts": {},
        },
    )

    # 1. Rollback policy triggers rollback to index 0 when target_to_fix is generate_design
    with patch("agent._call_llm_structured", return_value=mock_bug_report_design):
        res = generate_regression_bug_report(state_reg)
        assert res["current_req_index"] == 0
        assert res["next_action"] == "generate_design"
        assert res["rollback_counts"] == {"0": 1}
        assert res["design_updated"] is False

        # 2. Max rollbacks (2) reached, should halt
        state_max_rollbacks = cast(
            TDDState,
            {
                "goal": "calculator",
                "requirements": [{"id": "REQ001"}, {"id": "REQ002"}],
                "current_req_index": 1,
                "failed_files": ["test_calc_req001_unit.py"],
                "test_output": "FAILED test_calc_req001_unit.py",
                "regression_failure_policy": "rollback",
                "rollback_counts": {"0": 2},
            },
        )
        res_max = generate_regression_bug_report(state_max_rollbacks)
        assert res_max["next_action"] == "halt_regression_test_failure"

        # 3. Halt policy halts immediately
        state_halt = cast(
            TDDState,
            {
                "goal": "calculator",
                "requirements": [{"id": "REQ001"}, {"id": "REQ002"}],
                "current_req_index": 1,
                "failed_files": ["test_calc_req001_unit.py"],
                "test_output": "FAILED test_calc_req001_unit.py",
                "regression_failure_policy": "halt",
                "rollback_counts": {},
            },
        )
        res_halt = generate_regression_bug_report(state_halt)
        assert res_halt["next_action"] == "halt_regression_test_failure"

    # 4. Rollback is skipped when target_to_fix is implement_logic
    with patch("agent._call_llm_structured", return_value=mock_bug_report_impl):
        res_impl = generate_regression_bug_report(state_reg)
        assert "current_req_index" not in res_impl
        assert res_impl["next_action"] == "implement_logic"

    # 5. Test should_fix_regression_tests_or_impl routing
    # If policy is halt, should return halt_regression_test_failure
    state_should_halt = cast(
        TDDState,
        {
            "next_action": "generate_tests",
            "requirements": [{"id": "REQ001"}, {"id": "REQ002"}],
            "current_req_index": 1,
            "failed_files": ["test_calc_req001_unit.py"],
            "test_output": "FAILED test_calc_req001_unit.py",
            "regression_failure_policy": "halt",
        },
    )
    assert should_fix_regression_tests_or_impl(state_should_halt) == "halt_regression_test_failure"

    # If policy is rollback, should return update_design_for_req as fallback if it reaches here
    state_should_rollback = cast(
        TDDState,
        {
            "next_action": "generate_tests",
            "requirements": [{"id": "REQ001"}, {"id": "REQ002"}],
            "current_req_index": 1,
            "failed_files": ["test_calc_req001_unit.py"],
            "test_output": "FAILED test_calc_req001_unit.py",
            "regression_failure_policy": "rollback",
        },
    )
    assert should_fix_regression_tests_or_impl(state_should_rollback) == "update_design_for_req"


def test_oracle_verification_math_safeguards(monkeypatch):
    from unittest.mock import MagicMock

    import agent
    import config

    # Stub the verifier
    mock_verifier = MagicMock(return_value="5")
    monkeypatch.setattr(config, "ORACLE_VERIFIER", mock_verifier)

    # Stub the LLM expression extractor to avoid actual API calls during testing
    import re

    def mock_extract_llm(body, failing_line):
        expr_match = re.search(r'evaluate\(\s*["\'](.*?)["\']\s*\)', body)
        expected_match = re.search(r'expected\s*=\s*["\'](.*?)["\']', body)
        expr = expr_match.group(1) if expr_match else ""
        expected_val = expected_match.group(1).replace("\\n", "\n") if expected_match else ""
        return expr, expected_val, [expr]

    monkeypatch.setattr(agent, "_extract_oracle_target_llm", mock_extract_llm)

    # 1. Test _run_early_oracle_verification with alphabetical strings (should skip)
    test_plan_json = '{"test_cases": [{"action": "Lex variable \'x==y\'", "expected_outcome": "Outcome 5"}]}'
    discrepancies = agent._run_early_oracle_verification(test_plan_json)
    assert not discrepancies
    mock_verifier.assert_not_called()

    # 2. Test _run_oracle_verification_on_failures with alphabetical strings (should skip)
    test_output = "___________ test_non_math ___________\n>     assert evaluate('x==y') == expected\nE AssertionError\n"
    tests_code = """
def test_non_math():
    expected = "some_token_list"
    assert evaluate("x==y") == expected
"""
    res = agent._run_oracle_verification_on_failures(test_output, tests_code)
    assert res == "No assertion discrepancies detected by the oracle."
    mock_verifier.assert_not_called()

    # 2b. Test _run_oracle_verification_on_failures with single lowercase variable (should NOT skip)
    mock_verifier.reset_mock()
    mock_verifier.return_value = "0"
    test_output_var = "___________ test_var ___________\n>     assert evaluate('x%=3') == expected\nE AssertionError\n"
    tests_code_var = """
def test_var():
    expected = "1.5"
    assert evaluate("x%=3") == expected
"""
    res_var = agent._run_oracle_verification_on_failures(test_output_var, tests_code_var)
    assert "contains an assertion error" in res_var
    assert "Expected value hardcoded in test: `1.5`" in res_var
    assert "Actual correct oracle value: `0`" in res_var
    mock_verifier.assert_called_once_with("x%=3", expected="1.5")

    # 2c. Test _run_oracle_verification_on_failures with non-numeric expected value (should skip)
    mock_verifier.reset_mock()
    test_output_non_num = (
        "___________ test_non_math ___________\n>     assert evaluate('x==y') == expected\nE AssertionError\n"
    )
    tests_code_non_num = """
def test_non_math():
    expected = "some_token_list"
    assert evaluate("x==y") == expected
"""
    res_non_num = agent._run_oracle_verification_on_failures(test_output_non_num, tests_code_non_num)
    assert res_non_num == "No assertion discrepancies detected by the oracle."
    mock_verifier.assert_not_called()

    # 2d. Test _run_oracle_verification_on_failures with invalid alphabetic variables in expr (should skip)
    mock_verifier.reset_mock()
    test_output_invalid = (
        "___________ test_invalid ___________\n>     assert evaluate('foo%=3') == expected\nE AssertionError\n"
    )
    tests_code_invalid = """
def test_invalid():
    expected = "1.5"
    assert evaluate("foo%=3") == expected
"""
    res_invalid = agent._run_oracle_verification_on_failures(test_output_invalid, tests_code_invalid)
    assert res_invalid == "No assertion discrepancies detected by the oracle."
    mock_verifier.assert_not_called()

    # 3. Test _run_early_oracle_verification with float values (should skip)
    test_plan_json_float = '{"test_cases": [{"action": "Lex variable \'.5\'", "expected_outcome": "Outcome 5"}]}'
    discrepancies_float = agent._run_early_oracle_verification(test_plan_json_float)
    assert not discrepancies_float
    mock_verifier.assert_not_called()

    # 4. Test _run_oracle_verification_on_failures with float values (should skip)
    test_output_float = (
        "___________ test_float ___________\n>     assert evaluate('.5') == expected\nE AssertionError\n"
    )
    tests_code_float = """
def test_float():
    expected = "some_token_list"
    assert evaluate(".5") == expected
"""
    res_float = agent._run_oracle_verification_on_failures(test_output_float, tests_code_float)
    assert res_float == "No assertion discrepancies detected by the oracle."
    mock_verifier.assert_not_called()

    # 4b. Test _run_oracle_verification_on_failures with multi-line outputs (matching)
    mock_verifier.reset_mock()
    mock_verifier.return_value = "5\n.5"
    test_output_multiline = (
        "___________ test_multiline ___________\n>     assert evaluate('5; .5') == expected\nE AssertionError\n"
    )
    tests_code_multiline = """
def test_multiline():
    expected = "5\\n.5"
    assert evaluate("5; .5") == expected
"""
    res_multiline = agent._run_oracle_verification_on_failures(test_output_multiline, tests_code_multiline)
    assert res_multiline == "No assertion discrepancies detected by the oracle."
    mock_verifier.assert_called_once_with("5; .5", expected="5\n.5")

    # 4c. Test _run_oracle_verification_on_failures with multi-line outputs (discrepancy)
    mock_verifier.reset_mock()
    mock_verifier.return_value = "5\n.5"
    test_output_multiline_disc = (
        "___________ test_multiline_disc ___________\n>     assert evaluate('5; .5') == expected\nE AssertionError\n"
    )
    tests_code_multiline_disc = """
def test_multiline_disc():
    expected = "5\\n.6"
    assert evaluate("5; .5") == expected
"""
    res_multiline_disc = agent._run_oracle_verification_on_failures(
        test_output_multiline_disc, tests_code_multiline_disc
    )
    assert "Expected value hardcoded in test: `5\n.6`" in res_multiline_disc
    assert "Actual correct oracle value: `5\n.5`" in res_multiline_disc
    mock_verifier.assert_called_once_with("5; .5", expected="5\n.6")

    # 5. Test _run_early_oracle_verification with oracle_expression and oracle_expected fields
    mock_verifier.reset_mock()
    mock_verifier.return_value = "5"

    # 5a. Matching expected (no discrepancies)
    plan_matching = (
        '{"test_cases": [{"action": "calc", "expected_outcome": "5", '
        '"oracle_expression": "2+3", "oracle_expected": "5"}]}'
    )
    disc1 = agent._run_early_oracle_verification(plan_matching)
    assert not disc1
    mock_verifier.assert_called_once_with("2+3", expected="5")

    # 5b. Discrepancy (oracle returns "5" but plan expected "6")
    mock_verifier.reset_mock()
    plan_discrepancy = (
        '{"test_cases": [{"action": "calc", "expected_outcome": "6", '
        '"oracle_expression": "2+3", "oracle_expected": "6"}]}'
    )
    disc2 = agent._run_early_oracle_verification(plan_discrepancy)
    assert len(disc2) == 1
    assert disc2[0]["expected_val"] == "6"
    assert disc2[0]["oracle_val"] == "5"
    mock_verifier.assert_called_once_with("2+3", expected="6")

    # 5c. Multi-line matching expected
    mock_verifier.reset_mock()
    mock_verifier.return_value = "5\n.5"
    plan_multiline_match = (
        '{"test_cases": [{"action": "calc", "expected_outcome": "5 and 0.5", '
        '"oracle_expression": "5; .5", "oracle_expected": "5\\n.5"}]}'
    )
    disc3 = agent._run_early_oracle_verification(plan_multiline_match)
    assert not disc3
    mock_verifier.assert_called_once_with("5; .5", expected="5\n.5")

    # 5d. Multi-line discrepancy
    mock_verifier.reset_mock()
    mock_verifier.return_value = "5\n.5"
    plan_multiline_disc = (
        '{"test_cases": [{"action": "calc", "expected_outcome": "5 and 0.6", '
        '"oracle_expression": "5; .5", "oracle_expected": "5\\n.6"}]}'
    )
    disc4 = agent._run_early_oracle_verification(plan_multiline_disc)
    assert len(disc4) == 1
    assert disc4[0]["expected_val"] == "5\n.6"
    assert disc4[0]["oracle_val"] == "5\n.5"
    mock_verifier.assert_called_once_with("5; .5", expected="5\n.6")

    # 6. Test _get_dynamic_max_tokens with None state
    assert agent._get_dynamic_max_tokens(None) == 8192

    # 7. Test _get_dynamic_max_tokens with active state scaling
    from schema import TDDState

    state_large = TDDState(
        impl_code="A" * 150000,  # 150,000 chars / 4 = 37,500 estimated tokens. Buffer is 4096.
    )
    # Expected dynamic token count: 37500 + 4096 = 41596
    assert agent._get_dynamic_max_tokens(state_large) == 41596

    # Test dynamic max tokens limits clamping to the configuration maximum
    state_huge = TDDState(
        impl_code="A" * 1200000,  # 1,200,000 chars / 4 = 300,000 estimated tokens. Clamps to max 262144.
    )
    assert agent._get_dynamic_max_tokens(state_huge) == 262144

    # 8. Test _call_llm_with_reasoning wrapper routes max_tokens correctly
    with patch("agent.call_llm_with_reasoning") as mock_reasoning:
        mock_reasoning.return_value = "mock response"
        # Mock thread local state
        agent._thread_local.current_state = state_large
        try:
            res = agent._call_llm_with_reasoning("test prompt", thinking_level="MINIMAL")
            assert res == "mock response"
            mock_reasoning.assert_called_once_with(
                "test prompt",
                response_schema=None,
                tools=None,
                thinking_level="MINIMAL",
                temperature=0.0,
                max_tokens=41596,
            )
        finally:
            agent._thread_local.current_state = None


def test_robust_self_correction_rollback_improvements(tmp_path, monkeypatch):
    import builtins
    from unittest.mock import patch

    import agent
    import config
    from schema import BugReport, TDDState, TestPlan

    # Mock Artifacts Dir
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))

    # --- 1. Test update_design_for_req skip logic ---
    state_skip = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        oracle_discrepancy_only=True,
        design_doc="original design doc content",
    )
    res_skip = agent.update_design_for_req(state_skip)
    assert res_skip["design_updated"] is True
    assert res_skip["design_doc"] == "original design doc content"
    assert res_skip["oracle_discrepancy_only"] is False

    # Skip logic fallback when design_doc is empty but design.md file exists
    state_skip_file = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        oracle_discrepancy_only=True,
        design_doc="",
    )
    design_file = tmp_path / "design.md"
    design_file.write_text("file design doc content")
    res_skip_file = agent.update_design_for_req(state_skip_file)
    assert res_skip_file["design_doc"] == "file design doc content"

    # --- 2. Test plan_unit_tests & plan_integration_tests bug_report injection ---
    state_plan = TDDState(
        goal="goal",
        spec_content="spec",
        design_doc="design",
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        bug_report="ORACLE DISCREPANCY REPORT",
    )
    mock_test_plan = TestPlan(test_cases=[])
    with patch("agent._call_llm_structured", return_value=mock_test_plan) as mock_llm:
        agent.plan_unit_tests(state_plan)
        called_prompt_unit = mock_llm.call_args[0][0]
        assert "Previous Test Mismatch / Oracle Warnings" in called_prompt_unit
        assert "ORACLE DISCREPANCY REPORT" in called_prompt_unit

        agent.plan_integration_tests(state_plan)
        called_prompt_integ = mock_llm.call_args[0][0]
        assert "Previous Test Mismatch / Oracle Warnings" in called_prompt_integ
        assert "ORACLE DISCREPANCY REPORT" in called_prompt_integ

    # --- 3. Test _get_filtered_regression_test_code_context ---
    # We patch glob.glob to return a list containing a history path to cover line 2475 skip logic
    test_files = [
        str(tmp_path / "test_py_bc_req001_unit.py"),
        str(tmp_path / "history" / "test_py_bc_req001_unit.py"),
        str(tmp_path / "test_py_bc_req002_unit.py"),
    ]
    (tmp_path / "test_py_bc_req001_unit.py").write_text("def test_one(): pass")
    (tmp_path / "test_py_bc_req002_unit.py").write_text("def test_two(): pass")

    with patch("glob.glob", return_value=test_files):
        res_code = agent._get_filtered_regression_test_code_context(1)
        assert "test_py_bc_req001_unit.py" in res_code
        assert "test_py_bc_req002_unit.py" not in res_code

    # PermissionError fallback path coverage
    real_open = builtins.open

    def mock_open_err(file, mode="r", *args, **kwargs):
        if "test_py_bc_req001" in str(file):
            raise PermissionError("mock error")
        return real_open(file, mode, *args, **kwargs)

    with patch("builtins.open", side_effect=mock_open_err):
        res_code_err = agent._get_filtered_regression_test_code_context(1)
        assert "No active regression tests found" in res_code_err

    # --- 4. Test _get_filtered_test_output ---
    test_output = (
        "collected 2 items\n"
        "___________ test_req001_case ___________\n"
        "E AssertionError\n"
        "___________ test_req002_case ___________\n"
        "E AssertionError2\n"
        "=== short test summary info ===\n"
        "FAILED test_py_bc_req001_unit.py::test_req001_case\n"
        "FAILED test_py_bc_req002_unit.py::test_req002_case\n"
    )
    res_output = agent._get_filtered_test_output(test_output, 1)
    assert "test_req001_case" in res_output
    assert "test_req002_case" not in res_output
    assert "FAILED test_py_bc_req001_unit.py" in res_output
    assert "FAILED test_py_bc_req002_unit.py" not in res_output
    assert agent._get_filtered_test_output("", 1) == ""

    # --- 5. Test generate_regression_bug_report rollback, shutil.copy and design recovery ---
    state_report = TDDState(
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}],
        current_req_index=2,
        failed_files=["test_py_bc_req001_unit.py"],
        test_output="FAILED test_py_bc_req001_unit.py",
        regression_failure_policy="rollback",
        module_name="py_bc.py",
    )

    # Setup history dir and backup file
    history_dir = tmp_path / "history"
    history_dir.mkdir(exist_ok=True)
    backup_file = history_dir / "py_bc_req001_impl_iter001.py"
    backup_file.write_text("historical impl content")
    backup_design_file = history_dir / "design_req001_iter001.md"
    backup_design_file.write_text("historical design content")

    mock_bug_report_design = BugReport(
        failed_test_cases=["test_req001_case"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix it",
        target_to_fix="generate_design",
    )

    mock_bug_report_impl = BugReport(
        failed_test_cases=["test_req001_case"],
        expected_vs_actual="expected X, got Y",
        fix_instructions="fix it",
        target_to_fix="implement_logic",
    )

    # Test rollback path (target_to_fix is generate_design) -> restores impl and design
    with patch("agent._call_llm_structured", return_value=mock_bug_report_design):
        res_report = agent.generate_regression_bug_report(state_report)
        assert res_report["next_action"] == "generate_design"
        assert res_report["current_req_index"] == 0
        assert res_report["design_doc"] == "historical design content"

    # Test non-rollback path (target_to_fix is implement_logic) -> skips rollback
    with patch("agent._call_llm_structured", return_value=mock_bug_report_impl):
        res_report_impl = agent.generate_regression_bug_report(state_report)
        assert "current_req_index" not in res_report_impl
        assert res_report_impl["next_action"] == "implement_logic"

    # --- 6. Coverage for exceptions (193-194, 708-709) ---
    # Exception coverage for os.rename in save_history_snapshot (193-194)
    # Ensure history directory and the expected history file exist to trigger rotation
    (tmp_path / "history").mkdir(exist_ok=True)
    (tmp_path / "history" / "test_file_iter001.py").write_text("content v1")
    monkeypatch.setattr(agent.os, "rename", MagicMock(side_effect=OSError("mock rename error")))
    # This should log/print error but not crash
    agent.save_history_snapshot("test_file.py", "content v2", 1, state=state_report)

    # Exception coverage for design file read fallback in update_design_for_req (708-709)
    design_md_path = tmp_path / "design.md"
    if design_md_path.exists():
        design_md_path.unlink()

    state_skip_err = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        oracle_discrepancy_only=True,
        design_doc="",
    )
    # design.md does not exist, so reading it raises FileNotFoundError, catching and passing
    res_skip_err = agent.update_design_for_req(state_skip_err)
    assert res_skip_err["design_doc"] == ""
    assert res_skip_err["oracle_discrepancy_only"] is False
    # a. update_design_for_req: file read error fallback (652-653)
    state_skip_file_err = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        oracle_discrepancy_only=True,
        design_doc="",
    )
    with patch("builtins.open", side_effect=IOError("read error")):
        res_skip_file_err = agent.update_design_for_req(state_skip_file_err)
        assert res_skip_file_err["design_doc"] == ""

    # b. generate_regression_bug_report: rollback no backup warning coverage (2664-2667)
    state_report_no_backup = TDDState(
        goal="goal",
        spec_content="spec",
        design_doc="design",
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}],
        current_req_index=2,
        failed_files=["test_py_bc_req002_unit.py"],
        test_output="FAILED test_py_bc_req002_unit.py",
        regression_failure_policy="rollback",
        module_name="py_bc.py",
    )
    with patch("agent._call_llm_structured", return_value=mock_bug_report_design):
        res_report_no_backup = agent.generate_regression_bug_report(state_report_no_backup)
        assert res_report_no_backup["next_action"] == "generate_design"
        assert res_report_no_backup["current_req_index"] == 1

    # c. generate_regression_bug_report: shutil.copy exception path coverage (2661-2662)
    state_report_copy_err = TDDState(
        goal="goal",
        spec_content="spec",
        design_doc="design",
        requirements=[{"id": "REQ001"}, {"id": "REQ002"}, {"id": "REQ003"}],
        current_req_index=2,
        failed_files=["test_py_bc_req001_unit.py"],
        test_output="FAILED test_py_bc_req001_unit.py",
        regression_failure_policy="rollback",
        module_name="py_bc.py",
    )
    with patch("agent._call_llm_structured", return_value=mock_bug_report_design):
        with patch("shutil.copy", side_effect=IOError("copy failed")):
            res_report_err = agent.generate_regression_bug_report(state_report_copy_err)
            assert res_report_err["next_action"] == "generate_design"
            assert res_report_err["current_req_index"] == 0


def test_oracle_judgment_and_rollback_paths(monkeypatch):
    from unittest.mock import patch

    import agent
    from schema import OracleDiscrepancyJudgment, TDDState, TestPlanReviewReport

    # Mock the LLM calls
    mock_judgment_design = OracleDiscrepancyJudgment(
        is_design_error=True, reason="Core rules for multiplication are missing in design", corrected_expected=None
    )
    mock_judgment_test_plan = OracleDiscrepancyJudgment(
        is_design_error=False,
        reason="Simple natural language mismatch in expectation description",
        corrected_expected="16.20",
    )

    # 1. Test _judge_oracle_discrepancy_with_llm
    with patch("agent._call_llm_structured", return_value=mock_judgment_design):
        res = agent._judge_oracle_discrepancy_with_llm("design_doc", {"action": "eval"}, "16.2")
        assert res.is_design_error is True
        assert res.reason == "Core rules for multiplication are missing in design"

    # LLM exception path
    with patch("agent._call_llm_structured", side_effect=ValueError("LLM error")):
        res_err = agent._judge_oracle_discrepancy_with_llm("design_doc", {"action": "eval"}, "16.2")
        assert res_err.is_design_error is True
        assert "LLM error" in res_err.reason

    # 2. Test review_unit_test_plan / review_integration_test_plan under mismatches
    mismatches = [
        {
            "test_case": {
                "action": "calc 10+5",
                "expected_outcome": "Prints 15",
                "oracle_expression": "10+5",
                "oracle_expected": "15",
            },
            "expr": "10+5",
            "expected_val": "Prints 15",
            "oracle_val": "15",
        }
    ]

    mock_review_report = TestPlanReviewReport(missing_test_cases=[], estimated_coverage=95, feedback="Looks good")

    state = TDDState(
        goal="goal",
        spec_content="spec",
        design_doc="design",
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        unit_test_plan='{"test_cases": []}',
        integration_test_plan='{"test_cases": []}',
        test_plan="plan",
    )

    # Path A: Design error detected (should rollback to update_design_for_req)
    with patch("agent._run_early_oracle_verification", return_value=mismatches):
        with patch("agent._call_llm_structured") as mock_calls:
            mock_calls.side_effect = [mock_review_report, mock_judgment_design]

            res_unit = agent.review_unit_test_plan(state)
            assert res_unit["test_plan_review_decision"] == "update_design_for_req"
            assert res_unit["oracle_discrepancy_only"] is False

    # Path B: Test Plan notation error detected (should rollback to review_test_plan)
    with patch("agent._run_early_oracle_verification", return_value=mismatches):
        with patch("agent._call_llm_structured") as mock_calls:
            mock_calls.side_effect = [mock_review_report, mock_judgment_test_plan]

            res_unit = agent.review_unit_test_plan(state)
            assert res_unit["test_plan_review_decision"] == "review_test_plan"
            assert res_unit["oracle_discrepancy_only"] is True

    # Path C: Integration Test Plan verification matching tests
    with patch("agent._run_early_oracle_verification", return_value=mismatches):
        with patch("agent._call_llm_structured") as mock_calls:
            mock_calls.side_effect = [mock_review_report, mock_judgment_design]
            res_integ = agent.review_integration_test_plan(state)
            assert res_integ["test_plan_review_decision"] == "update_design_for_req"
            assert res_integ["oracle_discrepancy_only"] is False

    with patch("agent._run_early_oracle_verification", return_value=mismatches):
        with patch("agent._call_llm_structured") as mock_calls:
            mock_calls.side_effect = [mock_review_report, mock_judgment_test_plan]
            res_integ = agent.review_integration_test_plan(state)
            assert res_integ["test_plan_review_decision"] == "review_test_plan"
            assert res_integ["oracle_discrepancy_only"] is True

    # 3. Extra coverage paths for _run_early_oracle_verification
    # 3a. JSON loads failure
    assert agent._run_early_oracle_verification("{invalid json") == []

    # 3b. Evaluate pattern without numbers
    plan_no_num = '{"test_cases": [{"action": "execute", "expected_outcome": "[Evaluate: 1+2]"}]}'
    assert agent._run_early_oracle_verification(plan_no_num) == []

    # 3c. Verifier raising exception
    import config

    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: 1 / 0)
    plan_exc = (
        '{"test_cases": [{"action": "calc", "expected_outcome": "5", '
        '"oracle_expression": "2+3", "oracle_expected": "6"}]}'
    )
    assert agent._run_early_oracle_verification(plan_exc) == []

    # 3d. Coverage for oracle_discrepancy_only path in plan_unit_tests / plan_integration_tests
    from schema import TestCase, TestPlan

    mock_test_plan = TestPlan(test_cases=[TestCase(action="5 != 5", expected_outcome="0")])

    state_disc_fix = TDDState(
        goal="bc",
        spec_content="spec",
        design_doc="design",
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_plan_iterations=1,
        test_plan_review="some review",
        unit_test_plan='{"test_cases": [{"action": "5 != 5", "expected_outcome": "1"}]}',
        integration_test_plan='{"test_cases": [{"action": "5 != 5", "expected_outcome": "1"}]}',
        oracle_discrepancy_only=True,
    )

    with patch("agent._call_llm_structured", return_value=mock_test_plan):
        res_unit = agent.plan_unit_tests(state_disc_fix)
        assert "5 != 5" in res_unit["test_plan"]

        res_integ = agent.plan_integration_tests(state_disc_fix)
        assert "5 != 5" in res_integ["test_plan"]


def test_test_plan_review_circuit_breakers(monkeypatch):
    import config
    from agent import review_integration_test_plan, review_unit_test_plan
    from schema import OracleDiscrepancyJudgment, TestPlanReviewReport

    # Mock oracle verifier
    monkeypatch.setattr(config, "ORACLE_VERIFIER", lambda expr: "0")

    unit_plan_str = (
        '{"test_cases": [{"action": "5 != 5", "expected_outcome": "0", '
        '"oracle_expression": "5 != 5", "oracle_expected": "1"}]}'
    )
    state = TDDState(
        goal="bc",
        spec_content="spec",
        design_doc="design",
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        unit_test_plan=unit_plan_str,
        integration_test_plan=unit_plan_str,
        test_plan="1. Action: 5 != 5 | Expected: 1",
        rollback_counts={"REQ001": 3},
    )

    mock_report = TestPlanReviewReport(missing_test_cases=[], estimated_coverage=100, feedback="looks good")
    mock_judgment = OracleDiscrepancyJudgment(is_design_error=True, reason="Flaw", corrected_expected=None)

    with patch("agent._call_llm_structured") as mock_call:
        mock_call.side_effect = [mock_report, mock_judgment, mock_report, mock_judgment]

        # Test unit test plan review circuit breaker (should return review_test_plan decision)
        res_unit = review_unit_test_plan(state)
        assert res_unit["test_plan_review_decision"] == "review_test_plan"
        assert res_unit["oracle_discrepancy_only"] is True

        # Test integration test plan review circuit breaker (should return review_test_plan decision)
        res_integ = review_integration_test_plan(state)
        assert res_integ["test_plan_review_decision"] == "review_test_plan"
        assert res_integ["oracle_discrepancy_only"] is True


def test_oracle_expected_validation_relaxed():
    import pytest

    from schema import TestCase

    # 1. Valid raw values (numbers, decimals, raw strings)
    p1 = TestCase(action="calc", expected_outcome="5", oracle_expression="2+3", oracle_expected="5")
    assert p1.oracle_expected == "5"

    p2 = TestCase(action="calc", expected_outcome="16.2", oracle_expression="16.2", oracle_expected="16.2")
    assert p2.oracle_expected == "16.2"

    # 2. Raw comment block outputs (e.g. for testing comment behaviors)
    p3 = TestCase(
        action="evaluate comment",
        expected_outcome="empty",
        oracle_expression="/* comment */",
        oracle_expected="/* comment */",
    )
    assert p3.oracle_expected == "/* comment */"

    p4 = TestCase(
        action="evaluate comment",
        expected_outcome="empty",
        oracle_expression="# comment",
        oracle_expected="# comment",
    )
    assert p4.oracle_expected == "# comment"

    # 3. Valid multi-word outputs (like raw error messages)
    p5 = TestCase(
        action="evaluate div by zero",
        expected_outcome="error",
        oracle_expression="1/0",
        oracle_expected="Runtime error: divide by zero",
    )
    assert p5.oracle_expected == "Runtime error: divide by zero"

    # 4. Invalid natural language explanations (should still be blocked)
    with pytest.raises(ValueError, match="oracle_expected must be a raw string output"):
        TestCase(
            action="calc",
            expected_outcome="5",
            oracle_expression="2+3",
            oracle_expected="The output should prints 5",
        )

    with pytest.raises(ValueError, match="oracle_expected must be a raw string output"):
        TestCase(action="calc", expected_outcome="5", oracle_expression="2+3", oracle_expected="outputs 5")


def test_run_oracle_verification_tail_matching(monkeypatch):
    from unittest.mock import MagicMock, patch

    import agent
    import config
    from schema import OracleAssertionTarget

    # Mock verifier to return a multi-line response (e.g. preceding outputs)
    mock_verifier = MagicMock(return_value="15\n2500")
    monkeypatch.setattr(config, "ORACLE_VERIFIER", mock_verifier)

    test_output_sim = (
        "tests/test_app.py::test_sequential\n"
        "___________ test_sequential ___________\n"
        "> assert calc.evaluate('1000 + 1500') == ['2500']\n"
        "E AssertionError\n"
    )
    tests_code_sim = (
        "def test_sequential():\n    calc.execute('10 + 5')\n    assert calc.evaluate('1000 + 1500') == ['2500']\n"
    )

    # 1. Match case: expected is "2500" (matches the tail of "15\n2500")
    def mock_extract_match(prompt, schema, model_name=None):
        return OracleAssertionTarget(expression="1000 + 1500", expected="2500", preceding=["10 + 5"])

    with patch("agent._call_llm_structured", side_effect=mock_extract_match):
        feedback = agent._run_oracle_verification_on_failures(test_output_sim, tests_code_sim)
        # Should match, hence no mathematically INCORRECT feedback generated
        assert "mathematically INCORRECT" not in feedback
        assert "No assertion discrepancies detected by the oracle." in feedback

    # 2. Mismatch case: expected is "2501" (does not match tail)
    def mock_extract_mismatch(prompt, schema, model_name=None):
        return OracleAssertionTarget(expression="1000 + 1500", expected="2501", preceding=["10 + 5"])

    with patch("agent._call_llm_structured", side_effect=mock_extract_mismatch):
        feedback_mismatch = agent._run_oracle_verification_on_failures(test_output_sim, tests_code_sim)
        # Should not match, hence mathematical discrepancy feedback is generated
        assert "ORACLE VERIFICATION" in feedback_mismatch
        assert "mathematically INCORRECT" in feedback_mismatch


def test_robust_self_correction_new_features(tmp_path, monkeypatch):
    from unittest.mock import patch

    import agent
    import config
    from schema import BugReport, TDDState

    # 1. Setup mock artifacts directory
    monkeypatch.setattr(config, "ARTIFACTS_DIR", str(tmp_path))
    history_dir = tmp_path / "history"
    history_dir.mkdir(exist_ok=True)

    # 2. Test future test file filtering in _get_existing_tests_context and _get_combined_tests_code
    req001_test = tmp_path / "test_py_bc_req001_unit.py"
    req001_test.write_text("def test_req001(): pass")

    req003_test = tmp_path / "test_py_bc_req003_unit.py"
    req003_test.write_text("def test_req003(): pass")

    # When current_req_index is 1 (REQ002), active test is test_py_bc_req002_unit.py
    state = TDDState(
        current_req_index=1, test_module_name="test_py_bc_req002_unit.py", unit_tests_code="def test_req002(): pass"
    )

    existing_context = agent._get_existing_tests_context(state)
    # req003_test should be excluded since N=3 >= 1+2 (3 >= 3)
    assert "test_req001" in existing_context
    assert "test_req003" not in existing_context

    combined_code = agent._get_combined_tests_code(state)
    assert "test_req001" in combined_code
    assert "test_req003" not in combined_code

    # 3. Test future test file deletion on rollback
    # Ensure req003_test exists before rollback
    assert req003_test.exists()

    state_rollback = TDDState(
        requirements=[
            {"id": "REQ001", "description": "desc"},
            {"id": "REQ002", "description": "desc"},
            {"id": "REQ003", "description": "desc"},
        ],
        current_req_index=2,  # REQ003
        rollback_counts={},
        design_iterations=0,
        test_output="FAILED tests/test_py_bc_req002_unit.py::test_calc",
    )

    # Mock LLM Judge response to return a failure not target to implement_logic, triggering rollback
    mock_bug_report = BugReport(
        failed_test_cases=[], expected_vs_actual="", fix_instructions="", target_to_fix="generate_tests"
    )
    with (
        patch("agent._call_llm_structured", return_value=mock_bug_report),
        patch("agent.save_history_snapshot"),
        patch("shutil.copy"),
    ):
        agent.generate_regression_bug_report(state_rollback)

    # REQ003 test file should be deleted on rollback to REQ002 (oldest_failing_req_idx = 1)
    assert not req003_test.exists()

    # Test error handling when os.remove fails during rollback
    req004_test = tmp_path / "test_py_bc_req004_unit.py"
    req004_test.write_text("def test_req004(): pass")

    def mock_remove_fail(path):
        raise OSError("Permission denied")

    with (
        patch("agent._call_llm_structured", return_value=mock_bug_report),
        patch("agent.save_history_snapshot"),
        patch("shutil.copy"),
        patch("os.remove", side_effect=mock_remove_fail),
    ):
        # Should catch exception and print warning
        res = agent.generate_regression_bug_report(state_rollback)
        assert res["current_req_index"] == 1

    # 4. Test robust .bak.N rotation in _cleanup_history_on_rollback
    state_cleanup = TDDState(
        requirements=[{"id": "REQ001", "description": "desc"}],
        current_req_index=0,
        test_iterations=1,
    )

    # Create baseline snapshot
    f_base = history_dir / "impl_req001_test_iter001_impl_iter001.py"
    f_base.write_text("code v1")

    # Run first cleanup: should rename to .bak
    agent._cleanup_history_on_rollback(state_cleanup)
    f_bak = history_dir / "impl_req001_test_iter001_impl_iter001.py.bak"
    assert f_bak.exists()
    assert not f_base.exists()

    # Recreate baseline and run again: should rename to .bak.1
    f_base.write_text("code v2")
    agent._cleanup_history_on_rollback(state_cleanup)
    f_bak_1 = history_dir / "impl_req001_test_iter001_impl_iter001.py.bak.1"
    assert f_bak_1.exists()
    assert f_bak.exists()

    # Recreate baseline and run again: should rename to .bak.2
    f_base.write_text("code v3")
    agent._cleanup_history_on_rollback(state_cleanup)
    f_bak_2 = history_dir / "impl_req001_test_iter001_impl_iter001.py.bak.2"
    assert f_bak_2.exists()
    assert f_bak_1.exists()


@patch("agent._call_llm_structured")
def test_analyze_architecture(mock_llm):
    from unittest.mock import mock_open, patch

    from agent import analyze_architecture
    from schema import ArchitectureAudit

    mock_audit = ArchitectureAudit(
        classification="architectural_bottleneck",
        architectural_bottleneck="conflation of statement and expression",
        refactoring_plan="1. split executor",
        safeties_and_invariants="preserve semicolons",
    )
    mock_llm.return_value = mock_audit

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "arithmetic"}],
        current_req_index=0,
        impl_code="a=1",
        test_output="failed assertion",
    )

    # 1. Spec file doesn't exist (architectural_bottleneck)
    with patch("os.path.exists", return_value=False):
        res = analyze_architecture(state)
        assert res["loop_detected"] is True
        assert "conflation of statement and expression" in res["architecture_audit"]
        assert "split executor" in res["architecture_audit"]
        assert "preserve semicolons" in res["architecture_audit"]
        assert res["next_action"] == "update_design_for_req"
        assert res["audit_loop_count"] == 0

    # 2. Spec file exists (architectural_bottleneck)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", mock_open(read_data="POSIX bc specification")):
            res = analyze_architecture(state)
            assert res["loop_detected"] is True
            assert "conflation of statement and expression" in res["architecture_audit"]
            assert res["next_action"] == "update_design_for_req"

    # 2b. Spec file exists but fails to read (architectural_bottleneck)
    with patch("os.path.exists", return_value=True):
        with patch("builtins.open", side_effect=IOError("read error")):
            res = analyze_architecture(state)
            assert res["loop_detected"] is True
            assert "conflation of statement and expression" in res["architecture_audit"]
            assert res["next_action"] == "update_design_for_req"

    # 3. Local bug classification routing
    mock_audit_local = ArchitectureAudit(
        classification="local_bug",
        architectural_bottleneck="local typo in parser",
        refactoring_plan="fix token check at line 50",
        safeties_and_invariants="preserve parser state",
    )
    mock_llm.return_value = mock_audit_local

    state_local = TDDState(
        requirements=[{"id": "REQ001", "description": "arithmetic"}],
        current_req_index=0,
        impl_code="a=1",
        test_output="failed assertion",
        loop_origin_node="implement_integration_logic",
        audit_loop_count=0,
    )
    with patch("os.path.exists", return_value=False):
        res = analyze_architecture(state_local)
        assert res["loop_detected"] is True
        assert res["next_action"] == "implement_integration_logic"
        assert "local typo in parser" in res["bug_report"]
        assert "fix token check" in res["bug_report"]
        assert res["audit_loop_count"] == 1
        assert res["iterations"] == 0
        assert res["stagnant_iterations"] == 0

    # 4. Circuit breaker escalation (2nd attempt forces architectural_bottleneck)
    state_circuit = TDDState(
        requirements=[{"id": "REQ001", "description": "arithmetic"}],
        current_req_index=0,
        impl_code="a=1",
        test_output="failed assertion",
        loop_origin_node="implement_initial_logic",
        audit_loop_count=1,
    )
    with patch("os.path.exists", return_value=False):
        res = analyze_architecture(state_circuit)
        assert res["loop_detected"] is True
        # Escalates to update_design_for_req
        assert res["next_action"] == "update_design_for_req"
        assert res["audit_loop_count"] == 0
        # Classification in report is updated to architectural_bottleneck
        assert "Classification:\narchitectural_bottleneck" in res["architecture_audit"]


def test_agent_module_delattr():
    import sys

    import agent

    # 1. Nonexistent attribute deletion (covers the except block)
    delattr(agent, "nonexistent_mock_attr")

    # 2. Existing attribute deletion on submodules propagation
    setattr(agent, "test_propagation_attr", "prop_val")
    assert getattr(sys.modules["agent.nodes"], "test_propagation_attr") == "prop_val"
    delattr(agent, "test_propagation_attr")
    assert not hasattr(sys.modules["agent.nodes"], "test_propagation_attr")
