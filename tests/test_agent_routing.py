from unittest.mock import patch

from tddrobo.agent import (
    _has_implementation_exceptions,
    generate_integration_bug_report,
    generate_unit_bug_report,
    should_fix_integration_tests_or_impl,
    should_fix_unit_tests_or_impl,
)
from tddrobo.schema import BugReport, DesignDocument, TDDState


def test_has_implementation_exceptions():
    # 1. Valid implementation exception
    output_1 = """
=================================== FAILURES ===================================
_______ TestBCAssignmentIntegration.test_compound_addition_assignment __________
py_bc.py:237: SyntaxError
E       SyntaxError: Unexpected token: Token(type='ASSIGN_OP', value='=')
"""
    assert _has_implementation_exceptions(output_1, "py_bc.py") is True

    # 2. Exception but from test file, not implementation
    output_2 = """
=================================== FAILURES ===================================
_______ TestBCAssignmentIntegration.test_compound_addition_assignment __________
test_py_bc.py:237: AttributeError: 'NoneType' object has no attribute 'val'
"""
    assert _has_implementation_exceptions(output_2, "py_bc.py") is False

    # 3. Normal assertion failure (no exceptions)
    output_3 = """
=================================== FAILURES ===================================
_______ TestBCAssignmentIntegration.test_compound_addition_assignment __________
E       AssertionError: assert '7' == '8'
"""
    assert _has_implementation_exceptions(output_3, "py_bc.py") is False

    # 4. Empty output
    assert _has_implementation_exceptions("", "py_bc.py") is False


@patch("tddrobo.agent._run_oracle_verification_on_failures")
@patch("tddrobo.agent._call_llm_structured")
def test_generate_unit_bug_report_with_exception(mock_call_llm, mock_oracle):
    mock_oracle.return_value = "ORACLE VERIFICATION FEEDBACK: Discrepancy found."

    # Mock LLM returning implement_logic
    bug_report = BugReport(
        failed_test_cases=["test_case_1"],
        expected_vs_actual="Expected 7, got 8",
        fix_instructions="Fix Lexer regex",
        target_to_fix="implement_logic",
    )
    mock_call_llm.return_value = bug_report

    # Test state with implementation exception
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        unit_tests_code="def test_case_1(): pass",
        impl_code="print('ok')",
        test_output='File "py_bc.py", line 10\nSyntaxError: unexpected EOF',
        module_name="py_bc.py",
    )

    res = generate_unit_bug_report(state)
    # Since there is an exception in py_bc.py, target_to_fix should NOT be overridden to generate_tests
    assert res["next_action"] == "implement_logic"


@patch("tddrobo.agent._run_oracle_verification_on_failures")
@patch("tddrobo.agent._call_llm_structured")
def test_generate_unit_bug_report_without_exception(mock_call_llm, mock_oracle):
    mock_oracle.return_value = "ORACLE VERIFICATION FEEDBACK: Discrepancy found."

    # Mock LLM returning implement_logic
    bug_report = BugReport(
        failed_test_cases=["test_case_1"],
        expected_vs_actual="Expected 7, got 8",
        fix_instructions="Fix Lexer regex",
        target_to_fix="implement_logic",
    )
    mock_call_llm.return_value = bug_report

    # Test state WITHOUT implementation exception (normal assertion error)
    state = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        unit_tests_code="def test_case_1(): pass",
        impl_code="print('ok')",
        test_output="E AssertionError: assert '7' == '8'",
        module_name="py_bc.py",
    )

    res = generate_unit_bug_report(state)
    # Since there is no exception, target_to_fix should be overridden to generate_tests
    assert res["next_action"] == "generate_tests"


@patch("tddrobo.agent._run_oracle_verification_on_failures")
@patch("tddrobo.agent._call_llm_structured")
def test_generate_integration_bug_report_with_exception(mock_call_llm, mock_oracle):
    mock_oracle.return_value = "ORACLE VERIFICATION FEEDBACK: Discrepancy found."

    bug_report = BugReport(
        failed_test_cases=["test_case_1"],
        expected_vs_actual="Expected 7, got 8",
        fix_instructions="Fix Lexer regex",
        target_to_fix="implement_logic",
    )
    mock_call_llm.return_value = bug_report

    state = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        integration_tests_code="def test_case_1(): pass",
        impl_code="print('ok')",
        test_output="File \"py_bc.py\", line 10\nNameError: name 'x' is not defined",
        module_name="py_bc.py",
    )

    res = generate_integration_bug_report(state)
    assert res["next_action"] == "implement_logic"


@patch("tddrobo.agent._detect_toggle_loop")
def test_should_fix_unit_tests_or_impl_loop(mock_detect_loop):
    # If loop is detected, should routing go to analyze_architecture?
    mock_detect_loop.return_value = True

    # Case A: next_action is generate_tests
    state_a = TDDState(next_action="generate_tests")
    res_a = should_fix_unit_tests_or_impl(state_a)
    assert res_a == "analyze_architecture"

    # Case B: next_action is implement_logic
    state_b = TDDState(next_action="implement_logic")
    res_b = should_fix_unit_tests_or_impl(state_b)
    assert res_b == "analyze_architecture"


@patch("tddrobo.agent._detect_toggle_loop")
def test_should_fix_integration_tests_or_impl_loop(mock_detect_loop):
    mock_detect_loop.return_value = True

    # Case A: next_action is generate_tests
    state_a = TDDState(next_action="generate_tests")
    res_a = should_fix_integration_tests_or_impl(state_a)
    assert res_a == "analyze_architecture"

    # Case B: next_action is implement_logic
    state_b = TDDState(next_action="implement_logic")
    res_b = should_fix_integration_tests_or_impl(state_b)
    assert res_b == "analyze_architecture"


@patch("tddrobo.agent._detect_toggle_loop")
def test_should_fix_unit_tests_or_impl_no_loop(mock_detect_loop):
    mock_detect_loop.return_value = False

    # next_action = generate_tests should return generate_unit_tests
    state_a = TDDState(next_action="generate_tests")
    res_a = should_fix_unit_tests_or_impl(state_a)
    assert res_a == "generate_unit_tests"

    # next_action = implement_logic should return implement_initial_logic
    state_b = TDDState(next_action="implement_logic")
    res_b = should_fix_unit_tests_or_impl(state_b)
    assert res_b == "implement_initial_logic"


@patch("tddrobo.agent.open", create=True)
@patch("tddrobo.agent._call_llm_structured")
def test_update_design_for_req_skip_condition(mock_call_llm, mock_open):
    from tddrobo.agent import update_design_for_req

    # Mock LLM and file open
    mock_call_llm.return_value = DesignDocument(
        module_responsibilities="...",
        architecture_and_components="...",
        interface_definitions="...",
        data_structures="...",
        logic_and_algorithms="...",
        edge_cases_and_limitations="...",
        error_handling="...",
        command_line_interface="...",
    )

    # Case A: pure oracle discrepancy ONLY -> Should skip design update
    state_a = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        oracle_discrepancy_only=True,
        loop_detected=False,
        design_doc="Old Design",
    )
    res_a = update_design_for_req(state_a)
    # If skipped, it just returns design_doc and oracle_discrepancy_only = False
    assert res_a["oracle_discrepancy_only"] is False
    assert mock_call_llm.call_count == 0

    # Case B: oracle discrepancy BUT loop is detected -> Should NOT skip design update
    state_b = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        oracle_discrepancy_only=True,
        loop_detected=True,
        design_doc="Old Design",
    )
    update_design_for_req(state_b)
    # Should call LLM structured since skip is bypassed
    assert mock_call_llm.call_count == 1


@patch("tddrobo.agent._run_oracle_verification_on_failures")
@patch("tddrobo.agent._call_llm_structured")
@patch("tddrobo.agent._get_combined_tests_code")
def test_oracle_discrepancy_clear_lifecycle(mock_get_combined_tests, mock_call_llm, mock_oracle):
    from tddrobo.agent import analyze_architecture

    mock_oracle.return_value = "No discrepancies"
    bug_report = BugReport(
        failed_test_cases=["t1"], expected_vs_actual="...", fix_instructions="...", target_to_fix="implement_logic"
    )
    mock_call_llm.return_value = bug_report
    mock_get_combined_tests.return_value = "def test_1(): pass"

    # 1. generate_unit_bug_report clears flag
    state_unit = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        unit_tests_code="def test_1(): pass",
        oracle_discrepancy_only=True,
    )
    res_unit = generate_unit_bug_report(state_unit)
    assert res_unit["oracle_discrepancy_only"] is False

    # 2. generate_integration_bug_report clears flag
    state_integ = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}],
        current_req_index=0,
        integration_tests_code="def test_1(): pass",
        oracle_discrepancy_only=True,
    )
    res_integ = generate_integration_bug_report(state_integ)
    assert res_integ["oracle_discrepancy_only"] is False

    # 3. analyze_architecture clears flag
    state_audit = TDDState(
        requirements=[{"id": "REQ001", "description": "Desc"}], current_req_index=0, oracle_discrepancy_only=True
    )

    # Mock ArchitectureAudit LLM call
    from tddrobo.schema import ArchitectureAudit as AuditSchema

    mock_call_llm.return_value = AuditSchema(
        classification="architectural_bottleneck",
        architectural_bottleneck="...",
        refactoring_plan="...",
        safeties_and_invariants="...",
    )

    res_audit = analyze_architecture(state_audit)
    assert res_audit["oracle_discrepancy_only"] is False
