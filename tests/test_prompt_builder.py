from tddrobo.agent.prompt_builder import (
    build_architecture_audit_prompt,
    build_design_prompt,
    build_implementation_prompt,
    build_refactor_prompt,
    build_test_generation_prompt,
    build_test_plan_prompt,
    build_uniqueness_advice,
)


def test_build_test_plan_prompt():
    state = {
        "goal": "Calculator",
        "spec_content": "Add two numbers",
        "design_doc": "Use Calculator class",
        "unit_test_plan": "Test cases list",
        "bug_report": "Oracle mismatch",
    }

    # Fresh plan
    prompt = build_test_plan_prompt(state, "unit", "REQ001", "Math constraints", is_fix=False)
    assert "Calculator" in prompt
    assert "Oracle mismatch" in prompt

    # Plan fix
    prompt_fix = build_test_plan_prompt(
        state, "unit", "REQ001", "Math constraints", is_fix=True, review_feedback="Review comments"
    )
    assert "Review comments" in prompt_fix

    # Integration phase
    prompt_integ = build_test_plan_prompt(state, "integration", "REQ001", "Math constraints", is_fix=False)
    assert "Calculator" in prompt_integ

    # Integration phase fix
    prompt_integ_fix = build_test_plan_prompt(
        state, "integration", "REQ001", "Math constraints", is_fix=True, review_feedback="Integ comments"
    )
    assert "Integ comments" in prompt_integ_fix


def test_build_test_generation_prompt():
    state = {
        "design_doc": "DesignDoc",
        "impl_code": "def run(): pass",
        "unit_test_plan": "Test cases list",
        "integration_test_plan": "Integration test list",
    }

    # Unit
    prompt_unit = build_test_generation_prompt(
        state, "unit", "REQ001", "previous test code", "mismatch bug", "syntax error output"
    )
    assert "DesignDoc" in prompt_unit
    assert "mismatch bug" in prompt_unit
    assert "syntax error output" in prompt_unit

    # Integration
    prompt_integ = build_test_generation_prompt(
        state, "integration", "REQ001", "previous test code", "mismatch bug", "syntax error output"
    )
    assert "DesignDoc" in prompt_integ


def test_build_uniqueness_advice():
    # No uniqueness error
    assert build_uniqueness_advice("Success logic check", "def run(): pass") == ""

    # Uniqueness error - matches multiple times
    impl_check = (
        "Failed to apply Search/Replace block. matches multiple times. Matches found at line numbers: [0, 2, 6]"
    )
    existing = "line1\ntarget\nsame\nline3\nline4\ntarget\nsame\nline5"
    advice = build_uniqueness_advice(impl_check, existing)
    assert "CRITICAL ERROR" in advice
    assert "SEARCH" in advice

    # Distinguishing line selection test cases
    existing_dist = "foo\ntarget\nbar\ntarget\nbaz"
    advice_dist = build_uniqueness_advice(impl_check, existing_dist)
    assert "SEARCH" in advice_dist

    # Empty indices / empty matches list
    impl_check_empty = "Failed to apply Search/Replace block. matches multiple times. Matches found at line numbers: []"
    advice_empty = build_uniqueness_advice(impl_check_empty, existing)
    assert "multiple times" in advice_empty


def test_build_implementation_prompt():
    state = {
        "goal": "Build calculator",
        "design_doc": "DesignDoc",
        "bug_report": "",
        "loop_detected": True,
        "requirements_list_str": "checklist",
        "impl_code": "def run(): pass",
        "impl_check_output": (
            "Failed to apply Search/Replace block. matches multiple times. Matches found at line numbers: [2, 4]\n"
            "Target SEARCH block that failed to match:\ntarget_search_block_not_matched\n"
        ),
    }

    # Not bug fix, loop detected, check output fails, design not updated
    prompt = build_implementation_prompt(
        state,
        "unit",
        "REQ001",
        "existing code",
        "existing param",
        "tests code",
        "domain tips",
        "python tips",
        state["impl_check_output"],
        "impl.py",
        False,
    )
    assert "DesignDoc" in prompt
    assert "CRITICAL: PREVIOUS MODIFICATION CHECK FAILED!" in prompt
    assert "target_search_block_not_matched" in prompt

    # Design updated
    prompt_design = build_implementation_prompt(
        state,
        "unit",
        "REQ001",
        "existing code",
        "existing param",
        "tests code",
        "domain tips",
        "python tips",
        "",
        "impl.py",
        True,
    )
    assert "DESIGN WAS RECENTLY UPDATED" in prompt_design

    # Bug fix
    state["bug_report"] = "Standard error report"
    prompt_bug = build_implementation_prompt(
        state,
        "unit",
        "REQ001",
        "existing code",
        "existing param",
        "tests code",
        "domain tips",
        "python tips",
        "",
        "impl.py",
        False,
    )
    assert "Standard error report" in prompt_bug


def test_build_design_prompt():
    state = {
        "goal": "Calculator",
        "spec_content": "Spec",
        "loop_detected": True,
        "architecture_audit": "Audit Bottleneck Details",
    }
    prompt = build_design_prompt(state, "REQ001", "DesignContext", "ImplCode")
    assert "Calculator" in prompt
    assert "Audit Bottleneck Details" in prompt


def test_build_refactor_prompt():
    state = {
        "design_doc": "DesignDoc",
    }

    # Simple refactor
    prompt, reasons = build_refactor_prompt(state, "Better structure", "", "tips", "def run(): pass", 1, "impl.py")
    assert "Better structure" in prompt
    assert reasons == "Better structure"

    # Bug fix refactor
    prompt_fix, reasons_fix = build_refactor_prompt(
        state, "Better structure", "Failed test regression", "tips", "def run(): pass", 1, "impl.py"
    )
    assert "Failed test regression" in prompt_fix
    assert "PREVIOUS REFACTORING ATTEMPT BROKE EXISTING TESTS" in reasons_fix


def test_build_architecture_audit_prompt():
    prompt = build_architecture_audit_prompt("Spec", "REQ001", "ImplCode", "TestsCode", "TestOutput")
    assert "Spec" in prompt
    assert "REQ001" in prompt
