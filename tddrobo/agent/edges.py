import re
from typing import cast

from langgraph.graph import END

from tddrobo import config
from tddrobo.logger import print
from tddrobo.schema import TDDState

from .runner import _detect_toggle_loop

MAX_ITERATIONS = config.MAX_ITERATIONS


def should_review_design_initial_or_continue(state: TDDState) -> str:
    if state.get("design_review_feedback"):
        return "generate_design_initial"
    return "plan_unit_tests"


def should_review_design_incremental_or_continue(state: TDDState) -> str:
    if state.get("design_review_feedback"):
        return "update_design_for_req"
    return "plan_unit_tests"


def should_review_test_plan_or_continue(state: TDDState):
    """Determine whether to regenerate the test plan, rollback to design, or proceed to test generation."""
    decision = state.get("test_plan_review_decision")
    if decision == "review_test_plan":
        return "plan_tests"
    elif decision == "update_design_for_req":
        return "update_design_for_req"
    return "generate_tests"


def should_review_unit_tests_or_continue(state: TDDState):
    """Determine whether to regenerate unit tests or proceed to implementation."""
    if state.get("tests_check_output"):
        max_syntax = config.MAX_SYNTAX_ERROR_ITERATIONS
        if state.get("test_syntax_error_iterations", 0) >= max_syntax:
            print(
                "[TDD Robo] 🚨 Unit Test Syntax Error Loop: "
                f"Max retries reached ({max_syntax}). Forcing transition to generate_unit_bug_report."
            )
            return "generate_unit_bug_report"
        return "generate_unit_tests"
    return "implement_initial_logic"


def should_review_integration_tests_or_continue(state: TDDState):
    """Determine whether to regenerate integration tests or proceed to implementation."""
    if state.get("tests_check_output"):
        max_syntax = config.MAX_SYNTAX_ERROR_ITERATIONS
        if state.get("test_syntax_error_iterations", 0) >= max_syntax:
            print(
                "[TDD Robo] 🚨 Integration Test Syntax Error Loop: "
                f"Max retries reached ({max_syntax}). Forcing transition to generate_integration_bug_report."
            )
            return "generate_integration_bug_report"
        return "generate_integration_tests"
    return "implement_integration_logic"


def should_run_unit_tests(state: TDDState):
    """Determine whether to fix implementation syntax errors or run the unit tests."""
    if state.get("impl_check_output"):
        max_syntax = config.MAX_SYNTAX_ERROR_ITERATIONS
        if state.get("syntax_error_iterations", 0) >= max_syntax:
            print(
                f"[TDD Robo] 🚨 Syntax Error Loop: Max syntax retries reached ({max_syntax}). "
                "Forcing transition to update_design_for_req."
            )
            return "update_design_for_req"
        return "implement_initial_logic"
    return "run_unit_tests"


def should_run_integration_tests(state: TDDState):
    """Determine whether to fix implementation syntax errors or run the integration tests."""
    if state.get("impl_check_output"):
        max_syntax = config.MAX_SYNTAX_ERROR_ITERATIONS
        if state.get("syntax_error_iterations", 0) >= max_syntax:
            print(
                f"[TDD Robo] 🚨 Syntax Error Loop: Max syntax retries reached ({max_syntax}). "
                "Forcing transition to update_design_for_req."
            )
            return "update_design_for_req"
        return "implement_integration_logic"
    return "run_integration_tests"


def should_run_regression_tests(state: TDDState):
    """Determine whether to fix implementation syntax errors or run regression tests."""
    if state.get("impl_check_output"):
        max_syntax = config.MAX_SYNTAX_ERROR_ITERATIONS
        if state.get("syntax_error_iterations", 0) >= max_syntax:
            print(
                f"[TDD Robo] 🚨 Syntax Error Loop: Max syntax retries reached ({max_syntax}). "
                "Forcing transition to update_design_for_req."
            )
            return "update_design_for_req"
        return "implement_regression_logic"
    return "run_regression_tests"


def should_run_regression_after_refactor(state: TDDState):
    """Determine whether to fix refactored syntax errors or run regression tests."""
    if state.get("impl_check_output"):
        max_syntax = config.MAX_SYNTAX_ERROR_ITERATIONS
        if state.get("syntax_error_iterations", 0) >= max_syntax:
            print(
                f"[TDD Robo] 🚨 Syntax Error Loop: Max syntax retries reached ({max_syntax}). "
                "Forcing transition to generate_refactor_bug_report."
            )
            return "generate_refactor_bug_report"
        return "refactor_logic"
    return "run_regression_tests"


def should_continue_unit(state: TDDState):
    """Determine whether to advance to integration tests or report unit bugs."""
    if state.get("success", False):
        return "plan_integration_tests"
    max_iters = state.get("max_iterations", MAX_ITERATIONS)
    if state.get("iterations", 0) >= max_iters:
        return END
    return "generate_unit_bug_report"


def should_continue_integration(state: TDDState):
    """Determine whether to advance to regression tests or report integration bugs."""
    if state.get("success", False):
        return "run_regression_tests"
    max_iters = state.get("max_iterations", MAX_ITERATIONS)
    if state.get("iterations", 0) >= max_iters:
        return END
    return "generate_integration_bug_report"


def should_continue_regression(state: TDDState):
    """Determine whether to proceed to refactoring/advancement or report regression/refactor bugs."""
    if state.get("success", False):
        if state.get("refactor_decision") == "refactor":
            return "increment_requirement"
        return "decide_refactor"

    if state.get("refactor_decision") == "refactor":
        return "generate_refactor_bug_report"

    max_iters = state.get("max_iterations", MAX_ITERATIONS)
    if state.get("iterations", 0) >= max_iters:
        return END
    return "generate_regression_bug_report"


def should_fix_unit_tests_or_impl(state: TDDState):
    """Determine next action for unit failures, with toggle loop detection."""
    if _detect_toggle_loop(state):
        print("[TDD Robo] 🔄 Loop Detector Override: Stuck in loop. Transitioning to analyze_architecture.")
        return "analyze_architecture"

    next_act = state.get("next_action", "implement_initial_logic")
    if next_act == "implement_logic":
        next_act = "implement_initial_logic"
    if next_act == "generate_tests":
        return "generate_unit_tests"
    return next_act


def should_fix_integration_tests_or_impl(state: TDDState):
    """Determine next action for integration failures, with toggle loop detection."""
    if _detect_toggle_loop(state):
        print("[TDD Robo] 🔄 Loop Detector Override: Stuck in loop. Transitioning to analyze_architecture.")
        return "analyze_architecture"

    next_act = state.get("next_action", "implement_integration_logic")
    if next_act == "implement_logic":
        next_act = "implement_integration_logic"
    if next_act == "generate_tests":
        return "generate_integration_tests"
    return next_act


def should_fix_regression_tests_or_impl(state: TDDState):
    """Determine next action for regression failures, with toggle loop detection."""
    if _detect_toggle_loop(state):
        print("[TDD Robo] 🔄 Loop Detector Override: Stuck in loop. Transitioning to analyze_architecture.")
        return "analyze_architecture"

    next_act = state.get("next_action", "implement_regression_logic")
    if next_act == "implement_logic":
        next_act = "implement_regression_logic"
    if next_act == "generate_design":
        return "update_design_for_req"
    if next_act == "generate_tests":
        test_output = state.get("test_output", "")
        reqs = state.get("requirements", [])
        idx = state.get("current_req_index", 0)
        active_req_id = ""
        if reqs and idx < len(reqs):
            active_req_id = str(reqs[idx].get("id") or "").lower()

        failed_files = cast(list[str], state.get("failed_files", []))
        failed_req_nums = []
        for f in failed_files:
            match = re.search(r"test_[a-zA-Z0-9_]+_req(\d+)_(?:unit|integration)\.py", f)
            if match:
                failed_req_nums.append(int(match.group(1)))

        if not failed_req_nums:
            for line in test_output.splitlines():
                if "FAILED" in line or "ERROR" in line:
                    matches = re.findall(r"test_[a-zA-Z0-9_]+_req(\d+)_(?:unit|integration)\.py", line)
                    for m in matches:
                        failed_req_nums.append(int(m))

        policy = state.get("regression_failure_policy", "rollback")
        if failed_req_nums:
            for req_num in failed_req_nums:
                req_idx = req_num - 1
                if req_idx < idx:
                    if policy == "halt":
                        print(
                            f"[TDD Robo] 🚨 Circuit Breaker: Regression failure detected in a historical test "
                            f"(REQ{req_num:03d}). Policy is 'halt'. Halting workflow."
                        )
                        return "halt_regression_test_failure"
                    else:
                        print(
                            f"[TDD Robo] 🚨 Rollback: Regression failure detected in a historical test "
                            f"(REQ{req_num:03d}). Routing to update_design_for_req."
                        )
                        return "update_design_for_req"

        unit_test_pattern = f"_{active_req_id}_unit"
        integration_test_pattern = f"_{active_req_id}_integration"

        if unit_test_pattern in test_output:
            print(f"[TDD Robo] Routing regression failure fix to Unit Tests (matched {unit_test_pattern})")
            return "generate_unit_tests"
        elif integration_test_pattern in test_output:
            print(
                f"[TDD Robo] Routing regression failure fix to Integration Tests (matched {integration_test_pattern})"
            )
            return "generate_integration_tests"
        else:
            if "_unit" in test_output:
                print("[TDD Robo] Routing regression failure fix to Unit Tests (matched '_unit')")
                return "generate_unit_tests"
            print("[TDD Robo] Routing regression failure fix to Integration Tests (fallback)")
            return "generate_integration_tests"
    return next_act


def should_fix_refactor_or_continue(state: TDDState):
    """Determine next action for refactor failures (always refactor_logic unless rolled back)."""
    if state.get("next_action") == "rollback_continue":
        return "increment_requirement"
    return "refactor_logic"


def should_refactor(state: TDDState):
    """Route based on the refactor decision."""
    if state.get("refactor_decision") == "refactor":
        return "refactor_logic"
    return "increment_requirement"


def should_continue_workflow(state: TDDState):
    """Determine whether to process the next requirement or finish."""
    requirements = state.get("requirements", [])
    current_index = state.get("current_req_index", 0)
    if current_index < len(requirements):
        return "update_design_for_req"
    return "generate_readme"


def should_route_from_audit(state: TDDState):
    """Determine the next node to route to from architectural audit."""
    return state.get("next_action", "update_design_for_req")
