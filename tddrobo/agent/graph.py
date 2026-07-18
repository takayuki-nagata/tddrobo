# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

from langgraph.graph import END, StateGraph

from tddrobo.schema import TDDState

from .edges import (
    should_continue_integration,
    should_continue_regression,
    should_continue_unit,
    should_continue_workflow,
    should_fix_integration_tests_or_impl,
    should_fix_refactor_or_continue,
    should_fix_regression_tests_or_impl,
    should_fix_unit_tests_or_impl,
    should_refactor,
    should_review_design_incremental_or_continue,
    should_review_design_initial_or_continue,
    should_review_integration_tests_or_continue,
    should_review_test_plan_or_continue,
    should_review_unit_tests_or_continue,
    should_route_from_audit,
    should_run_integration_tests,
    should_run_regression_after_refactor,
    should_run_regression_tests,
    should_run_unit_tests,
)
from .nodes import (
    analyze_architecture,
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
    generate_requirements,
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
    review_design_incremental,
    review_design_initial,
    review_integration_test_plan,
    review_unit_test_plan,
    run_integration_tests,
    run_regression_tests,
    run_unit_tests,
    update_design_for_req,
)


class TDDAgent:
    """An agent that builds and executes a LangGraph workflow to perform Test-Driven Development."""

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer
        self.app = self._build_graph()

    def invoke(self, state, config=None):
        return self.app.invoke(state, config=config)

    def get_graph(self):
        return self.app.get_graph()

    def _build_graph(self):
        workflow = StateGraph(TDDState)

        # Core Spec & Requirements Nodes
        workflow.add_node("fetch_spec", fetch_spec)
        workflow.add_node("generate_requirements", generate_requirements)
        workflow.add_node("plan_files", plan_files)

        # Design Phase Nodes
        workflow.add_node("generate_design_initial", generate_design_initial)
        workflow.add_node("update_design_for_req", update_design_for_req)
        workflow.add_node("review_design_initial", review_design_initial)
        workflow.add_node("review_design_incremental", review_design_incremental)

        # Unit Test Planning & Execution Nodes
        workflow.add_node("plan_unit_tests", plan_unit_tests)
        workflow.add_node("review_unit_test_plan", review_unit_test_plan)
        workflow.add_node("generate_unit_tests", generate_unit_tests)
        workflow.add_node("check_unit_tests_syntax", check_unit_tests_syntax)
        workflow.add_node("implement_initial_logic", implement_initial_logic)
        workflow.add_node("check_initial_impl_syntax", check_initial_impl_syntax)
        workflow.add_node("run_unit_tests", run_unit_tests)
        workflow.add_node("generate_unit_bug_report", generate_unit_bug_report)

        # Integration Test Planning & Execution Nodes
        workflow.add_node("plan_integration_tests", plan_integration_tests)
        workflow.add_node("review_integration_test_plan", review_integration_test_plan)
        workflow.add_node("generate_integration_tests", generate_integration_tests)
        workflow.add_node("check_integration_tests_syntax", check_integration_tests_syntax)
        workflow.add_node("implement_integration_logic", implement_integration_logic)
        workflow.add_node("check_integration_impl_syntax", check_integration_impl_syntax)
        workflow.add_node("run_integration_tests", run_integration_tests)
        workflow.add_node("generate_integration_bug_report", generate_integration_bug_report)

        # Regression Test Nodes
        workflow.add_node("run_regression_tests", run_regression_tests)
        workflow.add_node("generate_regression_bug_report", generate_regression_bug_report)
        workflow.add_node("implement_regression_logic", implement_regression_logic)
        workflow.add_node("check_regression_impl_syntax", check_regression_impl_syntax)

        # Refactoring Nodes
        workflow.add_node("decide_refactor", decide_refactor)
        workflow.add_node("refactor_logic", refactor_logic)
        workflow.add_node("check_refactored_impl_syntax", check_refactored_impl_syntax)
        workflow.add_node("generate_refactor_bug_report", generate_refactor_bug_report)

        # Finalization & Loop Escalation Nodes
        workflow.add_node("increment_requirement", increment_requirement)
        workflow.add_node("generate_readme", generate_readme)
        workflow.add_node("analyze_architecture", analyze_architecture)

        # Graph wiring
        workflow.set_entry_point("fetch_spec")
        workflow.add_edge("fetch_spec", "generate_requirements")
        workflow.add_edge("generate_requirements", "plan_files")
        workflow.add_edge("plan_files", "generate_design_initial")
        workflow.add_edge("generate_design_initial", "review_design_initial")
        workflow.add_conditional_edges(
            "review_design_initial",
            should_review_design_initial_or_continue,
            {
                "generate_design_initial": "generate_design_initial",
                "plan_unit_tests": "plan_unit_tests",
            },
        )
        workflow.add_edge("update_design_for_req", "review_design_incremental")
        workflow.add_conditional_edges(
            "review_design_incremental",
            should_review_design_incremental_or_continue,
            {
                "update_design_for_req": "update_design_for_req",
                "plan_unit_tests": "plan_unit_tests",
            },
        )

        # Unit Testing Phase Routing
        workflow.add_edge("plan_unit_tests", "review_unit_test_plan")
        workflow.add_conditional_edges(
            "review_unit_test_plan",
            should_review_test_plan_or_continue,
            {
                "plan_tests": "plan_unit_tests",
                "update_design_for_req": "update_design_for_req",
                "generate_tests": "generate_unit_tests",
            },
        )
        workflow.add_edge("generate_unit_tests", "check_unit_tests_syntax")
        workflow.add_conditional_edges(
            "check_unit_tests_syntax",
            should_review_unit_tests_or_continue,
            {
                "generate_unit_tests": "generate_unit_tests",
                "implement_initial_logic": "implement_initial_logic",
                "generate_unit_bug_report": "generate_unit_bug_report",
            },
        )
        workflow.add_edge("implement_initial_logic", "check_initial_impl_syntax")
        workflow.add_conditional_edges(
            "check_initial_impl_syntax",
            should_run_unit_tests,
            {
                "implement_initial_logic": "implement_initial_logic",
                "run_unit_tests": "run_unit_tests",
                "update_design_for_req": "update_design_for_req",
            },
        )
        workflow.add_conditional_edges(
            "run_unit_tests",
            should_continue_unit,
            {
                "plan_integration_tests": "plan_integration_tests",
                "generate_unit_bug_report": "generate_unit_bug_report",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            "generate_unit_bug_report",
            should_fix_unit_tests_or_impl,
            {
                "implement_initial_logic": "implement_initial_logic",
                "generate_unit_tests": "generate_unit_tests",
                "analyze_architecture": "analyze_architecture",
            },
        )

        # Integration Testing Phase Routing
        workflow.add_edge("plan_integration_tests", "review_integration_test_plan")
        workflow.add_conditional_edges(
            "review_integration_test_plan",
            should_review_test_plan_or_continue,
            {
                "plan_tests": "plan_integration_tests",
                "update_design_for_req": "update_design_for_req",
                "generate_tests": "generate_integration_tests",
            },
        )
        workflow.add_edge("generate_integration_tests", "check_integration_tests_syntax")
        workflow.add_conditional_edges(
            "check_integration_tests_syntax",
            should_review_integration_tests_or_continue,
            {
                "generate_integration_tests": "generate_integration_tests",
                "implement_integration_logic": "implement_integration_logic",
                "generate_integration_bug_report": "generate_integration_bug_report",
            },
        )
        workflow.add_edge("implement_integration_logic", "check_integration_impl_syntax")
        workflow.add_conditional_edges(
            "check_integration_impl_syntax",
            should_run_integration_tests,
            {
                "implement_integration_logic": "implement_integration_logic",
                "run_integration_tests": "run_integration_tests",
                "update_design_for_req": "update_design_for_req",
            },
        )
        workflow.add_conditional_edges(
            "run_integration_tests",
            should_continue_integration,
            {
                "run_regression_tests": "run_regression_tests",
                "generate_integration_bug_report": "generate_integration_bug_report",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            "generate_integration_bug_report",
            should_fix_integration_tests_or_impl,
            {
                "implement_integration_logic": "implement_integration_logic",
                "generate_integration_tests": "generate_integration_tests",
                "analyze_architecture": "analyze_architecture",
            },
        )

        # Regression Testing Routing
        workflow.add_conditional_edges(
            "run_regression_tests",
            should_continue_regression,
            {
                "decide_refactor": "decide_refactor",
                "increment_requirement": "increment_requirement",
                "generate_regression_bug_report": "generate_regression_bug_report",
                "generate_refactor_bug_report": "generate_refactor_bug_report",
                END: END,
            },
        )
        workflow.add_conditional_edges(
            "generate_regression_bug_report",
            should_fix_regression_tests_or_impl,
            {
                "implement_regression_logic": "implement_regression_logic",
                "generate_unit_tests": "generate_unit_tests",
                "generate_integration_tests": "generate_integration_tests",
                "analyze_architecture": "analyze_architecture",
                "update_design_for_req": "update_design_for_req",
                "halt_regression_test_failure": "increment_requirement",
            },
        )
        workflow.add_edge("implement_regression_logic", "check_regression_impl_syntax")
        workflow.add_conditional_edges(
            "check_regression_impl_syntax",
            should_run_regression_tests,
            {
                "implement_regression_logic": "implement_regression_logic",
                "run_regression_tests": "run_regression_tests",
                "update_design_for_req": "update_design_for_req",
            },
        )

        # Refactoring Phase Routing
        workflow.add_conditional_edges(
            "decide_refactor",
            should_refactor,
            {
                "refactor_logic": "refactor_logic",
                "increment_requirement": "increment_requirement",
            },
        )
        workflow.add_edge("refactor_logic", "check_refactored_impl_syntax")
        workflow.add_conditional_edges(
            "check_refactored_impl_syntax",
            should_run_regression_after_refactor,
            {
                "refactor_logic": "refactor_logic",
                "run_regression_tests": "run_regression_tests",
                "generate_refactor_bug_report": "generate_refactor_bug_report",
            },
        )
        workflow.add_conditional_edges(
            "generate_refactor_bug_report",
            should_fix_refactor_or_continue,
            {
                "refactor_logic": "refactor_logic",
                "increment_requirement": "increment_requirement",
            },
        )

        # Architecture Audit Routing
        workflow.add_conditional_edges(
            "analyze_architecture",
            should_route_from_audit,
            {
                "update_design_for_req": "update_design_for_req",
                "implement_initial_logic": "implement_initial_logic",
                "implement_integration_logic": "implement_integration_logic",
                "implement_regression_logic": "implement_regression_logic",
            },
        )

        # Advancement Routing
        workflow.add_conditional_edges(
            "increment_requirement",
            should_continue_workflow,
            {
                "update_design_for_req": "update_design_for_req",
                "generate_readme": "generate_readme",
            },
        )
        workflow.add_edge("generate_readme", END)

        return workflow.compile(checkpointer=self.checkpointer)
