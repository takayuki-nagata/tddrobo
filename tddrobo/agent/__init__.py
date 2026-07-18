# ruff: noqa: F401

import glob

# 1. Standard library and third-party imports used in agent and mocked in tests
import os
import shutil
import subprocess
import sys
import time
import types

import requests

# Expose utils helpers imported/mocked directly via agent
from tddrobo.utils import (
    add_line_numbers,
    apply_search_replace_blocks,
    call_llm_standard,
    call_llm_with_reasoning,
    extract_code,
    read_artifact,
    save_artifact,
)

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

# 2. Package-level exports
from .graph import TDDAgent
from .history import (
    _backup_project_before_rollback,
    _cleanup_history_on_rollback,
    save_history_snapshot,
)

# Expose print wrapper if tests patch it
# Define internal implementation helpers to support legacy mock patching
from .nodes import (
    _implement_logic_helper,
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
    print,
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
from .oracle import (
    _extract_oracle_target_llm,
    _judge_oracle_discrepancy_with_llm,
    _run_early_oracle_verification,
    _run_oracle_verification_on_failures,
)
from .prompt_builder import build_uniqueness_advice as _build_uniqueness_advice
from .runner import (
    _call_llm_structured,
    _call_llm_text,
    _call_llm_with_reasoning,
    _detect_toggle_loop,
    _execute_tests_helper,
    _extract_failing_line,
    _extract_method_body,
    _find_failed_methods,
    _get_balanced_test_output_context,
    _get_combined_tests_code,
    _get_dynamic_max_tokens,
    _get_existing_tests_context,
    _get_filtered_regression_test_code_context,
    _get_filtered_test_output,
    _get_regression_test_code_context,
    _has_implementation_exceptions,
    _parse_pytest_summary,
    _run_syntax_check,
    _syntax_check_helper,
    _thread_local,
    _truncate_test_output_smart,
)


# 3. Custom Module Wrapper to propagate dynamic mocking attributes from tests
class AgentModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        # Populate initial dictionary from the current module's dictionary
        self.__dict__.update(sys.modules[name].__dict__)

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        # List of submodules to propagate mock attribute updates
        submodules = [
            "tddrobo.agent.nodes",
            "tddrobo.agent.runner",
            "tddrobo.agent.oracle",
            "tddrobo.agent.history",
            "tddrobo.agent.edges",
            "tddrobo.agent.prompt_builder",
            "tddrobo.agent.graph",
        ]
        for submod_name in submodules:
            if submod_name in sys.modules:
                submod = sys.modules[submod_name]
                # Propagate the mock/override to all submodules
                setattr(submod, name, value)

    def __delattr__(self, name):
        try:
            super().__delattr__(name)
        except AttributeError:
            pass
        submodules = [
            "tddrobo.agent.nodes",
            "tddrobo.agent.runner",
            "tddrobo.agent.oracle",
            "tddrobo.agent.history",
            "tddrobo.agent.edges",
            "tddrobo.agent.prompt_builder",
            "tddrobo.agent.graph",
        ]
        for submod_name in submodules:
            if submod_name in sys.modules:
                submod = sys.modules[submod_name]
                if hasattr(submod, name):
                    delattr(submod, name)


# Override sys.modules entry to use our custom class
sys.modules[__name__] = AgentModule(__name__)
