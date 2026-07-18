import difflib
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import cast

import markdownify
import requests

import config
from logger import logger, print
from prompts import (
    DECIDE_REFACTOR_PROMPT,
    GENERATE_INTEGRATION_BUG_REPORT_PROMPT,
    GENERATE_README_PROMPT,
    GENERATE_REFACTOR_BUG_REPORT_PROMPT,
    GENERATE_REGRESSION_BUG_REPORT_PROMPT,
    GENERATE_REQUIREMENTS_PROMPT,
    GENERATE_UNIT_BUG_REPORT_PROMPT,
    PLAN_FILES_PROMPT,
    REFACTOR_LOGIC_FALLBACK_WARNING,
    REFACTOR_LOGIC_PROMPT,
    REVIEW_DESIGN_PROMPT,
    REVIEW_TEST_PLAN_PROMPT,
    TEST_PLAN_ORACLE_CONSTRAINTS,
)
from schema import (
    ArchitectureAudit,
    BugReport,
    DesignDocument,
    DesignReviewReport,
    FilePlan,
    RefactorDecision,
    RequirementsList,
    TDDState,
    TestCase,
    TestPlan,
    TestPlanReviewReport,
)
from utils import (
    add_line_numbers,
    apply_search_replace_blocks,
    call_llm_standard,
    extract_code,
    extract_json,
    read_artifact,
    save_artifact,
)

from .history import (
    _backup_project_before_rollback,
    _cleanup_history_on_rollback,
    save_history_snapshot,
)
from .oracle import (
    _judge_oracle_discrepancy_with_llm,
    _run_early_oracle_verification,
    _run_oracle_verification_on_failures,
)
from .prompt_builder import (
    build_architecture_audit_prompt,
    build_design_prompt,
    build_implementation_prompt,
    build_refactor_prompt,
    build_test_generation_prompt,
    build_test_plan_prompt,
)
from .runner import (
    _call_llm_structured,
    _call_llm_text,
    _call_llm_with_reasoning,
    _execute_tests_helper,
    _get_balanced_test_output_context,
    _get_combined_tests_code,
    _get_existing_tests_context,
    _get_filtered_regression_test_code_context,
    _get_filtered_test_output,
    _get_regression_test_code_context,
    _has_implementation_exceptions,
    _syntax_check_helper,
)

DEFAULT_IMPL_NAME = config.DEFAULT_IMPL_NAME
DEFAULT_TEST_NAME = config.DEFAULT_TEST_NAME
FETCH_TIMEOUT_SEC = config.FETCH_TIMEOUT_SEC
MAX_ITERATIONS = config.MAX_ITERATIONS
MAX_TEST_PLAN_ITERATIONS = config.MAX_TEST_PLAN_ITERATIONS
MAX_TEST_ITERATIONS = config.MAX_TEST_ITERATIONS
TARGET_TEST_PLAN_COVERAGE = config.TARGET_TEST_PLAN_COVERAGE
TARGET_TEST_COVERAGE = config.TARGET_TEST_COVERAGE
TARGET_DESIGN_QUALITY = config.TARGET_DESIGN_QUALITY


def _update_req_progress(state: TDDState):
    """Update requirement progress indicators from state in a thread-safe manner."""
    reqs = state.get("requirements", [])
    if reqs:
        total = len(reqs)
        current = state.get("current_req_index", 0) + 1
        logger.update_progress(current, total)


# --- Sub-section: Spec & Requirements Phase Nodes ---


def fetch_spec(state: TDDState):
    """Fetch the specification from the provided URL or local cache."""
    _update_req_progress(state)
    spec_url = state.get("spec_url", "")
    is_url = spec_url.startswith("http://") or spec_url.startswith("https://")
    spec_path = os.path.join(config.ARTIFACTS_DIR, "specification.txt")

    if not is_url:
        print(f"[TDD Robo] 📄 Loading local specification from '{spec_url}'...")
        if not os.path.exists(spec_url):
            print(f"[TDD Robo] ❌ Local specification file not found: {spec_url}")
            return {"spec_content": f"Error: Local specification file not found: {spec_url}"}
        try:
            with open(spec_url, "r", encoding="utf-8") as f:
                content = f.read()
            if spec_url.endswith(".html") or spec_url.endswith(".htm"):
                content = markdownify.markdownify(content, heading_style="ATX")

            save_artifact("specification.txt", content)
            print(f"[TDD Robo] ✅ Loaded and saved specification to {spec_path}")
            return {"spec_content": content}
        except Exception as e:
            print(f"[TDD Robo] ❌ Failed to read local specification: {e}")
            return {"spec_content": f"Error reading local specification: {e}"}

    headers = {}
    if os.path.exists(spec_path):
        mtime = os.path.getmtime(spec_path)
        headers["If-Modified-Since"] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(mtime))

    print("[TDD Robo] 🌐 Checking and downloading specification...")
    try:
        response = requests.get(spec_url, headers=headers, timeout=FETCH_TIMEOUT_SEC)

        if response.status_code == 304:
            if config.VERBOSE:
                print("[TDD Robo] Remote file not modified. Loading cached specification.")
            spec_content = read_artifact("specification.txt")
            print(f"[TDD Robo] ✅ Loaded specification from cache ({spec_path})")
            return {"spec_content": spec_content}

        response.raise_for_status()
        content = response.text
    except requests.exceptions.RequestException as e:
        print(f"[TDD Robo] ❌ Failed to fetch specification: {e}")
        return {"spec_content": f"Error fetching specification: {e}"}

    if "text/html" in response.headers.get("Content-Type", "").lower() or spec_url.endswith(".html"):
        content = markdownify.markdownify(content, heading_style="ATX")

    spec_path = save_artifact("specification.txt", content)
    print(f"[TDD Robo] ✅ Saved specification to {spec_path}")
    return {"spec_content": content}


def generate_requirements(state: TDDState):
    """Analyze the specification and extract a list of verifiable functional requirements."""
    _update_req_progress(state)
    print("[TDD Robo] 📋 Analyzing specification and extracting requirements...")
    prompt = GENERATE_REQUIREMENTS_PROMPT.format(spec=state.get("spec_content", ""))

    response = call_llm_standard(prompt, response_schema=RequirementsList)
    reqs_data = json.loads(extract_json(response))

    requirements = []
    requirements_list_str = "Requirements Checklist:\n"
    for r in reqs_data.get("requirements", []):
        req_dict = {"id": r.get("id"), "description": r.get("description")}
        requirements.append(req_dict)
        requirements_list_str += f"- {req_dict['id']}: {req_dict['description']}\n"

    print(f"[TDD Robo] ✅ Extracted {len(requirements)} sequential requirements.")
    if config.VERBOSE:
        print(requirements_list_str)

    req_path = save_artifact("requirements.md", requirements_list_str)
    print(f"[TDD Robo] ✅ Saved requirements list to {req_path}")

    print("\n=================== 📊 TDD PROGRESS STATUS =================== ")
    for idx, req in enumerate(requirements):
        req_id = req.get("id", f"REQ{idx + 1:03d}")
        desc = req.get("description", "")
        desc_short = desc if len(desc) <= 50 else desc[:47] + "..."
        status = "⏳ Active " if idx == 0 else "💤 Pending"
        print(f"[{req_id}] {status} - {desc_short}")
    print("==============================================================\n")

    return {
        "requirements": requirements,
        "current_req_index": 0,
        "requirements_list_str": requirements_list_str,
        "loop_detected": False,
    }


def plan_files(state: TDDState):
    """Determine filenames for the implementation and test modules."""
    _update_req_progress(state)
    print("[TDD Robo] 📁 Determining filenames...")
    prompt = PLAN_FILES_PROMPT.format(goal=state.get("goal", ""))

    response = call_llm_standard(prompt, response_schema=FilePlan)

    plan = json.loads(extract_json(response))
    impl_name = plan.get("impl_filename", DEFAULT_IMPL_NAME)
    test_name = plan.get("test_filename", DEFAULT_TEST_NAME)

    print(f"[TDD Robo] ✅ Determined filenames: Implementation={impl_name}, Test={test_name}")
    return {"module_name": impl_name, "test_module_name": test_name}


# --- Sub-section: Design Phase Nodes ---


def generate_design_initial(state: TDDState):
    """Generate the initial system architecture Design Doc."""
    _update_req_progress(state)
    iters = state.get("design_iterations", 0) + 1
    print(f"[TDD Robo] 📐 Generating initial design (Iteration {iters})...")

    feedback = state.get("design_review_feedback", "")
    design_context = ""
    if feedback:
        design_context = (
            f"\n# Design Review Feedback\n"
            f"The previous design draft was audited and found incomplete. "
            f"You MUST address the following gaps in your updated design document:\n"
            f"{feedback}\n"
        )

    prompt = build_design_prompt(state, "initial", design_context, "")
    design_doc_obj = _call_llm_structured(prompt, DesignDocument)

    formatted_design = f"""# Software Design Document

## 1. Module Responsibilities
{design_doc_obj.module_responsibilities}

## 2. Architecture & Components
{design_doc_obj.architecture_and_components}

## 3. Interface Definitions
{design_doc_obj.interface_definitions}

## 4. Data Structures
{design_doc_obj.data_structures}

## 5. Logic & Algorithms
{design_doc_obj.logic_and_algorithms}

## 6. Edge Cases & Limitations
{design_doc_obj.edge_cases_and_limitations}

## 7. Error Handling
{design_doc_obj.error_handling}

## 8. Command-Line Interface (CLI)
{design_doc_obj.command_line_interface}
"""
    design_path = save_artifact("design.md", formatted_design)
    print(f"[TDD Robo] ✅ Saved initial design to {design_path}")
    save_history_snapshot("design.md", formatted_design, iters, state)

    return {
        "design_doc": formatted_design,
        "design_iterations": iters,
        "design_updated": True,
        "iterations": 0,
        "loop_detected": False,
        "stagnant_iterations": 0,
        "last_test_summary": "",
    }


def update_design_for_req(state: TDDState):
    """Update and refine Design Doc incrementally for the target requirement."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"ID: {target_req.get('id')}\nDescription: {target_req.get('description')}"

    if state.get("oracle_discrepancy_only", False) and not state.get("loop_detected", False):
        print(f"[TDD Robo] 📐 Skipping design update for {target_req.get('id')} (reason: pure oracle discrepancy).")
        formatted_design = state.get("design_doc", "")
        if not formatted_design:
            try:
                with open(os.path.join(config.ARTIFACTS_DIR, "design.md"), "r", encoding="utf-8") as f:
                    formatted_design = f.read()
            except Exception:
                pass

        return {
            "design_doc": formatted_design,
            "design_iterations": state.get("design_iterations", 0),
            "design_updated": True,
            "oracle_discrepancy_only": False,
            "iterations": 0,
            "loop_detected": False,
            "test_plan_iterations": 0,
            "test_iterations": 0,
            "unit_test_iterations": 0,
            "integration_test_iterations": 0,
            "regression_test_iterations": 0,
            "refactor_iterations": 0,
            "unit_test_plan": None,
            "integration_test_plan": None,
            "test_plan": None,
            "test_plan_review": "",
            "test_plan_review_decision": "",
        }

    if not state.get("design_updated", False):
        if state.get("loop_detected"):
            _cleanup_history_on_rollback(state)
        iters = state.get("design_iterations", 0) + 1
        print(f"[TDD Robo] 📐 Incrementally updating design for {target_req.get('id')} (Iteration {iters})...")

        design_context = (
            f"\n# Target Requirement Update\n"
            f"We are now designing components to support the following requirement:\n"
            f"{target_req_str}\n"
            "Please update the architectural components, internal structures, "
            "or component interfaces in the Design Document to support this requirement cleanly.\n"
        )

        feedback = state.get("design_review_feedback", "")
        if feedback:
            design_context += (
                f"\n## 🚨 Design Review Feedback\n"
                f"The previous design draft was audited and found incomplete. "
                f"You MUST address the following gaps in your updated design document:\n"
                f"{feedback}\n"
            )

        if state.get("loop_detected", False) or state.get("bug_report") or state.get("test_output"):
            latest_bug = state.get("bug_report", "")
            latest_test = state.get("test_output", "")
            if latest_bug or latest_test:
                defect_context = (
                    "\n## 🚨 Previous Implementation Failure Context\n"
                    "Our recent implementation attempts failed to pass tests. To break the loop, "
                    "you must adjust the architectural design, core component structures, "
                    "or interface signatures to resolve the failure.\n"
                )
                if latest_bug:
                    defect_context += f"### Latest Defect Bug Report:\n{latest_bug}\n"
                if latest_test:
                    truncated_test = _get_balanced_test_output_context(
                        latest_test, max_chars=config.BUG_REPORT_TEST_OUTPUT_MAX_CHARS
                    )
                    defect_context += f"### Latest Test Output (Truncated):\n{truncated_test}\n"
                design_context += defect_context

        last_green = state.get("last_green_impl_code", "")
        impl_code_for_design = last_green if last_green else state.get("impl_code", "")

        prompt = build_design_prompt(state, target_req.get("id"), design_context, impl_code_for_design)
        design_doc_obj = _call_llm_structured(prompt, DesignDocument)

        formatted_design = f"""# Software Design Document (Updated for {target_req.get("id")})

## 1. Module Responsibilities
{design_doc_obj.module_responsibilities}

## 2. Architecture & Components
{design_doc_obj.architecture_and_components}

## 3. Interface Definitions
{design_doc_obj.interface_definitions}

## 4. Data Structures
{design_doc_obj.data_structures}

## 5. Logic & Algorithms
{design_doc_obj.logic_and_algorithms}

## 6. Edge Cases & Limitations
{design_doc_obj.edge_cases_and_limitations}

## 7. Error Handling
{design_doc_obj.error_handling}

## 8. Command-Line Interface (CLI)
{design_doc_obj.command_line_interface}
"""
        design_path = save_artifact("design.md", formatted_design)
        print(f"[TDD Robo] ✅ Saved updated design to {design_path}")
        save_history_snapshot("design.md", formatted_design, iters, state)

        return {
            "design_doc": formatted_design,
            "design_iterations": iters,
            "design_updated": True,
            "iterations": 0,
            "loop_detected": False,
            "test_plan_iterations": 0,
            "test_iterations": 0,
            "unit_test_iterations": 0,
            "integration_test_iterations": 0,
            "regression_test_iterations": 0,
            "refactor_iterations": 0,
            "unit_test_plan": None,
            "integration_test_plan": None,
            "test_plan": None,
            "test_plan_review": None,
            "test_plan_review_decision": None,
            "stagnant_iterations": 0,
            "last_test_summary": "",
        }
    return {}


def _review_design_generic(state: TDDState, phase: str):
    """Audit the Software Design Document against its raw specification."""
    _update_req_progress(state)
    print(f"[TDD Robo] 📐 Auditing software design document ({phase} phase)...")

    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if reqs and 0 <= idx < len(reqs):
        target_req = reqs[idx]
        active_requirement = f"{target_req.get('id')}: {target_req.get('description')}"
    else:
        active_requirement = f"REQ{idx + 1:03d} (Active Index: {idx})"

    prompt = REVIEW_DESIGN_PROMPT.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        design_doc=state.get("design_doc", ""),
        active_requirement=active_requirement,
    )

    report = _call_llm_structured(prompt, DesignReviewReport, model_name=config.MODEL_PRIMARY)

    quality = report.estimated_quality
    comments = report.comments
    print(f"[TDD Robo] Design Audit Report ({phase}): Estimated Quality = {quality}%")
    print(f"[TDD Robo] Audit Comments:\n{comments}")

    iters = state.get("design_review_iterations", 0) + 1
    target_quality = state.get("target_design_quality", TARGET_DESIGN_QUALITY)

    if quality < target_quality and iters <= config.MAX_DESIGN_REVIEW_ITERATIONS:
        print(
            f"[TDD Robo] 🚨 Design quality check failed ({quality}% < {target_quality}%). "
            f"Looping back to refine design. Iteration {iters}/{config.MAX_DESIGN_REVIEW_ITERATIONS}"
        )
        return {
            "design_review_feedback": comments,
            "design_review_iterations": iters,
            "design_updated": False,
        }
    else:
        if quality < target_quality:
            print(
                f"[TDD Robo] ⚠️ Max design review iterations reached "
                f"({iters}/{config.MAX_DESIGN_REVIEW_ITERATIONS}). Proceeding with current design."
            )
        else:
            print("[TDD Robo] ✅ Design quality check passed.")
        return {
            "design_review_feedback": "",
            "design_review_iterations": 0,
        }


def review_design_initial(state: TDDState):
    """Audit the initial Software Design Document."""
    return _review_design_generic(state, phase="initial")


def review_design_incremental(state: TDDState):
    """Audit the updated/incremental Software Design Document."""
    return _review_design_generic(state, phase="incremental")


# --- Sub-section: Test Planning Nodes ---


def plan_unit_tests(state: TDDState):
    """Plan Unit test cases based on Design component interfaces."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    iters = state.get("test_plan_iterations", 0) + 1
    print(f"[TDD Robo] 📝 Planning Unit Tests for {target_req.get('id')} (Iteration {iters})...")

    review_feedback = state.get("test_plan_review", "")
    oracle_constraints = TEST_PLAN_ORACLE_CONSTRAINTS if getattr(config, "ORACLE_VERIFIER", None) is not None else ""

    is_fix = bool(review_feedback and state.get("unit_test_plan"))
    prompt = build_test_plan_prompt(state, "unit", target_req_str, oracle_constraints, is_fix, review_feedback)

    test_plan_obj = _call_llm_structured(prompt, TestPlan, model_name=config.MODEL_PRIMARY)

    if is_fix:
        if state.get("oracle_discrepancy_only"):
            pass
        else:
            try:
                existing_data = json.loads(state.get("unit_test_plan", "{}"))
                existing_cases = existing_data.get("test_cases", [])
            except Exception:
                existing_cases = []

            existing_cases_dict = {
                re.sub(r"\s+", " ", tc.get("action", "")).strip().lower(): tc
                for tc in existing_cases
                if tc.get("action")
            }
            for tc in test_plan_obj.test_cases:
                normalized_action = re.sub(r"\s+", " ", tc.action).strip().lower()
                if normalized_action:
                    existing_cases_dict[normalized_action] = tc.model_dump()

            test_plan_obj.test_cases = [TestCase(**tc) for tc in existing_cases_dict.values()]

    plan_md = f"# Unit Test Plan for {target_req.get('id')}\n\n"
    for idx_tc, tc in enumerate(test_plan_obj.test_cases, 1):
        plan_md += f"{idx_tc}. Action: {tc.action} | Expected: {tc.expected_outcome}\n"

    plan_path = save_artifact("test_plan.md", plan_md)
    print(f"[TDD Robo] ✅ Saved unit test plan to {plan_path}")

    test_plan_json = json.dumps(test_plan_obj.model_dump(), indent=2)
    return {
        "unit_test_plan": test_plan_json,
        "test_plan": plan_md,
        "test_plan_iterations": iters,
    }


def review_unit_test_plan(state: TDDState):
    """Review the generated Unit Test Plan against the design components."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    print(f"[TDD Robo] 🧐 Reviewing Unit Test Plan for {target_req.get('id')}...")

    prompt = REVIEW_TEST_PLAN_PROMPT.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        design=state.get("design_doc", ""),
        requirements_list_str=state.get("requirements_list_str", ""),
        target_requirement=target_req_str,
        test_plan=state.get("test_plan", ""),
        target_coverage=state.get("target_test_plan_coverage", TARGET_TEST_PLAN_COVERAGE),
    )

    review_report = _call_llm_structured(prompt, TestPlanReviewReport, model_name=config.MODEL_PRIMARY)
    coverage = review_report.estimated_coverage
    print(f"[TDD Robo] ✅ Unit test plan review completed. Estimated coverage: {coverage}%")

    test_plan_json = state.get("unit_test_plan", "")
    mismatches = _run_early_oracle_verification(test_plan_json)

    decision = "continue"
    feedback = review_report.feedback

    if mismatches:
        print(f"[TDD Robo] 🧐 Found {len(mismatches)} early oracle mismatches. Invoking LLM Judge...")
        design_doc = state.get("design_doc", "")
        design_error_detected = False
        discrepancy_logs = []

        for mis in mismatches:
            tc = mis["test_case"]
            oracle_val = mis["oracle_val"]
            expected_val = mis["expected_val"]
            action = tc.get("action", "")

            judgment = _judge_oracle_discrepancy_with_llm(design_doc, tc, oracle_val)
            print(f"[TDD Robo] LLM Judge: is_design_error={judgment.is_design_error}, reason={judgment.reason}")

            if judgment.is_design_error:
                design_error_detected = True
                discrepancy_logs.append(
                    f"Action '{action}' expects '{expected_val}', but Oracle evaluated to '{oracle_val}'. "
                    f"Design flaw identified: {judgment.reason}"
                )
            else:
                corrected_val = judgment.corrected_expected or oracle_val
                discrepancy_logs.append(
                    f"Action '{action}' expects '{expected_val}', but Oracle evaluated to '{oracle_val}'. "
                    f"Test plan notation error: {judgment.reason} (Corrected: '{corrected_val}')"
                )

        disc_str = "\n".join(f"- {log}" for log in discrepancy_logs)

        if design_error_detected:
            rollback_counts = state.get("rollback_counts", {})
            current_req_id = target_req.get("id", f"REQ{idx:03d}")
            current_rollback = rollback_counts.get(current_req_id, 0)

            if current_rollback >= config.MAX_DESIGN_ROLLBACKS:
                print(
                    f"[TDD Robo] 🛡️ Circuit Breaker: Max design rollbacks reached ({current_rollback}) "
                    f"for {current_req_id} during test-plan review. Routing to test plan review instead."
                )
                decision = "review_test_plan"
                feedback = (
                    f"Early Oracle Verification detected Design Errors, but max rollbacks reached. "
                    f"Treating as Test Plan notation errors. Please adjust tests to align with current design/oracle:\n"
                    f"{disc_str}"
                )
                return {
                    "test_plan_review": feedback,
                    "test_plan_review_decision": decision,
                    "oracle_discrepancy_only": True,
                }
            else:
                rollback_counts = dict(rollback_counts)
                rollback_counts[current_req_id] = current_rollback + 1
                decision = "update_design_for_req"
                feedback = f"Early Oracle Verification detected Design Errors:\n{disc_str}\n{review_report.feedback}"
                return {
                    "test_plan_review": feedback,
                    "test_plan_review_decision": decision,
                    "design_review_feedback": f"Test plan verification failed with design discrepancies:\n{disc_str}",
                    "design_updated": False,
                    "oracle_discrepancy_only": False,
                    "rollback_counts": rollback_counts,
                }
        else:
            decision = "review_test_plan"
            feedback = (
                f"Early Oracle Verification detected Test Plan notation errors. "
                f"Please update the expected outcomes or oracle fields to match the correct oracle format:\n{disc_str}"
            )
            return {
                "test_plan_review": feedback,
                "test_plan_review_decision": decision,
                "oracle_discrepancy_only": True,
            }

    target_cov = state.get("target_test_plan_coverage", TARGET_TEST_PLAN_COVERAGE)
    if coverage < target_cov:
        iters = state.get("test_plan_iterations", 0)
        max_iters = state.get("max_test_plan_iterations", MAX_TEST_PLAN_ITERATIONS)
        if iters >= max_iters:
            print("[TDD Robo] ⚠️ Max iterations reached. Proceeding to test generation.")
        else:
            decision = "review_test_plan"

    return {"test_plan_review": feedback, "test_plan_review_decision": decision}


def plan_integration_tests(state: TDDState):
    """Plan Integration/E2E test cases based on external requirements and CLI specification."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    iters = state.get("test_plan_iterations", 0) + 1
    print(f"[TDD Robo] 📝 Planning Integration Tests for {target_req.get('id')} (Iteration {iters})...")

    review_feedback = state.get("test_plan_review", "")
    oracle_constraints = TEST_PLAN_ORACLE_CONSTRAINTS if getattr(config, "ORACLE_VERIFIER", None) is not None else ""

    is_fix = bool(review_feedback and state.get("integration_test_plan"))
    prompt = build_test_plan_prompt(state, "integration", target_req_str, oracle_constraints, is_fix, review_feedback)

    test_plan_obj = _call_llm_structured(prompt, TestPlan, model_name=config.MODEL_PRIMARY)

    if is_fix:
        if state.get("oracle_discrepancy_only"):
            pass
        else:
            try:
                existing_data = json.loads(state.get("integration_test_plan", "{}"))
                existing_cases = existing_data.get("test_cases", [])
            except Exception:
                existing_cases = []

            existing_cases_dict = {
                re.sub(r"\s+", " ", tc.get("action", "")).strip().lower(): tc
                for tc in existing_cases
                if tc.get("action")
            }
            for tc in test_plan_obj.test_cases:
                normalized_action = re.sub(r"\s+", " ", tc.action).strip().lower()
                if normalized_action:
                    existing_cases_dict[normalized_action] = tc.model_dump()

            test_plan_obj.test_cases = [TestCase(**tc) for tc in existing_cases_dict.values()]

    plan_md = f"# Integration Test Plan for {target_req.get('id')}\n\n"
    for idx_tc, tc in enumerate(test_plan_obj.test_cases, 1):
        plan_md += f"{idx_tc}. Action: {tc.action} | Expected: {tc.expected_outcome}\n"

    plan_path = save_artifact("test_plan.md", plan_md)
    print(f"[TDD Robo] ✅ Saved integration test plan to {plan_path}")

    test_plan_json = json.dumps(test_plan_obj.model_dump(), indent=2)
    return {"integration_test_plan": test_plan_json, "test_plan": plan_md, "test_plan_iterations": iters}


def review_integration_test_plan(state: TDDState):
    """Review the generated Integration Test Plan against CLI and Spec."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    print(f"[TDD Robo] 🧐 Reviewing Integration Test Plan for {target_req.get('id')}...")

    prompt = REVIEW_TEST_PLAN_PROMPT.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        design=state.get("design_doc", ""),
        requirements_list_str=state.get("requirements_list_str", ""),
        target_requirement=target_req_str,
        test_plan=state.get("test_plan", ""),
        target_coverage=state.get("target_test_plan_coverage", TARGET_TEST_PLAN_COVERAGE),
    )

    review_report = _call_llm_structured(prompt, TestPlanReviewReport, model_name=config.MODEL_PRIMARY)
    coverage = review_report.estimated_coverage
    print(f"[TDD Robo] ✅ Integration test plan review completed. Estimated coverage: {coverage}%")

    test_plan_json = state.get("integration_test_plan", "")
    mismatches = _run_early_oracle_verification(test_plan_json)

    decision = "continue"
    feedback = review_report.feedback

    if mismatches:
        print(f"[TDD Robo] 🧐 Found {len(mismatches)} early oracle mismatches. Invoking LLM Judge...")
        design_doc = state.get("design_doc", "")
        design_error_detected = False
        discrepancy_logs = []

        for mis in mismatches:
            tc = mis["test_case"]
            oracle_val = mis["oracle_val"]
            expected_val = mis["expected_val"]
            action = tc.get("action", "")

            judgment = _judge_oracle_discrepancy_with_llm(design_doc, tc, oracle_val)
            print(f"[TDD Robo] LLM Judge: is_design_error={judgment.is_design_error}, reason={judgment.reason}")

            if judgment.is_design_error:
                design_error_detected = True
                discrepancy_logs.append(
                    f"Action '{action}' expects '{expected_val}', but Oracle evaluated to '{oracle_val}'. "
                    f"Design flaw identified: {judgment.reason}"
                )
            else:
                corrected_val = judgment.corrected_expected or oracle_val
                discrepancy_logs.append(
                    f"Action '{action}' expects '{expected_val}', but Oracle evaluated to '{oracle_val}'. "
                    f"Test plan notation error: {judgment.reason} (Corrected: '{corrected_val}')"
                )

        disc_str = "\n".join(f"- {log}" for log in discrepancy_logs)

        if design_error_detected:
            rollback_counts = state.get("rollback_counts", {})
            current_req_id = target_req.get("id", f"REQ{idx:03d}")
            current_rollback = rollback_counts.get(current_req_id, 0)

            if current_rollback >= config.MAX_DESIGN_ROLLBACKS:
                print(
                    f"[TDD Robo] 🛡️ Circuit Breaker: Max design rollbacks reached ({current_rollback}) "
                    f"for {current_req_id} during integration-test-plan review. Routing to test plan review instead."
                )
                decision = "review_test_plan"
                feedback = (
                    f"Early Oracle Verification detected Design Errors, but max rollbacks reached. "
                    f"Treating as Test Plan notation errors. Please adjust tests to align with current design/oracle:\n"
                    f"{disc_str}"
                )
                return {
                    "test_plan_review": feedback,
                    "test_plan_review_decision": decision,
                    "oracle_discrepancy_only": True,
                }
            else:
                rollback_counts = dict(rollback_counts)
                rollback_counts[current_req_id] = current_rollback + 1
                decision = "update_design_for_req"
                feedback = f"Early Oracle Verification detected Design Errors:\n{disc_str}\n{review_report.feedback}"
                return {
                    "test_plan_review": feedback,
                    "test_plan_review_decision": decision,
                    "design_review_feedback": f"Test plan verification failed with design discrepancies:\n{disc_str}",
                    "design_updated": False,
                    "oracle_discrepancy_only": False,
                    "rollback_counts": rollback_counts,
                }
        else:
            decision = "review_test_plan"
            feedback = (
                f"Early Oracle Verification detected Test Plan notation errors. "
                f"Please update the expected outcomes or oracle fields to match the correct oracle format:\n{disc_str}"
            )
            return {
                "test_plan_review": feedback,
                "test_plan_review_decision": decision,
                "oracle_discrepancy_only": True,
            }

    target_cov = state.get("target_test_plan_coverage", TARGET_TEST_PLAN_COVERAGE)
    if coverage < target_cov:
        iters = state.get("test_plan_iterations", 0)
        max_iters = state.get("max_test_plan_iterations", MAX_TEST_PLAN_ITERATIONS)
        if iters >= max_iters:
            print("[TDD Robo] ⚠️ Max iterations reached. Proceeding to test generation.")
        else:
            decision = "review_test_plan"

    return {"test_plan_review": feedback, "test_plan_review_decision": decision}


# --- Sub-section: Test Generation Nodes ---


def generate_unit_tests(state: TDDState):
    """Generate pytest unit tests based on the Unit Test Plan."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    iters = state.get("test_iterations", 0) + 1
    print(f"[TDD Robo] 🧪 Generating Unit Test Code for {target_req.get('id')} (Iteration {iters})...")

    previous_tests = ""
    bug_report = state.get("bug_report", "")
    tests_check_output = state.get("tests_check_output", "")

    if bug_report or tests_check_output:
        previous_tests = state.get("unit_tests_code", "")
        test_module_name = state.get("test_module_name")
        if not previous_tests and test_module_name:
            test_path = os.path.join(config.ARTIFACTS_DIR, test_module_name)
            if os.path.exists(test_path):
                try:
                    with open(test_path, "r", encoding="utf-8") as f:
                        previous_tests = f.read()
                except Exception:
                    pass

    prompt = build_test_generation_prompt(state, "unit", target_req_str, previous_tests, bug_report, tests_check_output)
    unit_tests_code = _call_llm_text(prompt, model_name=config.MODEL_PRIMARY)

    req_id = str(target_req.get("id") or "").lower()
    test_filename = f"test_{state.get('module_name', 'impl')[:-3]}_{req_id}_unit.py"
    test_path = save_artifact(test_filename, unit_tests_code)
    print(f"[TDD Robo] ✅ Saved unit test code to {test_path}")
    save_history_snapshot(test_filename, unit_tests_code, iters, state=state)

    return {
        "unit_tests_code": unit_tests_code,
        "test_module_name": test_filename,
        "test_iterations": iters,
        "success": False,
        "bug_report": state.get("bug_report", ""),
        "test_review": "",
        "tests_check_output": "",
        "test_syntax_error_iterations": 0,
    }


def check_unit_tests_syntax(state: TDDState):
    """Syntax check on Unit Test Code."""
    test_file = state.get("test_module_name") or ""
    return _syntax_check_helper(test_file, "test_syntax_error_iterations", state)


def generate_integration_tests(state: TDDState):
    """Generate pytest integration tests based on the Integration Test Plan."""
    _update_req_progress(state)
    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    iters = state.get("test_iterations", 0) + 1
    print(f"[TDD Robo] 🧪 Generating Integration Test Code for {target_req.get('id')} (Iteration {iters})...")

    previous_tests = ""
    bug_report = state.get("bug_report", "")
    tests_check_output = state.get("tests_check_output", "")

    if bug_report or tests_check_output:
        previous_tests = state.get("integration_tests_code", "")
        test_module_name = state.get("test_module_name")
        if not previous_tests and test_module_name:
            test_path = os.path.join(config.ARTIFACTS_DIR, test_module_name)
            if os.path.exists(test_path):
                try:
                    with open(test_path, "r", encoding="utf-8") as f:
                        previous_tests = f.read()
                except Exception:
                    pass

    prompt = build_test_generation_prompt(
        state, "integration", target_req_str, previous_tests, bug_report, tests_check_output
    )
    integration_tests_code = _call_llm_text(prompt, model_name=config.MODEL_PRIMARY)

    req_id = str(target_req.get("id") or "").lower()
    test_filename = f"test_{state.get('module_name', 'impl')[:-3]}_{req_id}_integration.py"
    test_path = save_artifact(test_filename, integration_tests_code)
    print(f"[TDD Robo] ✅ Saved integration test code to {test_path}")
    save_history_snapshot(test_filename, integration_tests_code, iters, state=state)

    return {
        "integration_tests_code": integration_tests_code,
        "test_module_name": test_filename,
        "test_iterations": iters,
        "success": False,
        "bug_report": state.get("bug_report", ""),
        "test_review": "",
        "tests_check_output": "",
        "test_syntax_error_iterations": 0,
    }


def check_integration_tests_syntax(state: TDDState):
    """Syntax check on Integration Test Code."""
    test_file = state.get("test_module_name") or ""
    return _syntax_check_helper(test_file, "test_syntax_error_iterations", state)


# --- Sub-section: Implementation Nodes ---


def _implement_logic_helper(state: TDDState, phase: str) -> dict:
    """Generic helper to generate/update implementation code for a specific phase."""
    _update_req_progress(state)

    global_iters = state.get("iterations", 0) + 1
    print(f"[TDD Robo] 💻 Generating implementation code for phase: {phase} (Iteration {global_iters})...")

    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    test_path = os.path.join(config.ARTIFACTS_DIR, test_name)

    if (
        os.path.exists(impl_path)
        and os.path.exists(test_path)
        and not state.get("bug_report")
        and not state.get("impl_check_output")
    ):
        try:
            test_res = subprocess.run(
                [sys.executable, "-m", "pytest", test_name, "--maxfail=1"],
                capture_output=True,
                cwd=config.ARTIFACTS_DIR,
                timeout=config.EARLY_TEST_CHECK_TIMEOUT_SEC,
            )
            if test_res.returncode == 0:
                print(f"[TDD Robo] 🎉 Existing {impl_name} passes active tests for {phase}! Skipping LLM generation.")
                with open(impl_path, "r", encoding="utf-8") as f:
                    code = f.read()
                save_history_snapshot(impl_name, code, state.get("iterations", 0), state=state, phase=phase)
                return {
                    "impl_code": code,
                    "impl_check_output": "",
                    "bug_report": "",
                    "design_updated": False,
                    "impl_updated": False,
                }
        except Exception:
            pass

    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if idx < len(reqs):
        target_req = reqs[idx]
        target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"
    else:
        target_req_str = "No active target requirement (all completed)."

    existing_impl = state.get("impl_code", "")
    if existing_impl:
        clean_lines = [line for line in existing_impl.splitlines() if not line.startswith("# TDD_ROBO_SYNTAX_ERROR")]
        existing_impl = "\n".join(clean_lines).lstrip()

    design_updated = state.get("design_updated", False)
    existing_impl_code_param = "" if design_updated else existing_impl

    if phase == "unit":
        active_tests = state.get("unit_tests_code", "")
        existing_tests = _get_existing_tests_context(state)
        if existing_tests:
            tests_code = active_tests + "\n\n# --- Existing Tests (MUST NOT break) ---\n" + existing_tests
        else:
            tests_code = active_tests
    elif phase == "integration":
        active_tests = state.get("integration_tests_code", "")
        existing_tests = _get_existing_tests_context(state)
        if existing_tests:
            tests_code = active_tests + "\n\n# --- Existing Tests (MUST NOT break) ---\n" + existing_tests
        else:
            tests_code = active_tests
    else:
        tests_code = _get_combined_tests_code(state)

    domain_tips = state.get("domain_tips", "")
    python_tips = state.get("python_tips", "")
    impl_check_output = state.get("impl_check_output", "")

    prompt = build_implementation_prompt(
        state,
        phase,
        target_req_str,
        existing_impl,
        existing_impl_code_param,
        tests_code,
        domain_tips,
        python_tips,
        impl_check_output,
        impl_name,
        design_updated,
    )

    temp = config.LLM_LOOP_TEMPERATURE if state.get("loop_detected") else config.LLM_DEFAULT_TEMPERATURE
    response = _call_llm_with_reasoning(prompt, thinking_level="MINIMAL", temperature=temp)

    if existing_impl and not design_updated:
        has_sr_markers = bool(re.search(r"<<<<<<<\s*SEARCH", response, re.IGNORECASE))
        try:
            code = apply_search_replace_blocks(existing_impl, response)
            print("[TDD Robo] ⚙️ Applied Search/Replace diff blocks to implementation.")
            try:
                compile(code, impl_name, "exec")
            except SyntaxError as syntax_err:
                raise ValueError(f"Applied Search/Replace blocks resulted in a SyntaxError: {syntax_err}")
        except ValueError as e:
            if has_sr_markers:
                raw_error = f"Failed to apply Search/Replace block: {e}\n"
                code = existing_impl
                print(f"[TDD Robo] ❌ Search/Replace block application failed: {e}.")
                impl_path = save_artifact(impl_name, code)
                save_history_snapshot(impl_name, code, global_iters, state=state, phase=phase)
                return {
                    "impl_code": code,
                    "impl_check_output": f"Error: {raw_error}",
                    "bug_report": "",
                    "design_updated": False,
                    "iterations": global_iters,
                    "impl_updated": True,
                }
            else:
                code = extract_code(response)
    else:
        code = extract_code(response)

    impl_path = save_artifact(impl_name, code)
    print(f"[TDD Robo] ✅ Saved implementation code to {impl_path}")
    save_history_snapshot(impl_name, code, global_iters, state=state, phase=phase)

    return {
        "impl_code": code,
        "impl_check_output": "",
        "bug_report": "",
        "design_updated": False,
        "iterations": global_iters,
        "impl_updated": True,
    }


def implement_initial_logic(state: TDDState):
    """Write initial implementation code satisfying unit tests."""
    return _implement_logic_helper(state, "unit")


def implement_integration_logic(state: TDDState):
    """Refine implementation code to satisfy integration tests."""
    return _implement_logic_helper(state, "integration")


def implement_regression_logic(state: TDDState):
    """Refine implementation code to satisfy regression tests."""
    return _implement_logic_helper(state, "regression")


def check_initial_impl_syntax(state: TDDState):
    """Verify syntax of implementation code in unit phase."""
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    if not state.get("impl_updated", True):
        print(f"[TDD Robo] ⏭️ Skipping syntax check for {impl_name} (no changes).")
        return {"syntax_error_iterations": 0, "impl_check_output": "", "impl_updated": False}
    res = _syntax_check_helper(impl_name, "syntax_error_iterations", state)
    res["impl_updated"] = False
    return res


def check_integration_impl_syntax(state: TDDState):
    """Verify syntax of implementation code in integration phase."""
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    if not state.get("impl_updated", True):
        print(f"[TDD Robo] ⏭️ Skipping syntax check for {impl_name} (no changes).")
        return {"syntax_error_iterations": 0, "impl_check_output": "", "impl_updated": False}
    res = _syntax_check_helper(impl_name, "syntax_error_iterations", state)
    res["impl_updated"] = False
    return res


def check_regression_impl_syntax(state: TDDState):
    """Verify syntax of implementation code in regression phase."""
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    if not state.get("impl_updated", True):
        print(f"[TDD Robo] ⏭️ Skipping syntax check for {impl_name} (no changes).")
        return {"syntax_error_iterations": 0, "impl_check_output": "", "impl_updated": False}
    res = _syntax_check_helper(impl_name, "syntax_error_iterations", state)
    res["impl_updated"] = False
    return res


def check_refactored_impl_syntax(state: TDDState):
    """Verify syntax of implementation code after refactoring."""
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    if not state.get("impl_updated", True):
        print(f"[TDD Robo] ⏭️ Skipping syntax check for {impl_name} (no changes).")
        return {"syntax_error_iterations": 0, "impl_check_output": "", "impl_updated": False}
    res = _syntax_check_helper(impl_name, "syntax_error_iterations", state)
    res["impl_updated"] = False
    return res


# --- Sub-section: Test Execution Nodes ---


def run_unit_tests(state: TDDState):
    """Execute unit tests and update iteration counters."""
    _update_req_progress(state)
    test_file = state.get("test_module_name")
    iters = state.get("unit_test_iterations", 0) + 1
    print(f"[TDD Robo] 🏃 Running Unit Tests (Iteration {iters})...")

    res = _execute_tests_helper(test_file, state)
    res["unit_test_iterations"] = iters
    return res


def run_integration_tests(state: TDDState):
    """Execute integration tests and update iteration counters."""
    _update_req_progress(state)
    test_file = state.get("test_module_name")
    iters = state.get("integration_test_iterations", 0) + 1
    print(f"[TDD Robo] 🏃 Running Integration Tests (Iteration {iters})...")

    res = _execute_tests_helper(test_file, state)
    res["integration_test_iterations"] = iters
    return res


def run_regression_tests(state: TDDState):
    """Execute full regression test suite (all unit and integration tests)."""
    _update_req_progress(state)
    iters = state.get("regression_test_iterations", 0) + 1
    print(f"[TDD Robo] 🏃 Running Regression Tests (Iteration {iters})...")

    res = _execute_tests_helper(None, state)
    res["regression_test_iterations"] = iters
    return res


# --- Sub-section: Diagnostics & Refactoring Nodes ---


def generate_unit_bug_report(state: TDDState):
    """Generate a bug report diagnosing unit test failures."""
    _update_req_progress(state)
    print("[TDD Robo] 🐛 Generating Unit Bug Report...")

    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    unit_tests_code = state.get("unit_tests_code", "")
    oracle_feedback = _run_oracle_verification_on_failures(state.get("test_output", ""), unit_tests_code, state)

    prompt = GENERATE_UNIT_BUG_REPORT_PROMPT.format(
        target_req=target_req_str,
        unit_test_plan=state.get("unit_test_plan", ""),
        oracle_verification_feedback=oracle_feedback,
        unit_test_code=add_line_numbers(unit_tests_code),
        impl_code=add_line_numbers(state.get("impl_code", "")),
        test_output=state.get("test_output", ""),
    )

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=config.MODEL_PRIMARY)

    if "ORACLE VERIFICATION FEEDBACK" in oracle_feedback:
        impl_name = state.get("module_name", "impl.py")
        if _has_implementation_exceptions(state.get("test_output", ""), impl_name):
            print(
                "[TDD Robo] 🔮 Oracle discrepancy detected, but implementation exceptions were found. "
                f"Prioritizing '{bug_report_obj.target_to_fix}'."
            )
        else:
            print("[TDD Robo] 🔮 Oracle discrepancy detected. Overriding target_to_fix to 'generate_tests'.")
            bug_report_obj.target_to_fix = "generate_tests"

    report_md = "### Failed Unit Test Cases\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Unit Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")
    return {
        "bug_report": report_md,
        "next_action": bug_report_obj.target_to_fix,
        "oracle_discrepancy_only": False,
        "loop_origin_node": "implement_initial_logic",
    }


def generate_integration_bug_report(state: TDDState):
    """Generate a bug report diagnosing integration test failures."""
    _update_req_progress(state)
    print("[TDD Robo] 🐛 Generating Integration Bug Report...")

    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    integration_tests_code = state.get("integration_tests_code", "")
    oracle_feedback = _run_oracle_verification_on_failures(state.get("test_output", ""), integration_tests_code, state)

    prompt = GENERATE_INTEGRATION_BUG_REPORT_PROMPT.format(
        target_req=target_req_str,
        integration_test_plan=state.get("integration_test_plan", ""),
        oracle_verification_feedback=oracle_feedback,
        integration_test_code=add_line_numbers(integration_tests_code),
        impl_code=add_line_numbers(state.get("impl_code", "")),
        test_output=state.get("test_output", ""),
    )

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=config.MODEL_PRIMARY)

    if "ORACLE VERIFICATION FEEDBACK" in oracle_feedback:
        impl_name = state.get("module_name", "impl.py")
        if _has_implementation_exceptions(state.get("test_output", ""), impl_name):
            print(
                "[TDD Robo] 🔮 Oracle discrepancy detected, but implementation exceptions were found. "
                f"Prioritizing '{bug_report_obj.target_to_fix}'."
            )
        else:
            print("[TDD Robo] 🔮 Oracle discrepancy detected. Overriding target_to_fix to 'generate_tests'.")
            bug_report_obj.target_to_fix = "generate_tests"

    report_md = "### Failed Integration Test Cases\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Integration Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")
    return {
        "bug_report": report_md,
        "next_action": bug_report_obj.target_to_fix,
        "oracle_discrepancy_only": False,
        "loop_origin_node": "implement_integration_logic",
    }


def generate_regression_bug_report(state: TDDState):
    """Generate a bug report diagnosing regression test failures."""
    _update_req_progress(state)
    print("[TDD Robo] 🐛 Generating Regression Bug Report...")

    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    if not reqs or idx >= len(reqs):
        return {}
    target_req = reqs[idx]
    target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"

    failed_files = cast(list[str], state.get("failed_files", []))
    failed_req_nums = []
    for f in failed_files:
        match = re.search(r"test_[a-zA-Z0-9_]+_req(\d+)_(?:unit|integration)\.py", f)
        if match:
            failed_req_nums.append(int(match.group(1)))

    if not failed_req_nums:
        test_output = state.get("test_output", "")
        for line in test_output.splitlines():
            if "FAILED" in line or "ERROR" in line:
                matches = re.findall(r"test_[a-zA-Z0-9_]+_req(\d+)_(?:unit|integration)\.py", line)
                for m in matches:
                    failed_req_nums.append(int(m))

    has_historical_regression = False
    oldest_failing_req_idx = None
    target_req_num = idx + 1
    if failed_req_nums:
        min_failed_req = min(failed_req_nums)
        target_req_num = min_failed_req
        if min_failed_req - 1 < idx:
            has_historical_regression = True
            oldest_failing_req_idx = min_failed_req - 1

    test_code = _get_filtered_regression_test_code_context(target_req_num)
    raw_test_output = state.get("test_output", "")
    filtered_test_output = _get_filtered_test_output(raw_test_output, target_req_num)

    oracle_feedback = _run_oracle_verification_on_failures(filtered_test_output, test_code, state)

    last_green = state.get("last_green_impl_code", "")
    current_impl = state.get("impl_code", "")
    impl_diff = ""
    if last_green:
        diff_lines = list(
            difflib.unified_diff(
                last_green.splitlines(),
                current_impl.splitlines(),
                fromfile="last_green_impl",
                tofile="current_impl",
                lineterm="",
            )
        )
        impl_diff = "\n".join(diff_lines[:200])

    prompt = GENERATE_REGRESSION_BUG_REPORT_PROMPT.format(
        target_req=target_req_str,
        failed_requirements=", ".join([f"REQ{n:03d}" for n in failed_req_nums]) if failed_req_nums else "None",
        oracle_verification_feedback=oracle_feedback,
        test_code=test_code,
        impl_code=add_line_numbers(current_impl),
        impl_diff=impl_diff,
        test_output=filtered_test_output,
    )

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=config.MODEL_PRIMARY)

    llm_target_req = bug_report_obj.target_req
    target_req_index = None
    if llm_target_req:
        match = re.search(r"REQ(\d+)", llm_target_req, re.IGNORECASE)
        if match:
            req_num = int(match.group(1))
            for i, req in enumerate(reqs):
                if req.get("id") == f"REQ{req_num:03d}":
                    target_req_index = i
                    break

    rollback_target_idx = oldest_failing_req_idx
    if target_req_index is not None and target_req_index < idx:
        rollback_target_idx = target_req_index

    if "ORACLE VERIFICATION FEEDBACK" in oracle_feedback:
        if has_historical_regression and rollback_target_idx is not None and rollback_target_idx < idx:
            if bug_report_obj.target_to_fix == "implement_logic":
                print(
                    "[TDD Robo] 🔮 Oracle discrepancy detected in a historical test, "
                    "but LLM Judge determined it is an implementation bug. "
                    "Preserving 'implement_logic' to avoid unnecessary rollback."
                )
            else:
                print(
                    "[TDD Robo] 🔮 Oracle discrepancy detected in a historical test. "
                    "LLM Judge determined it is a test bug. "
                    "Overriding target_to_fix to 'generate_tests'."
                )
                bug_report_obj.target_to_fix = "generate_tests"
        else:
            print("[TDD Robo] 🔮 Oracle discrepancy detected. Overriding target_to_fix to 'generate_tests'.")
            bug_report_obj.target_to_fix = "generate_tests"

    report_md = "### Failed Regression Test Cases\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Regression Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")

    policy = state.get("regression_failure_policy", "rollback")

    if (
        has_historical_regression
        and rollback_target_idx is not None
        and bug_report_obj.target_to_fix != "implement_logic"
    ):
        if policy == "halt":
            print(
                f"[TDD Robo] 🚨 Circuit Breaker: Regression failure detected in a historical test "
                f"(REQ{rollback_target_idx + 1:03d}). Policy is 'halt'. Halting execution."
            )
            return {
                "bug_report": report_md,
                "next_action": "halt_regression_test_failure",
            }
        else:
            rollback_counts = dict(cast(dict, state.get("rollback_counts") or {}))
            count = rollback_counts.get(str(rollback_target_idx), 0)
            if count >= config.MAX_REGRESSION_ROLLBACKS:
                print(
                    f"[TDD Robo] 🚨 Circuit Breaker: Max rollback attempts "
                    f"({config.MAX_REGRESSION_ROLLBACKS}) reached for "
                    f"REQ{rollback_target_idx + 1:03d}. Halting execution."
                )
                return {
                    "bug_report": report_md,
                    "next_action": "halt_regression_test_failure",
                }
            else:
                rollback_counts[str(rollback_target_idx)] = count + 1
                _backup_project_before_rollback(state, idx, rollback_target_idx)
                if target_req_index is not None and target_req_index < idx:
                    print(
                        f"[TDD Robo] 🔄 Rolling back current_req_index from {idx} to {rollback_target_idx} "
                        f"(Policy is 'rollback', target_req is LLM-specified: {llm_target_req}). "
                        f"Attempt {count + 1}/{config.MAX_REGRESSION_ROLLBACKS}."
                    )
                else:
                    print(
                        f"[TDD Robo] 🔄 Rolling back current_req_index from {idx} to {rollback_target_idx} "
                        f"(Policy is 'rollback', fallback to oldest failing req). "
                        f"Attempt {count + 1}/{config.MAX_REGRESSION_ROLLBACKS}."
                    )

                test_pattern = os.path.join(config.ARTIFACTS_DIR, "test_*.py")
                for tf in glob.glob(test_pattern):
                    basename = os.path.basename(tf)
                    match = re.search(r"req(\d+)", basename)
                    if match:
                        file_req_num = int(match.group(1))
                        if file_req_num >= rollback_target_idx + 2:
                            try:
                                os.remove(tf)
                                print(f"[TDD Robo] 🧹 Cleaned up future test file during rollback: {basename}")
                            except Exception as e:
                                print(f"[TDD Robo] ⚠️ Failed to remove future test file {basename}: {e}")

                target_req_pattern = f"py_bc_req{rollback_target_idx + 1:03d}_"
                history_dir = os.path.join(config.ARTIFACTS_DIR, "history")
                restored = False
                if os.path.exists(history_dir):
                    history_files = glob.glob(os.path.join(history_dir, f"{target_req_pattern}*"))
                    impl_backups = [
                        f
                        for f in history_files
                        if "impl_iter" in os.path.basename(f) or "refactor_iter" in os.path.basename(f)
                    ]
                    if impl_backups:
                        impl_backups.sort()
                        latest_backup = impl_backups[-1]
                        try:
                            impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
                            dest_path = os.path.join(config.ARTIFACTS_DIR, impl_name)
                            shutil.copy(latest_backup, dest_path)
                            backup_name = os.path.basename(latest_backup)
                            print(
                                f"[TDD Robo] 🔄 Restoring implementation code for REQ{rollback_target_idx + 1:03d} "
                                f"to last green snapshot: {backup_name}"
                            )
                            restored = True
                        except Exception as e:
                            print(f"[TDD Robo] ⚠️ Failed to copy historical backup: {e}")

                design_restored_content = ""
                if os.path.exists(history_dir):
                    design_pattern = f"design_req{rollback_target_idx + 1:03d}_*"
                    design_files = glob.glob(os.path.join(history_dir, design_pattern))
                    if design_files:
                        design_files.sort()
                        latest_design_backup = design_files[-1]
                        try:
                            design_dest_path = os.path.join(config.ARTIFACTS_DIR, "design.md")
                            shutil.copy(latest_design_backup, design_dest_path)
                            with open(design_dest_path, "r", encoding="utf-8") as df:
                                design_restored_content = df.read()
                            print(
                                f"[TDD Robo] 🔄 Restoring design document for REQ{rollback_target_idx + 1:03d} "
                                f"to last green snapshot: {os.path.basename(latest_design_backup)}"
                            )
                        except Exception as e:
                            print(f"[TDD Robo] ⚠️ Failed to copy historical design backup: {e}")

                if not restored:
                    print(f"[TDD Robo] ⚠️ No historical backup found for REQ{rollback_target_idx + 1:03d} in history/.")

                ret_val = {
                    "bug_report": report_md,
                    "next_action": "generate_design",
                    "current_req_index": rollback_target_idx,
                    "rollback_counts": rollback_counts,
                    "design_updated": False,
                    "oracle_discrepancy_only": "ORACLE VERIFICATION FEEDBACK" in oracle_feedback,
                    "loop_detected": True,
                    "design_iterations": 0,
                    "loop_origin_node": "implement_regression_logic",
                    "unit_test_plan": None,
                    "integration_test_plan": None,
                    "unit_tests_code": "",
                    "integration_tests_code": "",
                    "test_plan": "",
                    "test_plan_review": "",
                    "test_plan_review_decision": "",
                }
                if design_restored_content:
                    ret_val["design_doc"] = design_restored_content
                return ret_val

    return {
        "bug_report": report_md,
        "next_action": bug_report_obj.target_to_fix,
        "oracle_discrepancy_only": False,
        "loop_origin_node": "implement_regression_logic",
    }


def generate_refactor_bug_report(state: TDDState):
    """Generate a bug report diagnosing refactoring test failures."""
    _update_req_progress(state)

    refactor_iters = state.get("refactor_iterations", 0)
    max_refactor_iters = getattr(config, "MAX_REFACTOR_ITERATIONS", 5)
    stagnant_iters = state.get("stagnant_iterations", 0)
    max_stagnant_iters = getattr(config, "MAX_STAGNANT_ITERATIONS", 3)

    is_stagnant_over = stagnant_iters >= max_stagnant_iters
    is_refactor_over = refactor_iters >= max_refactor_iters

    if is_refactor_over or is_stagnant_over:
        reason_str = (
            f"iteration limit ({refactor_iters}/{max_refactor_iters})"
            if is_refactor_over
            else f"stagnant limit ({stagnant_iters}/{max_stagnant_iters})"
        )
        print(f"[TDD Robo] 🛑 Refactoring reached {reason_str}. Rolling back implementation code to last green state.")
        last_green = state.get("last_green_impl_code", "")
        impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
        save_artifact(impl_name, last_green)
        save_history_snapshot(impl_name, last_green, refactor_iters + 1, state=state, is_refactor=True)

        report_md = (
            "### Refactoring Halted\n"
            f"Refactoring reached {reason_str}. "
            "Implementation rolled back to the last stable state."
        )
        return {
            "bug_report": report_md,
            "next_action": "rollback_continue",
            "impl_code": last_green,
            "impl_updated": True,
        }

    print("[TDD Robo] 🐛 Generating Refactor Bug Report...")

    test_code = _get_regression_test_code_context()

    prompt = GENERATE_REFACTOR_BUG_REPORT_PROMPT.format(
        test_code=test_code,
        impl_code=add_line_numbers(state.get("impl_code", "")),
        test_output=state.get("test_output", ""),
        refactoring_reasons="\n".join(state.get("reasons", [])) or "Refactoring to improve structure.",
    )

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=config.MODEL_PRIMARY)

    report_md = "### Failed Test Cases after Refactoring\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Refactor Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")
    return {"bug_report": report_md, "next_action": bug_report_obj.target_to_fix}


def analyze_architecture(state: TDDState) -> dict:
    """Perform an Architectural Audit when stuck in an implementation deadlock."""
    print("[TDD Robo] 🔍 Initiating Architectural Audit to break the implementation deadlock...")

    audit_loop_count = state.get("audit_loop_count", 0) + 1
    max_audit_loop_count = getattr(config, "MAX_AUDIT_LOOP_COUNT", 2)

    spec_path = os.path.join(config.ARTIFACTS_DIR, "specification.txt")
    spec_content = ""
    if os.path.exists(spec_path):
        try:
            with open(spec_path, "r", encoding="utf-8") as f:
                spec_content = f.read()
        except Exception as e:
            print(f"[TDD Robo] ⚠️ Warning: Failed to read specification.txt: {e}")

    reqs = state.get("requirements", [])
    idx = state.get("current_req_index", 0)
    target_req_str = ""
    if idx < len(reqs):
        target_req = reqs[idx]
        target_req_str = f"ID: {target_req.get('id')}\nDescription: {target_req.get('description')}"

    impl_code = state.get("impl_code", "")
    tests_code = _get_combined_tests_code(state)
    test_output = state.get("test_output", "")

    prompt = build_architecture_audit_prompt(spec_content, target_req_str, impl_code, tests_code, test_output)

    audit_data = _call_llm_structured(prompt, ArchitectureAudit, model_name=config.MODEL_PRIMARY)

    classification = audit_data.classification
    if audit_loop_count >= max_audit_loop_count:
        print(
            f"[TDD Robo] 🚨 Circuit Breaker: Consecutive audits count {audit_loop_count} >= {max_audit_loop_count}. "
            "Escalating local bug to architectural_bottleneck to force design rollback."
        )
        classification = "architectural_bottleneck"

    audit_report_text = (
        f"Classification:\n{classification}\n\n"
        f"Architectural Bottleneck:\n{audit_data.architectural_bottleneck}\n\n"
        f"Refactoring Plan:\n{audit_data.refactoring_plan}\n\n"
        f"Safeties and Invariants:\n{audit_data.safeties_and_invariants}"
    )

    print("\n=== [Architectural Audit Report] ===")
    print(audit_report_text)
    print("====================================\n")

    if classification == "local_bug":
        next_action = state.get("loop_origin_node", "implement_initial_logic")
        bug_report = (
            f"### Loop Audit Diagnosis (Local Bug)\n{audit_data.architectural_bottleneck}\n\n"
            f"### Local Fix Plan\n{audit_data.refactoring_plan}\n\n"
            f"### Invariants & Safeties to Preserve\n{audit_data.safeties_and_invariants}"
        )
        print(
            f"[TDD Robo] 🔄 Routing back to {next_action} to apply local fixes. "
            f"Attempt {audit_loop_count}/{max_audit_loop_count}."
        )
        return {
            "architecture_audit": audit_report_text,
            "loop_detected": True,
            "oracle_discrepancy_only": False,
            "next_action": next_action,
            "bug_report": bug_report,
            "iterations": 0,
            "stagnant_iterations": 0,
            "audit_loop_count": audit_loop_count,
        }
    else:
        print("[TDD Robo] 📐 Routing to update_design_for_req to update design document.")
        return {
            "architecture_audit": audit_report_text,
            "loop_detected": True,
            "oracle_discrepancy_only": False,
            "next_action": "update_design_for_req",
            "audit_loop_count": 0,
        }


def decide_refactor(state: TDDState):
    """Analyze implementation code and decide if refactoring is needed."""
    _update_req_progress(state)
    print("[TDD Robo] ❓ Deciding if refactoring is needed...")

    prompt = DECIDE_REFACTOR_PROMPT.format(design_doc=state.get("design_doc", ""), impl_code=state.get("impl_code", ""))

    decision_obj = _call_llm_structured(prompt, RefactorDecision, model_name=config.MODEL_PRIMARY)
    print(f"[TDD Robo] ✅ Refactor decision: needed={decision_obj.refactor_needed}, reasons={decision_obj.reasons}")

    decision_str = "refactor" if decision_obj.refactor_needed else "continue"
    res = {
        "refactor_decision": decision_str,
        "reasons": decision_obj.reasons,
    }
    if decision_obj.refactor_needed:
        res["last_green_impl_code"] = state.get("impl_code", "")
    return res


def refactor_logic(state: TDDState):
    """Refactor the implementation code using the reasoning model."""
    _update_req_progress(state)
    iters = state.get("refactor_iterations", 0) + 1
    print(f"[TDD Robo] 🧹 Refactoring implementation code (Iteration {iters})...")

    reasons_str = "\n".join(state.get("reasons", [])) or "Refactoring to improve structure."
    bug_report = state.get("bug_report", "")
    python_tips = state.get("python_tips", "")

    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)

    existing_impl = state.get("impl_code", "")
    if not existing_impl and os.path.exists(impl_path):
        try:
            with open(impl_path, "r", encoding="utf-8") as f:
                existing_impl = f.read()
        except Exception:
            pass

    prompt, reasons_with_bug = build_refactor_prompt(
        state, reasons_str, bug_report, python_tips, existing_impl, iters, impl_name
    )

    response = _call_llm_text(prompt, model_name=config.MODEL_PRIMARY)

    is_bug_fix = bool(bug_report and existing_impl)
    if is_bug_fix:
        has_sr_markers = bool(re.search(r"<<<<<<<\s*SEARCH", response, re.IGNORECASE))
        try:
            refactored_code = apply_search_replace_blocks(existing_impl, response)
            print("[TDD Robo] ⚙️ Applied Search/Replace diff blocks to refactored implementation.")
            try:
                compile(refactored_code, impl_name, "exec")
            except SyntaxError as syntax_err:
                raise ValueError(f"Applied Search/Replace blocks resulted in a SyntaxError: {syntax_err}")
        except ValueError as e:
            if has_sr_markers:
                print(
                    f"[TDD Robo] ❌ Search/Replace block application failed: {e}. "
                    "Falling back to full code generation..."
                )
                save_history_snapshot(impl_name, existing_impl, iters, state=state, is_refactor=True)
                reasons_with_failure = reasons_with_bug + "\n\n" + REFACTOR_LOGIC_FALLBACK_WARNING.format(error=e)
                fallback_prompt = REFACTOR_LOGIC_PROMPT.format(
                    design_doc=state.get("design_doc", ""),
                    impl_code=existing_impl,
                    refactoring_reasons=reasons_with_failure,
                    python_tips=python_tips,
                )
                response = _call_llm_text(fallback_prompt, model_name=config.MODEL_PRIMARY)
                refactored_code = extract_code(response)
            else:
                refactored_code = extract_code(response)
    else:
        refactored_code = extract_code(response)

    impl_path = save_artifact(impl_name, refactored_code)
    print(f"[TDD Robo] ✅ Saved refactored implementation code to {impl_path}")
    save_history_snapshot(impl_name, refactored_code, iters, state=state, is_refactor=True)

    return {
        "impl_code": refactored_code,
        "refactor_iterations": iters,
        "impl_check_output": "",
        "bug_report": "",
        "impl_updated": True,
    }


# --- Sub-section: Finalization & Documentation Nodes ---


def generate_readme(state: TDDState):
    """Generate a README.md file based on the goal and implementation code."""
    _update_req_progress(state)
    print("[TDD Robo] 📝 Generating README.md...")

    prompt = GENERATE_README_PROMPT.format(
        goal=state.get("goal", ""),
        impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
        impl_code=state.get("impl_code", ""),
    )
    readme = call_llm_standard(prompt)
    readme_path = save_artifact("README.md", readme)
    print(f"[TDD Robo] ✅ Saved README.md to {readme_path}")
    return {"readme_content": readme}


def increment_requirement(state: TDDState):
    """State-updating node to advance to the next target requirement."""
    _update_req_progress(state)
    current_index = state.get("current_req_index", 0)
    requirements = state.get("requirements", [])
    next_index = current_index + 1

    print(f"[TDD Robo] ➡️ Advancing from requirement {current_index + 1} to {next_index + 1}/{len(requirements)}...")

    print("\n=================== 📊 TDD PROGRESS STATUS =================== ")
    for idx, req in enumerate(requirements):
        req_id = req.get("id", f"REQ{idx + 1:03d}")
        desc = req.get("description", "")
        desc_short = desc if len(desc) <= 50 else desc[:47] + "..."
        if idx < next_index:
            status = "✅ Passed "
        elif idx == next_index:
            status = "⏳ Active "
        else:
            status = "💤 Pending"
        print(f"[{req_id}] {status} - {desc_short}")
    print("==============================================================\n")
    ret_val: TDDState
    if next_index >= len(requirements):
        ret_val = {
            "current_req_index": next_index,
            "success": True,
        }
        if state.get("impl_code"):
            ret_val["last_green_impl_code"] = state.get("impl_code", "")
        return ret_val

    ret_val = {
        "current_req_index": next_index,
        "success": False,
        "design_updated": False,
        "test_plan": "",
        "test_plan_review": "",
        "test_plan_iterations": 0,
        "unit_test_plan": "",
        "integration_test_plan": "",
        "unit_tests_code": "",
        "integration_tests_code": "",
        "refactor_decision": "",
        "reasons": [],
        "unit_test_iterations": 0,
        "integration_test_iterations": 0,
        "regression_test_iterations": 0,
        "refactor_iterations": 0,
        "iterations": 0,
        "loop_detected": False,
        "stagnant_iterations": 0,
        "last_test_summary": "",
        "oracle_discrepancy_only": False,
        "audit_loop_count": 0,
    }
    if state.get("impl_code"):
        ret_val["last_green_impl_code"] = state.get("impl_code", "")
    return ret_val
