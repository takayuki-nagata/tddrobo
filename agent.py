import builtins
import difflib
import glob
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Any, cast

import markdownify
import requests
from langgraph.graph import END, StateGraph

import config
from config import MODEL_PRIMARY
from prompts import (
    DECIDE_REFACTOR_PROMPT,
    EXTRACT_ORACLE_TARGET_PROMPT,
    GENERATE_DESIGN_PROMPT,
    GENERATE_INTEGRATION_BUG_REPORT_PROMPT,
    GENERATE_INTEGRATION_TESTS_PROMPT,
    GENERATE_README_PROMPT,
    GENERATE_REFACTOR_BUG_REPORT_PROMPT,
    GENERATE_REGRESSION_BUG_REPORT_PROMPT,
    GENERATE_REQUIREMENTS_PROMPT,
    GENERATE_UNIT_BUG_REPORT_PROMPT,
    GENERATE_UNIT_TESTS_PROMPT,
    IMPLEMENT_LOGIC_PROMPT_FIX,
    IMPLEMENT_LOGIC_PROMPT_INITIAL,
    IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX,
    PLAN_FILES_PROMPT,
    PLAN_INTEGRATION_TESTS_PROMPT,
    PLAN_INTEGRATION_TESTS_PROMPT_FIX,
    PLAN_UNIT_TESTS_PROMPT,
    PLAN_UNIT_TESTS_PROMPT_FIX,
    REFACTOR_LOGIC_FIX_PROMPT,
    REFACTOR_LOGIC_PROMPT,
    REVIEW_DESIGN_PROMPT,
    REVIEW_TEST_PLAN_PROMPT,
    TEST_PLAN_ORACLE_CONSTRAINTS,
)
from schema import (
    BugReport,
    DesignDocument,
    DesignReviewReport,
    FilePlan,
    OracleAssertionTarget,
    OracleDiscrepancyJudgment,
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
    call_llm_structured,
    call_llm_text,
    call_llm_with_reasoning,
    extract_code,
    extract_json,
    get_prompt,
    read_artifact,
    save_artifact,
)

DEFAULT_IMPL_NAME = config.DEFAULT_IMPL_NAME
DEFAULT_TEST_NAME = config.DEFAULT_TEST_NAME
FETCH_TIMEOUT_SEC = config.FETCH_TIMEOUT_SEC
SYNTAX_CHECK_TIMEOUT_SEC = config.SYNTAX_CHECK_TIMEOUT_SEC
TEST_EXECUTION_TIMEOUT_SEC = config.TEST_EXECUTION_TIMEOUT_SEC
MAX_ITERATIONS = config.MAX_ITERATIONS
MAX_TEST_PLAN_ITERATIONS = config.MAX_TEST_PLAN_ITERATIONS
MAX_TEST_ITERATIONS = config.MAX_TEST_ITERATIONS
TARGET_TEST_PLAN_COVERAGE = config.TARGET_TEST_PLAN_COVERAGE
TARGET_TEST_COVERAGE = config.TARGET_TEST_COVERAGE
TARGET_DESIGN_QUALITY = config.TARGET_DESIGN_QUALITY


_thread_local = threading.local()


def _update_req_progress(state: TDDState):
    """Update requirement progress indicators from state in a thread-safe manner."""
    _thread_local.current_state = state
    reqs = state.get("requirements", [])
    if reqs:
        _thread_local.total_req_num = len(reqs)
        _thread_local.current_req_num = state.get("current_req_index", 0) + 1


def print(*args, **kwargs):
    """Override builtin print to add timestamps and requirement progress to [TDD Robo] messages."""
    if args and isinstance(args[0], str) and args[0].startswith("[TDD Robo]"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        progress_suffix = ""
        current_req = getattr(_thread_local, "current_req_num", 0)
        total_req = getattr(_thread_local, "total_req_num", 0)
        if current_req > 0 and total_req > 0:
            progress_suffix = f" ({current_req}/{total_req})"
        new_arg0 = args[0].replace("[TDD Robo]", f"[{timestamp}] [TDD Robo]{progress_suffix}", 1)
        args = (new_arg0,) + args[1:]
    builtins.print(*args, **kwargs)


def save_history_snapshot(
    filename: str,
    code: str,
    iteration: int,
    state: TDDState | None = None,
    is_refactor: bool = False,
    phase: str | None = None,
):
    """
    Saves a snapshot of the generated file to the artifacts/history/ directory.

    Args:
        filename (str): The original filename (e.g. 'impl.py').
        code (str): The code content to write.
        iteration (int): The current iteration number.
        state (TDDState): The current state of the TDD workflow (optional).
        is_refactor (bool): Whether the snapshot is from a refactoring iteration.
        phase (str): The active phase (e.g. 'unit', 'integration', 'regression').
    """
    name_parts = os.path.splitext(filename)
    is_test_file = filename.startswith("test_")
    is_design_file = filename == "design.md"

    # Fetch design iterations prefix (e.g., _d002) if available
    d_suffix = ""
    if state and state.get("design_iterations") is not None:
        d_suffix = f"_d{state.get('design_iterations', 0):03d}"

    req_id = None
    if state:
        if "requirements" in state and "current_req_index" in state:
            reqs = state["requirements"]
            idx = state["current_req_index"]
            if 0 <= idx < len(reqs):
                req_id = reqs[idx].get("id")
        if req_id is None and "current_req_index" in state:
            req_id = f"req{state['current_req_index'] + 1:03d}"

    if not is_test_file and not is_design_file and req_id and state:
        test_iterations = state.get("test_iterations", 1)
        iter_prefix = "refactor" if is_refactor else "impl"
        if phase:
            iter_prefix = f"{phase}_{iter_prefix}"
        history_filename = (
            f"{name_parts[0]}_{req_id.lower()}{d_suffix}_test_iter{test_iterations:03d}_"
            f"{iter_prefix}_iter{iteration:03d}{name_parts[1]}"
        )
    elif is_test_file and state:
        phase_suffix = ""
        base_name_str = name_parts[0]
        if base_name_str.endswith("_unit"):
            base_name_str = base_name_str[:-5]
            phase_suffix = "_unit"
        elif base_name_str.endswith("_integration"):
            base_name_str = base_name_str[:-12]
            phase_suffix = "_integration"
        history_filename = f"{base_name_str}{d_suffix}{phase_suffix}_iter{iteration:03d}{name_parts[1]}"
    elif is_design_file and state:
        if req_id:
            history_filename = f"design_{req_id.lower()}_iter{iteration:03d}.md"
        else:
            history_filename = f"design_iter{iteration:03d}.md"
    else:
        history_filename = f"{name_parts[0]}_iter{iteration:03d}{name_parts[1]}"

    history_dir = os.path.join(config.ARTIFACTS_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, history_filename)

    # Rotate existing file if it exists to prevent overwriting
    if os.path.exists(history_path):
        i = 1
        while os.path.exists(f"{history_path}.{i}"):
            i += 1
        try:
            os.rename(history_path, f"{history_path}.{i}")
            if getattr(config, "VERBOSE", False):
                print(f"[TDD Robo] 🔄 Rotated existing history snapshot to {history_path}.{i}")
        except Exception as e:
            print(f"Warning: Could not rotate history file {history_path}: {e}")

    try:
        with open(history_path, "w", encoding="utf-8") as f:
            f.write(code)
        if getattr(config, "VERBOSE", False):
            print(f"[TDD Robo] 📦 Saved history snapshot to {history_path}")
    except Exception as e:
        print(f"Warning: Could not save history snapshot to {history_path}: {e}")


# --- Common LLM & Runner Helpers ---
def _get_dynamic_max_tokens(state: TDDState | None) -> int:
    """Calculate a dynamic max token limit based on current code size."""
    if not state:
        return 8192

    current_codes = [
        state.get("impl_code", ""),
        state.get("tests_code", ""),
        state.get("design_doc", ""),
        state.get("unit_tests_code", ""),
        state.get("integration_tests_code", ""),
    ]
    max_char_len = max(len(c) for c in current_codes if c) if any(current_codes) else 0

    # 1 token roughly equals 4 characters. Add a 4096 token safety buffer.
    estimated_tokens = max_char_len // 4
    dynamic_limit = estimated_tokens + 4096

    # Clamp between 8192 (minimum default) and 32768 (hard ceiling)
    return min(max(8192, dynamic_limit), 32768)


def _call_llm_structured_wrapper(
    prompt: str, response_schema: Any, model_name: str = config.MODEL_PRIMARY, **kwargs
) -> Any:
    state = getattr(_thread_local, "current_state", None)
    max_tokens = _get_dynamic_max_tokens(state)
    return call_llm_structured(prompt, response_schema, model_name=model_name, max_tokens=max_tokens)


def _call_llm_text_wrapper(prompt: str, model_name: str = config.MODEL_PRIMARY, **kwargs) -> str:
    state = getattr(_thread_local, "current_state", None)
    max_tokens = _get_dynamic_max_tokens(state)
    return call_llm_text(prompt, model_name=model_name, max_tokens=max_tokens)


_call_llm_structured = _call_llm_structured_wrapper
_call_llm_text = _call_llm_text_wrapper


def _cleanup_history_on_rollback(state: TDDState):
    """
    Rename previous history snapshot files for the current requirement
    by appending a '.bak' suffix, to prevent the Loop Detector from matching
    and falsely triggering on design-rollback cycles.
    """
    import glob

    artifacts_dir = getattr(config, "ARTIFACTS_DIR", "artifacts")
    history_dir = os.path.join(artifacts_dir, "history")
    if not os.path.exists(history_dir):
        return

    module_name = state.get("module_name", "impl.py")
    base_name = os.path.splitext(module_name)[0]

    req_id = None
    if "requirements" in state and "current_req_index" in state:
        reqs = state["requirements"]
        idx = state["current_req_index"]
        if 0 <= idx < len(reqs):
            req_id = reqs[idx].get("id")
    if req_id is None and "current_req_index" in state:
        req_id = f"req{state['current_req_index'] + 1:03d}"

    test_iterations = state.get("test_iterations", 1)

    if req_id:
        pattern = os.path.join(
            history_dir, f"{base_name}_{req_id.lower()}*_test_iter{test_iterations:03d}*_impl_iter*.py"
        )
        files = glob.glob(pattern)
        for f in files:
            try:
                dest = f + ".bak"
                if os.path.exists(dest):
                    i = 1
                    while os.path.exists(f"{dest}.{i}"):
                        i += 1
                    dest = f"{dest}.{i}"
                os.rename(f, dest)
            except Exception as e:
                print(f"Warning: Failed to rename history file {f}: {e}")


def _execute_tests_helper(test_file: str | None, state: TDDState) -> dict:
    """Helper to execute pytest suite against specific target test file or all tests (regression)."""
    test_name = test_file if test_file else ""
    cwd_dir = config.ARTIFACTS_DIR

    try:
        run_env = os.environ.copy()
        run_env["PYTHONUNBUFFERED"] = "1"
        run_env["TDD_ROBO_DEBUG"] = config.TDD_ROBO_DEBUG

        if test_name:
            print(f"[TDD Robo] 🏃 Running tests ({test_name})...")
            cmd = [sys.executable, "-m", "pytest", test_name, "-v", "-s", "--tb=short", "--junitxml=report.xml"]
        else:
            print("[TDD Robo] 🏃 Running regression test suite (all tests)...")
            cmd = [
                sys.executable,
                "-m",
                "pytest",
                "-v",
                "-s",
                "--tb=short",
                "--ignore=history",
                "--junitxml=report.xml",
            ]

        if config.PYTEST_MAXFAIL > 0:
            cmd.append(f"--maxfail={config.PYTEST_MAXFAIL}")

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=cwd_dir,
            env=run_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=TEST_EXECUTION_TIMEOUT_SEC)
            success = process.returncode == 0
            output = str(stdout or "") + "\n" + str(stderr or "")
        except subprocess.TimeoutExpired:
            success = False
            try:
                if hasattr(os, "killpg"):
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            except OSError:
                pass
            stdout, stderr = process.communicate()
            output = f"Test execution timed out after {TEST_EXECUTION_TIMEOUT_SEC} seconds.\n"
            output += f"Partial stdout:\n{stdout or ''}\n"
            output += f"Partial stderr:\n{stderr or ''}"
    except Exception as e:
        success = False
        output = f"Failed to execute tests: {e}"

    # Parse JUnit XML if it exists
    import xml.etree.ElementTree as ET

    xml_path = os.path.join(cwd_dir, "report.xml")
    failed_methods = set()
    failed_files = set()
    failed_tests_detail = {}
    if os.path.exists(xml_path):
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for tc in root.iter("testcase"):
                failure_node = tc.find("failure")
                error_node = tc.find("error")
                err_node = failure_node if failure_node is not None else error_node
                if err_node is not None:
                    name = tc.get("name")
                    file_path = tc.get("file")
                    if name:
                        failed_methods.add(name)
                        failed_tests_detail[name] = err_node.text or ""
                    if file_path:
                        failed_files.add(os.path.basename(file_path))
            try:
                os.remove(xml_path)
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: Failed to parse JUnit XML report at {xml_path}: {e}")

    MAX_OUTPUT_CHARS = 8000
    if len(output) > MAX_OUTPUT_CHARS:
        output = _truncate_test_output_smart(output, MAX_OUTPUT_CHARS)

    if config.VERBOSE:
        print(output)

    active_summary = _parse_pytest_summary(output)
    summary_part = f" ({active_summary})" if active_summary else ""
    print(f"[TDD Robo] {'✅ Test suite passed!' if success else '❌ Test suite failed!'}{summary_part}")

    # stagnant_iterations update logic
    last_summary = state.get("last_test_summary", "")

    def parse_failed_error_counts(summary_str: str) -> tuple[int, int]:
        import re

        failed = 0
        errors = 0
        if not summary_str:
            return failed, errors
        failed_match = re.search(r"(\d+)\s+failed", summary_str)
        if failed_match:
            failed = int(failed_match.group(1))
        error_match = re.search(r"(\d+)\s+error", summary_str)
        if error_match:
            errors = int(error_match.group(1))
        return failed, errors

    curr_failed, curr_errors = parse_failed_error_counts(active_summary)
    last_failed, last_errors = parse_failed_error_counts(last_summary)

    has_progress = False
    if success:
        has_progress = True
    elif not last_summary:
        has_progress = True
    else:
        if curr_failed < last_failed or curr_errors < last_errors or active_summary != last_summary:
            has_progress = True
        else:
            has_progress = False

    stagnant_iters = state.get("stagnant_iterations", 0)
    new_stagnant_iters = 0 if has_progress else stagnant_iters + 1

    if success:
        res = {
            "test_output": output,
            "success": success,
            "syntax_error_iterations": 0,
            "last_test_summary": "",
            "stagnant_iterations": 0,
            "iterations": 0,
            "test_plan_iterations": 0,
            "failed_methods": list(failed_methods),
            "failed_files": list(failed_files),
            "failed_tests_detail": failed_tests_detail,
        }
        if state.get("impl_code"):
            res["last_green_impl_code"] = state.get("impl_code")
        return res

    return {
        "test_output": output,
        "success": success,
        "syntax_error_iterations": 0,
        "last_test_summary": active_summary,
        "stagnant_iterations": new_stagnant_iters,
        "failed_methods": list(failed_methods),
        "failed_files": list(failed_files),
        "failed_tests_detail": failed_tests_detail,
    }


def _syntax_check_helper(filename: str, state_key_iter: str, state: TDDState) -> dict:
    """Helper to perform syntax validation check on a file and update iteration state."""
    _update_req_progress(state)
    iters = int(str(state.get(state_key_iter, 0) or 0)) + 1
    is_test = filename.startswith("test")

    # If the file is implementation and has a previous Search/Replace application failure,
    # preserve the error and increment the iteration counter.
    existing_err = state.get("tests_check_output", "") if is_test else state.get("impl_check_output", "")
    if existing_err and "Failed to apply Search/Replace block" in existing_err:
        print(f"[TDD Robo] ❌ {filename} has Search/Replace application failure (Iteration {iters})!")
        save_history_snapshot(f"syntax_error_{filename}", existing_err, iters, state)
        return {state_key_iter: iters, "tests_check_output" if is_test else "impl_check_output": existing_err}

    abs_path = os.path.join(config.ARTIFACTS_DIR, filename)
    print(f"[TDD Robo] 🔍 Checking {filename} syntax (Iteration {iters})...")

    check_output = _run_syntax_check(abs_path, filename)
    passed = check_output == ""

    if passed:
        print(f"[TDD Robo] ✅ {filename} syntax check passed!")
        return {state_key_iter: 0, "tests_check_output" if is_test else "impl_check_output": ""}
    else:
        print(f"[TDD Robo] ❌ {filename} syntax check failed!")
        save_history_snapshot(f"syntax_error_{filename}", check_output, iters, state)
        return {state_key_iter: iters, "tests_check_output" if is_test else "impl_check_output": check_output}


# --- Node Definitions ---


def fetch_spec(state: TDDState):
    """
    Fetch the specification from the provided URL or local cache.

    Reads:
        - spec_url
    Writes:
        - spec_content
    """
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
    """
    Analyze the specification and extract a list of verifiable functional requirements.

    Reads:
        - spec_content
    Writes:
        - requirements
        - current_req_index
        - requirements_list_str
    """
    _update_req_progress(state)
    print("[TDD Robo] 📋 Analyzing specification and extracting requirements...")
    template = get_prompt("generate_requirements_prompt", GENERATE_REQUIREMENTS_PROMPT)
    prompt = template.format(spec=state.get("spec_content", ""))

    response = call_llm_standard(prompt, response_schema=RequirementsList)
    reqs_data = json.loads(extract_json(response))

    requirements = []
    requirements_list_str = "Requirements Checklist:\n"
    for r in reqs_data.get("requirements", []):
        req_dict = {"id": r.get("id"), "description": r.get("description")}
        requirements.append(req_dict)
        requirements_list_str += f"- {req_dict['id']}: {req_dict['description']}\n"

    _thread_local.total_req_num = len(requirements)
    _thread_local.current_req_num = 1

    print(f"[TDD Robo] ✅ Extracted {len(requirements)} sequential requirements.")
    if config.VERBOSE:
        print(requirements_list_str)

    req_path = save_artifact("requirements.md", requirements_list_str)
    print(f"[TDD Robo] ✅ Saved requirements list to {req_path}")

    # Print initial progress checklist
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
    """
    Determine filenames for the implementation and test modules.

    Reads:
        - goal
    Writes:
        - module_name
        - test_module_name
    """
    _update_req_progress(state)
    print("[TDD Robo] 📁 Determining filenames...")
    template = get_prompt("plan_files_prompt", PLAN_FILES_PROMPT)
    prompt = template.format(goal=state.get("goal", ""))

    response = call_llm_standard(prompt, response_schema=FilePlan)

    plan = json.loads(extract_json(response))
    impl_name = plan.get("impl_filename", DEFAULT_IMPL_NAME)
    test_name = plan.get("test_filename", DEFAULT_TEST_NAME)

    print(f"[TDD Robo] ✅ Determined filenames: Implementation={impl_name}, Test={test_name}")
    return {"module_name": impl_name, "test_module_name": test_name}


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

    prompt = GENERATE_DESIGN_PROMPT.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
        test_name=state.get("test_module_name", DEFAULT_TEST_NAME),
        design_context=design_context,
        impl_code="",
    )

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

    # Skip design updates if rollback was triggered purely by oracle mismatch
    if state.get("oracle_discrepancy_only", False):
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

        from prompts import GENERATE_DESIGN_PROMPT
        from schema import DesignDocument

        design_context = (
            f"\n# Target Requirement Update\n"
            f"We are now designing components to support the following requirement:\n"
            f"{target_req_str}\n"
            "Please update the architectural components, internal structures, "
            "or component interfaces in the Design Document to support this requirement cleanly.\n"
        )

        # Check if we were rolled back due to a loop or failure, or if we have design review feedback
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
                    # Truncate test output to avoid exceeding token limits using balanced bidirectional context
                    truncated_test = _get_balanced_test_output_context(latest_test, max_chars=3500)
                    defect_context += f"### Latest Test Output (Truncated):\n{truncated_test}\n"
                design_context += defect_context

        last_green = state.get("last_green_impl_code", "")
        impl_code_for_design = last_green if last_green else state.get("impl_code", "")

        prompt = GENERATE_DESIGN_PROMPT.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
            test_name=state.get("test_module_name", DEFAULT_TEST_NAME),
            design_context=design_context,
            impl_code=impl_code_for_design,
        )

        if state.get("loop_detected"):
            last_test_output = state.get("test_output", "")
            if last_test_output:
                snippet = _get_balanced_test_output_context(last_test_output, max_chars=3500)
                loop_note = (
                    "\n\n# ⚠️ DESIGN ROLLBACK TRIGGERED BY IMPLEMENTATION LOOP\n"
                    "The previous design revision led to a stuck implementation loop. "
                    "The following test output shows what was failing when the loop was detected. "
                    "Please identify and fix the architectural root cause that made it impossible "
                    "to satisfy all tests simultaneously.\n"
                    f"<loop_failure_context>\n{snippet}\n</loop_failure_context>\n"
                )
                prompt += loop_note

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

    # Load prompt and call LLM
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

    report = _call_llm_structured(prompt, DesignReviewReport, model_name=MODEL_PRIMARY)

    quality = report.estimated_quality
    comments = report.comments
    print(f"[TDD Robo] Design Audit Report ({phase}): Estimated Quality = {quality}%")
    print(f"[TDD Robo] Audit Comments:\n{comments}")

    iters = state.get("design_review_iterations", 0) + 1

    target_quality = state.get("target_design_quality", TARGET_DESIGN_QUALITY)

    # If quality < target_quality and we haven't exceeded 3 iterations, loop back.
    if quality < target_quality and iters <= 3:
        print(
            f"[TDD Robo] 🚨 Design quality check failed ({quality}% < {target_quality}%). "
            f"Looping back to refine design. Iteration {iters}/3"
        )
        return {
            "design_review_feedback": comments,
            "design_review_iterations": iters,
            "design_updated": False,
        }
    else:
        if quality < target_quality:
            print(f"[TDD Robo] ⚠️ Max design review iterations reached ({iters}/3). Proceeding with current design.")
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


def should_review_design_initial_or_continue(state: TDDState) -> str:
    if state.get("design_review_feedback"):
        return "generate_design_initial"
    return "plan_unit_tests"


def should_review_design_incremental_or_continue(state: TDDState) -> str:
    if state.get("design_review_feedback"):
        return "update_design_for_req"
    return "plan_unit_tests"


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
    is_fix = False
    oracle_constraints = TEST_PLAN_ORACLE_CONSTRAINTS if getattr(config, "ORACLE_VERIFIER", None) is not None else ""

    if review_feedback and state.get("unit_test_plan"):
        is_fix = True
        prompt = PLAN_UNIT_TESTS_PROMPT_FIX.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design_doc=state.get("design_doc", ""),
            target_req=target_req_str,
            test_plan=state.get("unit_test_plan", ""),
            test_plan_review=review_feedback,
            oracle_constraints=oracle_constraints,
        )
    else:
        prompt = PLAN_UNIT_TESTS_PROMPT.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design_doc=state.get("design_doc", ""),
            target_req=target_req_str,
            oracle_constraints=oracle_constraints,
        )

    bug_report = state.get("bug_report", "")
    if bug_report:
        oracle_warning = (
            "\n\n## 🚨 Previous Test Mismatch / Oracle Warnings\n"
            "During previous runs, the tests generated under this requirement had assertion failures "
            "where the expected outcomes in the tests did not match the correct oracle behavior.\n"
            "Please carefully analyze the failures below and ensure your new test plan's expected outcomes "
            "are mathematically correct and match the oracle expectations:\n"
            f"{bug_report}\n"
        )
        prompt += oracle_warning

    test_plan_obj = _call_llm_structured(prompt, TestPlan, model_name=MODEL_PRIMARY)

    if is_fix:
        if state.get("oracle_discrepancy_only"):
            # If rollback is purely due to oracle discrepancies, bypass merging to clean out incorrect/renamed cases
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
        "test_plan": plan_md,  # Populate legacy key for review compatibility
        "test_plan_iterations": iters,
    }


def _run_early_oracle_verification(test_plan_json: str) -> list[dict]:
    """
    Parse test plan JSON and verify calculation expectations against dynamic oracle.
    Returns a list of dictionaries with mismatch details:
    [{"test_case": tc, "expr": expr_cleaned, "expected_val": expected_val, "oracle_val": last_oracle_val}]
    """
    verifier = getattr(config, "ORACLE_VERIFIER", None)
    if verifier is None:
        return []

    try:
        data = json.loads(test_plan_json)
    except Exception:
        return []

    mismatches = []
    for tc in data.get("test_cases", []):
        action = tc.get("action", "")
        expected = tc.get("expected_outcome", "")
        oracle_expr = tc.get("oracle_expression")
        oracle_expected = tc.get("oracle_expected")

        expr = None
        expected_val = None

        if oracle_expr and oracle_expected:
            expr = oracle_expr
            expected_val = oracle_expected
        else:
            # Look for [Evaluate: expression]
            match = re.search(r"\[Evaluate:\s*(.*?)\s*\]", expected)
            if not match:
                match = re.search(r"\[Evaluate:\s*(.*?)\s*\]", action)

            if match:
                expr = match.group(1)
                cleaned_expected = expected.replace(match.group(0), "").strip()
                num_match = re.search(r"\b\d+(?:\.\d+)?\b", cleaned_expected)
                if num_match:
                    expected_val = num_match.group(0)
            else:
                quote_match = re.search(r"['\"](.*?)['\"]", action)
                if quote_match:
                    expr = quote_match.group(1)
                else:
                    math_match = re.search(r"\b\d+[\s\d+\-*/%;=.]+\d+\b", action)
                    if math_match:
                        expr = math_match.group(0)

                # Try to find expected outcome value
                num_match = re.search(r"\b\d+(?:\.\d+)?\b", expected)
                if num_match:
                    expected_val = num_match.group(0)

        if expr and expected_val:
            expr_cleaned = expr.strip().rstrip(".")
            # Filter out simple numbers (integers or floats) as they are not formulas to evaluate
            try:
                float(expr_cleaned)
                continue
            except ValueError:
                pass
            # Skip if the expression is not a valid mathematical formula
            # (i.e. contains letters other than the registers 'scale', 'ibase',
            # 'obase', 'last' or math library functions)
            cleaned_temp = re.sub(r"\\[a-zA-Z]", "", expr_cleaned)
            cleaned_temp = re.sub(r"\b(scale|ibase|obase|last)\b", "", cleaned_temp, flags=re.IGNORECASE)
            cleaned_temp = re.sub(r"\b[sclaej]\s*\(", "(", cleaned_temp, flags=re.IGNORECASE)
            if re.search(r"[a-zA-Z]", cleaned_temp):
                continue
            try:
                try:
                    oracle_result = verifier(expr_cleaned, expected=expected_val).strip()
                except TypeError:
                    oracle_result = verifier(expr_cleaned).strip()
                oracle_lines = [l.strip() for l in oracle_result.splitlines() if l.strip()]
                # Support multi-line outputs (e.g. semicolon-separated expressions in bc)
                # by joining lines with '\n', while keeping single-line fallback compatible.
                last_oracle_val = "\n".join(oracle_lines) if oracle_lines else ""

                if "Error" not in oracle_result and "Exception" not in oracle_result:
                    if last_oracle_val != expected_val:
                        mismatches.append(
                            {
                                "test_case": tc,
                                "expr": expr_cleaned,
                                "expected_val": expected_val,
                                "oracle_val": last_oracle_val,
                            }
                        )
            except Exception:
                pass

    return mismatches


def _judge_oracle_discrepancy_with_llm(
    design_doc: str, test_case_dict: dict, oracle_val: str
) -> OracleDiscrepancyJudgment:
    """
    Use the primary LLM to judge whether an oracle discrepancy is a core
    design flaw or a test plan representation issue.
    """
    from prompts import JUDGE_ORACLE_DISCREPANCY_PROMPT

    tc_formatted = json.dumps(test_case_dict, indent=2, ensure_ascii=False)

    prompt = JUDGE_ORACLE_DISCREPANCY_PROMPT.format(
        design_doc=design_doc, test_case=tc_formatted, actual_output=oracle_val
    )

    try:
        judgment = _call_llm_structured(prompt, OracleDiscrepancyJudgment, model_name=MODEL_PRIMARY)
        return judgment
    except Exception as e:
        print(f"Warning: Failed to call LLM Judge for discrepancy, defaulting to Design Error: {e}")
        return OracleDiscrepancyJudgment(
            is_design_error=True, reason=f"LLM Judge call failed: {str(e)}", corrected_expected=None
        )


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

    review_report = _call_llm_structured(prompt, TestPlanReviewReport, model_name=MODEL_PRIMARY)
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

            if current_rollback >= 3:
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
            decision = "review_test_plan"  # Rollback to plan_tests
            feedback = (
                f"Early Oracle Verification detected Test Plan notation errors. "
                f"Please update the expected outcomes or oracle fields to match the correct oracle format:\n{disc_str}"
            )
            return {
                "test_plan_review": feedback,
                "test_plan_review_decision": decision,
                "oracle_discrepancy_only": True,  # Skip design updates since it is only a test plan issue
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
    is_fix = False
    oracle_constraints = TEST_PLAN_ORACLE_CONSTRAINTS if getattr(config, "ORACLE_VERIFIER", None) is not None else ""

    if review_feedback and state.get("integration_test_plan"):
        is_fix = True
        prompt = PLAN_INTEGRATION_TESTS_PROMPT_FIX.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design_doc=state.get("design_doc", ""),
            target_req=target_req_str,
            test_plan=state.get("integration_test_plan", ""),
            test_plan_review=review_feedback,
            oracle_constraints=oracle_constraints,
        )
    else:
        prompt = PLAN_INTEGRATION_TESTS_PROMPT.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design_doc=state.get("design_doc", ""),
            target_req=target_req_str,
            oracle_constraints=oracle_constraints,
        )

    bug_report = state.get("bug_report", "")
    if bug_report:
        oracle_warning = (
            "\n\n## 🚨 Previous Test Mismatch / Oracle Warnings\n"
            "During previous runs, the tests generated under this requirement had assertion failures "
            "where the expected outcomes in the tests did not match the correct oracle behavior.\n"
            "Please carefully analyze the failures below and ensure your new test plan's expected outcomes "
            "are mathematically correct and match the oracle expectations:\n"
            f"{bug_report}\n"
        )
        prompt += oracle_warning

    test_plan_obj = _call_llm_structured(prompt, TestPlan, model_name=MODEL_PRIMARY)

    if is_fix:
        if state.get("oracle_discrepancy_only"):
            # If rollback is purely due to oracle discrepancies, bypass merging to clean out incorrect/renamed cases
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

    review_report = _call_llm_structured(prompt, TestPlanReviewReport, model_name=MODEL_PRIMARY)
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

            if current_rollback >= 3:
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
            decision = "review_test_plan"  # Rollback to plan_tests
            feedback = (
                f"Early Oracle Verification detected Test Plan notation errors. "
                f"Please update the expected outcomes or oracle fields to match the correct oracle format:\n{disc_str}"
            )
            return {
                "test_plan_review": feedback,
                "test_plan_review_decision": decision,
                "oracle_discrepancy_only": True,  # Skip design updates since it is only a test plan issue
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

    prompt = GENERATE_UNIT_TESTS_PROMPT.format(
        target_req=target_req_str,
        design_doc=state.get("design_doc", ""),
        unit_test_plan=state.get("unit_test_plan", ""),
        impl_code=state.get("impl_code", ""),
    )

    bug_report = state.get("bug_report", "")
    if bug_report:
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
        bug_fix_context = (
            "\n\n# Bug Fix Context\n"
            "A previous test execution failed because of bugs in the test suite. "
            "Please update the test suite to resolve these issues.\n\n"
            "## Previous Test Code\n"
            "<previous_test_code>\n"
            f"{previous_tests}\n"
            "</previous_test_code>\n\n"
            "## Bug Report\n"
            "<bug_report>\n"
            f"{bug_report}\n"
            "</bug_report>\n\n"
            "Please output the complete corrected python test code, ensuring that all test bugs described "
            "in the bug report are resolved.\n"
        )
        prompt += bug_fix_context

    tests_check_output = state.get("tests_check_output", "")
    if tests_check_output:
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
        syntax_fix_context = (
            "\n\n# Test Syntax Fix Context\n"
            "The previously generated test code had syntax errors. "
            "Please fix the syntax errors based on the error output below.\n\n"
            "## Previous Test Code\n"
            "<previous_test_code>\n"
            f"{previous_tests}\n"
            "</previous_test_code>\n\n"
            "## Syntax Checker Output\n"
            "<syntax_check_output>\n"
            f"{tests_check_output}\n"
            "</syntax_check_output>\n\n"
            "Please output the complete corrected python test code, ensuring that all syntax errors are resolved.\n"
        )
        prompt += syntax_fix_context

    unit_tests_code = _call_llm_text(prompt, model_name=MODEL_PRIMARY)

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

    prompt = GENERATE_INTEGRATION_TESTS_PROMPT.format(
        target_req=target_req_str,
        design_doc=state.get("design_doc", ""),
        integration_test_plan=state.get("integration_test_plan", ""),
        impl_code=state.get("impl_code", ""),
    )

    bug_report = state.get("bug_report", "")
    if bug_report:
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
        bug_fix_context = (
            "\n\n# Bug Fix Context\n"
            "A previous test execution failed because of bugs in the test suite. "
            "Please update the test suite to resolve these issues.\n\n"
            "## Previous Test Code\n"
            "<previous_test_code>\n"
            f"{previous_tests}\n"
            "</previous_test_code>\n\n"
            "## Bug Report\n"
            "<bug_report>\n"
            f"{bug_report}\n"
            "</bug_report>\n\n"
            "Please output the complete corrected python test code, ensuring that all test bugs described "
            "in the bug report are resolved.\n"
        )
        prompt += bug_fix_context

    tests_check_output = state.get("tests_check_output", "")
    if tests_check_output:
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
        syntax_fix_context = (
            "\n\n# Test Syntax Fix Context\n"
            "The previously generated test code had syntax errors. "
            "Please fix the syntax errors based on the error output below.\n\n"
            "## Previous Test Code\n"
            "<previous_test_code>\n"
            f"{previous_tests}\n"
            "</previous_test_code>\n\n"
            "## Syntax Checker Output\n"
            "<syntax_check_output>\n"
            f"{tests_check_output}\n"
            "</syntax_check_output>\n\n"
            "Please output the complete corrected python test code, ensuring that all syntax errors are resolved.\n"
        )
        prompt += syntax_fix_context

    integration_tests_code = _call_llm_text(prompt, model_name=MODEL_PRIMARY)

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


def _run_syntax_check(filename: str, label: str) -> str:
    """
    Run a basic syntax check on a given Python file using flake8.

    Args:
        filename (str): The path to the file to check.
        label (str): A descriptive label for logging output.

    Returns:
        str: Empty string if syntax is valid, otherwise the error output.
    """
    try:
        # Select rules defined in config.FLAKE8_SELECT
        result = subprocess.run(
            [sys.executable, "-m", "flake8", f"--select={config.FLAKE8_SELECT}", filename],
            capture_output=True,
            text=True,
            timeout=SYNTAX_CHECK_TIMEOUT_SEC,
        )
        success = result.returncode == 0
        output = str(result.stdout or "") + "\n" + str(result.stderr or "")
    except subprocess.TimeoutExpired as e:
        success = False
        output = f"Syntax check timed out after {e.timeout} seconds."
    except Exception as e:
        success = False
        output = str(e)
    if not success and config.VERBOSE:
        print(output)
    return "" if success else output


def _get_combined_tests_code(state: TDDState) -> str:
    """Helper to collect and combine all test_*.py files in artifacts directory."""
    current_req_idx = state.get("current_req_index", 0)
    all_tests_content = []
    if os.path.exists(config.ARTIFACTS_DIR):
        test_files = sorted(glob.glob(os.path.join(config.ARTIFACTS_DIR, "test_*.py")))
        for tf in test_files:
            tf_basename = os.path.basename(tf)

            # Skip future requirements tests
            match = re.search(r"req(\d+)", tf_basename)
            if match:
                file_req_num = int(match.group(1))
                if file_req_num >= current_req_idx + 2:
                    continue

            try:
                with open(tf, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    all_tests_content.append(f"# --- File: {tf_basename} ---\n{file_content}")
            except Exception:
                pass
    # Fallback to state code keys if no files found
    if not all_tests_content:
        unit = state.get("unit_tests_code", "")
        integ = state.get("integration_tests_code", "")
        if unit or integ:
            return f"# Unit Tests:\n{unit}\n\n# Integration Tests:\n{integ}"
    return "\n\n".join(all_tests_content) if all_tests_content else state.get("tests_code", "")


def _truncate_test_output_smart(text: str | None, max_chars: int = 8000) -> str:
    """Intelligently truncates pytest output to protect token context while preserving the FAILURES block."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    # Try to find FAILURES/ERRORS block to prioritize it
    failures_match = re.search(r"(={3,}\s*(?:FAILURES|ERRORS)\s*={3,}.*?)(?=\n={3,}|\Z)", text, re.DOTALL)
    failures_content = ""
    if failures_match:
        failures_content = failures_match.group(1).strip()

    # Reserve size for prefix (startup logs) and suffix (summary info)
    min_ends = max(20, int(max_chars * 0.1))
    reserved_for_ends = min(2000, max(min_ends, int(max_chars * 0.25)))
    max_failures_len = max(50, max_chars - reserved_for_ends - 100)
    if len(failures_content) > max_failures_len:
        half = max_failures_len // 2
        failures_content = (
            failures_content[:half] + "\n\n... [TRUNCATED MIDDLE OF FAILURES] ...\n\n" + failures_content[-half:]
        )

    overhead = 350
    available_for_ends = max(50, max_chars - len(failures_content) - overhead)
    prefix_len = max(10, int(available_for_ends * 0.35))
    suffix_len = max(10, available_for_ends - prefix_len)

    prefix = text[:prefix_len].strip()
    suffix = text[-suffix_len:].strip()

    parts = []
    if prefix:
        parts.append(prefix)
    if failures_content:
        parts.append("\n=== PRESERVED FAILURES/ERRORS BLOCK ===\n" + failures_content)
    else:
        parts.append("\n... [NO FAILURES/ERRORS BLOCK FOUND IN LOGS] ...")
    if suffix:
        parts.append(suffix)

    msg = f"\n\n... [TRUNCATED MIDDLE LOGS TO PROTECT CONTEXT (Original total length: {len(text)})] ...\n\n"
    result = msg.join(parts)
    if len(result) > max_chars:
        # Ultimate fallback to simple slice
        fallback_prefix = max(10, int(max_chars * 0.35))
        fallback_suffix = max(0, max_chars - fallback_prefix - 100)
        return (
            text[:fallback_prefix]
            + f"\n\n... [TRUNCATED {len(text) - max_chars} CHARACTERS] ...\n\n"
            + text[-fallback_suffix:]
        )
    return result


def _get_balanced_test_output_context(text: str, max_chars: int = 4000) -> str:
    """Preserves both the beginning and end of a test log, prioritising the FAILURES block."""
    if not text or len(text) <= max_chars:
        return text
    if max_chars < 150:
        return text[:max_chars]
    return _truncate_test_output_smart(text, max_chars)


def _get_existing_tests_context(state: TDDState) -> str:
    """Collect all test_*.py files in artifacts directory except the active test module."""
    active_test_name = state.get("test_module_name", "")
    active_basename = os.path.basename(active_test_name) if active_test_name else ""
    current_req_idx = state.get("current_req_index", 0)

    all_tests_content = []
    if os.path.exists(config.ARTIFACTS_DIR):
        test_files = sorted(glob.glob(os.path.join(config.ARTIFACTS_DIR, "test_*.py")))
        for tf in test_files:
            tf_basename = os.path.basename(tf)
            if active_basename and tf_basename == active_basename:
                continue

            # Skip future requirements tests
            match = re.search(r"req(\d+)", tf_basename)
            if match:
                file_req_num = int(match.group(1))
                if file_req_num >= current_req_idx + 2:
                    continue

            try:
                with open(tf, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    all_tests_content.append(f"# --- File: {tf_basename} ---\n{file_content}")
            except Exception:
                pass
    return "\n\n".join(all_tests_content) if all_tests_content else ""


def _build_uniqueness_advice(impl_check_output: str, existing_impl: str) -> str:
    """Analyze Search/Replace uniqueness errors and build context hints for the LLM."""
    if "Failed to apply Search/Replace block" not in impl_check_output:
        return ""

    advice = (
        "\n\n# 🚨 CRITICAL ERROR: PREVIOUS SEARCH/REPLACE APPLICATION FAILED! 🚨\n"
        "Your previous attempt to modify the code failed because the "
        "Search/Replace blocks were invalid or non-unique.\n"
        "Please follow these guidelines to fix your Search/Replace blocks:\n"
        "1. Read the `<existing_impl_code>` section line-by-line.\n"
        "2. Make sure your SEARCH block matches `<existing_impl_code>` character-for-character, "
        "including comments, spaces, newlines, and indentation.\n"
        "3. Do NOT add trailing periods or other punctuation at the end of your code blocks "
        "unless they are exactly present in `<existing_impl_code>`.\n"
        "4. **UNIQUENESS RULE**: A SEARCH block MUST match exactly one location in the file. "
        "If you are replacing a short or common line (e.g., `return None, True` or `self.consume()`), "
        "you MUST include 3-5 unique surrounding lines (like function definitions, "
        "preceding statements, or specific variables) "
        "in your SEARCH block to make it 100% unique. Do NOT keep it too small if it matches multiple places.\n"
    )

    if "matches multiple times" in impl_check_output:
        line_nums = []
        match = re.search(r"Matches found at line numbers: \[(.*?)\]", impl_check_output)
        if match:
            line_nums = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        context_hints = []
        if line_nums and existing_impl:
            lines = existing_impl.splitlines()
            indices = [min(len(lines) - 1, ln - 1) for ln in line_nums if ln > 0]
            if indices:
                first_idx = indices[0]
                block_len = 1
                while True:
                    all_same = True
                    for other_idx in indices:
                        if first_idx + block_len >= len(lines) or other_idx + block_len >= len(lines):
                            all_same = False
                            break
                        if lines[first_idx + block_len] != lines[other_idx + block_len]:
                            all_same = False
                            break
                    if all_same:
                        block_len += 1
                    else:
                        break

                pre_offset_is_distinguishing = {}
                for d in [-3, -2, -1]:
                    pre_lines_at_offset: list[str | None] = []
                    for other_idx in indices:
                        o_idx = other_idx + d
                        if 0 <= o_idx < len(lines):
                            pre_lines_at_offset.append(lines[o_idx])
                        else:
                            pre_lines_at_offset.append(None)
                    pre_offset_is_distinguishing[d] = len(set(pre_lines_at_offset)) > 1

                post_offset_is_distinguishing = {}
                for d in [0, 1, 2]:
                    post_lines_at_offset: list[str | None] = []
                    for other_idx in indices:
                        o_idx = other_idx + block_len + d
                        if 0 <= o_idx < len(lines):
                            post_lines_at_offset.append(lines[o_idx])
                        else:
                            post_lines_at_offset.append(None)
                    post_offset_is_distinguishing[d] = len(set(post_lines_at_offset)) > 1

                for line_num in line_nums:
                    idx = min(len(lines) - 1, line_num - 1)
                    if idx < 0:
                        continue
                    pre_start_idx = idx
                    for d in [-3, -2, -1]:
                        if pre_offset_is_distinguishing.get(d, False):
                            pre_start_idx = idx + d
                            break
                    post_end_idx = idx + block_len
                    for d in reversed([0, 1, 2]):
                        if post_offset_is_distinguishing.get(d, False):
                            post_end_idx = idx + block_len + d + 1
                            break
                    if pre_start_idx == idx and post_end_idx == idx + block_len:
                        pre_start_idx = max(0, idx - 2)
                    search_lines = lines[pre_start_idx:post_end_idx]
                    search_block_text = "\n".join(search_lines)
                    hint = (
                        f"- For the match at line {line_num}, you MUST use this unique SEARCH block template "
                        f"(do NOT omit context lines):\n"
                        f"<<<<<<< SEARCH\n{search_block_text}\n=======\n"
                    )
                    context_hints.append(hint)

        if context_hints:
            joined_hints = "\n".join(context_hints)
            advice += (
                "5. **UNIQUENESS ERROR**: Your SEARCH block matched multiple times. "
                "A SEARCH block MUST match exactly one unique location in the file. "
                "Specifically, you MUST format your SEARCH blocks with unique context lines "
                "to avoid multiple matches. Here are the exact templates you should use:\n"
                f"{joined_hints}\n"
            )
        else:
            advice += (
                "5. **UNIQUENESS ERROR**: Your SEARCH block matched multiple times. "
                "A SEARCH block MUST match exactly one unique location in the file. "
            )

    return advice


def _build_impl_prompt(
    state: TDDState,
    phase: str,
    target_req_str: str,
    existing_impl: str,
    existing_impl_code_param: str,
    tests_code: str,
    domain_tips: str,
    python_tips: str,
    impl_check_output: str,
    impl_name: str,
    design_updated: bool,
) -> str:
    """Build the final implementation prompt with context and loop warning additions."""
    if impl_check_output and state.get("impl_code"):
        template = IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX
        prompt = template.format(
            goal=state.get("goal", ""),
            design=state.get("design_doc", ""),
            tests_code=tests_code,
            impl_name=impl_name,
            impl_check_output=impl_check_output,
            bug_report=state.get("bug_report", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=add_line_numbers(existing_impl_code_param),
            domain_tips=domain_tips,
            python_tips=python_tips,
        )
    elif state.get("bug_report") and state.get("impl_code"):
        template = IMPLEMENT_LOGIC_PROMPT_FIX
        prompt = template.format(
            goal=state.get("goal", ""),
            design=state.get("design_doc", ""),
            tests_code=tests_code,
            impl_name=impl_name,
            bug_report=state.get("bug_report", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl_code_param,
            domain_tips=domain_tips,
            python_tips=python_tips,
            impl_check_output=impl_check_output,
        )
    else:
        template = IMPLEMENT_LOGIC_PROMPT_INITIAL
        prompt = template.format(
            goal=state.get("goal", ""),
            design=state.get("design_doc", ""),
            tests_code=tests_code,
            impl_name=impl_name,
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl_code_param,
            domain_tips=domain_tips,
            python_tips=python_tips,
        )

    if state.get("loop_detected"):
        loop_warning = (
            "\n\n# WARNING: IMPLEMENTATION TOGGLE LOOP DETECTED!\n"
            "The workflow is stuck in a cycle. Analyze all unit/regression tests and find a unified solution.\n"
        )
        prompt += loop_warning

    if design_updated:
        design_update_instruction = (
            "\n\n# CRITICAL: DESIGN WAS RECENTLY UPDATED!\n"
            "Do NOT use Search/Replace blocks. Write the COMPLETE updated Python code "
            "inside a single ```python block.\n"
            f"<existing_code_reference>\n{existing_impl}\n</existing_code_reference>\n"
        )
        prompt += design_update_instruction

    return prompt


def _implement_logic_helper(state: TDDState, phase: str) -> dict:
    """Generic helper to generate/update implementation code for a specific phase."""
    _update_req_progress(state)

    global_iters = state.get("iterations", 0) + 1
    print(f"[TDD Robo] 💻 Generating implementation code for phase: {phase} (Iteration {global_iters})...")

    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    test_path = os.path.join(config.ARTIFACTS_DIR, test_name)

    # 1. Skip check if already passing
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
                timeout=5,
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

    # 2. Gather requirement and implementation context
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

    # Determine the test suite input for the LLM
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
    else:  # regression
        tests_code = _get_combined_tests_code(state)

    domain_tips = state.get("domain_tips", "")
    python_tips = state.get("python_tips", "")
    impl_check_output = state.get("impl_check_output", "")

    # Format Search/Replace block application errors
    if impl_check_output:
        impl_check_output = (
            impl_check_output.replace("<<<<<<< SEARCH", "[PREVIOUS SEARCH]")
            .replace("=======", "[PREVIOUS DIVIDER]")
            .replace(">>>>>>> REPLACE", "[PREVIOUS REPLACE]")
        )
        advice = _build_uniqueness_advice(impl_check_output, existing_impl)
        impl_check_output += advice

        # Extract the failing SEARCH block text and re-present it explicitly
        failed_block_match = re.search(
            r"Target SEARCH block that failed to match:\n(.*?)(?=\n\[TDD Robo\]|\Z)",
            state.get("impl_check_output", ""),
            re.DOTALL,
        )
        if failed_block_match:
            failed_block_text = failed_block_match.group(1).strip()
            impl_check_output += (
                "\n\n# ⚠️ YOUR FAILING SEARCH BLOCK (exact text that did not match):\n"
                f"```\n{failed_block_text}\n```\n"
                "Compare this character-by-character with <existing_impl_code>. "
                "Common causes of mismatch: trailing punctuation (e.g. a stray period at "
                "end of a statement), extra or missing whitespace, comments, or the block "
                "was already modified by a preceding Search/Replace block in the same response.\n"
            )

    # 3. LLM Prompt Construction
    prompt = _build_impl_prompt(
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

    temp = 0.5 if state.get("loop_detected") else 0.0
    response = call_llm_with_reasoning(prompt, thinking_level="MINIMAL", temperature=temp)

    # 4. Apply changes and save
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


def _parse_pytest_summary(output: str) -> str:
    summary_lines = []
    for line in output.splitlines():
        line_strip = line.strip()
        if line_strip.startswith("===") and line_strip.endswith("==="):
            if any(word in line_strip for word in ["passed", "failed", "error", "skipped"]):
                summary_lines.append(line_strip.strip("=").strip())
    if summary_lines:
        return summary_lines[-1]
    return ""


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


def _find_failed_methods(test_output: str, state: TDDState | None = None) -> set[str]:
    """Find all failed test method names from the test output."""
    if state and state.get("failed_methods"):
        return set(cast(list[str], state.get("failed_methods")))
    failed_methods = set()
    for line in test_output.splitlines():
        if " PASSED" in line:
            continue
        match = re.search(r"::(test_\w+)", line)
        if match:
            failed_methods.add(match.group(1))
        else:
            if "FAILED" in line or "ERROR" in line or (line.startswith("___") and line.endswith("___")):
                match2 = re.search(r"\b(test_\w+)\b", line)
                if match2:
                    failed_methods.add(match2.group(1))
    return failed_methods


def _extract_method_body(method: str, tests_code: str) -> str:
    """Extract test method body from tests_code."""
    pattern = rf"def\s+{method}\s*\([^)]*\)\s*:"
    match = re.search(pattern, tests_code)
    if not match:
        return ""

    start_idx = match.start()
    remaining_code = tests_code[start_idx:]
    lines = remaining_code.splitlines()
    body_lines = [lines[0]]

    # Determine indentation of the body
    base_indent = None
    for line in lines[1:]:
        stripped = line.lstrip()
        if not stripped:
            body_lines.append(line)
            continue
        if stripped.startswith("def ") or stripped.startswith("class "):
            break
        indent = len(line) - len(stripped)
        if base_indent is None:
            base_indent = indent
        elif indent < base_indent:
            break
        body_lines.append(line)

    return "\n".join(body_lines)


def _extract_failing_line(method: str, test_output: str, state: TDDState | None = None) -> str | None:
    """Locate the traceback section for this method in test_output and extract failing line."""
    traceback_text = None
    if state:
        failed_tests_detail = cast(dict[str, str], state.get("failed_tests_detail", {}))
        if method in failed_tests_detail:
            traceback_text = failed_tests_detail[method]

    if not traceback_text:
        header_pattern = rf"_{{3,}}.*{re.escape(method)}.*_{{3,}}"
        header_match = re.search(header_pattern, test_output)
        if not header_match:
            return None

        traceback_text = test_output[header_match.end() :]
        next_header_match = re.search(r"_{3,}.*_{3,}", traceback_text)
        if next_header_match:
            traceback_text = traceback_text[: next_header_match.start()]
        else:
            summary_match = re.search(r"={3,} short test summary info ={3,}", traceback_text)
            if summary_match:
                traceback_text = traceback_text[: summary_match.start()]

    # Look for lines starting with '>' in the traceback
    failing_lines = []
    for line in traceback_text.splitlines():
        if line.strip().startswith(">"):
            failing_lines.append(line.strip().lstrip(">").strip())
    if not failing_lines:
        for line in traceback_text.splitlines():
            if line.strip().startswith("E "):
                cleaned = line.strip()[1:].strip()
                if not cleaned.startswith("AssertionError"):
                    failing_lines.append(cleaned)
    return failing_lines[0] if failing_lines else None


def _extract_oracle_target_llm(
    method_body: str,
    failing_line_in_body: str | None,
) -> tuple[str | None, str | None, list[str]]:
    """Extract target expression and expected value using LLM reasoning model."""
    try:
        prompt = EXTRACT_ORACLE_TARGET_PROMPT.format(
            method_body=method_body,
            failing_line=failing_line_in_body or "",
        )
        res = _call_llm_structured(prompt, OracleAssertionTarget, model_name=MODEL_PRIMARY)
        expr = res.expression if res.expression else None
        expected = res.expected if res.expected else None
        preceding = res.preceding if res.preceding else []
        return expr, expected, preceding
    except Exception as e:  # pragma: no cover
        print(f"Warning: Failed to extract oracle target via LLM: {e}")
        return None, None, []


def _run_oracle_verification_on_failures(test_output: str, tests_code: str, state: TDDState | None = None) -> str:
    """
    Parse test failures, extract expressions from the failed test cases,
    and verify them using the registered dynamic oracle verifier to detect incorrect test assertions.
    """
    verifier = config.ORACLE_VERIFIER
    if verifier is None:
        return ""

    failed_methods = _find_failed_methods(test_output, state)
    if not failed_methods:
        return ""

    feedback_lines = []

    for method in failed_methods:
        method_body = _extract_method_body(method, tests_code)
        if not method_body:
            continue

        failing_line_in_body = _extract_failing_line(method, test_output, state)
        expr, expected_val, preceding_exprs = _extract_oracle_target_llm(method_body, failing_line_in_body)

        if expr and expected_val:
            # Skip if the expected value is non-numeric (contains non-hex letters or tokens)
            # Allow common test dummy expectation values
            clean_expected = expected_val.strip().replace("\\n", "\n").strip()
            if clean_expected not in ("correct_val", "different_val", "unresolved_var"):
                if re.search(r"[g-zG-Z_]", clean_expected):
                    continue

            # Clean preceding_exprs to remove the current expr if it was duplicated due to traceback mismatch
            if preceding_exprs and preceding_exprs[-1] == expr:
                preceding_exprs_cleaned = preceding_exprs[:-1]  # pragma: no cover
            else:
                preceding_exprs_cleaned = preceding_exprs

            # Filter out simple numbers (integers or floats) as they are not formulas to evaluate,
            # but only if there are no preceding setup statements (like setting ibase or scale).
            if not preceding_exprs_cleaned:
                expr_cleaned = expr.strip().rstrip(".")
                try:
                    float(expr_cleaned)
                    continue
                except ValueError:
                    pass
            # Skip if the expression is not a valid mathematical formula
            # (i.e. contains letters other than the registers 'scale', 'ibase',
            # 'obase', 'last' or math library functions)
            cleaned_temp = re.sub(r"\\[a-zA-Z]", "", expr)
            cleaned_temp = re.sub(r"\b(scale|ibase|obase|last)\b", "", cleaned_temp, flags=re.IGNORECASE)
            cleaned_temp = re.sub(r"\b[sclaej]\s*\(", "(", cleaned_temp, flags=re.IGNORECASE)
            # Allow single lowercase letters as they represent bc variables (a-z)
            cleaned_temp = re.sub(r"\b[a-z]\b", "", cleaned_temp)
            if re.search(r"[a-zA-Z]", cleaned_temp):
                continue
            try:
                # Combine preceding expressions and current assertion expression
                if preceding_exprs_cleaned:
                    combined_expr = "; ".join(preceding_exprs_cleaned) + "; " + expr
                else:
                    combined_expr = expr

                try:
                    oracle_result = verifier(combined_expr, expected=expected_val).strip()
                except TypeError:
                    oracle_result = verifier(combined_expr).strip()
                expected_lines = [l.strip() for l in expected_val.splitlines() if l.strip()]
                num_expected_lines = len(expected_lines) if expected_lines else 1

                oracle_lines = [l.strip() for l in oracle_result.splitlines() if l.strip()]
                last_oracle_lines = oracle_lines[-num_expected_lines:] if oracle_lines else []
                last_oracle_val = "\n".join(last_oracle_lines) if last_oracle_lines else ""

                clean_oracle = last_oracle_val.strip().replace("\\n", "\n").strip()
                clean_expected = expected_val.strip().replace("\\n", "\n").strip()

                if "Error" not in oracle_result and "Exception" not in oracle_result:
                    if clean_oracle != clean_expected:
                        feedback_lines.append(
                            f"- Test case `{method}` contains an assertion error:\n"
                            f"  * Expression evaluated: `{combined_expr}`\n"
                            f"  * Expected value hardcoded in test: `{clean_expected}`\n"
                            f"  * Actual correct oracle value: `{clean_oracle}`\n"
                            f"  * Rationale: The test expectation `{clean_expected}` is mathematically INCORRECT. "
                            f"The correct output should be `{clean_oracle}`. Therefore, the bug is in the test "
                            f"code itself, not the implementation logic."
                        )
            except Exception:
                pass

    if feedback_lines:
        feedback_text = (
            "\n### ORACLE VERIFICATION FEEDBACK\n"
            "The verification oracle has analyzed the failed test cases and found the "
            "following discrepancies in test expectations:\n"
            + "\n".join(feedback_lines)
            + "\n\nCRITICAL DIRECTIVE: If the oracle shows a discrepancy, the test code assertion is wrong. "
            'You MUST specify `target_to_fix` as "generate_tests" to correct the test code.\n'
        )
        return feedback_text

    return "No assertion discrepancies detected by the oracle."


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

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=MODEL_PRIMARY)

    if "ORACLE VERIFICATION FEEDBACK" in oracle_feedback:
        print("[TDD Robo] 🔮 Oracle discrepancy detected. Overriding target_to_fix to 'generate_tests'.")
        bug_report_obj.target_to_fix = "generate_tests"

    report_md = "### Failed Unit Test Cases\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Unit Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")
    return {"bug_report": report_md, "next_action": bug_report_obj.target_to_fix}


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

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=MODEL_PRIMARY)

    if "ORACLE VERIFICATION FEEDBACK" in oracle_feedback:
        print("[TDD Robo] 🔮 Oracle discrepancy detected. Overriding target_to_fix to 'generate_tests'.")
        bug_report_obj.target_to_fix = "generate_tests"

    report_md = "### Failed Integration Test Cases\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Integration Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")
    return {"bug_report": report_md, "next_action": bug_report_obj.target_to_fix}


def _get_regression_test_code_context() -> str:
    test_files = glob.glob(os.path.join(config.ARTIFACTS_DIR, "test_*.py"))
    test_code_context = ""
    for tf in sorted(test_files):
        if "history" in tf or "__pycache__" in tf:
            continue
        try:
            with open(tf, "r", encoding="utf-8") as f:
                content = f.read()
            basename = os.path.basename(tf)
            test_code_context += f"### File: {basename}\n```python\n{content}\n```\n\n"
        except Exception:
            pass
    return test_code_context or "No active regression tests found."


def _get_filtered_regression_test_code_context(target_req_num: int) -> str:
    test_files = glob.glob(os.path.join(config.ARTIFACTS_DIR, "test_*.py"))
    test_code_context = ""
    target_pattern = f"req{target_req_num:03d}"
    for tf in sorted(test_files):
        if "history" in tf or "__pycache__" in tf:
            continue
        if target_pattern not in os.path.basename(tf).lower():
            continue
        try:
            with open(tf, "r", encoding="utf-8") as f:
                content = f.read()
            basename = os.path.basename(tf)
            test_code_context += f"### File: {basename}\n```python\n{content}\n```\n\n"
        except Exception:
            pass
    return test_code_context or "No active regression tests found for target requirement."


def _get_filtered_test_output(test_output: str, target_req_num: int) -> str:
    if not test_output:
        return ""
    target_str = f"req{target_req_num:03d}"
    filtered_lines = []

    # Split summary info first to avoid matching target_str in summary within section body
    main_output = test_output
    summary_text = ""
    summary_match = re.search(r"(=== short test summary info ===.*)", test_output, flags=re.DOTALL)
    if summary_match:
        summary_text = summary_match.group(1)
        main_output = test_output[: summary_match.start()]

    # Split the main output by "____" traceback headers
    sections = re.split(r"(^_{3,}.*?_{3,}\n)", main_output, flags=re.MULTILINE)
    if sections:
        filtered_lines.append(sections[0].strip())

    for i in range(1, len(sections), 2):
        header = sections[i]
        body = sections[i + 1] if i + 1 < len(sections) else ""
        if target_str in header.lower() or target_str in body.lower():
            filtered_lines.append(header + body)

    if summary_text:
        summary_lines = []
        for line in summary_text.splitlines():
            if line.startswith("===") or target_str in line.lower():
                summary_lines.append(line)
        filtered_lines.append("\n".join(summary_lines))

    return "\n".join(filtered_lines)


def _backup_project_before_rollback(state: TDDState, current_idx: int, target_idx: int):
    """
    Backs up the current project files (py_bc.py, all tests, test_output, bug_report)
    to a designated rollback_backups directory in history before files are modified/deleted.
    """
    import datetime
    import glob
    import json
    import shutil

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir_name = f"rollback_from_req{current_idx + 1}_to_req{target_idx + 1}_{timestamp}"
    backup_path = os.path.join(config.ARTIFACTS_DIR, "history", "rollback_backups", backup_dir_name)

    try:
        os.makedirs(backup_path, exist_ok=True)
        print(f"[TDD Robo] 📦 Creating rollback snapshot backup in: {backup_path}")

        # 1. Copy implementation code (if exists)
        impl_path = os.path.join(config.ARTIFACTS_DIR, "py_bc.py")
        if os.path.exists(impl_path):
            shutil.copy2(impl_path, os.path.join(backup_path, "py_bc.py"))

        # 2. Copy all test files in artifacts
        test_pattern = os.path.join(config.ARTIFACTS_DIR, "test_*.py")
        for tf in glob.glob(test_pattern):
            shutil.copy2(tf, os.path.join(backup_path, os.path.basename(tf)))

        # 3. Write test output
        test_output = state.get("test_output", "")
        if test_output:
            with open(os.path.join(backup_path, "test_output.log"), "w", encoding="utf-8") as f:
                f.write(test_output)

        # 4. Write state summary / bug report
        summary_data = {
            "timestamp": timestamp,
            "current_req_index": current_idx,
            "target_req_index": target_idx,
            "bug_report": state.get("bug_report", ""),
            "failed_files": state.get("failed_files", []),
        }
        with open(os.path.join(backup_path, "state_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print("[TDD Robo] ✅ Rollback snapshot backup completed successfully.")
    except Exception as e:
        print(f"[TDD Robo] ⚠️ Failed to create rollback snapshot backup: {e}")


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

    # --- Identify rollback target and determine active requirement ---
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

    # Filter regression context to focus LLM purely on the target requirement
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

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=MODEL_PRIMARY)

    # Parse target_req from LLM to determine pinpoint rollback target
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
        # If it is a historical regression (failed test was passed in the past)
        # and LLM Judge determined it is actually an implementation bug (implement_logic):
        # We respect LLM Judge's decision and DO NOT override it to generate_tests.
        # This prevents unnecessary/destructive rollbacks on implementation regressions.
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
            if count >= 2:
                print(
                    f"[TDD Robo] 🚨 Circuit Breaker: Max rollback attempts (2) reached for "
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
                        f"(Policy is 'rollback', target_req is LLM-specified: {llm_target_req}). Attempt {count + 1}/2."
                    )
                else:
                    print(
                        f"[TDD Robo] 🔄 Rolling back current_req_index from {idx} to {rollback_target_idx} "
                        f"(Policy is 'rollback', fallback to oldest failing req). Attempt {count + 1}/2."
                    )

                # Clean up future test files from artifacts directory
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

                # --- Restore implementation code for the target rollback requirement ---
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

                # --- Restore design document for the target rollback requirement ---
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

    return {"bug_report": report_md, "next_action": bug_report_obj.target_to_fix}


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

    bug_report_obj = _call_llm_structured(prompt, BugReport, model_name=MODEL_PRIMARY)

    report_md = "### Failed Test Cases after Refactoring\n"
    for t in bug_report_obj.failed_test_cases:
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_report_obj.expected_vs_actual}\n\n"
    report_md += f"### Fix Instructions\n{bug_report_obj.fix_instructions}\n"

    print(f"[TDD Robo] ✅ Refactor Bug report generated. Target to fix: {bug_report_obj.target_to_fix}")
    return {"bug_report": report_md, "next_action": bug_report_obj.target_to_fix}


def decide_refactor(state: TDDState):
    """Analyze implementation code and decide if refactoring is needed."""
    _update_req_progress(state)
    print("[TDD Robo] ❓ Deciding if refactoring is needed...")

    prompt = DECIDE_REFACTOR_PROMPT.format(design_doc=state.get("design_doc", ""), impl_code=state.get("impl_code", ""))

    decision_obj = _call_llm_structured(prompt, RefactorDecision, model_name=MODEL_PRIMARY)
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

    if bug_report:
        reasons_str += (
            "\n\n# WARNING: PREVIOUS REFACTORING ATTEMPT BROKE EXISTING TESTS!\n"
            "Your previous refactored code failed the tests. Here is the bug report detailing the failures:\n"
            f"<refactor_bug_report>\n{bug_report}\n</refactor_bug_report>\n"
            "Please fix these issues and ensure the refactored code passes all tests."
        )

    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)

    existing_impl = state.get("impl_code", "")
    if not existing_impl and os.path.exists(impl_path):
        try:
            with open(impl_path, "r", encoding="utf-8") as f:
                existing_impl = f.read()
        except Exception:
            pass

    is_bug_fix = bool(bug_report and existing_impl)

    if is_bug_fix:
        prompt = REFACTOR_LOGIC_FIX_PROMPT.format(
            design_doc=state.get("design_doc", ""),
            existing_impl_code=existing_impl,
            bug_report=bug_report,
            python_tips=python_tips,
        )
    else:
        prompt = REFACTOR_LOGIC_PROMPT.format(
            design_doc=state.get("design_doc", ""),
            impl_code=existing_impl,
            refactoring_reasons=reasons_str,
            python_tips=python_tips,
        )

    response = _call_llm_text(prompt, model_name=MODEL_PRIMARY)

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
                reasons_with_failure = reasons_str + (
                    f"\n\n# WARNING: PREVIOUS SEARCH/REPLACE APPLICATION FAILED!\n"
                    f"The Search/Replace block application failed with error: {e}.\n"
                    "Please output the COMPLETE refactored implementation code "
                    "inside a single python block."
                )
                fallback_prompt = REFACTOR_LOGIC_PROMPT.format(
                    design_doc=state.get("design_doc", ""),
                    impl_code=existing_impl,
                    refactoring_reasons=reasons_with_failure,
                    python_tips=python_tips,
                )
                response = _call_llm_text(fallback_prompt, model_name=MODEL_PRIMARY)
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


def generate_readme(state: TDDState):
    """Generate a README.md file based on the goal and implementation code."""
    _update_req_progress(state)
    print("[TDD Robo] 📝 Generating README.md...")

    template = get_prompt("generate_readme_prompt", GENERATE_README_PROMPT)
    prompt = template.format(
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
    }
    if state.get("impl_code"):
        ret_val["last_green_impl_code"] = state.get("impl_code", "")
    return ret_val


def _detect_toggle_loop(state: TDDState) -> bool:
    """
    Detect if the implementation is stuck in a toggle loop (e.g. A-B-A-B)
    or a repeating state loop across recent iterations.
    """
    # Force design revision if iterations reach a threshold or if stagnant iterations reach threshold
    current_iterations = state.get("iterations", 0)
    loop_threshold = getattr(config, "LOOP_DETECTION_THRESHOLD", 8)
    if current_iterations >= loop_threshold:
        print(
            f"[TDD Robo] 🚨 Loop Detector: Iterations reached threshold ({current_iterations} >= {loop_threshold}). "
            "Forcing rollback to design."
        )
        state["loop_detected"] = True
        return True

    stagnant_iters = state.get("stagnant_iterations", 0)
    max_stagnant_iters = getattr(config, "MAX_STAGNANT_ITERATIONS", 3)
    if stagnant_iters >= max_stagnant_iters:
        print(
            f"[TDD Robo] 🚨 Loop Detector: Test results have not improved for {stagnant_iters} iterations. "
            "Forcing rollback to design."
        )
        state["loop_detected"] = True
        return True

    artifacts_dir = getattr(config, "ARTIFACTS_DIR", "artifacts")
    history_dir = os.path.join(artifacts_dir, "history")
    if not os.path.exists(history_dir):
        return False

    module_name = state.get("module_name", "impl.py")
    base_name = os.path.splitext(module_name)[0]

    test_module_name = state.get("test_module_name", getattr(config, "DEFAULT_TEST_NAME", "test_impl.py"))
    test_base_name = os.path.splitext(test_module_name)[0]
    if test_base_name.endswith("_unit"):
        test_pattern_base = test_base_name[:-5]
        test_pattern = os.path.join(history_dir, f"{test_pattern_base}*_unit_iter*.py")
    elif test_base_name.endswith("_integration"):
        test_pattern_base = test_base_name[:-12]
        test_pattern = os.path.join(history_dir, f"{test_pattern_base}*_integration_iter*.py")
    else:
        test_pattern = os.path.join(history_dir, f"{test_base_name}*_iter*.py")
    test_snapshots = glob.glob(test_pattern)
    latest_test_mtime = 0.0
    if test_snapshots:
        latest_test_mtime = max(os.path.getmtime(f) for f in test_snapshots)

    req_id = None
    if "requirements" in state and "current_req_index" in state:
        reqs = state["requirements"]
        idx = state["current_req_index"]
        if 0 <= idx < len(reqs):
            req_id = reqs[idx].get("id")
    if req_id is None and "current_req_index" in state:
        req_id = f"req{state['current_req_index'] + 1:03d}"

    test_iterations = state.get("test_iterations", 1)

    if req_id:
        pattern = os.path.join(
            history_dir, f"{base_name}_{req_id.lower()}*_test_iter{test_iterations:03d}*_impl_iter*.py"
        )
    else:
        pattern = os.path.join(history_dir, f"{base_name}_iter*.py")

    snapshot_files = sorted(glob.glob(pattern), key=os.path.getmtime)

    if latest_test_mtime > 0.0:
        snapshot_files = [f for f in snapshot_files if os.path.getmtime(f) > latest_test_mtime]

    if len(snapshot_files) < 4:
        return False

    hashes = []
    for fpath in snapshot_files[-6:]:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                content = re.sub(r"#.*$", "", content, flags=re.MULTILINE)
                content = re.sub(r'"""[\s\S]*?"""', "", content)
                content = re.sub(r"'''[\s\S]*?'''", "", content)
                normalized = re.sub(r"\s+", "", content)
                hashes.append(hashlib.md5(normalized.encode("utf-8")).hexdigest())
        except Exception:
            continue

    if len(hashes) < 4:
        return False

    if hashes[-1] == hashes[-3] and hashes[-2] == hashes[-4]:
        print("[TDD Robo] 🚨 Loop Detector: Detected toggle loop (A-B-A-B) in normalized implementation snapshots!")
        state["loop_detected"] = True
        return True

    if len(hashes) >= 5 and hashes[-1] == hashes[-3] and hashes[-1] == hashes[-5]:
        print(
            "[TDD Robo] 🚨 Loop Detector: Detected repeating state (A-X-A-Y-A) in normalized implementation snapshots!"
        )
        state["loop_detected"] = True
        return True

    return False


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
        max_syntax = getattr(config, "MAX_SYNTAX_ERROR_ITERATIONS", 3)
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
        max_syntax = getattr(config, "MAX_SYNTAX_ERROR_ITERATIONS", 3)
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
        max_syntax = getattr(config, "MAX_SYNTAX_ERROR_ITERATIONS", 3)
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
        max_syntax = getattr(config, "MAX_SYNTAX_ERROR_ITERATIONS", 3)
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
        max_syntax = getattr(config, "MAX_SYNTAX_ERROR_ITERATIONS", 3)
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
        max_syntax = getattr(config, "MAX_SYNTAX_ERROR_ITERATIONS", 3)
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
    next_act = state.get("next_action", "implement_initial_logic")
    if next_act == "implement_logic":
        next_act = "implement_initial_logic"
    if next_act == "implement_initial_logic":
        if _detect_toggle_loop(state):
            print(
                "[TDD Robo] 🔄 Loop Detector Override: Stuck in implementation loop. "
                "Rolling back to update_design_for_req."
            )
            return "update_design_for_req"
    if next_act == "generate_tests":
        return "generate_unit_tests"
    return next_act


def should_fix_integration_tests_or_impl(state: TDDState):
    """Determine next action for integration failures, with toggle loop detection."""
    next_act = state.get("next_action", "implement_integration_logic")
    if next_act == "implement_logic":
        next_act = "implement_integration_logic"
    if next_act == "implement_integration_logic":
        if _detect_toggle_loop(state):
            print(
                "[TDD Robo] 🔄 Loop Detector Override: Stuck in implementation loop. "
                "Rolling back to update_design_for_req."
            )
            return "update_design_for_req"
    if next_act == "generate_tests":
        return "generate_integration_tests"
    return next_act


def should_fix_regression_tests_or_impl(state: TDDState):
    """Determine next action for regression failures, with toggle loop detection."""
    next_act = state.get("next_action", "implement_regression_logic")
    if next_act == "implement_logic":
        next_act = "implement_regression_logic"
    if next_act == "implement_regression_logic":
        if _detect_toggle_loop(state):
            print(
                "[TDD Robo] 🔄 Loop Detector Override: Stuck in implementation loop. "
                "Rolling back to update_design_for_req."
            )
            return "update_design_for_req"
    if next_act == "generate_design":
        return "update_design_for_req"
    if next_act == "generate_tests":
        # Check test_output to determine whether unit tests or integration tests failed
        test_output = state.get("test_output", "")
        reqs = state.get("requirements", [])
        idx = state.get("current_req_index", 0)
        active_req_id = ""
        if reqs and idx < len(reqs):
            active_req_id = str(reqs[idx].get("id") or "").lower()

        # Circuit Breaker: Check if any failing test belongs to a previous requirement
        import re

        failed_files = cast(list[str], state.get("failed_files", []))
        failed_req_nums = []
        for f in failed_files:
            match = re.search(r"test_[a-zA-Z0-9_]+_req(\d+)_(?:unit|integration)\.py", f)
            if match:
                failed_req_nums.append(int(match.group(1)))

        # Fallback to scanning FAILED/ERROR lines of test_output if failed_files is empty
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


# --- Graph Construction ---
class TDDAgent:
    """
    An agent that builds and executes a LangGraph workflow to perform Test-Driven Development.
    """

    def __init__(self, checkpointer=None):
        self.checkpointer = checkpointer
        self.app = self._build_graph()

    def invoke(self, state, config=None):
        return self.app.invoke(state, config=config)

    def get_graph(self):
        return self.app.get_graph()

    def _build_graph(self):
        workflow = StateGraph(TDDState)

        workflow.add_node("fetch_spec", fetch_spec)
        workflow.add_node("generate_requirements", generate_requirements)
        workflow.add_node("plan_files", plan_files)
        workflow.add_node("generate_design_initial", generate_design_initial)
        workflow.add_node("update_design_for_req", update_design_for_req)
        workflow.add_node("review_design_initial", review_design_initial)
        workflow.add_node("review_design_incremental", review_design_incremental)

        # Unit test nodes
        workflow.add_node("plan_unit_tests", plan_unit_tests)
        workflow.add_node("review_unit_test_plan", review_unit_test_plan)
        workflow.add_node("generate_unit_tests", generate_unit_tests)
        workflow.add_node("check_unit_tests_syntax", check_unit_tests_syntax)
        workflow.add_node("implement_initial_logic", implement_initial_logic)
        workflow.add_node("check_initial_impl_syntax", check_initial_impl_syntax)
        workflow.add_node("run_unit_tests", run_unit_tests)
        workflow.add_node("generate_unit_bug_report", generate_unit_bug_report)

        # Integration test nodes
        workflow.add_node("plan_integration_tests", plan_integration_tests)
        workflow.add_node("review_integration_test_plan", review_integration_test_plan)
        workflow.add_node("generate_integration_tests", generate_integration_tests)
        workflow.add_node("check_integration_tests_syntax", check_integration_tests_syntax)
        workflow.add_node("implement_integration_logic", implement_integration_logic)
        workflow.add_node("check_integration_impl_syntax", check_integration_impl_syntax)
        workflow.add_node("run_integration_tests", run_integration_tests)
        workflow.add_node("generate_integration_bug_report", generate_integration_bug_report)

        # Regression nodes
        workflow.add_node("run_regression_tests", run_regression_tests)
        workflow.add_node("generate_regression_bug_report", generate_regression_bug_report)
        workflow.add_node("implement_regression_logic", implement_regression_logic)
        workflow.add_node("check_regression_impl_syntax", check_regression_impl_syntax)

        # Refactoring nodes
        workflow.add_node("decide_refactor", decide_refactor)
        workflow.add_node("refactor_logic", refactor_logic)
        workflow.add_node("check_refactored_impl_syntax", check_refactored_impl_syntax)
        workflow.add_node("generate_refactor_bug_report", generate_refactor_bug_report)

        # Finalization nodes
        workflow.add_node("increment_requirement", increment_requirement)
        workflow.add_node("generate_readme", generate_readme)

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

        workflow.add_edge("plan_unit_tests", "review_unit_test_plan")
        workflow.add_conditional_edges(
            "review_unit_test_plan",
            should_review_test_plan_or_continue,
            {
                "plan_tests": "plan_unit_tests",
                "generate_tests": "generate_unit_tests",
                "update_design_for_req": "update_design_for_req",
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
                "update_design_for_req": "update_design_for_req",
            },
        )

        # Integration Phase
        workflow.add_edge("plan_integration_tests", "review_integration_test_plan")
        workflow.add_conditional_edges(
            "review_integration_test_plan",
            should_review_test_plan_or_continue,
            {
                "plan_tests": "plan_integration_tests",
                "generate_tests": "generate_integration_tests",
                "update_design_for_req": "update_design_for_req",
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
                "update_design_for_req": "update_design_for_req",
            },
        )

        # Regression Phase
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
                "update_design_for_req": "update_design_for_req",
                "generate_integration_tests": "generate_integration_tests",
                "generate_unit_tests": "generate_unit_tests",
                "halt_regression_test_failure": END,
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

        # Refactoring Phase
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
            {"refactor_logic": "refactor_logic", "increment_requirement": "increment_requirement"},
        )

        # Advancement
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
