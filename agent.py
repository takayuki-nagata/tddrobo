import json
import os
import subprocess
import sys
import time

import markdownify
import requests
from langgraph.graph import END, StateGraph

import config
from prompts import (
    EXTRACT_CALCULATIONS_PROMPT,
    GENERATE_BUG_REPORT_PROMPT,
    GENERATE_DESIGN_PROMPT,
    GENERATE_README_PROMPT,
    GENERATE_REQUIREMENTS_PROMPT,
    GENERATE_TEST_PLAN_PROMPT,
    GENERATE_TEST_PLAN_PROMPT_FIX,
    GENERATE_TESTS_PROMPT_BUG_FIX,
    GENERATE_TESTS_PROMPT_FIX,
    GENERATE_TESTS_PROMPT_INITIAL,
    GENERATE_TESTS_PROMPT_REVIEW_FIX,
    IMPLEMENT_LOGIC_PROMPT_FIX,
    IMPLEMENT_LOGIC_PROMPT_INITIAL,
    IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX,
    PLAN_FILES_PROMPT,
    REVIEW_TEST_PLAN_PROMPT,
    REVIEW_TESTS_PROMPT,
)
from schema import (
    BugReport,
    CalculationTestPlan,
    DesignDocument,
    FilePlan,
    RequirementsList,
    TDDState,
    TestPlan,
    TestPlanReviewReport,
    TestReviewReport,
)
from utils import (
    add_line_numbers,
    apply_search_replace_blocks,
    call_llm_standard,
    call_llm_with_reasoning,
    evaluate_math_expression,
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


_CURRENT_REQ_NUM = 0
_TOTAL_REQ_NUM = 0


def _update_req_progress(state: TDDState):
    """Update global requirement progress indicators from state."""
    global _CURRENT_REQ_NUM, _TOTAL_REQ_NUM
    reqs = state.get("requirements", [])
    if reqs:
        _TOTAL_REQ_NUM = len(reqs)
        _CURRENT_REQ_NUM = state.get("current_req_index", 0) + 1


def print(*args, **kwargs):
    """Override builtin print to add timestamps and requirement progress to [TDD Robo] messages."""
    import builtins
    import time

    if args and isinstance(args[0], str) and args[0].startswith("[TDD Robo]"):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        progress_suffix = ""
        if _CURRENT_REQ_NUM > 0 and _TOTAL_REQ_NUM > 0:
            progress_suffix = f" ({_CURRENT_REQ_NUM}/{_TOTAL_REQ_NUM})"
        new_arg0 = args[0].replace("[TDD Robo]", f"[{timestamp}] [TDD Robo]{progress_suffix}", 1)
        args = (new_arg0,) + args[1:]
    builtins.print(*args, **kwargs)


def save_history_snapshot(filename: str, code: str, iteration: int):
    """
    Saves a snapshot of the generated file to the artifacts/history/ directory.

    Args:
        filename (str): The original filename (e.g. 'impl.py').
        code (str): The code content to write.
        iteration (int): The current iteration number.
    """
    name_parts = os.path.splitext(filename)
    history_filename = f"{name_parts[0]}_iter{iteration:03d}{name_parts[1]}"
    history_dir = os.path.join(config.ARTIFACTS_DIR, "history")
    os.makedirs(history_dir, exist_ok=True)
    history_path = os.path.join(history_dir, history_filename)
    try:
        with open(history_path, "w", encoding="utf-8") as f:
            f.write(code)
        if config.VERBOSE:
            print(f"[TDD Robo] 📦 Saved history snapshot to {history_path}")
    except Exception as e:
        print(f"Warning: Could not save history snapshot to {history_path}: {e}")


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

    global _CURRENT_REQ_NUM, _TOTAL_REQ_NUM
    _TOTAL_REQ_NUM = len(requirements)
    _CURRENT_REQ_NUM = 1

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


def generate_design(state: TDDState):
    """
    Generate a system design document based on the goal and specification.

    Reads:
        - goal
        - spec_content
        - module_name
        - test_module_name
    Writes:
        - design_doc
    """
    _update_req_progress(state)
    print("[TDD Robo] 📐 Generating design document...")
    template = get_prompt("generate_design_prompt", GENERATE_DESIGN_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
        test_name=state.get("test_module_name", DEFAULT_TEST_NAME),
    )

    response = call_llm_with_reasoning(prompt, response_schema=DesignDocument, thinking_level="HIGH")

    design_data = json.loads(extract_json(response))
    design = "# Software Design Document\n\n"
    for idx, (key, value) in enumerate(design_data.items(), 1):
        title = key.replace("_", " ").title()
        design += f"## {idx}. {title}\n{value}\n\n"

    design_path = save_artifact("design.md", design)
    print(f"[TDD Robo] ✅ Saved design document to {design_path}")
    return {"design_doc": design}


def plan_tests(state: TDDState):
    """
    Generate a comprehensive test plan based on the specification and design.

    Reads:
        - goal
        - spec_content
        - design_doc
        - module_name
        - test_plan
        - test_plan_review
        - test_plan_iterations
        - requirements
        - current_req_index
        - requirements_list_str
    Writes:
        - test_plan
        - test_plan_iterations
    """
    _update_req_progress(state)
    iters = state.get("test_plan_iterations", 0) + 1
    requirements = state.get("requirements", [])
    current_index = state.get("current_req_index", 0)

    if current_index < len(requirements):
        target_req = requirements[current_index]
        target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"
    else:
        target_req_str = "No active target requirement (all completed)."

    print(f"[TDD Robo] 📋 Generating test plan for requirement {target_req_str} (Iteration {iters})...")

    if state.get("test_plan_review") and state.get("test_plan"):
        template = get_prompt("generate_test_plan_prompt_fix", GENERATE_TEST_PLAN_PROMPT_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design=state.get("design_doc", ""),
            impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
            test_plan=state.get("test_plan", ""),
            test_plan_review=state.get("test_plan_review", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
        )
    else:
        template = get_prompt("generate_test_plan_prompt", GENERATE_TEST_PLAN_PROMPT)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design=state.get("design_doc", ""),
            impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
        )

    response = call_llm_with_reasoning(prompt, response_schema=TestPlan, thinking_level="HIGH")

    plan_data = json.loads(extract_json(response))
    plan_md = "# Test Plan\n\n"
    seen = set()
    unique_test_cases = []
    for tc in plan_data.get("test_cases", []):
        tc_str = f"Action: {tc.get('action', '')} | Expected: {tc.get('expected_outcome', '')}"
        if tc_str not in seen:
            seen.add(tc_str)
            unique_test_cases.append(tc_str)
    for idx, tc in enumerate(unique_test_cases, 1):
        plan_md += f"{idx}. {tc}\n"

    plan_path = save_artifact("test_plan.md", plan_md)
    print(f"[TDD Robo] ✅ Saved test plan to {plan_path}")
    return {"test_plan": plan_md, "test_plan_iterations": iters}


def review_test_plan(state: TDDState):
    """
    Review the generated test plan for completeness and missing edge cases.

    Reads:
        - goal
        - spec_content
        - design_doc
        - test_plan
        - target_test_plan_coverage
        - test_plan_iterations
        - max_test_plan_iterations
        - requirements
        - current_req_index
        - requirements_list_str
    Writes:
        - test_plan_review
        - test_plan_review_decision
    """
    _update_req_progress(state)
    requirements = state.get("requirements", [])
    current_index = state.get("current_req_index", 0)

    if current_index < len(requirements):
        target_req = requirements[current_index]
        target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"
    else:
        target_req_str = "No active target requirement (all completed)."

    print(f"[TDD Robo] 🧐 Reviewing test plan for requirement {target_req_str}...")
    template = get_prompt("review_test_plan_prompt", REVIEW_TEST_PLAN_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        design=state.get("design_doc", ""),
        test_plan=state.get("test_plan", ""),
        target_coverage=state.get("target_test_plan_coverage", TARGET_TEST_PLAN_COVERAGE),
        requirements_list_str=state.get("requirements_list_str", ""),
        target_requirement=target_req_str,
    )

    response = call_llm_with_reasoning(prompt, response_schema=TestPlanReviewReport, thinking_level="HIGH")

    review_data = json.loads(extract_json(response))
    coverage = review_data.get("estimated_coverage", 100)
    report_md = f"### Estimated Coverage: {coverage}%\n\n### Missing Test Cases:\n"
    seen = set()
    for m in review_data.get("missing_test_cases", []):
        if m not in seen:
            seen.add(m)
            report_md += f"- {m}\n"
    report_md += f"\n### Feedback:\n{review_data.get('feedback', '')}"

    print(f"[TDD Robo] ✅ Test plan review completed. Estimated coverage: {coverage}%")
    if config.VERBOSE:
        print("-----------------------")
        print(report_md)
        print("-----------------------")

    target_coverage = state.get("target_test_plan_coverage", TARGET_TEST_PLAN_COVERAGE)
    if coverage < target_coverage:
        iters = state.get("test_plan_iterations", 0)
        max_iters = state.get("max_test_plan_iterations", MAX_TEST_PLAN_ITERATIONS)
        if iters >= max_iters:
            print(
                f"[TDD Robo] ⚠️ Test plan coverage is insufficient ({coverage}% < {target_coverage}%), "
                f"but max iterations ({iters}/{max_iters}) reached. Proceeding to test generation."
            )
            return {"test_plan_review": "", "test_plan_review_decision": "generate_tests"}
        else:
            print(
                f"[TDD Robo] ⚠️ Test plan coverage is insufficient ({coverage}% < {target_coverage}%). "
                f"Regenerating test plan..."
            )
            return {"test_plan_review": report_md, "test_plan_review_decision": "plan_tests"}

    print(f"[TDD Robo] ✅ Test plan coverage is sufficient ({coverage}%). Proceeding to test generation.")
    return {"test_plan_review": "", "test_plan_review_decision": "generate_tests"}


def generate_tests(state: TDDState):
    """
    Generate a pytest suite following the test plan, specification, and design.

    Reads:
        - goal
        - spec_content
        - design_doc
        - test_plan
        - module_name
        - test_module_name
        - tests_check_output
        - tests_code
        - bug_report
        - test_review
        - test_iterations
        - requirements
        - current_req_index
        - requirements_list_str
    Writes:
        - tests_code
        - test_iterations
        - iterations
        - success
        - bug_report
        - test_review
    """
    _update_req_progress(state)
    iters = state.get("test_iterations", 0) + 1
    print(f"[TDD Robo] 🧪 Generating test code (Iteration {iters})...")
    spec = state.get("spec_content", "")
    design = state.get("design_doc", "")
    test_plan = state.get("test_plan", "")
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)

    # Step 1: Verify test plan mathematical outcomes using Python with subprocess bc
    print("[TDD Robo] 🧮 Verifying math outcomes in the Test Plan using bc...")
    extract_template = get_prompt("extract_calculations_prompt", EXTRACT_CALCULATIONS_PROMPT)
    extract_prompt = extract_template.format(
        goal=state.get("goal", ""),
        test_plan=test_plan,
    )
    # Extract calculations using standard model (tools=None, response_schema=CalculationTestPlan)
    # This is extremely fast (under 5 seconds) and never hangs because it doesn't use tools in LLM
    extract_response = call_llm_standard(extract_prompt, response_schema=CalculationTestPlan)

    verified_plan = test_plan
    try:
        extract_data = json.loads(extract_json(extract_response))
        items = extract_data.get("items", [])

        if config.VERBOSE:
            print(f"Found {len(items)} mathematical test cases to verify.")
        plan_lines = test_plan.splitlines()

        for item in items:
            num = item.get("test_case_number")
            expr = item.get("expression", "").strip()
            if num is not None and expr:
                try:
                    num_int = int(num)
                except ValueError:
                    continue

                # Compute using our mathematical verification oracle tool
                bc_result = evaluate_math_expression(expr).strip()
                if "Error" not in bc_result and "Exception" not in bc_result:
                    # Find the line that matches the case number prefix (e.g. "2." or "2. ")
                    target_prefix = f"{num_int}."
                    for idx, line in enumerate(plan_lines):
                        if line.strip().startswith(target_prefix):
                            # Update expectation part in this specific line
                            if "Expected:" in line:
                                prefix = line.split("Expected:")[0]
                                plan_lines[idx] = f"{prefix}Expected: Output {bc_result}"
                                if config.VERBOSE:
                                    print(f"Verified case #{num_int}: '{expr}' -> '{bc_result}'")
                            break

        verified_plan = "\n".join(plan_lines)
        print("[TDD Robo] ✅ Corrected math expectations verified via bc tool successfully!")
    except Exception as e:
        if config.VERBOSE:
            print(f"⚠️ Failed to parse/verify calculations ({e}). Using original test plan.")
        verified_plan = test_plan

    # Step 2: Generate test code using the reasoning model (gemma-4-31b-it) without tools (safe from hangs)
    requirements = state.get("requirements", [])
    current_index = state.get("current_req_index", 0)

    if current_index < len(requirements):
        target_req = requirements[current_index]
        req_id = target_req.get("id", f"REQ{current_index + 1:03d}")
        target_req_str = f"{req_id}: {target_req.get('description')}"
    else:
        req_id = f"REQ{current_index + 1:03d}"
        target_req_str = "No active target requirement (all completed)."

    base_test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    import re

    # Clean any existing requirement suffixes to get the clean base name (e.g., test_bc_clone)
    base_name_clean = re.sub(r"_(req\d+)+", "", base_test_name.replace(".py", ""))
    # Append ONLY the current requirement ID (e.g. test_bc_clone_req001.py)
    test_name = f"{base_name_clean}_{req_id.lower()}.py"

    # If we are iterating on the tests for this requirement,
    # load the existing test code from the requirement-specific file.
    test_path = os.path.join(config.ARTIFACTS_DIR, test_name)
    if iters > 1 and os.path.exists(test_path):
        try:
            with open(test_path, "r", encoding="utf-8") as f:
                existing_tests = f.read()
        except Exception:
            existing_tests = ""
    else:
        existing_tests = ""

    if state.get("tests_check_output") and state.get("tests_code"):
        template = get_prompt("generate_tests_prompt_fix", GENERATE_TESTS_PROMPT_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=verified_plan,
            impl_name=impl_name,
            tests_check_output=state.get("tests_check_output", ""),
            tests_code=state.get("tests_code", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_test_code=existing_tests,
        )
    elif state.get("bug_report") and state.get("tests_code"):
        template = get_prompt("generate_tests_prompt_bug_fix", GENERATE_TESTS_PROMPT_BUG_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=verified_plan,
            impl_name=impl_name,
            bug_report=state.get("bug_report", ""),
            tests_code=state.get("tests_code", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_test_code=existing_tests,
        )
    elif state.get("test_review") and state.get("tests_code"):
        template = get_prompt("generate_tests_prompt_review_fix", GENERATE_TESTS_PROMPT_REVIEW_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=verified_plan,
            impl_name=impl_name,
            test_review=state.get("test_review", ""),
            tests_code=state.get("tests_code", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_test_code=existing_tests,
        )
    else:
        template = get_prompt("generate_tests_prompt_initial", GENERATE_TESTS_PROMPT_INITIAL)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=verified_plan,
            impl_name=impl_name,
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_test_code=existing_tests,
        )

    print("[TDD Robo] 🧪 Generating final test code with reasoning model...")
    # Call primary reasoning model without tools to guarantee stability and prevent any connection hangs
    response = call_llm_with_reasoning(prompt)
    code = extract_code(response)
    test_path = save_artifact(test_name, code)
    print(f"[TDD Robo] ✅ Saved test code to {test_path}")
    save_history_snapshot(test_name, code, iters)
    # Clear previous bug reports to prevent misbehavior in subsequent implementation nodes after modifying the test
    return {
        "tests_code": code,
        "test_module_name": test_name,
        "test_iterations": iters,
        "iterations": 0,
        "success": False,
        "bug_report": "",
        "test_review": "",
        "tests_check_output": "",
    }


def _run_syntax_check(filename: str, label: str) -> str:
    """
    Run a basic syntax check on a given Python file using flake8.

    Args:
        filename (str): The path to the file to check.
        label (str): A descriptive label for logging output.

    Returns:
        str: Empty string if syntax is valid, otherwise the error output.
    """
    print(f"[TDD Robo] 🔍 Checking {label} syntax...")
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
    if success:
        print(f"[TDD Robo] ✅ {label} syntax check passed!")
    else:
        print(f"[TDD Robo] ❌ {label} syntax error found!")
        if config.VERBOSE:
            print(output)
    return "" if success else output


def check_tests_syntax(state: TDDState):
    """
    Check the syntax of the generated test code.

    Reads:
        - test_module_name
        - tests_code
    Writes:
        - tests_check_output
    """
    _update_req_progress(state)
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    test_path = os.path.join(config.ARTIFACTS_DIR, test_name)

    tests_code = state.get("tests_code", "").strip()
    if not tests_code:
        print("[TDD Robo] ❌ Test code is empty!")
        return {"tests_check_output": "Error: The generated test code is empty. Please provide valid Python code."}

    output = _run_syntax_check(test_path, "test code")
    return {"tests_check_output": output}


def review_tests(state: TDDState):
    """
    Review the generated test code against the test plan to ensure coverage.

    Reads:
        - test_plan
        - tests_code
        - target_test_coverage
        - test_iterations
        - max_test_iterations
    Writes:
        - test_review
        - test_review_decision
    """
    _update_req_progress(state)
    print("[TDD Robo] 🧐 Reviewing test code...")

    # Load regression test codes from other requirements
    base_test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    import re

    base_name_clean = re.sub(r"_(req\d+)+", "", base_test_name.replace(".py", ""))

    requirements = state.get("requirements", [])
    current_index = state.get("current_req_index", 0)
    if current_index < len(requirements):
        target_req = requirements[current_index]
        req_id = target_req.get("id", f"REQ{current_index + 1:03d}")
    else:
        req_id = f"REQ{current_index + 1:03d}"
    current_test_name = f"{base_name_clean}_{req_id.lower()}.py"

    regression_tests_list = []
    artifacts_dir = config.ARTIFACTS_DIR
    if os.path.exists(artifacts_dir):
        for f_name in sorted(os.listdir(artifacts_dir)):
            if f_name.startswith(base_name_clean + "_req") and f_name.endswith(".py") and f_name != current_test_name:
                f_path = os.path.join(artifacts_dir, f_name)
                try:
                    with open(f_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                    regression_tests_list.append(f"# File: {f_name}\n{file_content}")
                except Exception as e:
                    print(f"[TDD Robo] ⚠️ Failed to read regression test file {f_name}: {e}")

    regression_tests_code = (
        "\n\n".join(regression_tests_list) if regression_tests_list else "No regression tests exist yet."
    )

    template = get_prompt("review_tests_prompt", REVIEW_TESTS_PROMPT)
    prompt = template.format(
        test_plan=state.get("test_plan", ""),
        tests_code=add_line_numbers(state.get("tests_code", "")),
        regression_tests_code=regression_tests_code,
        target_coverage=state.get("target_test_coverage", TARGET_TEST_COVERAGE),
    )

    response = call_llm_with_reasoning(prompt, response_schema=TestReviewReport, thinking_level="HIGH")

    review_data = json.loads(extract_json(response))
    coverage = review_data.get("estimated_coverage", 100)
    report_md = f"### Estimated Coverage: {coverage}%\n\n### Missing Test Cases:\n"
    seen = set()
    for m in review_data.get("missing_test_cases", []):
        if m not in seen:
            seen.add(m)
            report_md += f"- {m}\n"
    report_md += f"\n### Feedback:\n{review_data.get('feedback', '')}"

    print(f"[TDD Robo] ✅ Test review completed. Estimated coverage: {coverage}%")
    if config.VERBOSE:
        print("-----------------------")
        print(report_md)
        print("-----------------------")

    target_coverage = state.get("target_test_coverage", TARGET_TEST_COVERAGE)
    if coverage < target_coverage:
        iters = state.get("test_iterations", 0)
        max_iters = state.get("max_test_iterations", MAX_TEST_ITERATIONS)
        if iters >= max_iters:
            print(
                f"[TDD Robo] ⚠️ Test coverage is insufficient ({coverage}% < {target_coverage}%), "
                f"but max iterations ({iters}/{max_iters}) reached. Proceeding to implementation."
            )
            return {"test_review": "", "test_review_decision": "implement_logic"}
        else:
            print(
                f"[TDD Robo] ⚠️ Test coverage is insufficient ({coverage}% < {target_coverage}%). Regenerating tests..."
            )
            return {"test_review": report_md, "test_review_decision": "generate_tests"}

    print(f"[TDD Robo] ✅ Test coverage is sufficient ({coverage}%). Proceeding to implementation.")
    return {"test_review": "", "test_review_decision": "implement_logic"}


def _get_combined_tests_code(state: TDDState) -> str:
    """Helper to collect and combine all test_bc_clone_req*.py files in artifacts directory."""
    import glob

    all_tests_content = []
    if os.path.exists(config.ARTIFACTS_DIR):
        test_files = sorted(glob.glob(os.path.join(config.ARTIFACTS_DIR, "test_bc_clone_req*.py")))
        for tf in test_files:
            try:
                with open(tf, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    all_tests_content.append(f"# --- File: {os.path.basename(tf)} ---\n{file_content}")
            except Exception:
                pass
    return "\n\n".join(all_tests_content) if all_tests_content else state.get("tests_code", "")


def implement_logic(state: TDDState):
    """
    Generate the implementation code that satisfies the design and tests.

    Reads:
        - goal
        - design_doc
        - module_name
        - tests_code
        - impl_check_output
        - impl_code
        - bug_report
        - iterations
        - requirements
        - current_req_index
        - requirements_list_str
    Writes:
        - impl_code
    """
    _update_req_progress(state)
    print(f"[TDD Robo] 💻 Generating implementation code (Iteration {state.get('iterations', 0) + 1})...")
    design = state.get("design_doc", "")
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)

    impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    test_path = os.path.join(config.ARTIFACTS_DIR, test_name)
    if os.path.exists(impl_path) and os.path.exists(test_path) and not state.get("bug_report"):
        try:
            test_res = subprocess.run(
                [sys.executable, "-m", "pytest", test_name, "--maxfail=1"],
                capture_output=True,
                cwd=config.ARTIFACTS_DIR,
                timeout=5,
            )
            if test_res.returncode == 0:
                print(f"[TDD Robo] 🎉 Existing {impl_name} passes all tests! Skipping LLM generation.")
                with open(impl_path, "r", encoding="utf-8") as f:
                    code = f.read()
                save_history_snapshot(impl_name, code, state.get("iterations", 0) + 1)
                return {"impl_code": code, "impl_check_output": "", "bug_report": ""}
        except Exception:
            pass

    requirements = state.get("requirements", [])
    current_index = state.get("current_req_index", 0)

    if current_index < len(requirements):
        target_req = requirements[current_index]
        target_req_str = f"{target_req.get('id')}: {target_req.get('description')}"
    else:
        target_req_str = "No active target requirement (all completed)."

    existing_impl = state.get("impl_code", "")
    # Clean any pre-existing syntax error markers to keep context clean
    if existing_impl:
        clean_lines = [line for line in existing_impl.splitlines() if not line.startswith("# TDD_ROBO_SYNTAX_ERROR")]
        existing_impl = "\n".join(clean_lines).lstrip()

    combined_tests = _get_combined_tests_code(state)

    domain_tips = state.get("domain_tips", "")

    impl_check_output = state.get("impl_check_output", "")
    if impl_check_output:
        impl_check_output = (
            impl_check_output.replace("<<<<<<< SEARCH", "[PREVIOUS SEARCH]")
            .replace("=======", "[PREVIOUS DIVIDER]")
            .replace(">>>>>>> REPLACE", "[PREVIOUS REPLACE]")
        )
        if "Failed to apply Search/Replace block" in impl_check_output:
            impl_check_output += (
                "\n\n# CRITICAL ADVICE FOR SEARCH/REPLACE BLOCKS\n"
                "Your previous Search/Replace block failed because it did not match the existing code exactly.\n"
                "Please follow these guidelines to fix this:\n"
                "1. Read the `<existing_impl_code>` section line-by-line.\n"
                "2. Make sure your SEARCH block matches `<existing_impl_code>` character-for-character, "
                "including comments, spaces, newlines, and indentation.\n"
                "3. Do NOT add trailing periods or other punctuation at the end of your code blocks "
                "unless they are exactly present in `<existing_impl_code>`.\n"
                "4. Keep your SEARCH blocks as small and targeted as possible. Do not include large blocks "
                "of unchanged code. Try to replace only the specific lines you want to change.\n"
            )

    if impl_check_output and state.get("impl_code"):
        template = get_prompt("implement_logic_prompt_syntax_fix", IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            design=design,
            tests_code=combined_tests,
            impl_name=impl_name,
            impl_check_output=impl_check_output,
            impl_code=existing_impl,
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl,
            domain_tips=domain_tips,
        )
    elif state.get("bug_report") and state.get("impl_code"):
        template = get_prompt("implement_logic_prompt_fix", IMPLEMENT_LOGIC_PROMPT_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            design=design,
            tests_code=combined_tests,
            impl_name=impl_name,
            bug_report=state.get("bug_report", ""),
            impl_code=existing_impl,
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl,
            domain_tips=domain_tips,
        )
    else:
        template = get_prompt("implement_logic_prompt_initial", IMPLEMENT_LOGIC_PROMPT_INITIAL)
        prompt = template.format(
            goal=state.get("goal", ""),
            design=design,
            tests_code=combined_tests,
            impl_name=impl_name,
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl,
            domain_tips=domain_tips,
        )
    response = call_llm_with_reasoning(prompt, thinking_level="MINIMAL")

    if existing_impl:
        has_sr_markers = "<<<<<<< SEARCH" in response
        try:
            code = apply_search_replace_blocks(existing_impl, response)
            print("[TDD Robo] ⚙️ Applied Search/Replace diff blocks to implementation.")
            # Verify Python syntax of the resulting code to prevent saving broken files
            try:
                compile(code, impl_name, "exec")
            except SyntaxError as syntax_err:
                raise ValueError(f"Applied Search/Replace blocks resulted in a SyntaxError: {syntax_err}")
        except ValueError as e:
            if has_sr_markers:
                raw_error = (
                    f"Failed to apply Search/Replace block: {e}\n"
                    "Please ensure the SEARCH block matches the existing code EXACTLY, "
                    "including indentation.\n"
                )
                # Keep clean implementation code and avoid comments injection
                code = existing_impl
                print(f"[TDD Robo] ❌ Search/Replace block application failed: {e}.")
                impl_path = save_artifact(impl_name, code)
                print(f"[TDD Robo] ✅ Saved implementation code to {impl_path}")
                save_history_snapshot(impl_name, code, state.get("iterations", 0) + 1)
                return {"impl_code": code, "impl_check_output": f"Error: {raw_error}", "bug_report": ""}
            else:
                if config.VERBOSE:
                    print(f"[TDD Robo] Search/Replace block application skipped/failed (no markers): {e}")
                code = extract_code(response)
    else:
        code = extract_code(response)

    impl_path = save_artifact(impl_name, code)
    print(f"[TDD Robo] ✅ Saved implementation code to {impl_path}")
    save_history_snapshot(impl_name, code, state.get("iterations", 0) + 1)
    return {"impl_code": code, "impl_check_output": "", "bug_report": ""}


def check_impl_syntax(state: TDDState):
    """
    Check the syntax of the generated implementation code.

    Reads:
        - module_name
        - impl_code
    Writes:
        - impl_check_output
    """
    _update_req_progress(state)
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)

    impl_code = state.get("impl_code", "").strip()
    if not impl_code:
        print("[TDD Robo] ❌ Implementation code is empty!")
        return {
            "impl_check_output": "Error: The generated implementation code is empty. Please provide valid Python code."
        }

    # Intercept custom Search/Replace error marker (for backward compatibility)
    if "TDD_ROBO_SYNTAX_ERROR:" in impl_code:
        for line in impl_code.splitlines():
            if "TDD_ROBO_SYNTAX_ERROR:" in line:
                error_msg = line.split("TDD_ROBO_SYNTAX_ERROR:", 1)[1].strip()
                print(f"[TDD Robo] ❌ Syntax check intercepted: {error_msg}")
                return {"impl_check_output": f"Error: {error_msg}"}

    # If the previous step already set a Search/Replace application error, keep it
    impl_check_output = state.get("impl_check_output", "")
    if impl_check_output and "Failed to apply Search/Replace block" in impl_check_output:
        print(f"[TDD Robo] ❌ Syntax check intercepted (pre-existing): {impl_check_output}")
        return {"impl_check_output": impl_check_output}

    output = _run_syntax_check(impl_path, "implementation code")
    return {"impl_check_output": output}


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


def run_tests(state: TDDState):
    """
    Execute the pytest suite against the implementation.

    Reads:
        - test_module_name
        - iterations
    Writes:
        - test_output
        - success
        - iterations
    """
    _update_req_progress(state)
    iters = state.get("iterations", 0) + 1
    print(f"[TDD Robo] 🏃 Running tests (Iteration {iters})...")
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    try:
        # Step 1: Run the active test file first
        print(f"[TDD Robo] 🏃 Running active tests ({test_name})...")
        cmd1 = [sys.executable, "-m", "pytest", test_name, "-v", "--tb=short"]
        if config.PYTEST_MAXFAIL > 0:
            cmd1.append(f"--maxfail={config.PYTEST_MAXFAIL}")
        result = subprocess.run(
            cmd1,
            capture_output=True,
            text=True,
            timeout=TEST_EXECUTION_TIMEOUT_SEC,
            cwd=config.ARTIFACTS_DIR,
        )
        success = result.returncode == 0
        output = str(result.stdout or "") + "\n" + str(result.stderr or "")

        # Step 2: If active tests pass, run the regression test suite (all tests)
        if success:
            print("[TDD Robo] 🏃 Running regression test suite (all tests)...")
            cmd2 = [sys.executable, "-m", "pytest", "-v", "--tb=short", "--ignore=history"]
            if config.PYTEST_MAXFAIL > 0:
                cmd2.append(f"--maxfail={config.PYTEST_MAXFAIL}")
            reg_result = subprocess.run(
                cmd2,
                capture_output=True,
                text=True,
                timeout=TEST_EXECUTION_TIMEOUT_SEC,
                cwd=config.ARTIFACTS_DIR,
            )
            if reg_result.returncode != 0:
                success = False
                output = (
                    "REGRESSION ERROR: The current active tests passed, but other existing tests failed!\n"
                    + str(reg_result.stdout or "")
                    + "\n"
                    + str(reg_result.stderr or "")
                )
    except subprocess.TimeoutExpired as e:
        success = False
        stdout_str = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr_str = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        output = f"Test execution timed out after {e.timeout} seconds.\n"
        output += f"Partial stdout:\n{stdout_str}\n"
        output += f"Partial stderr:\n{stderr_str}"

    # Guard against excessively large test output to protect the LLM context window
    MAX_OUTPUT_CHARS = 8000
    if len(output) > MAX_OUTPUT_CHARS:
        prefix_len = 2000
        suffix_len = 5000
        truncated_msg = (
            f"\n\n... [TRUNCATED {len(output) - prefix_len - suffix_len} CHARACTERS TO PROTECT CONTEXT] ...\n\n"
        )
        output = output[:prefix_len] + truncated_msg + output[-suffix_len:]

    if config.VERBOSE:
        print(output)
    active_summary = _parse_pytest_summary(output)
    summary_part = f" ({active_summary})" if active_summary else ""
    print(f"[TDD Robo] {'✅ Test suite passed!' if success else '❌ Test suite failed!'}{summary_part}")
    return {"test_output": output, "success": success, "iterations": state.get("iterations", 0) + 1}


def _run_oracle_verification_on_failures(test_output: str, tests_code: str) -> str:
    """
    Parse test failures, extract expressions from the failed test cases,
    and verify them using the registered dynamic oracle verifier to detect incorrect test assertions.
    """
    import re

    verifier = config.ORACLE_VERIFIER
    if verifier is None:
        return ""

    # 1. Find all failed test method names from the test output
    failed_methods = set()
    for line in test_output.splitlines():
        match = re.search(r"::(test_\w+)", line)
        if match:
            failed_methods.add(match.group(1))
        else:
            match2 = re.search(r"\b(test_\w+)\b", line)
            if match2:
                failed_methods.add(match2.group(1))

    if not failed_methods:
        return ""

    feedback_lines = []

    # 2. For each failed method, extract its body from tests_code and search for assertions
    for method in failed_methods:
        pattern = rf"def\s+{method}\s*\([^)]*\)\s*:"
        match = re.search(pattern, tests_code)
        if not match:
            continue

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

        method_body = "\n".join(body_lines)

        # 3. Look for assertion statements and execute dynamic verifier
        expr = None
        expected_val = None

        code_match = re.search(r'code\s*=\s*["\']([^"\']+)["\']', method_body)
        expected_match = re.search(r'expected\s*=\s*\[\s*["\']([^"\']+)["\']\s*\]', method_body)

        if code_match and expected_match:
            expr = code_match.group(1).strip()
            expected_val = expected_match.group(1).strip()
        else:
            execute_match = re.search(
                r'execute\(\s*["\']([^"\']+)["\']\s*\)\s*==\s*\[\s*["\']([^"\']+)["\']\s*\]', method_body
            )
            if execute_match:
                expr = execute_match.group(1).strip()
                expected_val = execute_match.group(2).strip()

        if expr and expected_val:
            try:
                oracle_result = verifier(expr).strip()
                if "Error" not in oracle_result and "Exception" not in oracle_result:
                    if oracle_result != expected_val:
                        feedback_lines.append(
                            f"- Test case `{method}` contains an assertion error:\n"
                            f"  * Expression evaluated: `{expr}`\n"
                            f"  * Expected value hardcoded in test: `{expected_val}`\n"
                            f"  * Actual correct oracle value: `{oracle_result}`\n"
                            f"  * Rationale: The test expectation `{expected_val}` is mathematically INCORRECT. "
                            f"The correct output should be `{oracle_result}`. Therefore, the bug is in the test "
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


def generate_bug_report(state: TDDState):
    """
    Generate a bug report diagnosing the failures from the test execution.

    Reads:
        - goal
        - test_output
        - tests_code
        - impl_code
    Writes:
        - bug_report
        - next_action
    """
    _update_req_progress(state)
    print(f"[TDD Robo] 🐛 Generating bug report (Iteration {state.get('iterations', 0)})...")
    combined_tests = _get_combined_tests_code(state)
    impl_code = state.get("impl_code", "")
    if impl_code:
        clean_lines = [line for line in impl_code.splitlines() if not line.startswith("# TDD_ROBO_SYNTAX_ERROR")]
        impl_code = "\n".join(clean_lines).lstrip()

    # Perform oracle verification check on failures
    oracle_feedback = _run_oracle_verification_on_failures(state.get("test_output", ""), combined_tests)

    template = get_prompt("generate_bug_report_prompt", GENERATE_BUG_REPORT_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        oracle_verification_feedback=oracle_feedback,
        test_output=state.get("test_output", ""),
        tests_code=add_line_numbers(combined_tests),
        impl_code=add_line_numbers(impl_code),
    )

    response = call_llm_with_reasoning(prompt, response_schema=BugReport, thinking_level="HIGH")

    bug_data = json.loads(extract_json(response))
    report_md = "### Failed Test Cases\n"
    for t in bug_data.get("failed_test_cases", []):
        report_md += f"- {t}\n"
    report_md += f"\n### Expected vs Actual\n{bug_data.get('expected_vs_actual', '')}\n\n"
    report_md += f"### Fix Instructions\n{bug_data.get('fix_instructions', '')}\n"
    target_to_fix = bug_data.get("target_to_fix", "implement_logic")

    print(f"[TDD Robo] ✅ Bug report generated (Target: {target_to_fix})")
    if config.VERBOSE:
        print("-----------------------")
        print(report_md)
        print("-----------------------")
    return {"bug_report": report_md, "next_action": target_to_fix}


def generate_readme(state: TDDState):
    """
    Generate a README.md file based on the goal and implementation code.

    Reads:
        - goal
        - module_name
        - impl_code
    Writes:
        - readme_content
    """
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


# --- Edge (Conditional Branch) Definitions ---
def should_review_test_plan_or_continue(state: TDDState):
    """Determine whether to regenerate the test plan or proceed to test generation."""
    if state.get("test_plan_review_decision") == "plan_tests":
        return "plan_tests"
    return "generate_tests"


def should_review_tests_or_continue(state: TDDState):
    """Determine whether to regenerate tests based on syntax errors or proceed to review."""
    if state.get("tests_check_output"):
        return "generate_tests"
    return "review_tests"


def should_implement_logic(state: TDDState):
    """Determine whether to regenerate tests based on review feedback, run tests, or proceed to implementation."""
    if state.get("test_review_decision") == "generate_tests":
        return "generate_tests"
    if state.get("impl_code"):
        return "run_tests"
    return "implement_logic"


def should_run_tests(state: TDDState):
    """Determine whether to fix implementation syntax errors or run the tests."""
    if state.get("impl_check_output"):
        return "implement_logic"
    return "run_tests"


def increment_requirement(state: TDDState):
    """
    State-updating node to advance to the next target requirement.
    """
    _update_req_progress(state)
    current_index = state.get("current_req_index", 0)
    requirements = state.get("requirements", [])
    next_index = current_index + 1

    print(f"[TDD Robo] ➡️ Advancing from requirement {current_index + 1} to {next_index + 1}/{len(requirements)}...")

    # Print the progress checklist using the next index as active
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

    return {
        "current_req_index": next_index,
        "success": False,
        "test_plan": "",
        "test_plan_review": "",
        "test_plan_iterations": 0,
        "test_review": "",
        "test_iterations": 0,
        "tests_code": "",  # Clear test code for the next requirement
    }


def should_continue(state: TDDState):
    """Determine whether the workflow succeeded, max iterations reached, or a bug report is needed."""
    if state.get("success", False):
        requirements = state.get("requirements", [])
        current_index = state.get("current_req_index", 0)
        if current_index < len(requirements) - 1:
            return "increment_requirement"
        return "generate_readme"
    max_iters = state.get("max_iterations", MAX_ITERATIONS)
    if state.get("iterations", 0) >= max_iters:
        return END
    return "generate_bug_report"


def _detect_toggle_loop(state: TDDState) -> bool:
    """
    Detect whether the implementation code snapshots are repeating or toggling,
    which indicates the agent is stuck in an infinite debugging loop.
    """
    import glob
    import hashlib
    import os

    artifacts_dir = getattr(config, "ARTIFACTS_DIR", "artifacts")
    history_dir = os.path.join(artifacts_dir, "history")
    if not os.path.exists(history_dir):
        return False

    module_name = state.get("module_name", "impl.py")
    base_name = os.path.splitext(module_name)[0]

    # Get all implementation snapshots
    pattern = os.path.join(history_dir, f"{base_name}_iter*.py")
    snapshot_files = sorted(glob.glob(pattern))

    if len(snapshot_files) < 4:
        return False

    # Read last 6 snapshot hashes
    hashes = []
    for fpath in snapshot_files[-6:]:
        try:
            with open(fpath, "rb") as f:
                content = f.read()
                hashes.append(hashlib.md5(content).hexdigest())
        except Exception:
            continue

    if len(hashes) < 4:
        return False

    # Check for toggle loop: A-B-A-B or A-B-C-A-B-C
    if hashes[-1] == hashes[-3] and hashes[-2] == hashes[-4]:
        print("[TDD Robo] 🚨 Loop Detector: Detected toggle loop (A-B-A-B) in implementation snapshots!")
        return True

    # Check for multi-iteration repeating state
    # Check for multi-iteration repeating state
    if len(hashes) >= 5 and hashes[-1] == hashes[-3] and hashes[-1] == hashes[-5]:
        print("[TDD Robo] 🚨 Loop Detector: Detected repeating state (A-X-A-Y-A) in implementation snapshots!")
        return True

    return False


def should_fix_tests_or_impl(state: TDDState):
    """Determine the next action based on the bug report."""
    next_act = state.get("next_action", "implement_logic")

    # If the LLM wants to modify the implementation logic, check if it's already stuck in a loop
    if next_act == "implement_logic":
        if _detect_toggle_loop(state):
            print(
                "[TDD Robo] 🔄 Loop Detector Override: Stuck in implementation loop. "
                "Directing flow to 'generate_tests' to review tests."
            )
            return "generate_tests"

    return next_act


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
        workflow.add_node("generate_design", generate_design)
        workflow.add_node("plan_tests", plan_tests)
        workflow.add_node("review_test_plan", review_test_plan)
        workflow.add_node("generate_tests", generate_tests)
        workflow.add_node("check_tests_syntax", check_tests_syntax)
        workflow.add_node("review_tests", review_tests)
        workflow.add_node("implement_logic", implement_logic)
        workflow.add_node("check_impl_syntax", check_impl_syntax)
        workflow.add_node("run_tests", run_tests)
        workflow.add_node("increment_requirement", increment_requirement)
        workflow.add_node("generate_bug_report", generate_bug_report)
        workflow.add_node("generate_readme", generate_readme)

        workflow.set_entry_point("fetch_spec")
        workflow.add_edge("fetch_spec", "generate_requirements")
        workflow.add_edge("generate_requirements", "plan_files")
        workflow.add_edge("plan_files", "generate_design")
        workflow.add_edge("generate_design", "plan_tests")
        workflow.add_edge("plan_tests", "review_test_plan")
        workflow.add_conditional_edges(
            "review_test_plan",
            should_review_test_plan_or_continue,
            {"plan_tests": "plan_tests", "generate_tests": "generate_tests"},
        )
        workflow.add_edge("generate_tests", "check_tests_syntax")
        workflow.add_conditional_edges(
            "check_tests_syntax",
            should_review_tests_or_continue,
            {"generate_tests": "generate_tests", "review_tests": "review_tests"},
        )
        workflow.add_conditional_edges(
            "review_tests",
            should_implement_logic,
            {"generate_tests": "generate_tests", "run_tests": "run_tests", "implement_logic": "implement_logic"},
        )
        workflow.add_edge("implement_logic", "check_impl_syntax")
        workflow.add_conditional_edges(
            "check_impl_syntax",
            should_run_tests,
            {"implement_logic": "implement_logic", "run_tests": "run_tests"},
        )
        workflow.add_conditional_edges(
            "run_tests",
            should_continue,
            {
                "increment_requirement": "increment_requirement",
                "generate_readme": "generate_readme",
                "generate_bug_report": "generate_bug_report",
                END: END,
            },
        )
        workflow.add_edge("increment_requirement", "plan_tests")
        workflow.add_conditional_edges(
            "generate_bug_report",
            should_fix_tests_or_impl,
            {"implement_logic": "implement_logic", "generate_tests": "generate_tests"},
        )
        workflow.add_edge("generate_readme", END)

        return workflow.compile(checkpointer=self.checkpointer)
