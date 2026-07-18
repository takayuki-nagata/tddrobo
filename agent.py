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
    GENERATE_BUG_REPORT_PROMPT,
    GENERATE_DESIGN_PROMPT,
    GENERATE_README_PROMPT,
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
from schema import BugReport, DesignDocument, FilePlan, TDDState, TestPlan, TestPlanReviewReport, TestReviewReport
from utils import (
    ARTIFACTS_DIR,
    add_line_numbers,
    extract_code,
    extract_json,
    get_prompt,
    llm_gencode,
    llm_gendoc,
    read_artifact,
    run_bc_command,
    save_artifact,
)

DEFAULT_IMPL_NAME = config.DEFAULT_IMPL_NAME
DEFAULT_TEST_NAME = config.DEFAULT_TEST_NAME
FETCH_TIMEOUT_SEC = config.FETCH_TIMEOUT_SEC
SYNTAX_CHECK_TIMEOUT_SEC = config.SYNTAX_CHECK_TIMEOUT_SEC
TEST_EXECUTION_TIMEOUT_SEC = config.TEST_EXECUTION_TIMEOUT_SEC


# --- Node Definitions ---
def fetch_spec(state: TDDState):
    """
    Fetch the specification from the provided URL or local cache.

    Reads:
        - spec_url
    Writes:
        - spec_content
    """
    spec_path = os.path.join(ARTIFACTS_DIR, "specification.txt")
    headers = {}
    if os.path.exists(spec_path):
        mtime = os.path.getmtime(spec_path)
        headers["If-Modified-Since"] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime(mtime))

    print("\n--- 🌐 Checking and downloading specification ---")
    try:
        response = requests.get(state.get("spec_url", ""), headers=headers, timeout=FETCH_TIMEOUT_SEC)

        if response.status_code == 304:
            print("--- 🌐 Remote file not modified. Loading cached specification ---")
            spec_content = read_artifact("specification.txt")
            print(f"✅ Loaded specification from {spec_path}")
            return {"spec_content": spec_content}

        response.raise_for_status()
        content = response.text
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to fetch specification: {e}")
        return {"spec_content": f"Error fetching specification: {e}"}

    if "text/html" in response.headers.get("Content-Type", "").lower() or state.get("spec_url", "").endswith(".html"):
        content = markdownify.markdownify(content, heading_style="ATX")

    spec_path = save_artifact("specification.txt", content)
    print(f"✅ Saved specification to {spec_path}")
    return {"spec_content": content}


def plan_files(state: TDDState):
    """
    Determine filenames for the implementation and test modules.

    Reads:
        - goal
    Writes:
        - module_name
        - test_module_name
    """
    print("\n--- 📁 Determining filenames ---")
    template = get_prompt("plan_files_prompt", PLAN_FILES_PROMPT)
    prompt = template.format(goal=state.get("goal", ""))

    response = llm_gendoc(prompt, response_schema=FilePlan)

    try:
        plan = json.loads(extract_json(response))
        impl_name = plan.get("impl_filename", DEFAULT_IMPL_NAME)
        test_name = plan.get("test_filename", DEFAULT_TEST_NAME)
    except json.JSONDecodeError:
        print("⚠️ Failed to parse structured output. Falling back to default names.")
        impl_name = DEFAULT_IMPL_NAME
        test_name = DEFAULT_TEST_NAME

    print(f"Determined filenames: Implementation={impl_name}, Test={test_name}")
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
    print("\n--- 📐 Generating design document ---")
    template = get_prompt("generate_design_prompt", GENERATE_DESIGN_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
        test_name=state.get("test_module_name", DEFAULT_TEST_NAME),
    )

    response = llm_gencode(prompt, response_schema=DesignDocument)

    try:
        design_data = json.loads(extract_json(response))
        design = "# Software Design Document\n\n"
        for idx, (key, value) in enumerate(design_data.items(), 1):
            title = key.replace("_", " ").title()
            design += f"## {idx}. {title}\n{value}\n\n"
    except json.JSONDecodeError:
        print("⚠️ Failed to parse structured output. Using raw response.")
        design = response

    design_path = save_artifact("design.md", design)
    print(f"✅ Saved design document to {design_path}")
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
    Writes:
        - test_plan
        - test_plan_iterations
    """
    iters = state.get("test_plan_iterations", 0) + 1
    print(f"\n--- 📋 Generating test plan (Iteration {iters}) ---")
    if state.get("test_plan_review") and state.get("test_plan"):
        template = get_prompt("generate_test_plan_prompt_fix", GENERATE_TEST_PLAN_PROMPT_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design=state.get("design_doc", ""),
            impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
            test_plan=state.get("test_plan", ""),
            test_plan_review=state.get("test_plan_review", ""),
        )
    else:
        template = get_prompt("generate_test_plan_prompt", GENERATE_TEST_PLAN_PROMPT)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=state.get("spec_content", ""),
            design=state.get("design_doc", ""),
            impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
        )

    response = llm_gendoc(prompt, response_schema=TestPlan)

    try:
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
    except json.JSONDecodeError:
        print("⚠️ Failed to parse structured output. Using raw response.")
        plan_md = response

    plan_path = save_artifact("test_plan.md", plan_md)
    print(f"✅ Saved test plan to {plan_path}")
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
    Writes:
        - test_plan_review
        - test_plan_review_decision
    """
    print("\n--- 🧐 Reviewing Test Plan ---")
    template = get_prompt("review_test_plan_prompt", REVIEW_TEST_PLAN_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        spec=state.get("spec_content", ""),
        design=state.get("design_doc", ""),
        test_plan=state.get("test_plan", ""),
        target_coverage=state.get("target_test_plan_coverage", 95),
    )

    response = llm_gencode(prompt, response_schema=TestPlanReviewReport)

    try:
        review_data = json.loads(extract_json(response))
        coverage = review_data.get("estimated_coverage", 100)
        report_md = f"### Estimated Coverage: {coverage}%\n\n### Missing Test Cases:\n"
        seen = set()
        for m in review_data.get("missing_test_cases", []):
            if m not in seen:
                seen.add(m)
                report_md += f"- {m}\n"
        report_md += f"\n### Feedback:\n{review_data.get('feedback', '')}"

        print("✅ Test plan review completed:")
        print("-----------------------")
        print(report_md)
        print("-----------------------")

        target_coverage = state.get("target_test_plan_coverage", 95)
        if coverage < target_coverage:
            iters = state.get("test_plan_iterations", 0)
            max_iters = state.get("max_test_plan_iterations", 3)
            if iters >= max_iters:
                print(
                    f"⚠️ Test plan coverage is insufficient ({coverage}% < {target_coverage}%), "
                    f"but max iterations ({iters}/{max_iters}) reached. Proceeding to test generation."
                )
                return {"test_plan_review": "", "test_plan_review_decision": "generate_tests"}
            else:
                print(
                    f"⚠️ Test plan coverage is insufficient ({coverage}% < {target_coverage}%). "
                    "Regenerating test plan..."
                )
                return {"test_plan_review": report_md, "test_plan_review_decision": "plan_tests"}

        print(f"✅ Test plan coverage is sufficient ({coverage}%). Proceeding to test generation.")
        return {"test_plan_review": "", "test_plan_review_decision": "generate_tests"}
    except Exception as e:
        print(f"⚠️ Failed to parse review report: {e}. Proceeding to test generation.")
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
    Writes:
        - tests_code
        - test_iterations
        - iterations
        - success
        - bug_report
        - test_review
    """
    iters = state.get("test_iterations", 0) + 1
    print(f"\n--- 🧪 Generating test code (Iteration {iters}) ---")
    spec = state.get("spec_content", "")
    design = state.get("design_doc", "")
    test_plan = state.get("test_plan", "")
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    if state.get("tests_check_output") and state.get("tests_code"):
        template = get_prompt("generate_tests_prompt_fix", GENERATE_TESTS_PROMPT_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=test_plan,
            impl_name=impl_name,
            tests_check_output=state.get("tests_check_output", ""),
            tests_code=state.get("tests_code", ""),
        )
    elif state.get("bug_report") and state.get("tests_code"):
        template = get_prompt("generate_tests_prompt_bug_fix", GENERATE_TESTS_PROMPT_BUG_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=test_plan,
            impl_name=impl_name,
            bug_report=state.get("bug_report", ""),
            tests_code=state.get("tests_code", ""),
        )
    elif state.get("test_review") and state.get("tests_code"):
        template = get_prompt("generate_tests_prompt_review_fix", GENERATE_TESTS_PROMPT_REVIEW_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            spec=spec,
            design=design,
            test_plan=test_plan,
            impl_name=impl_name,
            test_review=state.get("test_review", ""),
            tests_code=state.get("tests_code", ""),
        )
    else:
        template = get_prompt("generate_tests_prompt_initial", GENERATE_TESTS_PROMPT_INITIAL)
        prompt = template.format(
            goal=state.get("goal", ""), spec=spec, design=design, test_plan=test_plan, impl_name=impl_name
        )
    response = llm_gencode(prompt, tools=[run_bc_command])
    code = extract_code(response)
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    test_path = save_artifact(test_name, code)
    print(f"✅ Saved test code to {test_path}")
    # Clear previous bug reports to prevent misbehavior in subsequent implementation nodes after modifying the test
    return {
        "tests_code": code,
        "test_iterations": iters,
        "iterations": 0,
        "success": False,
        "bug_report": "",
        "test_review": "",
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
    print(f"\n--- 🔍 Checking {label} syntax ---")
    try:
        # E999: SyntaxError, F821: Undefined name, F822: Undefined name in __all__
        result = subprocess.run(
            [sys.executable, "-m", "flake8", "--select=E999,F821,F822", filename],
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
        print("✅ Syntax check passed!")
    else:
        print("❌ Syntax error found!")
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
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    test_path = os.path.join(ARTIFACTS_DIR, test_name)

    tests_code = state.get("tests_code", "").strip()
    if not tests_code:
        print("❌ Test code is empty!")
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
    print("\n--- 🧐 Reviewing Test Code ---")
    template = get_prompt("review_tests_prompt", REVIEW_TESTS_PROMPT)
    prompt = template.format(
        test_plan=state.get("test_plan", ""),
        tests_code=add_line_numbers(state.get("tests_code", "")),
        target_coverage=state.get("target_test_coverage", 90),
    )

    response = llm_gencode(prompt, response_schema=TestReviewReport)

    try:
        review_data = json.loads(extract_json(response))
        coverage = review_data.get("estimated_coverage", 100)
        report_md = f"### Estimated Coverage: {coverage}%\n\n### Missing Test Cases:\n"
        for m in review_data.get("missing_test_cases", []):
            report_md += f"- {m}\n"
        report_md += f"\n### Feedback:\n{review_data.get('feedback', '')}"

        print("✅ Test review completed:")
        print("-----------------------")
        print(report_md)
        print("-----------------------")

        target_coverage = state.get("target_test_coverage", 90)
        if coverage < target_coverage:
            iters = state.get("test_iterations", 0)
            max_iters = state.get("max_test_iterations", 3)
            if iters >= max_iters:
                print(
                    f"⚠️ Test coverage is insufficient ({coverage}% < {target_coverage}%), "
                    f"but max iterations ({iters}/{max_iters}) reached. Proceeding to implementation."
                )
                return {"test_review": "", "test_review_decision": "implement_logic"}
            else:
                print(f"⚠️ Test coverage is insufficient ({coverage}% < {target_coverage}%). Regenerating tests...")
                return {"test_review": report_md, "test_review_decision": "generate_tests"}

        print(f"✅ Test coverage is sufficient ({coverage}%). Proceeding to implementation.")
        return {"test_review": "", "test_review_decision": "implement_logic"}
    except Exception as e:
        print(f"⚠️ Failed to parse review report: {e}. Proceeding to implementation.")
        return {"test_review": "", "test_review_decision": "implement_logic"}


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
    Writes:
        - impl_code
    """
    print(f"\n--- 💻 Generating implementation code (Iteration {state.get('iterations', 0) + 1}) ---")
    design = state.get("design_doc", "")
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    if state.get("impl_check_output") and state.get("impl_code"):
        template = get_prompt("implement_logic_prompt_syntax_fix", IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            design=design,
            tests_code=state.get("tests_code", ""),
            impl_name=impl_name,
            impl_check_output=state.get("impl_check_output", ""),
            impl_code=state.get("impl_code", ""),
        )
    elif state.get("bug_report") and state.get("impl_code"):
        template = get_prompt("implement_logic_prompt_fix", IMPLEMENT_LOGIC_PROMPT_FIX)
        prompt = template.format(
            goal=state.get("goal", ""),
            design=design,
            tests_code=state.get("tests_code", ""),
            impl_name=impl_name,
            bug_report=state.get("bug_report", ""),
            impl_code=state.get("impl_code", ""),
        )
    else:
        template = get_prompt("implement_logic_prompt_initial", IMPLEMENT_LOGIC_PROMPT_INITIAL)
        prompt = template.format(
            goal=state.get("goal", ""), design=design, tests_code=state.get("tests_code", ""), impl_name=impl_name
        )
    response = llm_gencode(prompt)
    code = extract_code(response)
    impl_path = save_artifact(impl_name, code)
    print(f"✅ Saved implementation code to {impl_path}")
    return {"impl_code": code}


def check_impl_syntax(state: TDDState):
    """
    Check the syntax of the generated implementation code.

    Reads:
        - module_name
        - impl_code
    Writes:
        - impl_check_output
    """
    impl_name = state.get("module_name", DEFAULT_IMPL_NAME)
    impl_path = os.path.join(ARTIFACTS_DIR, impl_name)

    impl_code = state.get("impl_code", "").strip()
    if not impl_code:
        print("❌ Implementation code is empty!")
        return {
            "impl_check_output": "Error: The generated implementation code is empty. Please provide valid Python code."
        }

    output = _run_syntax_check(impl_path, "implementation code")
    return {"impl_check_output": output}


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
    print("\n--- 🏃 Running tests ---")
    test_name = state.get("test_module_name", DEFAULT_TEST_NAME)
    try:
        # Add --maxfail=3 and --tb=short to prevent the error log from bloating and protect the LLM context
        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_name, "-v", "--tb=short", "--maxfail=3"],
            capture_output=True,
            text=True,
            timeout=TEST_EXECUTION_TIMEOUT_SEC,
            cwd=ARTIFACTS_DIR,
        )
        success = result.returncode == 0
        output = str(result.stdout or "") + "\n" + str(result.stderr or "")
    except subprocess.TimeoutExpired as e:
        success = False
        stdout_str = e.stdout.decode("utf-8", errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        stderr_str = e.stderr.decode("utf-8", errors="replace") if isinstance(e.stderr, bytes) else (e.stderr or "")
        output = f"Test execution timed out after {e.timeout} seconds.\n"
        output += f"Partial stdout:\n{stdout_str}\n"
        output += f"Partial stderr:\n{stderr_str}"
    print(output)
    print(f"Test result: {'✅ Success' if success else '❌ Failure'}")
    return {"test_output": output, "success": success, "iterations": state.get("iterations", 0) + 1}


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
    print(f"\n--- 🐛 Generating bug report (Iteration {state.get('iterations', 0)}) ---")
    template = get_prompt("generate_bug_report_prompt", GENERATE_BUG_REPORT_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        test_output=state.get("test_output", ""),
        tests_code=add_line_numbers(state.get("tests_code", "")),
        impl_code=add_line_numbers(state.get("impl_code", "")),
    )

    response = llm_gencode(prompt, response_schema=BugReport)

    try:
        bug_data = json.loads(extract_json(response))
        report_md = "### Failed Test Cases\n"
        for t in bug_data.get("failed_test_cases", []):
            report_md += f"- {t}\n"
        report_md += f"\n### Expected vs Actual\n{bug_data.get('expected_vs_actual', '')}\n\n"
        report_md += f"### Fix Instructions\n{bug_data.get('fix_instructions', '')}\n"
        target_to_fix = bug_data.get("target_to_fix", "implement_logic")
    except json.JSONDecodeError:
        print("⚠️ Failed to parse structured output. Using raw response.")
        report_md = response
        target_to_fix = "implement_logic"

    print(f"✅ Bug report generated (Target: {target_to_fix})")
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
    print("\n--- 📝 Generating README.md ---")
    template = get_prompt("generate_readme_prompt", GENERATE_README_PROMPT)
    prompt = template.format(
        goal=state.get("goal", ""),
        impl_name=state.get("module_name", DEFAULT_IMPL_NAME),
        impl_code=state.get("impl_code", ""),
    )
    readme = llm_gendoc(prompt)
    readme_path = save_artifact("README.md", readme)
    print(f"✅ Saved README.md to {readme_path}")
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


def should_continue(state: TDDState):
    """Determine whether the workflow succeeded, max iterations reached, or a bug report is needed."""
    if state.get("success", False):
        return "generate_readme"
    max_iters = state.get("max_iterations", 3)
    if state.get("iterations", 0) >= max_iters:
        return END
    return "generate_bug_report"


def should_fix_tests_or_impl(state: TDDState):
    """Determine the next action based on the bug report."""
    return state.get("next_action", "implement_logic")


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
        workflow.add_node("generate_bug_report", generate_bug_report)
        workflow.add_node("generate_readme", generate_readme)

        workflow.set_entry_point("fetch_spec")
        workflow.add_edge("fetch_spec", "plan_files")
        workflow.add_edge("plan_files", "generate_design")
        workflow.add_edge("generate_design", "plan_tests")
        workflow.add_edge("plan_tests", "review_test_plan")
        workflow.add_conditional_edges("review_test_plan", should_review_test_plan_or_continue)
        workflow.add_edge("generate_tests", "check_tests_syntax")
        workflow.add_conditional_edges("check_tests_syntax", should_review_tests_or_continue)
        workflow.add_conditional_edges("review_tests", should_implement_logic)
        workflow.add_edge("implement_logic", "check_impl_syntax")
        workflow.add_conditional_edges("check_impl_syntax", should_run_tests)
        workflow.add_conditional_edges("run_tests", should_continue)
        workflow.add_conditional_edges("generate_bug_report", should_fix_tests_or_impl)
        workflow.add_edge("generate_readme", END)

        return workflow.compile(checkpointer=self.checkpointer)
