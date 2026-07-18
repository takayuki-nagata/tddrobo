import glob
import hashlib
import os
import re
import signal
import subprocess
import sys
import threading
from typing import Any, cast

import config
from logger import print
from schema import TDDState
from utils import call_llm_structured, call_llm_text, call_llm_with_reasoning

from .history import save_history_snapshot

TEST_EXECUTION_TIMEOUT_SEC = config.TEST_EXECUTION_TIMEOUT_SEC
SYNTAX_CHECK_TIMEOUT_SEC = config.SYNTAX_CHECK_TIMEOUT_SEC

_thread_local = threading.local()


def _get_dynamic_max_tokens(state: TDDState | None) -> int:
    """Calculate a dynamic max token limit based on current code size."""
    if not state:
        return config.LLM_DYNAMIC_MIN_TOKENS

    current_codes = [
        state.get("impl_code", ""),
        state.get("tests_code", ""),
        state.get("design_doc", ""),
        state.get("unit_tests_code", ""),
        state.get("integration_tests_code", ""),
    ]
    max_char_len = max(len(c) for c in current_codes if c) if any(current_codes) else 0

    estimated_tokens = max_char_len // config.CHARS_TO_TOKENS_RATIO
    dynamic_limit = estimated_tokens + config.LLM_DYNAMIC_BUFFER_TOKENS

    return min(max(config.LLM_DYNAMIC_MIN_TOKENS, dynamic_limit), config.LLM_DYNAMIC_MAX_TOKENS)


def _call_llm_structured(prompt: str, response_schema: Any, model_name: str = config.MODEL_PRIMARY, **kwargs) -> Any:
    state = getattr(_thread_local, "current_state", None)
    max_tokens = _get_dynamic_max_tokens(state)
    return call_llm_structured(prompt, response_schema, model_name=model_name, max_tokens=max_tokens)


def _call_llm_text(prompt: str, model_name: str = config.MODEL_PRIMARY, **kwargs) -> str:
    state = getattr(_thread_local, "current_state", None)
    max_tokens = _get_dynamic_max_tokens(state)
    return call_llm_text(prompt, model_name=model_name, max_tokens=max_tokens)


def _call_llm_with_reasoning(
    prompt: str,
    response_schema: Any = None,
    tools: list[Any] | None = None,
    thinking_level: str | None = None,
    temperature: float = 0.0,
    **kwargs,
) -> str:
    state = getattr(_thread_local, "current_state", None)
    max_tokens = _get_dynamic_max_tokens(state)
    return call_llm_with_reasoning(
        prompt,
        response_schema=response_schema,
        tools=tools,
        thinking_level=thinking_level,
        temperature=temperature,
        max_tokens=max_tokens,
    )


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

    MAX_OUTPUT_CHARS = config.MAX_TEST_OUTPUT_CHARS
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
            "audit_loop_count": 0,
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
    # Note: _update_req_progress will be called by nodes.py before syntax helper
    iters = int(str(state.get(state_key_iter, 0) or 0)) + 1
    is_test = filename.startswith("test")

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


def _run_syntax_check(filename: str, label: str) -> str:
    """Run a basic syntax check on a given Python file using flake8."""
    try:
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


def _truncate_test_output_smart(text: str | None, max_chars: int = config.MAX_TEST_OUTPUT_CHARS) -> str:
    """Intelligently truncates pytest output to protect token context while preserving the FAILURES block."""
    if not text:
        return ""
    if len(text) <= max_chars:
        return text

    failures_match = re.search(r"(={3,}\s*(?:FAILURES|ERRORS)\s*={3,}.*?)(?=\n={3,}|\Z)", text, re.DOTALL)
    failures_content = ""
    if failures_match:
        failures_content = failures_match.group(1).strip()

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
        fallback_prefix = max(10, int(max_chars * 0.35))
        fallback_suffix = max(0, max_chars - fallback_prefix - 100)
        return (
            text[:fallback_prefix]
            + f"\n\n... [TRUNCATED {len(text) - max_chars} CHARACTERS] ...\n\n"
            + text[-fallback_suffix:]
        )
    return result


def _get_balanced_test_output_context(text: str, max_chars: int = config.BALANCED_TEST_OUTPUT_MAX_CHARS) -> str:
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


def _get_combined_tests_code(state: TDDState) -> str:
    """Helper to collect and combine all test_*.py files in artifacts directory."""
    current_req_idx = state.get("current_req_index", 0)
    all_tests_content = []
    if os.path.exists(config.ARTIFACTS_DIR):
        test_files = sorted(glob.glob(os.path.join(config.ARTIFACTS_DIR, "test_*.py")))
        for tf in test_files:
            tf_basename = os.path.basename(tf)

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
    if not all_tests_content:
        unit = state.get("unit_tests_code", "")
        integ = state.get("integration_tests_code", "")
        if unit or integ:
            return f"# Unit Tests:\n{unit}\n\n# Integration Tests:\n{integ}"
    return "\n\n".join(all_tests_content) if all_tests_content else state.get("tests_code", "")


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


def _has_implementation_exceptions(test_output: str, impl_name: str) -> bool:
    """Check if the test output contains unhandled Python exceptions from the implementation code."""
    if not test_output:
        return False
    exception_patterns = [
        r"SyntaxError:",
        r"NameError:",
        r"AttributeError:",
        r"RuntimeError:",
        r"TypeError:",
        r"ZeroDivisionError:",
        r"IndexError:",
        r"KeyError:",
        r"ValueError:",
    ]
    has_exc_message = any(re.search(pat, test_output) for pat in exception_patterns)
    pattern = r"(?:^|/|\\|\s|['\"])" + re.escape(impl_name) + r"(?:$|\s|['\"]|:)"
    references_impl = bool(re.search(pattern, test_output))
    return has_exc_message and references_impl


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

    main_output = test_output
    summary_text = ""
    summary_match = re.search(r"(=== short test summary info ===.*)", test_output, flags=re.DOTALL)
    if summary_match:
        summary_text = summary_match.group(1)
        main_output = test_output[: summary_match.start()]

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


def _detect_toggle_loop(state: TDDState) -> bool:
    """Detect if the implementation is stuck in a toggle loop or repeating state loop."""
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
