import datetime
import glob
import json
import os
import shutil

from tddrobo import config
from tddrobo.logger import print
from tddrobo.schema import TDDState


def save_history_snapshot(
    filename: str,
    code: str,
    iteration: int,
    state: TDDState | None = None,
    is_refactor: bool = False,
    phase: str | None = None,
):
    """Saves a snapshot of the generated file to the artifacts/history/ directory."""
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


def _cleanup_history_on_rollback(state: TDDState):
    """
    Rename previous history snapshot files for the current requirement
    by appending a '.bak' suffix, to prevent the Loop Detector from matching
    and falsely triggering on design-rollback cycles.
    """
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


def _backup_project_before_rollback(state: TDDState, current_idx: int, target_idx: int):
    """
    Backs up the current project files (py_bc.py, all tests, test_output, bug_report)
    to a designated rollback_backups directory in history before files are modified/deleted.
    """
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
