import argparse
import json
import os
import shutil
import sys
import time

import mlflow

import config
from agent import TDDAgent
from logger import logger, print
from utils import FileMemorySaver

DEFAULT_IMPL_NAME = config.DEFAULT_IMPL_NAME
DEFAULT_TEST_NAME = config.DEFAULT_TEST_NAME


def parse_arguments(args_list=None):
    """
    Parses command line arguments and handles domain/python tips file reading.
    """
    parser = argparse.ArgumentParser(description="TDD Agent Workflow")
    parser.add_argument("--resume", action="store_true", help="Resume workflow from the last checkpoint")
    parser.add_argument("--goal", type=str, default=config.GOAL, help="Goal description")
    parser.add_argument("--spec-url", type=str, default=config.SPEC_URL, help="Specification URL")
    parser.add_argument("--max-iterations", type=int, default=config.MAX_ITERATIONS, help="Maximum workflow iterations")
    parser.add_argument(
        "--max-test-plan-iterations", type=int, default=config.MAX_TEST_PLAN_ITERATIONS, help="Max test plan iterations"
    )
    parser.add_argument(
        "--max-test-iterations", type=int, default=config.MAX_TEST_ITERATIONS, help="Max test iterations"
    )
    parser.add_argument(
        "--target-test-plan-coverage",
        type=int,
        default=config.TARGET_TEST_PLAN_COVERAGE,
        help="Target test plan coverage",
    )
    parser.add_argument(
        "--target-test-coverage", type=int, default=config.TARGET_TEST_COVERAGE, help="Target test coverage"
    )
    parser.add_argument(
        "--target-design-quality",
        type=int,
        default=config.TARGET_DESIGN_QUALITY,
        help="Target design quality threshold percentage (0-100)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose log outputs")
    parser.add_argument("--session-id", type=str, default=config.SESSION_ID, help="Session ID / subdirectory name")
    parser.add_argument("--domain-tips", type=str, default=config.DOMAIN_TIPS, help="Implementation domain tips text")
    parser.add_argument("--domain-tips-file", type=str, default="", help="Path to file containing domain tips text")
    parser.add_argument(
        "--python-tips", type=str, default=config.DEFAULT_PYTHON_TIPS, help="Python environment tips text"
    )
    parser.add_argument(
        "--python-tips-file", type=str, default="", help="Path to file containing python environment tips text"
    )
    parser.add_argument(
        "--regression-failure-policy",
        type=str,
        choices=["rollback", "halt"],
        default="rollback",
        help="Policy on regression failure: rollback or halt",
    )
    parser.add_argument(
        "--draw-graph",
        action="store_true",
        help="Draw and save the workflow graph image to the current directory ('workflow_graph.png') and exit",
    )
    args = parser.parse_args(args_list)

    if args.domain_tips_file:
        try:
            with open(args.domain_tips_file, "r", encoding="utf-8") as f:
                args.domain_tips = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read domain tips file {args.domain_tips_file}: {e}")

    if args.python_tips_file:
        try:
            with open(args.python_tips_file, "r", encoding="utf-8") as f:
                args.python_tips = f.read()
        except Exception as e:
            print(f"⚠️ Failed to read python tips file {args.python_tips_file}: {e}")

    if args.verbose:
        config.VERBOSE = True

    return args, parser


def prepare_session_id(args):
    """Resolves and returns the session_id to use."""
    session_id = args.session_id
    if args.resume and (not args.session_id or args.session_id == config.SESSION_ID):
        base_dir = config.TDD_BASE_ARTIFACTS_DIR
        latest_session = None
        latest_mtime = 0.0
        if os.path.exists(base_dir):
            for entry in os.listdir(base_dir):
                entry_path = os.path.join(base_dir, entry)
                if os.path.isdir(entry_path):
                    chk_path = os.path.join(entry_path, "checkpoint.pkl")
                    if os.path.exists(chk_path):
                        try:
                            mtime = os.path.getmtime(chk_path)
                            if mtime > latest_mtime:
                                latest_mtime = mtime
                                latest_session = entry
                        except Exception:
                            continue
        if latest_session:
            session_id = latest_session
            print(f"🔄 Auto-detected latest session to resume: '{session_id}'")
        else:
            session_id = args.session_id or config.SESSION_ID
            print(f"⚠️ No past session with checkpoint.pkl found to resume. Defaulting to session: '{session_id}'")
    return session_id


def acquire_session_lock(artifacts_dir, session_id):
    """Establishes session-wide file lock using fcntl to prevent double running on the same session."""
    lock_file_path = os.path.join(artifacts_dir, ".session.lock")
    try:
        import fcntl

        lock_file_handle = open(lock_file_path, "w")
        # Try to lock it. If blocked, raise BlockingIOError
        fcntl.flock(lock_file_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lock_file_handle
    except BlockingIOError:
        print(f"\n⚠️ ERROR: Session '{session_id}' is already locked and running in another instance.")
        print("To run multiple workflows, please specify different session IDs or run them in separate sessions.\n")
        sys.exit(1)
    except Exception as e:
        if config.VERBOSE:
            print(f"Warning: Could not establish session lock: {e}")
        return None


def prepare_artifacts_directory(artifacts_dir, should_resume):
    """Handles backup and cleaning of the artifacts directory if not resuming."""
    if not should_resume:
        # Check if directory has any real workflow artifacts other than the lock file or spec
        has_run_artifacts = False
        if os.path.exists(artifacts_dir):
            for entry_name in os.listdir(artifacts_dir):
                if entry_name not in ("specification.txt", ".session.lock"):
                    has_run_artifacts = True
                    break
        if has_run_artifacts:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            backup_dir = f"{artifacts_dir}_{timestamp}"
            os.makedirs(backup_dir, exist_ok=True)
            for entry_name in os.listdir(artifacts_dir):
                if entry_name != ".session.lock":
                    file_path = os.path.join(artifacts_dir, entry_name)
                    shutil.move(file_path, os.path.join(backup_dir, entry_name))
            print(f"📦 Backed up existing artifacts to {backup_dir}")

            old_spec_path = os.path.join(backup_dir, "specification.txt")
            if os.path.exists(old_spec_path):
                shutil.copy2(old_spec_path, os.path.join(artifacts_dir, "specification.txt"))


def initialize_mlflow(should_resume, run_name, artifacts_dir):
    """Configures MLflow tracking with automatic local fallback and autologging."""
    mlflow_uri = config.MLFLOW_DEFAULT_URI
    is_server_online = False

    import socket

    try:
        # Test connection with configured ping timeout
        # Parse host and port from URL
        from urllib.parse import urlparse

        parsed_url = urlparse(mlflow_uri)
        host = parsed_url.hostname or "localhost"
        port = parsed_url.port or 5000
        with socket.create_connection((host, port), timeout=config.MLFLOW_PING_TIMEOUT_SEC):
            is_server_online = True
    except Exception:
        pass

    env_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if env_uri:
        mlflow.set_tracking_uri(env_uri)
    elif is_server_online:
        mlflow.set_tracking_uri(mlflow_uri)
    else:
        local_db_path = os.path.join(artifacts_dir, "mlflow.db")
        local_db_uri = f"sqlite:///{os.path.abspath(local_db_path)}"
        print(f"\nℹ️ MLflow server at {mlflow_uri} is offline. Falling back to local database ({local_db_uri}).")
        mlflow.set_tracking_uri(local_db_uri)

        # Set custom local artifact location so mlruns is created inside artifacts_dir
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            exp = client.get_experiment_by_name(config.MLFLOW_EXPERIMENT_NAME)
            if exp is None:
                artifact_uri = f"file:///{os.path.abspath(os.path.join(artifacts_dir, 'mlruns'))}"
                mlflow.create_experiment(config.MLFLOW_EXPERIMENT_NAME, artifact_location=artifact_uri)
        except Exception as e:
            print(f"Warning: Could not create local MLflow experiment: {e}")

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    try:
        mlflow.gemini.autolog()
    except Exception as e:
        print(f"Warning: Could not configure MLflow Gemini Autolog: {e}")

    # Prevent MLflow's LangChain Tracer from crashing due to unsupported on_resume callback
    if not should_resume:
        try:
            mlflow.langchain.autolog()
        except Exception as e:
            print(f"Warning: Could not configure MLflow LangChain Autolog: {e}")


def display_workflow_results(final_state, artifacts_dir):
    """Displays the workflow result summary (success or failure)."""
    if final_state.get("success", False):
        print("\n=== 🎉 TDD and Specification Retrieval Workflow Complete ===")
        impl_name = final_state.get("module_name", DEFAULT_IMPL_NAME)
        test_name = final_state.get("test_module_name", DEFAULT_TEST_NAME)
        impl_path = os.path.join(artifacts_dir, impl_name)
        test_path = os.path.join(artifacts_dir, test_name)

        print(f"\n### 📄 Implementation Code (`{impl_path}`)")
        print(f"```python\n{final_state.get('impl_code', '')}\n```")

        print(f"\n### 🧪 Test Code (`{test_path}`)")
        print(f"```python\n{final_state.get('tests_code', '')}\n```")

        print(f"\n### 📝 {os.path.join(artifacts_dir, 'README.md')}")
        print(final_state.get("readme_content", ""))

    else:
        print("\n=== ❌ Workflow Failed ===")
        print("\n### 🐞 Last Test Output")
        print(f"```text\n{final_state.get('test_output', 'No test output available.')}\n```")


# --- Execution Logic ---
def main(args_list=None):
    """
    Main entry point for the TDD Agent Workflow.
    Parses arguments, sets up the workspace, initializes the agent, and runs the LangGraph workflow.
    """
    args, parser = parse_arguments(args_list)

    if args.draw_graph:
        tdd_agent = TDDAgent()
        try:
            graph_path = "workflow_graph.png"
            with open(graph_path, "wb") as graph_f:
                graph_f.write(tdd_agent.get_graph().draw_mermaid_png())
            print(f"✅ Saved workflow graph to {graph_path}")
            return None
        except Exception as e:
            print(f"❌ Could not save workflow graph image: {e}")
            sys.exit(1)

    # Resolve session ID and paths
    session_id = prepare_session_id(args)

    # Update global config paths dynamically
    config.SESSION_ID = session_id
    config.ARTIFACTS_DIR = os.path.join(config.TDD_BASE_ARTIFACTS_DIR, session_id)
    artifacts_dir = config.ARTIFACTS_DIR

    # Ensure artifacts directory exists
    os.makedirs(artifacts_dir, exist_ok=True)

    # Establish session lock
    lock_file_handle = acquire_session_lock(artifacts_dir, session_id)

    checkpoint_path = os.path.join(artifacts_dir, "checkpoint.pkl")
    checkpoint_exists = os.path.exists(checkpoint_path)
    should_resume = args.resume and checkpoint_exists

    if args.resume and not checkpoint_exists:
        print("⚠️ No checkpoint found in session directory. Starting a new workflow.")

    if not should_resume:
        # Prompt for goal if missing and running in interactive terminal
        if not args.goal and sys.stdin.isatty():
            try:
                print("\n💡 TDD Agent: Starting a new workflow.")
                user_goal = input("Enter your goal description (e.g., 'Build a calculator'): ").strip()
                if user_goal:
                    args.goal = user_goal
            except KeyboardInterrupt:
                print("\n🛑 Interrupted. Exiting...")
                sys.exit(130)

        if not args.goal:
            parser.error("--goal is required when starting a new workflow")

        # Prompt for spec_url if missing and running in interactive terminal
        if not args.spec_url and sys.stdin.isatty():
            try:
                user_spec = input("Enter the specification URL or local file path: ").strip()
                if user_spec:
                    args.spec_url = user_spec
            except KeyboardInterrupt:
                print("\n🛑 Interrupted. Exiting...")
                sys.exit(130)

        if not args.spec_url:
            parser.error("--spec-url is required when starting a new workflow")

    config_meta_path = os.path.join(artifacts_dir, "config_meta.json")
    current_config_meta = {
        "GOAL": args.goal,
        "SPEC_URL": args.spec_url,
        "MODEL_PRIMARY": config.MODEL_PRIMARY,
        "MODEL_SECONDARY": config.MODEL_SECONDARY,
        "LLM_SEED": config.LLM_SEED,
        "TARGET_TEST_PLAN_COVERAGE": args.target_test_plan_coverage,
        "TARGET_TEST_COVERAGE": args.target_test_coverage,
        "DOMAIN_TIPS": args.domain_tips,
        "PYTHON_TIPS": args.python_tips,
    }

    if should_resume:
        print(f"🔄 Resuming session '{session_id}' from the previous run...")
        if os.path.exists(config_meta_path):
            with open(config_meta_path, "r", encoding="utf-8") as meta_f:
                old_config_meta = json.load(meta_f)

            mismatches = []
            for k, v in old_config_meta.items():
                if k in current_config_meta and current_config_meta[k] != v:
                    mismatches.append(f"{k}: '{v}' -> '{current_config_meta[k]}'")

            if mismatches:
                print("\n⚠️ WARNING: Configuration changes detected since the last checkpoint:")
                for m in mismatches:
                    print(f"  - {m}")
                print("Applying these changes to the resumed workflow may cause inconsistent behavior.\n")
                time.sleep(
                    config.CONFIG_MISMATCH_WARN_DELAY_SEC
                )  # Give the user time to read the warning and cancel with Ctrl+C
    else:
        prepare_artifacts_directory(artifacts_dir, should_resume)

    # Initialize file logging after preparing the directory (to avoid backing up the newly created log file
    # and preventing it from being moved to the archived directory during a fresh run setup)
    log_file_path = os.path.join(artifacts_dir, "workflow.log")
    logger.add_file_handler(log_file_path)

    # Update config metadata for future resumes
    with open(config_meta_path, "w", encoding="utf-8") as meta_f:
        json.dump(current_config_meta, meta_f, indent=2)

    initial_state = {
        "goal": args.goal,
        "spec_url": args.spec_url,
        "max_iterations": args.max_iterations,
        "max_test_plan_iterations": args.max_test_plan_iterations,
        "max_test_iterations": args.max_test_iterations,
        "target_test_plan_coverage": args.target_test_plan_coverage,
        "target_test_coverage": args.target_test_coverage,
        "target_design_quality": args.target_design_quality,
        "domain_tips": args.domain_tips or "",
        "python_tips": args.python_tips or "",
        "design_iterations": 0,
        "syntax_error_iterations": 0,
        "test_syntax_error_iterations": 0,
        "regression_failure_policy": args.regression_failure_policy,
    }

    config_opts = {
        "configurable": {"thread_id": config.SESSION_ID, "recursion_limit": config.LANGGRAPH_RECURSION_LIMIT}
    }

    # Initialize Agent
    memory_saver = FileMemorySaver(checkpoint_path)
    tdd_agent = TDDAgent(checkpointer=memory_saver)

    # Bind dynamic oracle verifier for assertions
    from utils import evaluate_math_expression

    config.ORACLE_VERIFIER = evaluate_math_expression

    # Configure MLflow
    initialize_mlflow(should_resume, config.RUN_NAME, artifacts_dir)

    print("\n🚀 Invoking the TDD Agent Workflow...")
    with mlflow.start_run(run_name=config.RUN_NAME):
        mlflow.log_param("goal", args.goal)
        mlflow.log_param("spec_url", args.spec_url)
        mlflow.log_param("max_iterations", initial_state["max_iterations"])
        mlflow.log_param(
            "max_test_plan_iterations", initial_state.get("max_test_plan_iterations", config.MAX_TEST_PLAN_ITERATIONS)
        )
        mlflow.log_param("max_test_iterations", initial_state.get("max_test_iterations", config.MAX_TEST_ITERATIONS))
        mlflow.log_param(
            "target_test_plan_coverage",
            initial_state.get("target_test_plan_coverage", config.TARGET_TEST_PLAN_COVERAGE),
        )
        mlflow.log_param("target_test_coverage", initial_state.get("target_test_coverage", config.TARGET_TEST_COVERAGE))
        mlflow.log_param("resumed", should_resume)

        if should_resume:
            # Reset loop detection counters to prevent immediate rollback on resume
            try:
                state_snapshot = tdd_agent.app.get_state(config_opts)
                if state_snapshot and state_snapshot.values:
                    reset_data = {
                        "iterations": 0,
                        "stagnant_iterations": 0,
                        "syntax_error_iterations": 0,
                        "test_syntax_error_iterations": 0,
                        "loop_detected": False,
                        "bug_report": "",
                        "domain_tips": args.domain_tips or "",
                        "python_tips": args.python_tips or "",
                    }
                    tdd_agent.app.update_state(config_opts, reset_data)
                    print("⚙️ Reset loop safety counters in checkpoint to ensure a clean resume.")
            except Exception as e:
                print(f"Warning: Could not reset safety counters on resume: {e}")

            final_state = tdd_agent.invoke(None, config=config_opts)
        else:
            final_state = tdd_agent.invoke(initial_state, config=config_opts)

        mlflow.log_metric("iterations", final_state.get("iterations", 0))
        mlflow.log_param("success", final_state.get("success", False))

        if "design_doc" in final_state:
            mlflow.log_text(final_state["design_doc"], "design.md")
        if "test_plan" in final_state:
            mlflow.log_text(final_state["test_plan"], "test_plan.md")
        if "tests_code" in final_state:
            mlflow.log_text(final_state["tests_code"], final_state.get("test_module_name", DEFAULT_TEST_NAME))
        if "impl_code" in final_state:
            mlflow.log_text(final_state["impl_code"], final_state.get("module_name", DEFAULT_IMPL_NAME))
        if "readme_content" in final_state:
            mlflow.log_text(final_state["readme_content"], "README.md")
        if "bug_report" in final_state:
            mlflow.log_text(final_state["bug_report"], "bug_report.md")
        if "test_output" in final_state:
            mlflow.log_text(final_state["test_output"], "test_output.txt")

    # Display Results & Demo
    display_workflow_results(final_state, artifacts_dir)

    # Clean up lock file handle cleanly
    if lock_file_handle:
        try:
            lock_file_handle.close()
        except Exception:
            pass

    return final_state


if __name__ == "__main__":
    try:
        final_state = main()
        if final_state and not final_state.get("success", False):
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n🛑 Execution interrupted by user. Exiting gracefully.")
        sys.exit(130)
