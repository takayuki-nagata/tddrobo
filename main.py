import argparse
import json
import os
import shutil
import time

import mlflow

import config
from agent import TDDAgent
from utils import FileMemorySaver

ARTIFACTS_DIR = config.ARTIFACTS_DIR
DEFAULT_IMPL_NAME = config.DEFAULT_IMPL_NAME
DEFAULT_TEST_NAME = config.DEFAULT_TEST_NAME


# --- Execution Logic ---
def main(args_list=None):
    """
    Main entry point for the TDD Agent Workflow.
    Parses arguments, sets up the workspace, initializes the agent, and runs the LangGraph workflow.
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
    args = parser.parse_args(args_list)

    dir_exists = os.path.exists(ARTIFACTS_DIR)
    checkpoint_exists = dir_exists and os.path.exists(os.path.join(ARTIFACTS_DIR, "checkpoint.pkl"))
    should_resume = args.resume and checkpoint_exists

    if args.resume and not checkpoint_exists:
        print("⚠️ No artifacts directory or checkpoint found. Starting a new workflow.")

    if not should_resume:
        if not args.goal:
            parser.error("--goal is required when starting a new workflow")
        if not args.spec_url:
            parser.error("--spec-url is required when starting a new workflow")

    config_meta_path = os.path.join(ARTIFACTS_DIR, "config_meta.json")
    current_config_meta = {
        "GOAL": args.goal,
        "SPEC_URL": args.spec_url,
        "MODEL_GENCODE": config.MODEL_GENCODE,
        "MODEL_GENDOC": config.MODEL_GENDOC,
        "LLM_SEED": config.LLM_SEED,
        "TARGET_TEST_PLAN_COVERAGE": args.target_test_plan_coverage,
        "TARGET_TEST_COVERAGE": args.target_test_coverage,
    }

    if should_resume:
        print("🔄 Resuming from the previous run...")
        if os.path.exists(config_meta_path):
            with open(config_meta_path, "r", encoding="utf-8") as f:
                old_config_meta = json.load(f)

            mismatches = []
            for k, v in old_config_meta.items():
                if k in current_config_meta and current_config_meta[k] != v:
                    mismatches.append(f"{k}: '{v}' -> '{current_config_meta[k]}'")

            if mismatches:
                print("\n⚠️ WARNING: Configuration changes detected since the last checkpoint:")
                for m in mismatches:
                    print(f"  - {m}")
                print("Applying these changes to the resumed workflow may cause inconsistent behavior.\n")
                time.sleep(10)  # Give the user time to read the warning and cancel with Ctrl+C
    else:
        if dir_exists:
            readme_path = os.path.join(ARTIFACTS_DIR, "README.md")
            if os.path.exists(readme_path):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                backup_dir = f"{ARTIFACTS_DIR}_{timestamp}"
                os.rename(ARTIFACTS_DIR, backup_dir)
                os.makedirs(ARTIFACTS_DIR, exist_ok=True)
                print(f"📦 Backed up existing artifacts to {backup_dir}")

                old_spec_path = os.path.join(backup_dir, "specification.txt")
                if os.path.exists(old_spec_path):
                    shutil.copy2(old_spec_path, os.path.join(ARTIFACTS_DIR, "specification.txt"))
            else:
                for filename in os.listdir(ARTIFACTS_DIR):
                    if filename != "specification.txt":
                        file_path = os.path.join(ARTIFACTS_DIR, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.remove(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            print(f"⚠️ Failed to delete {file_path}. Reason: {e}")
        else:
            os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Update config metadata for future resumes
    with open(config_meta_path, "w", encoding="utf-8") as f:
        json.dump(current_config_meta, f, indent=2)

    initial_state = {
        "goal": args.goal,
        "spec_url": args.spec_url,
        "max_iterations": args.max_iterations,
        "max_test_plan_iterations": args.max_test_plan_iterations,
        "max_test_iterations": args.max_test_iterations,
        "target_test_plan_coverage": args.target_test_plan_coverage,
        "target_test_coverage": args.target_test_coverage,
    }

    config_opts = {"configurable": {"thread_id": config.SESSION_ID, "recursion_limit": 150}}

    # Initialize Agent
    checkpoint_path = os.path.join(ARTIFACTS_DIR, "checkpoint.pkl")
    memory_saver = FileMemorySaver(checkpoint_path)
    tdd_agent = TDDAgent(checkpointer=memory_saver)

    # Save compiled graph as an image (Optional, if Mermaid rendering is supported)
    try:
        graph_path = os.path.join(ARTIFACTS_DIR, "workflow_graph.png")
        with open(graph_path, "wb") as f:
            f.write(tdd_agent.get_graph().draw_mermaid_png())
        print(f"✅ Saved workflow graph to {graph_path}")
    except Exception as e:
        print(f"Could not save workflow graph image: {e}")

    # Configure MLflow tracking
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("TDD_Agent_Experiment")
    mlflow.gemini.autolog()

    # Prevent MLflow's LangChain Tracer from crashing due to unsupported on_resume callback
    if not should_resume:
        mlflow.langchain.autolog()

    print("\n🚀 Invoking the TDD Agent Workflow...")
    with mlflow.start_run(run_name=config.RUN_NAME):
        mlflow.log_param("goal", args.goal)
        mlflow.log_param("spec_url", args.spec_url)
        mlflow.log_param("max_iterations", initial_state["max_iterations"])
        mlflow.log_param("max_test_plan_iterations", initial_state.get("max_test_plan_iterations", 3))
        mlflow.log_param("max_test_iterations", initial_state.get("max_test_iterations", 3))
        mlflow.log_param("target_test_plan_coverage", initial_state.get("target_test_plan_coverage", 95))
        mlflow.log_param("target_test_coverage", initial_state.get("target_test_coverage", 90))
        mlflow.log_param("resumed", should_resume)

        if should_resume:
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
    if final_state.get("success", False):
        print("\n=== 🎉 TDD and Specification Retrieval Workflow Complete ===")
        impl_name = final_state.get("module_name", DEFAULT_IMPL_NAME)
        test_name = final_state.get("test_module_name", DEFAULT_TEST_NAME)
        impl_path = os.path.join(ARTIFACTS_DIR, impl_name)
        test_path = os.path.join(ARTIFACTS_DIR, test_name)

        print(f"\n### 📄 Implementation Code (`{impl_path}`)")
        print(f"```python\n{final_state.get('impl_code', '')}\n```")

        print(f"\n### 🧪 Test Code (`{test_path}`)")
        print(f"```python\n{final_state.get('tests_code', '')}\n```")

        print(f"\n### 📝 {os.path.join(ARTIFACTS_DIR, 'README.md')}")
        print(final_state.get("readme_content", ""))

    else:
        print("\n=== ❌ Workflow Failed ===")
        print("\n### 🐞 Last Test Output")
        print(f"```text\n{final_state.get('test_output', 'No test output available.')}\n```")

    return final_state


if __name__ == "__main__":
    main()
