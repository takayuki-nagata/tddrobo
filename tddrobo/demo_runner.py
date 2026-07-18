import argparse
import os
import typing

from cli import main as cli_main
from tddrobo import config


def run_demo(
    default_goal: str,
    default_spec_url: str,
    default_domain_tips: str,
    default_session_id: str,
    run_demo_verification_func: typing.Callable[[typing.Dict[str, typing.Any]], None],
):
    """Generic runner for E2E demo tasks.

    Handles --fresh option, auto-resume, CLI execution, and runs the
    verification callback.
    """
    # 1. Parse custom args for the demo runner
    # We use add_help=False to not conflict with cli.py help flag
    parser = argparse.ArgumentParser(description="TDD Agent Demo Runner", add_help=False)
    parser.add_argument("--fresh", action="store_true", help="Clean session artifacts and start fresh")
    parser.add_argument("--session-id", type=str, default=default_session_id)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    # Parse only known args to not interfere with cli.py specific arguments (like --verbose, etc.)
    demo_args, remaining_args = parser.parse_known_args()

    session_id = demo_args.session_id

    # Resolve paths
    artifacts_dir = os.path.join(config.TDD_BASE_ARTIFACTS_DIR, session_id)
    checkpoint_path = os.path.join(artifacts_dir, "checkpoint.pkl")

    # 2. Handle --fresh option
    if demo_args.fresh:
        print(f"🧹 --fresh specified. Delegating cleanup/backup for session '{session_id}' to cli.py...")

    # 3. Handle Auto-Resume
    has_checkpoint = os.path.exists(checkpoint_path)
    auto_resume = has_checkpoint and not demo_args.fresh

    # Rebuild arguments for cli.py
    cli_args = []

    # Inject defaults if not already present in remaining_args
    if "--goal" not in remaining_args:
        cli_args.extend(["--goal", default_goal])
    if "--spec-url" not in remaining_args:
        cli_args.extend(["--spec-url", default_spec_url])
    if "--domain-tips" not in remaining_args:
        cli_args.extend(["--domain-tips", default_domain_tips])
    if "--session-id" not in remaining_args:
        cli_args.extend(["--session-id", session_id])

    # Inject --resume if auto-resume triggers or user requested it explicitly
    if auto_resume or demo_args.resume:
        if "--resume" not in cli_args and "--resume" not in remaining_args:
            cli_args.append("--resume")
            if auto_resume:
                print(f"🔄 Auto-detected checkpoint at '{checkpoint_path}'. Enabling --resume.")

    # Pass all remaining args through (like -v, --max-iterations, etc.)
    cli_args.extend(remaining_args)

    # 4. Invoke CLI main
    final_state = cli_main(cli_args)

    # 5. Invoke Verification Hook if workflow succeeded
    if final_state and final_state.get("success", False):
        run_demo_verification_func(final_state)
