# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import argparse
import os
import sys

# Ensure tddrobo directory is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tddrobo.agent import TDDAgent
from tddrobo.utils import FileMemorySaver


def main():
    parser = argparse.ArgumentParser(description="TDD Agent Checkpoint Inspector")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="artifacts/bc_clone_session/checkpoint.pkl",
        help="Path to checkpoint.pkl file",
    )
    parser.add_argument("--show-all", action="store_true", help="Show all details")
    parser.add_argument("--show-req", action="store_true", help="Show current requirement")
    parser.add_argument("--show-bug-report", action="store_true", help="Show last bug report")
    parser.add_argument("--show-test-output", action="store_true", help="Show last test output")
    parser.add_argument("--test-output-limit", type=int, default=2000, help="Character limit for test output printing")
    parser.add_argument("--show-impl", action="store_true", help="Show current implementation code")
    parser.add_argument("--list-keys", action="store_true", help="List all keys in state")
    parser.add_argument("--dump-key", type=str, help="Dump the value of a specific key in state")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint file not found at '{args.checkpoint}'")
        sys.exit(1)

    print(f"🔍 Loading checkpoint from '{args.checkpoint}'...")
    try:
        memory_saver = FileMemorySaver(args.checkpoint)
        tdd_agent = TDDAgent(checkpointer=memory_saver)
        # Assuming the session ID is default or read from config/checkpoint
        # Normally checkpoint key is (thread_id, checkpoint_id)
        # We can extract the latest state using standard LangGraph getConfig/getState
        config_opts = {"configurable": {"thread_id": "bc_clone_session", "recursion_limit": 150}}
        state_obj = tdd_agent.app.get_state(config_opts)
        state = state_obj.values
    except Exception as e:
        print(f"❌ Failed to load or parse checkpoint: {e}")
        sys.exit(1)

    if not state:
        print("⚠️ Warning: No state values found in checkpoint. The workflow may not have started yet.")
        return

    if args.list_keys:
        print("🔑 Checkpoint State Keys:")
        for k in sorted(state.keys()):
            print(f"  - {k}")
        return

    if args.dump_key:
        if args.dump_key not in state:
            print(f"❌ Error: Key '{args.dump_key}' not found in state.", file=sys.stderr)
            sys.exit(1)
        val = state[args.dump_key]
        print(f"=== Dump of Key '{args.dump_key}' ===")
        if isinstance(val, (dict, list)):
            import json

            try:
                print(json.dumps(val, indent=2))
            except Exception:
                print(val)
        else:
            print(val)
        return

    print("==================================================")
    print("⚙️ Checkpoint Metadata:")
    print(f"  Iterations: {state.get('iterations', 0)}")
    print(f"  Syntax Error Iterations: {state.get('syntax_error_iterations', 0)}")
    print(f"  Current Requirement Index: {state.get('current_req_index', 0)}")
    print("==================================================")

    # Show Requirement
    if args.show_req or args.show_all:
        requirements = state.get("requirements", [])
        idx = state.get("current_req_index", 0)
        if requirements and idx < len(requirements):
            r = requirements[idx]
            print("📋 Current Target Requirement:")
            print(f"  ID: {r.get('id')}")
            print(f"  Description: {r.get('description')}")
            print("==================================================")

    # Show Bug Report
    if args.show_bug_report or args.show_all:
        bug_report = state.get("bug_report")
        if bug_report:
            print("🐞 Last Bug Report:")
            print(bug_report)
            print("==================================================")
        else:
            print("🐞 Last Bug Report: None (or all tests passed in last run)")
            print("==================================================")

    # Show Test Output
    if args.show_test_output or args.show_all:
        test_output = state.get("test_output")
        if test_output:
            print(f"🧪 Last Test Output (first {args.test_output_limit} chars):")
            print(test_output[: args.test_output_limit])
            if len(test_output) > args.test_output_limit:
                print(f"... [Truncated. Total length: {len(test_output)}]")
            print("==================================================")
        else:
            print("🧪 Last Test Output: None")
            print("==================================================")

    # Show Implementation Code
    if args.show_impl or args.show_all:
        impl_code = state.get("impl_code")
        if impl_code:
            print("💻 Current Implementation Code:")
            print(impl_code)
            print("==================================================")
        else:
            print("💻 Current Implementation Code: Empty")
            print("==================================================")


if __name__ == "__main__":
    main()
