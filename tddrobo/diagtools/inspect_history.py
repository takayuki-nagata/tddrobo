# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import argparse
import difflib
import os
import re
import sys
from typing import Dict, List


def parse_history_file(filename: str) -> Dict[str, str]:
    """Parse history filename components."""
    info = {"filename": filename, "type": "unknown", "req": "", "design": "", "test_iter": "", "impl_iter": ""}

    if filename.startswith("design_iter"):
        m = re.match(r"design_iter(\d+)\.md", filename)
        if m:
            info["type"] = "design"
            info["impl_iter"] = m.group(1)
        return info

    if filename.startswith("syntax_error_"):
        info["type"] = "syntax_error"
        # Extract requirement
        m = re.search(r"req(\d+)", filename)
        if m:
            info["req"] = f"REQ{m.group(1)}"
        return info

    if filename.startswith("test_py_bc_"):
        info["type"] = "test"
        m = re.match(r"test_py_bc_req(\d+)_d(\d+)_(unit|integration)_iter(\d+)\.py", filename)
        if m:
            info["req"] = f"REQ{m.group(1)}"
            info["design"] = m.group(2)
            info["test_iter"] = m.group(4)
        return info

    if filename.startswith("py_bc_"):
        # Could be implementation or refactor
        m_ref = re.match(r"py_bc_req(\d+)_d(\d+)_test_iter(\d+)_refactor_iter(\d+)\.py", filename)
        if m_ref:
            info["type"] = "refactor"
            info["req"] = f"REQ{m_ref.group(1)}"
            info["design"] = m_ref.group(2)
            info["impl_iter"] = m_ref.group(4)
            return info

        m_impl = re.match(r"py_bc_req(\d+)_d(\d+)_test_iter(\d+)_(unit|integration)_impl_iter(\d+)\.py", filename)
        if m_impl:
            info["type"] = "implementation"
            info["req"] = f"REQ{m_impl.group(1)}"
            info["design"] = m_impl.group(2)
            info["test_iter"] = m_impl.group(3)
            info["impl_iter"] = m_impl.group(5)
            return info

    return info


def print_diff(file1: str, file2: str):
    """Print unified diff between two files."""
    if not os.path.exists(file1):
        print(f"❌ Error: File not found at '{file1}'", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(file2):
        print(f"❌ Error: File not found at '{file2}'", file=sys.stderr)
        sys.exit(1)

    print(f"📄 Diff: {os.path.basename(file1)} ➡️ {os.path.basename(file2)}")
    print("=" * 60)

    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        diff = difflib.unified_diff(
            f1.readlines(), f2.readlines(), fromfile=os.path.basename(file1), tofile=os.path.basename(file2), n=3
        )
        has_diff = False
        for line in diff:
            has_diff = True
            print(line, end="")
        if not has_diff:
            print("ℹ️ No differences found.")


def main():
    parser = argparse.ArgumentParser(description="TDD Agent History snapshot inspector")
    parser.add_argument(
        "--history-dir",
        type=str,
        default="artifacts/bc_clone_session/history",
        help="Path to the history folder",
    )
    parser.add_argument(
        "--req",
        type=str,
        default="",
        help="Filter snapshots by requirement ID (e.g. REQ002)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all snapshots grouped by type",
    )
    parser.add_argument(
        "--diff",
        type=int,
        nargs=2,
        metavar=("ITER1", "ITER2"),
        help="Diff two implementation iterations for a target requirement (e.g. --diff 1 2)",
    )
    parser.add_argument(
        "--file-diff",
        type=str,
        nargs=2,
        metavar=("FILE1", "FILE2"),
        help="Diff two specific files directly",
    )
    args = parser.parse_args()

    # If file-diff is requested, perform it immediately (does not require history-dir)
    if args.file_diff:
        print_diff(args.file_diff[0], args.file_diff[1])
        return

    if not os.path.exists(args.history_dir):
        print(f"❌ Error: History directory not found at '{args.history_dir}'", file=sys.stderr)
        sys.exit(1)

    # Read and parse history files
    try:
        filenames = sorted(os.listdir(args.history_dir))
    except Exception as e:
        print(f"❌ Error reading history directory: {e}", file=sys.stderr)
        sys.exit(1)

    parsed_files = [parse_history_file(f) for f in filenames]

    # Filter by requirement if specified
    if args.req:
        target_req = args.req.upper()
        # For design, we don't filter out unless listed since they don't have req in name
        parsed_files = [
            f for f in parsed_files if f["req"].upper() == target_req or (f["type"] == "design" and args.list)
        ]

    # Handle List action
    if args.list:
        categories: Dict[str, List[Dict[str, str]]] = {
            "design": [],
            "implementation": [],
            "refactor": [],
            "test": [],
            "syntax_error": [],
            "unknown": [],
        }
        for pf in parsed_files:
            categories[pf["type"]].append(pf)

        print("==================================================")
        print(f"📂 History Directory: {args.history_dir}")
        if args.req:
            print(f"🎯 Filter: {args.req.upper()}")
        print("==================================================")

        for cat, items in categories.items():
            if not items:
                continue
            print(f"\n📁 Category: {cat.upper()} ({len(items)} files)")
            for item in items:
                req_str = f" | {item['req']}" if item["req"] else ""
                design_str = f" | Design: d{item['design']}" if item["design"] else ""
                iter_str = ""
                if item["impl_iter"]:
                    iter_str = f" | Iteration: {item['impl_iter']}"
                elif item["test_iter"]:
                    iter_str = f" | Test Iteration: {item['test_iter']}"
                print(f"  - {item['filename']}{req_str}{design_str}{iter_str}")
        return

    # Handle Diff action
    if args.diff:
        if not args.req:
            print(
                "❌ Error: Diffs require specifying a target requirement via --req (e.g. --req REQ002)", file=sys.stderr
            )
            sys.exit(1)

        iter1, iter2 = args.diff
        target_req = args.req.upper()

        # Find implementation files matching target_req and the iterations
        impl_files = [
            f for f in parsed_files if f["type"] in ("implementation", "refactor") and f["req"].upper() == target_req
        ]

        file1_info = [f for f in impl_files if int(f["impl_iter"]) == iter1]
        file2_info = [f for f in impl_files if int(f["impl_iter"]) == iter2]

        if not file1_info:
            print(f"❌ Error: No implementation snapshot found for {target_req} iteration {iter1}", file=sys.stderr)
            sys.exit(1)
        if not file2_info:
            print(f"❌ Error: No implementation snapshot found for {target_req} iteration {iter2}", file=sys.stderr)
            sys.exit(1)

        path1 = os.path.join(args.history_dir, file1_info[0]["filename"])
        path2 = os.path.join(args.history_dir, file2_info[0]["filename"])
        print_diff(path1, path2)
        return

    # Default: print usage help
    parser.print_help()


if __name__ == "__main__":
    main()
