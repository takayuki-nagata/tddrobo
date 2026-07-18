# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import argparse
import os
import sys
from datetime import datetime

import mlflow


def main():
    parser = argparse.ArgumentParser(description="TDD Agent MLflow Trace Inspector")
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default="http://localhost:5000",
        help="MLflow tracking server URI (e.g. http://localhost:5000 or sqlite:///mlflow.db)",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="TDD_Agent_Experiment",
        help="Name of the MLflow experiment",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=10,
        help="Max number of traces to search",
    )
    parser.add_argument(
        "--filter-query",
        type=str,
        default="",
        help="Search query to filter prompts/responses containing this text",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="scratch",
        help="Directory to save full trace text files",
    )
    parser.add_argument(
        "--save-prefix",
        type=str,
        default="",
        help="Prefix to prepended to saved trace file names (e.g. 'exp1_'). If empty, defaults to 'trace_{trace_id}_'.",
    )
    parser.add_argument(
        "--trace-id",
        type=str,
        default="",
        help="Specific trace ID (or comma-separated list of trace IDs) to inspect",
    )
    parser.add_argument(
        "--show-details",
        action="store_true",
        help="Print full prompt and response details to stdout",
    )
    parser.add_argument(
        "--truncate-limit",
        type=int,
        default=2000,
        help="Character limit for printing prompt/response. Set to -1 or 0 to show full content.",
    )
    parser.add_argument(
        "--extract-spans",
        action="store_true",
        help="Extract nested child spans (LLM prompts) and save them in dict format compatible with replay_prompt.py",
    )
    args = parser.parse_args()

    # Check if MLflow server is online; fall back to local sqlite db if offline and using the default URI
    def is_server_online(uri: str) -> bool:
        import socket
        import urllib.parse

        try:
            parsed = urllib.parse.urlparse(uri)
            if not parsed.netloc:
                return True  # Probably local SQLite URI already
            host_port = parsed.netloc.split(":")
            host = host_port[0]
            port = int(host_port[1]) if len(host_port) > 1 else (443 if parsed.scheme == "https" else 80)
            with socket.create_connection((host, port), timeout=1.5):
                return True
        except Exception:
            return False

    if args.tracking_uri == "http://localhost:5000" and not is_server_online(args.tracking_uri):
        print("ℹ️ MLflow server is offline. Falling back to local database (sqlite:///mlflow.db).")
        args.tracking_uri = "sqlite:///mlflow.db"

    print(f"📡 Connecting to MLflow at '{args.tracking_uri}'...")
    mlflow.set_tracking_uri(args.tracking_uri)

    try:
        exp = mlflow.get_experiment_by_name(args.experiment_name)
    except Exception as e:
        print(f"❌ Error connecting to MLflow tracking: {e}")
        sys.exit(1)

    if not exp:
        print(f"❌ Experiment '{args.experiment_name}' not found.")
        sys.exit(1)

    print(f"🧪 Experiment ID: {exp.experiment_id}")
    print(f"🔍 Searching for traces (max={args.max_results})...")

    try:
        # locations is the modern parameter replacing experiment_ids in search_traces
        traces = mlflow.search_traces(locations=[exp.experiment_id], max_results=args.max_results)
    except TypeError:
        # Fallback to experiment_ids if old version
        traces = mlflow.search_traces(experiment_ids=[exp.experiment_id], max_results=args.max_results)
    except Exception as e:
        print(f"❌ Failed to search traces: {e}")
        sys.exit(1)

    if traces.empty:
        print("ℹ️ No traces found in experiment.")
        return

    # Sort traces by request_time descending (latest first)
    traces_sorted = traces.sort_values(by="request_time", ascending=False)

    # Filter by trace ID if provided
    if args.trace_id:
        target_ids = [tid.strip() for tid in args.trace_id.split(",") if tid.strip()]
        traces_sorted = traces_sorted[traces_sorted["trace_id"].isin(target_ids)]

    print(f"Found {len(traces_sorted)} traces. Listing in reverse chronological order:")
    match_count = 0

    for idx, row in traces_sorted.iterrows():
        trace_id = row["trace_id"]
        req_time_ms = row["request_time"]
        req_time_str = datetime.fromtimestamp(req_time_ms / 1000.0).strftime("%Y-%m-%d %H:%M:%S")
        status = row.get("state")

        request = row.get("request", {})
        response = row.get("response", {})

        prompt = ""
        model_name = ""
        if isinstance(request, dict):
            prompt = request.get("prompt", "")
            model_name = request.get("model_name", "")

        resp_str = str(response) if response else ""

        # Filter by keyword if provided
        if args.filter_query:
            if args.filter_query.lower() not in prompt.lower() and args.filter_query.lower() not in resp_str.lower():
                continue

        match_count += 1
        print("--------------------------------------------------")
        print(f"[{match_count}] Trace ID: {trace_id}")
        print(f"    Time: {req_time_str} | Status: {status} | Model: {model_name}")

        spans = row.get("spans", [])
        if spans:
            print(f"    Spans ({len(spans)}): {[s.get('name') for s in spans]}")

        # Save to save_dir if specified
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            if args.save_prefix:
                req_file = os.path.join(args.save_dir, f"{args.save_prefix}req_{trace_id}.txt")
                resp_file = os.path.join(args.save_dir, f"{args.save_prefix}resp_{trace_id}.txt")
            else:
                req_file = os.path.join(args.save_dir, f"trace_{trace_id}_req.txt")
                resp_file = os.path.join(args.save_dir, f"trace_{trace_id}_resp.txt")

            with open(req_file, "w", encoding="utf-8") as f:
                f.write(str(request))
            with open(resp_file, "w", encoding="utf-8") as f:
                f.write(resp_str)

            print(f"    Saved details to: {req_file} and {resp_file}")

            # Extract child spans if requested
            if args.extract_spans and spans:
                span_counts: dict[str, int] = {}
                for span in spans:
                    if isinstance(span, dict):
                        s_name = span.get("name", "")
                        attrs = span.get("attributes", {})
                    else:
                        s_name = getattr(span, "name", "")
                        attrs = getattr(span, "attributes", {})

                    inputs = attrs.get("mlflow.spanInputs")
                    outputs = attrs.get("mlflow.spanOutputs")

                    if inputs:
                        span_counts[s_name] = span_counts.get(s_name, 0) + 1
                        idx_str = f"_{span_counts[s_name]:02d}"

                        # Determine file name
                        prefix = args.save_prefix if args.save_prefix else f"trace_{trace_id}_"
                        span_file = os.path.join(args.save_dir, f"{prefix}span_{s_name}{idx_str}.txt")

                        # Convert to dict with 'prompt' key if it's a string, or use the dict
                        if isinstance(inputs, dict):
                            prompt_dict = inputs
                        else:
                            prompt_dict = {"prompt": str(inputs)}

                        with open(span_file, "w", encoding="utf-8") as f:
                            f.write(str(prompt_dict))
                        print(f"    Saved span input for {s_name}{idx_str} to: {span_file}")

                        if outputs:
                            span_resp_file = os.path.join(args.save_dir, f"{prefix}span_{s_name}{idx_str}_resp.txt")
                            with open(span_resp_file, "w", encoding="utf-8") as f:
                                f.write(str(outputs))

        if args.show_details:
            limit = args.truncate_limit
            if limit <= 0:
                print("\n    --- Request Prompt ---")
                print(prompt)
                print("\n    --- Response ---")
                print(resp_str)
            else:
                print(f"\n    --- Request Prompt (first {limit} chars) ---")
                print(prompt[:limit])
                if len(prompt) > limit:
                    print(f"    ... [Request truncated. Total length: {len(prompt)}]")

                print(f"\n    --- Response (first {limit} chars) ---")
                print(resp_str[:limit])
                if len(resp_str) > limit:
                    print(f"    ... [Response truncated. Total length: {len(resp_str)}]")
            print()

    if args.filter_query and match_count == 0:
        print(f"ℹ️ No traces matched the filter query '{args.filter_query}'.")


if __name__ == "__main__":
    main()
