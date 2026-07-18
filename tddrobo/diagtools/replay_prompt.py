# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import argparse
import ast
import json
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Replay a prompt trace or text file through the GenAI model configuration."
    )
    parser.add_argument(
        "--trace",
        help="Path to the trace request text file (e.g. scratch/trace_..._req.txt)",
    )
    parser.add_argument(
        "--prompt-file",
        help="Path to a raw prompt text file to send (alternative to --trace)",
    )
    parser.add_argument(
        "--replace",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        help="Replace OLD with NEW in the prompt. Can be specified multiple times.",
    )
    parser.add_argument(
        "--replace-file",
        help="Path to a JSON file containing search-and-replace mappings (dictionary of OLD -> NEW strings).",
    )
    parser.add_argument(
        "--model",
        choices=["primary", "secondary", "gemma-4-31b-it", "gemma-4-26b-a4b-it"],
        help=(
            "Override the model to use. primary/gemma-4-31b-it uses the reasoning model, "
            "secondary/gemma-4-26b-a4b-it uses the standard model."
        ),
    )
    parser.add_argument(
        "--standard",
        action="store_true",
        help="Force using the standard (fast) model instead of the reasoning one.",
    )
    parser.add_argument(
        "--schema",
        help="Pydantic schema class name from schema module to apply (e.g. TestPlan)",
    )
    parser.add_argument(
        "--temp",
        type=float,
        help="Override temperature",
    )
    parser.add_argument(
        "--output",
        help="Path to a file where the LLM response will be saved.",
    )
    args = parser.parse_args()

    # Import config and utils locally to delay model initialization
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, root_dir)
    from tddrobo import config

    # Ensure config initializes correctly
    config.DEBUG_MODE = True
    from tddrobo.utils import call_llm_standard, call_llm_with_reasoning

    if not args.trace and not args.prompt_file:
        print("Error: Either --trace or --prompt-file must be specified.", file=sys.stderr)
        sys.exit(1)
    if args.trace and args.prompt_file:
        print("Error: Cannot specify both --trace and --prompt-file.", file=sys.stderr)
        sys.exit(1)

    if args.trace:
        if not os.path.exists(args.trace):
            print(f"Error: Trace file not found: {args.trace}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.trace, "r", encoding="utf-8") as f:
                content = f.read()
                # Try parsing as python dictionary literal
                try:
                    trace_data = ast.literal_eval(content)
                except Exception:
                    # Fallback to json if literal_eval fails
                    trace_data = json.loads(content)
        except Exception as e:
            print(f"Error reading/parsing trace file: {e}", file=sys.stderr)
            sys.exit(1)

        if not isinstance(trace_data, dict) or "prompt" not in trace_data:
            print(
                "Error: Invalid trace request file format. Must be a dict containing 'prompt'.",
                file=sys.stderr,
            )
            sys.exit(1)

        prompt = trace_data["prompt"]
        model_name = args.model or trace_data.get("model_name", "gemma-4-31b-it")
    else:
        if not os.path.exists(args.prompt_file):
            print(f"Error: Prompt file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)

        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt = f.read()
        except Exception as e:
            print(f"Error reading prompt file: {e}", file=sys.stderr)
            sys.exit(1)
        model_name = args.model or "gemma-4-31b-it"

    # Apply replacements from JSON file if specified
    if args.replace_file:
        try:
            with open(args.replace_file, "r", encoding="utf-8") as rf:
                replacements = json.load(rf)
            if not isinstance(replacements, dict):
                print(
                    "Error: Replacements file must be a JSON object mapping search strings to replacement strings.",
                    file=sys.stderr,
                )
                sys.exit(1)
            for old, new in replacements.items():
                prompt = prompt.replace(old, new)
            print(f"Loaded {len(replacements)} replacements from JSON file.")
        except Exception as e:
            print(f"Error reading/parsing replacements file: {e}", file=sys.stderr)
            sys.exit(1)

    # Apply any requested replacements from command line
    if args.replace:
        for old, new in args.replace:
            prompt = prompt.replace(old, new)

    # Apply temperature override if specified
    temperature = args.temp if args.temp is not None else 0.0

    # Load response schema if specified
    response_schema = None
    if args.schema:
        from tddrobo import schema

        if not hasattr(schema, args.schema):
            print(f"Error: Schema class '{args.schema}' not found in schema module.", file=sys.stderr)
            sys.exit(1)
        response_schema = getattr(schema, args.schema)

    # Decide which LLM wrapper to call
    if args.standard:
        is_reasoning = False
    elif args.model:
        is_reasoning = model_name in ["primary", "gemma-4-31b-it"]
    else:
        is_reasoning = "gemma-4-31b-it" in model_name

    print(f"📡 Replaying prompt using model: {model_name} (Reasoning: {is_reasoning})")
    print(f"Original prompt length: {len(prompt)} chars")
    print(f"Adjusted prompt length: {len(prompt)} chars")
    print("-" * 50)
    print("Sending request to Gemini...")

    try:
        if is_reasoning:
            response = call_llm_with_reasoning(
                prompt,
                response_schema=response_schema,
                thinking_level="MINIMAL",
                temperature=temperature,
            )
        else:
            response = call_llm_standard(
                prompt,
                response_schema=response_schema,
                temperature=temperature,
            )
        print("\n=== LLM Response ===")
        if response_schema and hasattr(response, "model_dump_json"):
            print(response.model_dump_json(indent=2))
        elif response_schema and isinstance(response, dict):
            print(json.dumps(response, indent=2))
        else:
            print(response)

        if args.output:
            with open(args.output, "w", encoding="utf-8") as of:
                if response_schema and hasattr(response, "model_dump_json"):
                    of.write(response.model_dump_json(indent=2))
                elif response_schema and isinstance(response, dict):
                    of.write(json.dumps(response, indent=2))
                else:
                    of.write(str(response))
            print(f"\n💾 Response saved to: {args.output}")

    except Exception as e:
        print(f"Error calling LLM: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
