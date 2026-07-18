import os
import sys

# Ensure tddrobo root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import config
from schema import DesignDocument
from utils import call_llm_structured


def run_test(prompt_template_path: str, label: str):
    print(f"\n=================== Running LLM Call: {label} ===================")

    # Load spec
    spec_path = os.path.join(os.path.dirname(__file__), "../artifacts/bc_clone_session/specification.txt")
    if not os.path.exists(spec_path):
        spec_path = os.path.join(os.path.dirname(__file__), "../tests/test_data/specification.txt")

    with open(spec_path, "r") as f:
        spec = f.read()

    spec_sliced = "\n".join(spec.splitlines()[:450])

    # Load prompt template
    with open(prompt_template_path, "r") as f:
        template = f.read()

    # Format prompt
    prompt = template.format(
        goal="Build a POSIX-compliant bc clone in Python.",
        spec=spec_sliced,
        impl_code="",
        design_context="",
        impl_name="py_bc.py",
        test_name="test_py_bc.py",
    )

    # Call primary model (Gemini)
    print("Calling LLM, please wait...")
    design_doc = call_llm_structured(prompt, DesignDocument, model_name=config.MODEL_PRIMARY)

    print("\n[ALL FIELDS]")
    for field, val in design_doc.model_dump().items():
        print(f"\n=== {field.upper()} ===")
        print(val)

    # Check for specific search keys across all fields
    check_keys = ["lowercase", "uppercase", "case", "A-F", "a-f"]
    found = {}
    for field, val in design_doc.model_dump().items():
        matches = [k for k in check_keys if k.lower() in val.lower()]
        if matches:
            found[field] = matches
    print(f"\nCase-sensitivity key words found: {found}")

    return design_doc


if __name__ == "__main__":
    template_path = os.path.join(os.path.dirname(__file__), "../prompts/generate_design.md")
    run_test(template_path, "Modified Prompt Full Test")
