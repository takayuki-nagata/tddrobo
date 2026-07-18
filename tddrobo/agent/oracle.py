# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import json
import re

from tddrobo import config
from tddrobo.logger import print
from tddrobo.prompts import EXTRACT_ORACLE_TARGET_PROMPT
from tddrobo.schema import OracleAssertionTarget, OracleDiscrepancyJudgment, TDDState

from .runner import (
    _call_llm_structured,
    _extract_failing_line,
    _extract_method_body,
    _find_failed_methods,
)

MODEL_PRIMARY = config.MODEL_PRIMARY


def _run_early_oracle_verification(test_plan_json: str) -> list[dict]:
    """Parse test plan JSON and verify calculation expectations against dynamic oracle."""
    verifier = getattr(config, "ORACLE_VERIFIER", None)
    if verifier is None:
        return []

    try:
        data = json.loads(test_plan_json)
    except Exception:
        return []

    mismatches = []
    for tc in data.get("test_cases", []):
        action = tc.get("action", "")
        expected = tc.get("expected_outcome", "")
        oracle_expr = tc.get("oracle_expression")
        oracle_expected = tc.get("oracle_expected")

        expr = None
        expected_val = None

        if oracle_expr and oracle_expected:
            expr = oracle_expr
            expected_val = oracle_expected
        else:
            match = re.search(r"\[Evaluate:\s*(.*?)\s*\]", expected)
            if not match:
                match = re.search(r"\[Evaluate:\s*(.*?)\s*\]", action)

            if match:
                expr = match.group(1)
                cleaned_expected = expected.replace(match.group(0), "").strip()
                num_match = re.search(r"\b\d+(?:\.\d+)?\b", cleaned_expected)
                if num_match:
                    expected_val = num_match.group(0)
            else:
                quote_match = re.search(r"['\"](.*?)['\"]", action)
                if quote_match:
                    expr = quote_match.group(1)
                else:
                    math_match = re.search(r"\b\d+[\s\d+\-*/%;=.]+\d+\b", action)
                    if math_match:
                        expr = math_match.group(0)

                num_match = re.search(r"\b\d+(?:\.\d+)?\b", expected)
                if num_match:
                    expected_val = num_match.group(0)

        if expr and expected_val:
            expr_cleaned = expr.strip().rstrip(".")
            try:
                float(expr_cleaned)
                continue
            except ValueError:
                pass
            cleaned_temp = re.sub(r"\\\\[a-zA-Z]", "", expr_cleaned)
            cleaned_temp = re.sub(r"\b(scale|ibase|obase|last)\b", "", cleaned_temp, flags=re.IGNORECASE)
            cleaned_temp = re.sub(r"\b[sclaej]\s*\(", "(", cleaned_temp, flags=re.IGNORECASE)
            if re.search(r"[a-zA-Z]", cleaned_temp):
                continue
            try:
                try:
                    oracle_result = verifier(expr_cleaned, expected=expected_val).strip()
                except TypeError:
                    oracle_result = verifier(expr_cleaned).strip()
                oracle_lines = [l.strip() for l in oracle_result.splitlines() if l.strip()]
                last_oracle_val = "\n".join(oracle_lines) if oracle_lines else ""

                expects_error = any(w in expected_val.lower() for w in ["error", "exception", "syntax", "fail"])
                oracle_has_error = "error" in oracle_result.lower() or "exception" in oracle_result.lower()
                is_context_missing = "undefined" in oracle_result.lower()

                if oracle_has_error:
                    if is_context_missing:
                        continue
                    elif expects_error:
                        continue
                    else:
                        mismatches.append(
                            {
                                "test_case": tc,
                                "expr": expr_cleaned,
                                "expected_val": expected_val,
                                "oracle_val": last_oracle_val,
                            }
                        )
                else:
                    if expects_error:
                        mismatches.append(
                            {
                                "test_case": tc,
                                "expr": expr_cleaned,
                                "expected_val": expected_val,
                                "oracle_val": last_oracle_val,
                            }
                        )
                    else:
                        if last_oracle_val != expected_val:
                            mismatches.append(
                                {
                                    "test_case": tc,
                                    "expr": expr_cleaned,
                                    "expected_val": expected_val,
                                    "oracle_val": last_oracle_val,
                                }
                            )
            except Exception:
                pass

    return mismatches


def _judge_oracle_discrepancy_with_llm(
    design_doc: str, test_case_dict: dict, oracle_val: str
) -> OracleDiscrepancyJudgment:
    """Use the primary LLM to judge whether an oracle discrepancy is a core design
    flaw or a test plan representation issue.
    """
    from tddrobo.prompts import JUDGE_ORACLE_DISCREPANCY_PROMPT

    tc_formatted = json.dumps(test_case_dict, indent=2, ensure_ascii=False)

    prompt = JUDGE_ORACLE_DISCREPANCY_PROMPT.format(
        design_doc=design_doc, test_case=tc_formatted, actual_output=oracle_val
    )

    try:
        judgment = _call_llm_structured(prompt, OracleDiscrepancyJudgment, model_name=MODEL_PRIMARY)
        return judgment
    except Exception as e:
        print(f"Warning: Failed to call LLM Judge for discrepancy, defaulting to Design Error: {e}")
        return OracleDiscrepancyJudgment(
            is_design_error=True, reason=f"LLM Judge call failed: {str(e)}", corrected_expected=None
        )


def _extract_oracle_target_llm(
    method_body: str,
    failing_line_in_body: str | None,
) -> tuple[str | None, str | None, list[str]]:
    """Extract target expression and expected value using LLM reasoning model."""
    try:
        prompt = EXTRACT_ORACLE_TARGET_PROMPT.format(
            method_body=method_body,
            failing_line=failing_line_in_body or "",
        )
        res = _call_llm_structured(prompt, OracleAssertionTarget, model_name=MODEL_PRIMARY)
        expr = res.expression if res.expression else None
        expected = res.expected if res.expected else None
        preceding = res.preceding if res.preceding else []
        return expr, expected, preceding
    except Exception as e:  # pragma: no cover
        print(f"Warning: Failed to extract oracle target via LLM: {e}")
        return None, None, []


def _run_oracle_verification_on_failures(test_output: str, tests_code: str, state: TDDState | None = None) -> str:
    """Parse test failures, extract expressions from the failed test cases, and verify
    them using the registered dynamic oracle verifier.
    """
    verifier = config.ORACLE_VERIFIER
    if verifier is None:
        return ""

    failed_methods = _find_failed_methods(test_output, state)
    if not failed_methods:
        return ""

    feedback_lines = []

    for method in failed_methods:
        method_body = _extract_method_body(method, tests_code)
        if not method_body:
            continue

        failing_line_in_body = _extract_failing_line(method, test_output, state)
        expr, expected_val, preceding_exprs = _extract_oracle_target_llm(method_body, failing_line_in_body)

        if expr and expected_val:
            clean_expected = expected_val.strip().replace("\\n", "\n").strip()
            if clean_expected not in ("correct_val", "different_val", "unresolved_var"):
                if re.search(r"[g-zG-Z_]", clean_expected):
                    continue

            if preceding_exprs and preceding_exprs[-1] == expr:
                preceding_exprs_cleaned = preceding_exprs[:-1]
            else:
                preceding_exprs_cleaned = preceding_exprs

            if not preceding_exprs_cleaned:
                expr_cleaned = expr.strip().rstrip(".")
                try:
                    float(expr_cleaned)
                    continue
                except ValueError:
                    pass
            cleaned_temp = re.sub(r"\\[a-zA-Z]", "", expr)
            cleaned_temp = re.sub(r"\b(scale|ibase|obase|last)\b", "", cleaned_temp, flags=re.IGNORECASE)
            cleaned_temp = re.sub(r"\b[sclaej]\s*\(", "(", cleaned_temp, flags=re.IGNORECASE)
            cleaned_temp = re.sub(r"\b[a-z]\b", "", cleaned_temp)
            if re.search(r"[a-zA-Z]", cleaned_temp):
                continue
            try:
                if preceding_exprs_cleaned:
                    combined_expr = "; ".join(preceding_exprs_cleaned) + "; " + expr
                else:
                    combined_expr = expr

                try:
                    oracle_result = verifier(combined_expr, expected=expected_val).strip()
                except TypeError:
                    oracle_result = verifier(combined_expr).strip()
                expected_lines = [l.strip() for l in expected_val.splitlines() if l.strip()]
                num_expected_lines = len(expected_lines) if expected_lines else 1

                oracle_lines = [l.strip() for l in oracle_result.splitlines() if l.strip()]
                last_oracle_lines = oracle_lines[-num_expected_lines:] if oracle_lines else []
                last_oracle_val = "\n".join(last_oracle_lines) if last_oracle_lines else ""

                clean_oracle = last_oracle_val.strip().replace("\\n", "\n").strip()
                clean_expected = expected_val.strip().replace("\\n", "\n").strip()

                if "Error" not in oracle_result and "Exception" not in oracle_result:
                    if clean_oracle != clean_expected:
                        feedback_lines.append(
                            f"- Test case `{method}` contains an assertion error:\n"
                            f"  * Expression evaluated: `{combined_expr}`\n"
                            f"  * Expected value hardcoded in test: `{clean_expected}`\n"
                            f"  * Actual correct oracle value: `{clean_oracle}`\n"
                            f"  * Rationale: The test expectation `{clean_expected}` is mathematically INCORRECT. "
                            f"The correct output should be `{clean_oracle}`. Therefore, the bug is in the test "
                            f"code itself, not the implementation logic."
                        )
            except Exception:
                pass

    if feedback_lines:
        feedback_text = (
            "\n### ORACLE VERIFICATION FEEDBACK\n"
            "The verification oracle has analyzed the failed test cases and found the "
            "following discrepancies in test expectations:\n"
            + "\n".join(feedback_lines)
            + "\n\nCRITICAL DIRECTIVE: If the oracle shows a discrepancy, the test code assertion is wrong. "
            'You MUST specify `target_to_fix` as "generate_tests" to correct the test code.\n'
        )
        return feedback_text

    return "No assertion discrepancies detected by the oracle."
