# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

from tddrobo import config
from tddrobo.prompts import (
    ANALYZE_ARCHITECTURE_PROMPT,
    GENERATE_DESIGN_PROMPT,
    GENERATE_DESIGN_REFACTOR_DIRECTIVE,
    GENERATE_INTEGRATION_TESTS_PROMPT,
    GENERATE_TESTS_BUG_FIX_CONTEXT,
    GENERATE_TESTS_SYNTAX_FIX_CONTEXT,
    GENERATE_UNIT_TESTS_PROMPT,
    IMPLEMENT_LOGIC_DESIGN_UPDATE_INSTRUCTION,
    IMPLEMENT_LOGIC_FAILING_SEARCH_BLOCK_ADVICE,
    IMPLEMENT_LOGIC_PROMPT_FIX,
    IMPLEMENT_LOGIC_PROMPT_INITIAL,
    IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX,
    IMPLEMENT_LOGIC_REFACTOR_MANDATE,
    IMPLEMENT_LOGIC_UNIQUENESS_ADVICE,
    IMPLEMENT_LOGIC_UNIQUENESS_MULTI_MATCH,
    IMPLEMENT_LOGIC_UNIQUENESS_MULTI_MATCH_SIMPLE,
    PLAN_INTEGRATION_TESTS_PROMPT,
    PLAN_INTEGRATION_TESTS_PROMPT_FIX,
    PLAN_TESTS_ORACLE_WARNING,
    PLAN_UNIT_TESTS_PROMPT,
    PLAN_UNIT_TESTS_PROMPT_FIX,
    REFACTOR_LOGIC_BUG_FIX_WARNING,
    REFACTOR_LOGIC_FIX_PROMPT,
    REFACTOR_LOGIC_PROMPT,
)
from tddrobo.utils import add_line_numbers


def build_test_plan_prompt(state, phase, target_req_str, oracle_constraints, is_fix=False, review_feedback=""):
    """Assemble prompt for planning unit or integration tests."""
    goal = state.get("goal", "")
    spec = state.get("spec_content", "")
    design_doc = state.get("design_doc", "")

    if is_fix:
        existing_plan = state.get("unit_test_plan" if phase == "unit" else "integration_test_plan", "")
        template = PLAN_UNIT_TESTS_PROMPT_FIX if phase == "unit" else PLAN_INTEGRATION_TESTS_PROMPT_FIX
        prompt = template.format(
            goal=goal,
            spec=spec,
            design_doc=design_doc,
            target_req=target_req_str,
            test_plan=existing_plan,
            test_plan_review=review_feedback,
            oracle_constraints=oracle_constraints,
        )
    else:
        template = PLAN_UNIT_TESTS_PROMPT if phase == "unit" else PLAN_INTEGRATION_TESTS_PROMPT
        prompt = template.format(
            goal=goal,
            spec=spec,
            design_doc=design_doc,
            target_req=target_req_str,
            oracle_constraints=oracle_constraints,
        )

    bug_report = state.get("bug_report", "")
    if bug_report:
        prompt += "\n\n" + PLAN_TESTS_ORACLE_WARNING.format(bug_report=bug_report)

    return prompt


def build_test_generation_prompt(state, phase, target_req_str, previous_tests, bug_report, tests_check_output):
    """Assemble prompt for generating unit or integration test code."""
    design_doc = state.get("design_doc", "")
    impl_code = state.get("impl_code", "")

    if phase == "unit":
        prompt = GENERATE_UNIT_TESTS_PROMPT.format(
            target_req=target_req_str,
            design_doc=design_doc,
            unit_test_plan=state.get("unit_test_plan", ""),
            impl_code=impl_code,
        )
    else:
        prompt = GENERATE_INTEGRATION_TESTS_PROMPT.format(
            target_req=target_req_str,
            design_doc=design_doc,
            integration_test_plan=state.get("integration_test_plan", ""),
            impl_code=impl_code,
        )

    if bug_report:
        prompt += "\n\n" + GENERATE_TESTS_BUG_FIX_CONTEXT.format(previous_tests=previous_tests, bug_report=bug_report)

    if tests_check_output:
        prompt += "\n\n" + GENERATE_TESTS_SYNTAX_FIX_CONTEXT.format(
            previous_tests=previous_tests, tests_check_output=tests_check_output
        )

    return prompt


def build_uniqueness_advice(impl_check_output, existing_impl):
    """Build guidance if Search/Replace fails to match due to non-uniqueness."""
    import re

    if "Failed to apply Search/Replace block" not in impl_check_output:
        return ""

    advice = "\n\n" + IMPLEMENT_LOGIC_UNIQUENESS_ADVICE

    if "matches multiple times" in impl_check_output:
        line_nums = []
        match = re.search(r"Matches found at line numbers: \[(.*?)\]", impl_check_output)
        if match:
            line_nums = [int(x.strip()) for x in match.group(1).split(",") if x.strip()]
        context_hints = []
        if line_nums and existing_impl:
            lines = existing_impl.splitlines()
            indices = [min(len(lines) - 1, ln - 1) for ln in line_nums if ln > 0]
            if indices:
                first_idx = indices[0]
                block_len = 1
                while True:
                    all_same = True
                    for other_idx in indices:
                        if first_idx + block_len >= len(lines) or other_idx + block_len >= len(lines):
                            all_same = False
                            break
                        if lines[first_idx + block_len] != lines[other_idx + block_len]:
                            all_same = False
                            break
                    if all_same:
                        block_len += 1
                    else:
                        break

                pre_offset_is_distinguishing = {}
                for d in [-3, -2, -1]:
                    pre_lines_at_offset = []
                    for other_idx in indices:
                        o_idx = other_idx + d
                        if 0 <= o_idx < len(lines):
                            pre_lines_at_offset.append(lines[o_idx])
                        else:
                            pre_lines_at_offset.append(None)
                    pre_offset_is_distinguishing[d] = len(set(pre_lines_at_offset)) > 1

                post_offset_is_distinguishing = {}
                for d in [0, 1, 2]:
                    post_lines_at_offset = []
                    for other_idx in indices:
                        o_idx = other_idx + block_len + d
                        if 0 <= o_idx < len(lines):
                            post_lines_at_offset.append(lines[o_idx])
                        else:
                            post_lines_at_offset.append(None)
                    post_offset_is_distinguishing[d] = len(set(post_lines_at_offset)) > 1

                for line_num in line_nums:
                    idx = min(len(lines) - 1, line_num - 1)
                    if idx < 0:
                        continue
                    pre_start_idx = idx
                    for d in [-3, -2, -1]:
                        if pre_offset_is_distinguishing.get(d, False):
                            pre_start_idx = idx + d
                            break
                    post_end_idx = idx + block_len
                    for d in reversed([0, 1, 2]):
                        if post_offset_is_distinguishing.get(d, False):
                            post_end_idx = idx + block_len + d + 1
                            break
                    if pre_start_idx == idx and post_end_idx == idx + block_len:
                        pre_start_idx = max(0, idx - 2)
                    search_lines = lines[pre_start_idx:post_end_idx]
                    search_block_text = "\n".join(search_lines)
                    hint = (
                        f"- For the match at line {line_num}, you MUST use this unique SEARCH block template "
                        f"(do NOT omit context lines):\n"
                        f"<<<<<<< SEARCH\n{search_block_text}\n=======\n"
                    )
                    context_hints.append(hint)

        if context_hints:
            joined_hints = "\n".join(context_hints)
            advice += "\n" + IMPLEMENT_LOGIC_UNIQUENESS_MULTI_MATCH.format(joined_hints=joined_hints)
        else:
            advice += "\n" + IMPLEMENT_LOGIC_UNIQUENESS_MULTI_MATCH_SIMPLE

    return advice


def build_implementation_prompt(
    state,
    phase,
    target_req_str,
    existing_impl,
    existing_impl_code_param,
    tests_code,
    domain_tips,
    python_tips,
    impl_check_output,
    impl_name,
    design_updated,
):
    """Assemble prompt for writing implementation code."""
    import re

    bug_report = state.get("bug_report", "")

    if impl_check_output and state.get("impl_code"):
        template = IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX
        prompt = template.format(
            goal=state.get("goal", ""),
            design=state.get("design_doc", ""),
            tests_code=tests_code,
            impl_name=impl_name,
            impl_check_output=impl_check_output,
            bug_report=state.get("bug_report", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=add_line_numbers(existing_impl_code_param),
            domain_tips=domain_tips,
            python_tips=python_tips,
        )
    elif bug_report and state.get("impl_code"):
        template = IMPLEMENT_LOGIC_PROMPT_FIX
        prompt = template.format(
            goal=state.get("goal", ""),
            design=state.get("design_doc", ""),
            tests_code=tests_code,
            impl_name=impl_name,
            bug_report=state.get("bug_report", ""),
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl_code_param,
            domain_tips=domain_tips,
            python_tips=python_tips,
            impl_check_output=impl_check_output,
        )
    else:
        template = IMPLEMENT_LOGIC_PROMPT_INITIAL
        prompt = template.format(
            goal=state.get("goal", ""),
            design=state.get("design_doc", ""),
            tests_code=tests_code,
            impl_name=impl_name,
            requirements_list_str=state.get("requirements_list_str", ""),
            target_requirement=target_req_str,
            existing_impl_code=existing_impl_code_param,
            domain_tips=domain_tips,
            python_tips=python_tips,
        )

    if state.get("loop_detected"):
        prompt += "\n\n" + IMPLEMENT_LOGIC_REFACTOR_MANDATE

    if design_updated:
        prompt += "\n\n" + IMPLEMENT_LOGIC_DESIGN_UPDATE_INSTRUCTION.format(existing_impl=existing_impl)

    if impl_check_output:
        cleaned_check_output = (
            impl_check_output.replace("<<<<<<< SEARCH", "[PREVIOUS SEARCH]")
            .replace("=======", "[PREVIOUS DIVIDER]")
            .replace(">>>>>>> REPLACE", "[PREVIOUS REPLACE]")
        )
        advice = build_uniqueness_advice(cleaned_check_output, existing_impl)
        cleaned_check_output += advice

        failed_block_match = re.search(
            r"Target SEARCH block that failed to match:\n(.*?)(?=\n\[TDD Robo\]|\Z)",
            state.get("impl_check_output", ""),
            re.DOTALL,
        )
        if failed_block_match:
            failed_block_text = failed_block_match.group(1).strip()
            cleaned_check_output += "\n\n" + IMPLEMENT_LOGIC_FAILING_SEARCH_BLOCK_ADVICE.format(
                failed_block_text=failed_block_text
            )
        prompt += f"\n\n# 🚨 CRITICAL: PREVIOUS MODIFICATION CHECK FAILED!\n{cleaned_check_output}"

    return prompt


def build_design_prompt(state, target_req_id, design_context, impl_code_for_design):
    """Assemble prompt for writing/updating the design document."""
    goal = state.get("goal", "")
    spec = state.get("spec_content", "")
    impl_name = state.get("module_name", config.DEFAULT_IMPL_NAME)
    test_name = state.get("test_module_name", config.DEFAULT_TEST_NAME)

    prompt = GENERATE_DESIGN_PROMPT.format(
        goal=goal,
        spec=spec,
        impl_name=impl_name,
        test_name=test_name,
        design_context=design_context,
        impl_code=impl_code_for_design,
    )

    if state.get("loop_detected"):
        prompt += "\n\n" + GENERATE_DESIGN_REFACTOR_DIRECTIVE.format(
            architecture_audit=state.get("architecture_audit", "")
        )

    return prompt


def build_refactor_prompt(state, reasons_str, bug_report, python_tips, existing_impl, iters, impl_name):
    """Assemble prompt for code refactoring node."""
    if bug_report:
        reasons_with_bug = reasons_str + "\n\n" + REFACTOR_LOGIC_BUG_FIX_WARNING.format(bug_report=bug_report)
        prompt = REFACTOR_LOGIC_FIX_PROMPT.format(
            design_doc=state.get("design_doc", ""),
            existing_impl_code=existing_impl,
            bug_report=bug_report,
            python_tips=python_tips,
        )
    else:
        reasons_with_bug = reasons_str
        prompt = REFACTOR_LOGIC_PROMPT.format(
            design_doc=state.get("design_doc", ""),
            impl_code=existing_impl,
            refactoring_reasons=reasons_str,
            python_tips=python_tips,
        )
    return prompt, reasons_with_bug


def build_architecture_audit_prompt(spec_content, target_req_str, impl_code, tests_code, test_output):
    """Assemble prompt for architectural audit node."""
    return ANALYZE_ARCHITECTURE_PROMPT.format(
        spec_content=spec_content,
        target_req_str=target_req_str,
        impl_code=impl_code,
        tests_code=tests_code,
        test_output=test_output,
    )
