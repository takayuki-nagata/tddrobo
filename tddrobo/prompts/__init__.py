# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent


def _load_prompt(*paths: str) -> str:
    path = PROMPTS_DIR.joinpath(*paths)
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


# 1. 共通制約の読み込み
BUG_REPORT_COMMON_CONSTRAINTS = _load_prompt("common", "bug_report_constraints.md")
TEST_PLAN_COMMON_CONSTRAINTS = _load_prompt("common", "test_plan_constraints.md")
TEST_PLAN_ORACLE_CONSTRAINTS = _load_prompt("common", "test_plan_oracle_constraints.md")
TEST_PLAN_FIX_COMMON_DIRECTIVE = _load_prompt("common", "test_plan_fix_directive.md")

# 2. 個別プロンプトの読み込み
PLAN_FILES_PROMPT = _load_prompt("plan_files.md")
GENERATE_DESIGN_PROMPT = _load_prompt("generate_design.md")
REVIEW_DESIGN_PROMPT = _load_prompt("review_design.md")
REVIEW_TEST_PLAN_PROMPT = _load_prompt("review_test_plan.md")
GENERATE_README_PROMPT = _load_prompt("generate_readme.md")
GENERATE_REQUIREMENTS_PROMPT = _load_prompt("generate_requirements.md")
GENERATE_UNIT_TESTS_PROMPT = _load_prompt("generate_unit_tests.md")
GENERATE_INTEGRATION_TESTS_PROMPT = _load_prompt("generate_integration_tests.md")
DECIDE_REFACTOR_PROMPT = _load_prompt("decide_refactor.md")
REFACTOR_LOGIC_PROMPT = _load_prompt("refactor_logic.md")
REFACTOR_LOGIC_FIX_PROMPT = _load_prompt("refactor_logic_fix.md")

# 3. 実装プロンプト群の結合
_impl_logic_base = _load_prompt("implement_logic_base.md")
_impl_logic_output = _load_prompt("implement_logic_output.md")

IMPLEMENT_LOGIC_PROMPT_INITIAL = _impl_logic_base + "\n" + _impl_logic_output
IMPLEMENT_LOGIC_PROMPT_SYNTAX_FIX = (
    _impl_logic_base + "\n" + _load_prompt("implement_logic_syntax_fix_context.md") + "\n" + _impl_logic_output
)
IMPLEMENT_LOGIC_PROMPT_FIX = (
    _impl_logic_base + "\n" + _load_prompt("implement_logic_bug_fix_context.md") + "\n" + _impl_logic_output
)

# 4. テスト計画系プロンプトの結合
PLAN_UNIT_TESTS_PROMPT = (
    _load_prompt("plan_unit_tests_base.md")
    .replace("{common_constraints}", TEST_PLAN_COMMON_CONSTRAINTS)
    .replace("{oracle_constraints}", TEST_PLAN_ORACLE_CONSTRAINTS)
)
PLAN_UNIT_TESTS_PROMPT_FIX = (
    PLAN_UNIT_TESTS_PROMPT
    + "\n"
    + _load_prompt("plan_unit_tests_fix_context.md")
    + "\n"
    + TEST_PLAN_FIX_COMMON_DIRECTIVE
)

PLAN_INTEGRATION_TESTS_PROMPT = (
    _load_prompt("plan_integration_tests_base.md")
    .replace("{common_constraints}", TEST_PLAN_COMMON_CONSTRAINTS)
    .replace("{oracle_constraints}", TEST_PLAN_ORACLE_CONSTRAINTS)
)
PLAN_INTEGRATION_TESTS_PROMPT_FIX = (
    PLAN_INTEGRATION_TESTS_PROMPT
    + "\n"
    + _load_prompt("plan_integration_tests_fix_context.md")
    + "\n"
    + TEST_PLAN_FIX_COMMON_DIRECTIVE
)

# 5. バグレポート系プロンプトの結合
GENERATE_UNIT_BUG_REPORT_PROMPT = _load_prompt("generate_unit_bug_report.md").replace(
    "{common_constraints}", BUG_REPORT_COMMON_CONSTRAINTS
)
GENERATE_INTEGRATION_BUG_REPORT_PROMPT = _load_prompt("generate_integration_bug_report.md").replace(
    "{common_constraints}", BUG_REPORT_COMMON_CONSTRAINTS
)
GENERATE_REGRESSION_BUG_REPORT_PROMPT = _load_prompt("generate_regression_bug_report.md").replace(
    "{common_constraints}", BUG_REPORT_COMMON_CONSTRAINTS
)
GENERATE_REFACTOR_BUG_REPORT_PROMPT = _load_prompt("generate_refactor_bug_report.md").replace(
    "{common_constraints}", BUG_REPORT_COMMON_CONSTRAINTS
)

EXTRACT_ORACLE_TARGET_PROMPT = _load_prompt("extract_oracle_target.md")
JUDGE_ORACLE_DISCREPANCY_PROMPT = _load_prompt("judge_oracle_discrepancy.md")

ANALYZE_ARCHITECTURE_PROMPT = _load_prompt("analyze_architecture.md")
IMPLEMENT_LOGIC_REFACTOR_MANDATE = _load_prompt("implement_logic_refactor_mandate.md")
GENERATE_DESIGN_REFACTOR_DIRECTIVE = _load_prompt("generate_design_refactor_directive.md")

PLAN_TESTS_ORACLE_WARNING = _load_prompt("plan_tests_oracle_warning.md")
GENERATE_TESTS_BUG_FIX_CONTEXT = _load_prompt("generate_tests_bug_fix_context.md")
GENERATE_TESTS_SYNTAX_FIX_CONTEXT = _load_prompt("generate_tests_syntax_fix_context.md")
IMPLEMENT_LOGIC_DESIGN_UPDATE_INSTRUCTION = _load_prompt("implement_logic_design_update_instruction.md")
IMPLEMENT_LOGIC_UNIQUENESS_ADVICE = _load_prompt("implement_logic_uniqueness_advice.md")
IMPLEMENT_LOGIC_UNIQUENESS_MULTI_MATCH = _load_prompt("implement_logic_uniqueness_multi_match.md")
IMPLEMENT_LOGIC_UNIQUENESS_MULTI_MATCH_SIMPLE = _load_prompt("implement_logic_uniqueness_multi_match_simple.md")
IMPLEMENT_LOGIC_FAILING_SEARCH_BLOCK_ADVICE = _load_prompt("implement_logic_failing_search_block_advice.md")
REFACTOR_LOGIC_BUG_FIX_WARNING = _load_prompt("refactor_logic_bug_fix_warning.md")
REFACTOR_LOGIC_FALLBACK_WARNING = _load_prompt("refactor_logic_fallback_warning.md")
