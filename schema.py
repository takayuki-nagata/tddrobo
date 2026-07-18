import re
from typing import Literal, TypedDict

from pydantic import BaseModel, Field, field_validator


class TDDState(TypedDict, total=False):
    """Represents the state of the TDD workflow, passed between nodes."""

    goal: str
    spec_url: str
    spec_content: str
    requirements: list[dict]
    current_req_index: int
    requirements_list_str: str
    design_doc: str
    test_plan: str
    test_plan_review: str
    test_plan_review_decision: str
    test_plan_iterations: int
    module_name: str
    test_module_name: str
    tests_code: str
    tests_check_output: str
    test_review: str
    test_review_decision: str
    test_iterations: int
    impl_code: str
    impl_check_output: str
    test_output: str
    bug_report: str
    next_action: str
    iterations: int
    max_iterations: int
    max_test_plan_iterations: int
    max_test_iterations: int
    target_test_plan_coverage: int
    target_test_coverage: int
    target_design_quality: int
    success: bool
    readme_content: str
    domain_tips: str
    python_tips: str
    loop_detected: bool
    design_updated: bool
    design_iterations: int
    syntax_error_iterations: int
    impl_updated: bool
    test_syntax_error_iterations: int
    unit_test_plan: str
    integration_test_plan: str
    unit_tests_code: str
    integration_tests_code: str
    refactor_decision: str
    unit_test_iterations: int
    integration_test_iterations: int
    regression_test_iterations: int
    refactor_iterations: int
    reasons: list[str]
    last_green_impl_code: str
    stagnant_iterations: int
    last_test_summary: str
    design_review_feedback: str
    design_review_iterations: int
    regression_failure_policy: str
    oracle_discrepancy_only: bool
    failed_files: list[str]
    rollback_counts: dict[str, int]


class DesignReviewReport(BaseModel):
    estimated_quality: int
    comments: str


class FilePlan(BaseModel):
    impl_filename: str
    test_filename: str


class DesignDocument(BaseModel):
    module_responsibilities: str
    architecture_and_components: str
    interface_definitions: str
    data_structures: str
    logic_and_algorithms: str
    edge_cases_and_limitations: str
    error_handling: str
    command_line_interface: str


class TestCase(BaseModel):
    __test__ = False
    action: str = Field(description="Extremely concise description of the action. Max 10 words. DO NOT repeat words.")
    expected_outcome: str = Field(description="Extremely concise expected outcome. Max 10 words. DO NOT repeat words.")
    oracle_expression: str | None = Field(
        default=None,
        description=(
            "Optional semicolon-separated expression sequence for dynamic oracle verification (e.g. 'scale=5; scale')."
        ),
    )
    oracle_expected: str | None = Field(
        default=None,
        description="Optional expected output value from oracle (e.g. '5').",
    )

    @field_validator("oracle_expected")
    @classmethod
    def validate_oracle_expected(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v_stripped = v.strip()
        forbidden_keywords = {"prints", "outputs", "returns", "expected", "should", "actual", "value", "stdout"}

        # Split by lines and validate each line individually to allow multi-line outputs (e.g., 'A\nA\n10')
        lines = [line.strip() for line in v_stripped.splitlines() if line.strip()]
        for line in lines:
            words = [w.lower() for w in re.split(r"\s+", line) if w]
            has_forbidden = any(kw in words for kw in forbidden_keywords)
            if has_forbidden:
                raise ValueError(
                    f"oracle_expected must be a raw string output from the oracle (e.g., '16.2', '0.00', 'error'), "
                    f"not a natural language description. If the expression produces no output in POSIX bc (such as "
                    f"comments or variable assignments without prints), oracle_expected must be an empty string ''. "
                    f"Got: '{v}'"
                )
        return v_stripped


class TestPlan(BaseModel):
    __test__ = False
    test_cases: list[TestCase]


class TestPlanReviewReport(BaseModel):
    __test__ = False
    missing_test_cases: list[str]
    estimated_coverage: int
    feedback: str


class TestReviewReport(BaseModel):
    __test__ = False
    missing_test_cases: list[str]
    estimated_coverage: int
    feedback: str


class BugReport(BaseModel):
    failed_test_cases: list[str]
    expected_vs_actual: str
    fix_instructions: str
    target_to_fix: Literal["implement_logic", "generate_tests", "generate_design"]
    target_req: str | None = Field(
        default=None,
        description=(
            "The specific requirement ID to target for fix/rollback (e.g., 'REQxxx'). "
            "Must point to the requirement where the test case design or logic bug actually belongs. "
            "Leave null if not applicable."
        ),
    )


class VerifiedTestCase(BaseModel):
    __test__ = False
    action: str = Field(description="The action being tested.")
    verified_expected_outcome: str = Field(
        description="The correct expected outcome verified by the run_bc_command tool."
    )


class VerifiedTestPlan(BaseModel):
    __test__ = False
    test_cases: list[VerifiedTestCase]


class CalculationTestPlanItem(BaseModel):
    __test__ = False
    test_case_number: int = Field(description="The exact case number from the Test Plan list, e.g. 1, 2, 129.")
    expression: str = Field(description="The exact bc mathematical expression to compute, e.g., 'scale=2; 1/3'")


class CalculationTestPlan(BaseModel):
    __test__ = False
    items: list[CalculationTestPlanItem]


class RequirementSpec(BaseModel):
    id: str = Field(description="REQ001, REQ002, etc.")
    description: str = Field(description="Requirement description")


class RequirementsList(BaseModel):
    requirements: list[RequirementSpec] = Field(description="List of verified requirements.")


class RefactorDecision(BaseModel):
    chain_of_thought: str
    refactor_needed: bool
    reasons: list[str]


class OracleAssertionTarget(BaseModel):
    __test__ = False
    expression: str = Field(description="The mathematical expression or statement sequence to evaluate.")
    expected: str = Field(description="The expected string or numeric output of the expression.")
    preceding: list[str] = Field(
        default_factory=list,
        description="Preceding execution statements required to set up the state (e.g. scale=2, ibase=8).",
    )


class OracleDiscrepancyJudgment(BaseModel):
    is_design_error: bool = Field(
        description=(
            "True if the mismatch is due to a flaw or gap in the Design Document (design.md) itself, "
            "requiring a design update. False if it is a minor notation difference or a simple typo/mistake "
            "in the test plan."
        )
    )
    reason: str = Field(
        description="Detailed explanation of why this mismatch is categorized as a design error or a test plan error."
    )
    corrected_expected: str | None = Field(
        default=None,
        description=(
            "The corrected expected value if it is a test plan error (e.g. '16.2' instead of 'Prints 16.2'). "
            "Should align with the actual oracle output."
        ),
    )
