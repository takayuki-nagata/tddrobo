from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class TDDState(TypedDict, total=False):
    """Represents the state of the TDD workflow, passed between nodes."""

    goal: str
    spec_url: str
    spec_content: str
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
    success: bool
    readme_content: str


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


class TestPlan(BaseModel):
    test_cases: list[TestCase]


class TestPlanReviewReport(BaseModel):
    missing_test_cases: list[str]
    estimated_coverage: int
    feedback: str


class TestReviewReport(BaseModel):
    missing_test_cases: list[str]
    estimated_coverage: int
    feedback: str


class BugReport(BaseModel):
    failed_test_cases: list[str]
    expected_vs_actual: str
    fix_instructions: str
    target_to_fix: Literal["implement_logic", "generate_tests"]
