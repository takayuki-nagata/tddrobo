import pytest
from pydantic import ValidationError

from schema import FilePlan, TestCase


def test_file_plan_schema():
    plan = FilePlan(impl_filename="app.py", test_filename="test_app.py")
    assert plan.impl_filename == "app.py"
    assert plan.test_filename == "test_app.py"


def test_test_case_validation():
    # Valid case
    tc = TestCase(action="Do something", expected_outcome="Success")
    assert tc.action == "Do something"

    # Invalid case (missing required field)
    with pytest.raises(ValidationError):
        TestCase(action="Missing expected outcome")  # type: ignore
