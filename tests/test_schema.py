# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import pytest
from pydantic import ValidationError

from tddrobo.schema import FilePlan, TestCase


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


def test_test_case_oracle_validation():
    # Valid raw values
    tc1 = TestCase(action="eval", expected_outcome="ok", oracle_expected="16.20")
    assert tc1.oracle_expected == "16.20"

    tc2 = TestCase(action="eval", expected_outcome="ok", oracle_expected=".45")
    assert tc2.oracle_expected == ".45"

    tc3 = TestCase(action="eval", expected_outcome="ok", oracle_expected="error")
    assert tc3.oracle_expected == "error"

    tc4 = TestCase(action="eval", expected_outcome="ok", oracle_expected=None)
    assert tc4.oracle_expected is None

    # Valid multi-line outputs (e.g. hex outputs)
    tc5 = TestCase(action="eval", expected_outcome="ok", oracle_expected="A\nA\n10")
    assert tc5.oracle_expected == "A\nA\n10"

    tc6 = TestCase(action="eval", expected_outcome="ok", oracle_expected="F\nB\n0")
    assert tc6.oracle_expected == "F\nB\n0"

    # Invalid descriptions with natural language or descriptive keywords
    with pytest.raises(ValidationError):
        TestCase(action="eval", expected_outcome="ok", oracle_expected="Prints 4.00")

    with pytest.raises(ValidationError):
        TestCase(action="eval", expected_outcome="ok", oracle_expected="Outputs 0")

    with pytest.raises(ValidationError):
        TestCase(action="eval", expected_outcome="ok", oracle_expected="should be negative value")

    # Invalid multi-line containing natural language description in any line
    with pytest.raises(ValidationError):
        TestCase(action="eval", expected_outcome="ok", oracle_expected="A\nOutputs A\n10")

    with pytest.raises(ValidationError):
        TestCase(action="eval", expected_outcome="ok", oracle_expected="A\nshould be 10\n10")
