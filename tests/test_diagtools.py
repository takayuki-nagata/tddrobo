# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Takayuki Nagata

import json
import os
import sys
from unittest.mock import patch

import pytest

# Ensure diagtools is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tddrobo.diagtools.inspect_checkpoint import main as inspect_checkpoint_main
from tddrobo.diagtools.replay_prompt import main


def test_replay_prompt_missing_file(capsys):
    test_args = ["replay_prompt.py", "--trace", "non_existent_file.txt"]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Trace file not found" in captured.err


def test_replay_prompt_invalid_format(tmp_path, capsys):
    trace_file = tmp_path / "invalid_trace.txt"
    # A valid Python literal (list), but not a dict
    trace_file.write_text("['not', 'a', 'dict']", encoding="utf-8")

    test_args = ["replay_prompt.py", "--trace", str(trace_file)]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Invalid trace request file format" in captured.err


@patch("tddrobo.utils.call_llm_with_reasoning")
@patch("tddrobo.utils.call_llm_standard")
def test_replay_prompt_success(mock_call_standard, mock_call_reasoning, tmp_path, capsys):
    mock_call_reasoning.return_value = "reasoning response"
    mock_call_standard.return_value = "standard response"

    # Create dummy trace files
    trace_dict = {"model_name": "gemma-4-31b-it", "prompt": "hello world"}
    trace_file_ast = tmp_path / "trace_ast.txt"
    trace_file_ast.write_text(str(trace_dict), encoding="utf-8")

    trace_file_json = tmp_path / "trace_json.txt"
    trace_file_json.write_text(json.dumps(trace_dict), encoding="utf-8")

    # 1. Test reasoning model via AST parsing
    test_args_ast = ["replay_prompt.py", "--trace", str(trace_file_ast)]
    with patch("sys.argv", test_args_ast):
        main()
        captured = capsys.readouterr()
        assert "reasoning response" in captured.out
        mock_call_reasoning.assert_called_once_with(
            "hello world", response_schema=None, thinking_level="MINIMAL", temperature=0.0
        )

    mock_call_reasoning.reset_mock()

    # 2. Test standard model via JSON parsing
    test_args_json = [
        "replay_prompt.py",
        "--trace",
        str(trace_file_json),
        "--standard",
    ]
    with patch("sys.argv", test_args_json):
        main()
        captured = capsys.readouterr()
        assert "standard response" in captured.out
        mock_call_standard.assert_called_once_with("hello world", response_schema=None, temperature=0.0)

    mock_call_standard.reset_mock()

    # 3. Test replacements
    test_args_replace = [
        "replay_prompt.py",
        "--trace",
        str(trace_file_ast),
        "--replace",
        "world",
        "gemini",
        "--standard",
    ]
    with patch("sys.argv", test_args_replace):
        main()
        captured = capsys.readouterr()
        assert "standard response" in captured.out
        mock_call_standard.assert_called_once_with("hello gemini", response_schema=None, temperature=0.0)


@patch("tddrobo.utils.call_llm_standard")
def test_replay_prompt_llm_error(mock_call_standard, tmp_path, capsys):
    mock_call_standard.side_effect = Exception("API Error")

    trace_dict = {"model_name": "gemma-4-26b-a4b-it", "prompt": "test"}
    trace_file = tmp_path / "trace.txt"
    trace_file.write_text(str(trace_dict), encoding="utf-8")

    test_args = ["replay_prompt.py", "--trace", str(trace_file)]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error calling LLM: API Error" in captured.err


@patch("tddrobo.utils.call_llm_with_reasoning")
@patch("tddrobo.utils.call_llm_standard")
def test_replay_prompt_extended(mock_call_standard, mock_call_reasoning, tmp_path, capsys):
    mock_call_reasoning.return_value = "reasoning response"
    mock_call_standard.return_value = "standard response"

    # Create dummy trace files
    trace_dict = {"model_name": "gemma-4-31b-it", "prompt": "hello world"}
    trace_file = tmp_path / "trace.txt"
    trace_file.write_text(str(trace_dict), encoding="utf-8")

    # 1. Test replace-file mapping
    replace_dict = {"world": "earth"}
    replace_file = tmp_path / "replace.json"
    replace_file.write_text(json.dumps(replace_dict), encoding="utf-8")

    output_file = tmp_path / "response.txt"

    test_args = [
        "replay_prompt.py",
        "--trace",
        str(trace_file),
        "--replace-file",
        str(replace_file),
        "--model",
        "primary",
        "--output",
        str(output_file),
    ]

    with patch("sys.argv", test_args):
        main()
        captured = capsys.readouterr()
        assert "reasoning response" in captured.out
        mock_call_reasoning.assert_called_once_with(
            "hello earth", response_schema=None, thinking_level="MINIMAL", temperature=0.0
        )
        assert output_file.read_text(encoding="utf-8") == "reasoning response"

    mock_call_reasoning.reset_mock()

    # 2. Test replace-file invalid (not a dict)
    replace_file_invalid = tmp_path / "replace_invalid.json"
    replace_file_invalid.write_text("[]", encoding="utf-8")

    test_args_invalid = [
        "replay_prompt.py",
        "--trace",
        str(trace_file),
        "--replace-file",
        str(replace_file_invalid),
    ]
    with patch("sys.argv", test_args_invalid):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    # 3. Test replace-file missing / read error
    test_args_missing = [
        "replay_prompt.py",
        "--trace",
        str(trace_file),
        "--replace-file",
        "non_existent.json",
    ]
    with patch("sys.argv", test_args_missing):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    # 4. Test model standard override via gemma-4-26b-a4b-it
    test_args_secondary = [
        "replay_prompt.py",
        "--trace",
        str(trace_file),
        "--model",
        "gemma-4-26b-a4b-it",
    ]
    with patch("sys.argv", test_args_secondary):
        main()
        captured = capsys.readouterr()
        assert "standard response" in captured.out
        mock_call_standard.assert_called_once_with("hello world", response_schema=None, temperature=0.0)


# inspect_checkpoint tests
def test_inspect_checkpoint_missing_file():
    test_args = ["inspect_checkpoint.py", "--checkpoint", "non_existent.pkl"]
    with patch("sys.argv", test_args):
        with pytest.raises(SystemExit) as excinfo:
            inspect_checkpoint_main()
        assert excinfo.value.code == 1


@patch("tddrobo.diagtools.inspect_checkpoint.FileMemorySaver")
@patch("tddrobo.diagtools.inspect_checkpoint.TDDAgent")
def test_inspect_checkpoint_keys_and_dump(mock_agent_class, mock_saver_class, capsys):
    mock_agent = mock_agent_class.return_value
    mock_state_obj = mock_agent.app.get_state.return_value
    mock_state_obj.values = {
        "goal": "test goal",
        "iterations": 5,
        "success": True,
        "requirements": [{"id": "REQ001", "description": "req desc"}],
    }

    with patch("os.path.exists", return_value=True):
        # 1. Test --list-keys
        test_args_list = ["inspect_checkpoint.py", "--list-keys"]
        with patch("sys.argv", test_args_list):
            inspect_checkpoint_main()
            captured = capsys.readouterr()
            assert "🔑 Checkpoint State Keys:" in captured.out
            assert "  - goal" in captured.out

        # 2. Test --dump-key (non-existent)
        test_args_dump_missing = ["inspect_checkpoint.py", "--dump-key", "non_existent_key"]
        with patch("sys.argv", test_args_dump_missing):
            with pytest.raises(SystemExit) as excinfo:
                inspect_checkpoint_main()
            assert excinfo.value.code == 1
            captured = capsys.readouterr()
            assert "Error: Key 'non_existent_key' not found in state." in captured.err

        # 3. Test --dump-key (string val)
        test_args_dump_str = ["inspect_checkpoint.py", "--dump-key", "goal"]
        with patch("sys.argv", test_args_dump_str):
            inspect_checkpoint_main()
            captured = capsys.readouterr()
            assert "=== Dump of Key 'goal' ===" in captured.out
            assert "test goal" in captured.out

        # 4. Test --dump-key (dict/list json val)
        test_args_dump_json = ["inspect_checkpoint.py", "--dump-key", "requirements"]
        with patch("sys.argv", test_args_dump_json):
            inspect_checkpoint_main()
            captured = capsys.readouterr()
            assert "=== Dump of Key 'requirements' ===" in captured.out
            assert "REQ001" in captured.out

        # 5. Test empty state warning
        mock_state_obj.values = {}
        test_args_empty = ["inspect_checkpoint.py"]
        with patch("sys.argv", test_args_empty):
            inspect_checkpoint_main()
            captured = capsys.readouterr()
            assert "Warning: No state values found in checkpoint" in captured.out


@patch("tddrobo.utils.call_llm_with_reasoning")
@patch("tddrobo.utils.call_llm_standard")
def test_replay_prompt_new_features(mock_call_standard, mock_call_reasoning, tmp_path, capsys):
    mock_call_reasoning.return_value = "reasoning response"
    mock_call_standard.return_value = "standard response"

    # Create dummy prompt text file
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("hello world prompt file", encoding="utf-8")

    # 1. Error: both trace and prompt-file missing
    test_args_both_missing = ["replay_prompt.py"]
    with patch("sys.argv", test_args_both_missing):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Either --trace or --prompt-file must be specified." in captured.err

    # 2. Error: both trace and prompt-file specified
    trace_file = tmp_path / "trace.txt"
    trace_file.write_text("{'prompt': 'trace'}", encoding="utf-8")
    test_args_both_specified = ["replay_prompt.py", "--trace", str(trace_file), "--prompt-file", str(prompt_file)]
    with patch("sys.argv", test_args_both_specified):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Cannot specify both --trace and --prompt-file." in captured.err

    # 3. Success with --prompt-file (default reasoning)
    test_args_prompt_file = ["replay_prompt.py", "--prompt-file", str(prompt_file)]
    with patch("sys.argv", test_args_prompt_file):
        main()
        captured = capsys.readouterr()
        assert "reasoning response" in captured.out
        mock_call_reasoning.assert_called_once_with(
            "hello world prompt file", response_schema=None, thinking_level="MINIMAL", temperature=0.0
        )
    mock_call_reasoning.reset_mock()

    # 4. Error: --prompt-file missing
    test_args_prompt_missing = ["replay_prompt.py", "--prompt-file", "non_existent_prompt.txt"]
    with patch("sys.argv", test_args_prompt_missing):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1

    # 5. Success with --schema and --temp
    test_args_schema_temp = [
        "replay_prompt.py",
        "--prompt-file",
        str(prompt_file),
        "--schema",
        "TestPlan",
        "--temp",
        "0.7",
        "--standard",
    ]
    from tddrobo.schema import TestPlan

    with patch("sys.argv", test_args_schema_temp):
        main()
        captured = capsys.readouterr()
        assert "standard response" in captured.out
        mock_call_standard.assert_called_once_with("hello world prompt file", response_schema=TestPlan, temperature=0.7)
    mock_call_standard.reset_mock()

    # 6. Error: Schema not found
    test_args_invalid_schema = ["replay_prompt.py", "--prompt-file", str(prompt_file), "--schema", "NonExistentSchema"]
    with patch("sys.argv", test_args_invalid_schema):
        with pytest.raises(SystemExit) as excinfo:
            main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Schema class 'NonExistentSchema' not found in schema module." in captured.err


# inspect_traces tests
@patch("mlflow.set_tracking_uri")
@patch("mlflow.get_experiment_by_name")
@patch("mlflow.search_traces")
@patch("socket.create_connection")
def test_inspect_traces_offline_fallback(mock_connect, mock_search, mock_get_exp, mock_set_uri, capsys):
    # Simulate offline by raising OSError on connection
    mock_connect.side_effect = OSError("Connection refused")

    # Mock experiment and empty search results
    mock_exp = mock_get_exp.return_value
    mock_exp.experiment_id = "1"

    import pandas as pd

    mock_search.return_value = pd.DataFrame()  # empty DataFrame to terminate main early

    test_args = ["inspect_traces.py"]
    with patch("sys.argv", test_args):
        from tddrobo.diagtools.inspect_traces import main as inspect_traces_main

        inspect_traces_main()
        captured = capsys.readouterr()
        assert "ℹ️ MLflow server is offline. Falling back to local database (sqlite:///mlflow.db)." in captured.out
        mock_set_uri.assert_called_with("sqlite:///mlflow.db")


# inspect_history tests
def test_inspect_history_directory_missing():
    test_args = ["inspect_history.py", "--history-dir", "non_existent_directory"]
    with patch("sys.argv", test_args):
        from tddrobo.diagtools.inspect_history import main as inspect_history_main

        with pytest.raises(SystemExit) as excinfo:
            inspect_history_main()
        assert excinfo.value.code == 1


def test_inspect_history_list_and_diff(tmp_path, capsys):
    # Create mock history snapshots
    history_dir = tmp_path / "history"
    history_dir.mkdir()

    # Snapshot 1: Design
    design_file = history_dir / "design_iter001.md"
    design_file.write_text("# Initial Design", encoding="utf-8")

    # Snapshot 2: Implementation Iteration 1
    impl_v1 = history_dir / "py_bc_req002_d002_test_iter001_unit_impl_iter001.py"
    impl_v1.write_text("def run():\n    print('v1')\n", encoding="utf-8")

    # Snapshot 3: Implementation Iteration 2
    impl_v2 = history_dir / "py_bc_req002_d002_test_iter001_unit_impl_iter002.py"
    impl_v2.write_text("def run():\n    print('v2')\n", encoding="utf-8")

    # Snapshot 4: Test
    test_file = history_dir / "test_py_bc_req002_d002_unit_iter001.py"
    test_file.write_text("def test_run(): pass\n", encoding="utf-8")

    from tddrobo.diagtools.inspect_history import main as inspect_history_main

    # 1. Test --list
    test_args_list = ["inspect_history.py", "--history-dir", str(history_dir), "--list"]
    with patch("sys.argv", test_args_list):
        inspect_history_main()
        captured = capsys.readouterr()
        assert "📁 Category: DESIGN" in captured.out
        assert "design_iter001.md" in captured.out
        assert "📁 Category: IMPLEMENTATION" in captured.out
        assert "py_bc_req002_d002_test_iter001_unit_impl_iter001.py" in captured.out
        assert "📁 Category: TEST" in captured.out
        assert "test_py_bc_req002_d002_unit_iter001.py" in captured.out

    # 2. Test --diff (success)
    test_args_diff = [
        "inspect_history.py",
        "--history-dir",
        str(history_dir),
        "--req",
        "REQ002",
        "--diff",
        "1",
        "2",
    ]
    with patch("sys.argv", test_args_diff):
        inspect_history_main()
        captured = capsys.readouterr()
        assert (
            "📄 Diff: py_bc_req002_d002_test_iter001_unit_impl_iter001.py ➡️ "
            "py_bc_req002_d002_test_iter001_unit_impl_iter002.py" in captured.out
        )
        assert "-    print('v1')" in captured.out
        assert "+    print('v2')" in captured.out

    # 3. Test --diff without --req (error)
    test_args_diff_no_req = [
        "inspect_history.py",
        "--history-dir",
        str(history_dir),
        "--diff",
        "1",
        "2",
    ]
    with patch("sys.argv", test_args_diff_no_req):
        with pytest.raises(SystemExit) as excinfo:
            inspect_history_main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Diffs require specifying a target requirement" in captured.err

    # 4. Test --diff missing iteration (error)
    test_args_diff_missing = [
        "inspect_history.py",
        "--history-dir",
        str(history_dir),
        "--req",
        "REQ002",
        "--diff",
        "1",
        "3",
    ]
    with patch("sys.argv", test_args_diff_missing):
        with pytest.raises(SystemExit) as excinfo:
            inspect_history_main()
        assert excinfo.value.code == 1
        captured = capsys.readouterr()
        assert "No implementation snapshot found for REQ002 iteration 3" in captured.err

    # 5. Test --file-diff (success)
    test_args_file_diff = [
        "inspect_history.py",
        "--file-diff",
        str(impl_v1),
        str(impl_v2),
    ]
    with patch("sys.argv", test_args_file_diff):
        inspect_history_main()
        captured = capsys.readouterr()
        assert (
            "📄 Diff: py_bc_req002_d002_test_iter001_unit_impl_iter001.py ➡️ "
            "py_bc_req002_d002_test_iter001_unit_impl_iter002.py" in captured.out
        )
        assert "-    print('v1')" in captured.out
