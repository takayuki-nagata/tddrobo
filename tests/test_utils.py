# mypy: ignore-errors
import os
from unittest.mock import MagicMock, patch

import pytest

from utils import (
    FileMemorySaver,
    Workspace,
    add_line_numbers,
    call_llm_with_reasoning,
    evaluate_math_expression,
    extract_code,
    extract_json,
    save_artifact,
)


def test_extract_code():
    markdown = "Here is the code:\n```python\nprint('hello')\n```\nEnd."
    assert extract_code(markdown) == "print('hello')"
    assert extract_code("No code block here") == "No code block here"
    assert extract_code("") == ""


def test_extract_none():
    from utils import add_line_numbers, extract_code, extract_json

    assert extract_code(None) == ""
    assert extract_json(None) == ""
    assert add_line_numbers(None) == ""


def test_extract_json():
    markdown = 'Result:\n```json\n{"key": "value"}\n```'
    assert extract_json(markdown) == '{"key": "value"}'
    assert extract_json("Plain text") == "Plain text"
    assert extract_json("") == ""
    # Test trailing markdown backticks only
    assert extract_json('{"key": "value"}\n```') == '{"key": "value"}'
    # Test array format with prefix/suffix text
    assert extract_json('Some prefix [{"item": 1}] and suffix') == '[{"item": 1}]'
    # Test fallback when no braces exist but brackets exist
    assert extract_json("[1, 2, 3]") == "[1, 2, 3]"


def test_add_line_numbers():
    code = "def foo():\n    pass"
    expected = "   1 | def foo():\n   2 |     pass"
    assert add_line_numbers(code) == expected
    assert add_line_numbers("") == ""


@patch("utils.subprocess.run")
def test_evaluate_math_expression_success(mock_run):
    import config

    mock_run.return_value = MagicMock(returncode=0, stdout="3.1415\n", stderr="")
    with patch.object(config, "VERBOSE", True):
        result = evaluate_math_expression("scale=4; 22/7")
    assert result == "3.1415"
    mock_run.assert_called_once()


@patch("utils.subprocess.run")
def test_evaluate_math_expression_error(mock_run):
    import config

    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="parse error\n")
    with patch.object(config, "VERBOSE", True):
        result = evaluate_math_expression("invalid math")
    assert result == "Error: parse error"


@patch("utils.subprocess.run")
def test_evaluate_math_expression_error_zero_returncode(mock_run):
    import config

    # Exit code is 0, but stderr has content (e.g. syntax error in bc for loops)
    mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="syntax error\n")
    with patch.object(config, "VERBOSE", True):
        result = evaluate_math_expression("while(i=0; i<1; i++) { 1 }")
    assert result == "Error: syntax error"


def test_evaluate_math_expression_timeout():
    import subprocess

    with patch("utils.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="bc", timeout=5)):
        result = evaluate_math_expression("1+1")
        assert "timed out after 5" in result


def test_evaluate_math_expression_exception():
    with patch("utils.subprocess.run", side_effect=Exception("mocked error")):
        result = evaluate_math_expression("1+1")
        assert "Exception occurred" in result
        assert "mocked error" in result


def test_workspace(tmp_path):
    ws = Workspace(str(tmp_path))
    content = "test content"
    filename = "test.txt"

    # Save
    saved_path = ws.save_artifact(filename, content)
    assert os.path.exists(saved_path)
    assert saved_path == os.path.join(str(tmp_path), filename)

    # Read
    read_content = ws.read_artifact(filename)
    assert read_content == content


def test_file_memory_saver_save_load(tmp_path):
    checkpoint_path = str(tmp_path / "test_cp.pkl")

    saver1 = FileMemorySaver(checkpoint_path)
    saver1.storage["test_key"] = "test_value"
    saver1._save()

    assert os.path.exists(checkpoint_path)

    saver2 = FileMemorySaver(checkpoint_path)
    assert getattr(saver2, "storage", {}).get("test_key") == "test_value"


def test_file_memory_saver_load_other_attrs(tmp_path):
    import pickle

    from utils import FileMemorySaver

    checkpoint_path = str(tmp_path / "test_cp.pkl")
    data = {"new_attr": "value1", "file_path": "overwritten"}
    with open(checkpoint_path, "wb") as f:
        pickle.dump(data, f)

    saver = FileMemorySaver(checkpoint_path)
    assert getattr(saver, "new_attr", None) == "value1"
    assert getattr(saver, "file_path", None) == "overwritten"


def test_file_memory_saver_unpicklable(tmp_path):
    import threading

    from utils import FileMemorySaver

    checkpoint_path = str(tmp_path / "test_cp.pkl")
    saver = FileMemorySaver(checkpoint_path)
    saver.storage["bad"] = threading.Lock()
    saver._save()
    assert os.path.exists(checkpoint_path)


def test_file_memory_saver_load_exception(tmp_path):
    from utils import FileMemorySaver

    checkpoint_path = str(tmp_path / "test_cp.pkl")
    with open(checkpoint_path, "wb") as f:
        f.write(b"not a pickle")
    saver = FileMemorySaver(checkpoint_path)
    assert getattr(saver, "storage", {}) == {}


def test_file_memory_saver_save_exception(tmp_path):
    from utils import FileMemorySaver

    checkpoint_path = str(tmp_path / "test_cp.pkl")
    saver = FileMemorySaver(checkpoint_path)
    with patch("os.replace", side_effect=Exception("replace error")):
        saver.storage["key"] = "val"
        saver._save()  # should catch and print warning


def test_file_memory_saver_put_and_writes(tmp_path):
    from utils import FileMemorySaver

    checkpoint_path = str(tmp_path / "test_cp.pkl")
    saver = FileMemorySaver(checkpoint_path)

    with patch("langgraph.checkpoint.memory.MemorySaver.put", return_value="put_res"):
        assert saver.put(MagicMock(), MagicMock(), MagicMock(), MagicMock()) == "put_res"

    with patch("langgraph.checkpoint.memory.MemorySaver.put_writes", return_value="put_writes_res"):
        assert saver.put_writes(MagicMock(), MagicMock(), MagicMock(), MagicMock()) == "put_writes_res"


def test_genai_client_generate_methods():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        with patch.object(llm, "call_with_retry", side_effect=["code_res", "doc_res"]) as mock_call:
            assert llm.generate_with_reasoning("prompt") == "code_res"
            assert llm.generate_standard("prompt") == "doc_res"
            assert mock_call.call_count == 2


@patch("utils._default_llm.generate_with_reasoning")
def test_call_llm_with_reasoning_facade(mock_generate_with_reasoning):
    mock_generate_with_reasoning.return_value = "test_code"
    assert call_llm_with_reasoning("prompt") == "test_code"
    mock_generate_with_reasoning.assert_called_once_with("prompt")


@patch("utils._default_workspace.save_artifact")
def test_save_artifact_facade(mock_save_artifact):
    mock_save_artifact.return_value = "path"
    assert save_artifact("file", "content") == "path"
    mock_save_artifact.assert_called_once_with("file", "content")


@patch("utils._default_llm.generate_standard")
def test_call_llm_standard_facade(mock_generate_standard):
    from utils import call_llm_standard

    mock_generate_standard.return_value = "doc_res"
    assert call_llm_standard("prompt") == "doc_res"


def test_genai_client_no_api_keys():
    from utils import GenAIClient

    with patch.dict("os.environ", {}, clear=True), patch("utils.genai.Client"):
        client = GenAIClient(debug_mode=False)
        assert client is not None


def test_genai_client_init_exception():
    from utils import GenAIClient

    with patch("utils.genai.Client") as MockClient:
        mock_client = MockClient.return_value
        mock_client.models.list.side_effect = Exception("list error")
        GenAIClient(debug_mode=False)  # Should catch and print error


def test_genai_client_retry_logic():
    from google.genai.errors import APIError

    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)

        # Mock count_tokens
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        # Simulate a 500 error on the first call, and success on the second
        error = APIError("Server Error", {})
        error.code = 500
        good_chunk = MagicMock(text="success_code", candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.side_effect = [error, [good_chunk]]

        with patch("utils.time.sleep") as mock_sleep:
            res = llm.call_with_retry("model", "prompt", retries=3)
            assert res == "success_code"
            mock_sleep.assert_called_once()


def test_genai_client_retry_logic_json_error():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        # Stream invalid JSON on the first call, and valid JSON on the second
        bad_chunk = MagicMock(text="invalid json", candidates=[MagicMock(finish_reason="STOP")])
        good_chunk = MagicMock(text='{"ok": "yes"}', candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.side_effect = [[bad_chunk], [good_chunk]]

        with patch("utils.time.sleep") as mock_sleep:
            res = llm.call_with_retry("model", "prompt", response_schema={"type": "object"})
            assert '{"ok": "yes"}' in res
            mock_sleep.assert_called_once()


@patch("utils.time.sleep")
def test_genai_client_streaming_loop_numeric_exceed(mock_sleep):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = "012345678901234" * 11
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason=None)])
        llm.client.models.generate_content_stream.side_effect = [
            [chunk],
            [MagicMock(text="ok", candidates=[MagicMock(finish_reason="STOP")])],
        ]

        res = llm.call_with_retry("model", "prompt", retries=2)
        assert res == "ok"


@patch("utils.time.sleep")
def test_genai_client_streaming_loop_numeric_allowed(mock_sleep):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = "012345678901234" * 9
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        res = llm.call_with_retry("model", "prompt", retries=1)
        assert text in res


def test_genai_client_streaming_malformed_brackets():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = '{"a": 1'
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)


def test_genai_client_sync_loop_numeric_allowed():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        text = "01234567890123" * 9
        mock_response = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        res = llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)
        assert text in res


def test_genai_client_sync_debug_mode():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        mock_response = MagicMock(text="test sync debug", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        res = llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)
        assert res == "test sync debug"


def test_genai_client_streaming_json_object_incomplete():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="}{", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)


def test_genai_client_streaming_debug_mode():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="debug output", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        res = llm.call_with_retry("model", "prompt", retries=1)
        assert res == "debug output"


def test_genai_client_streaming_long_normal_text():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        long_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 2
        chunk = MagicMock(text=long_str, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        res = llm.call_with_retry("model", "prompt", retries=1)
        assert long_str in res


def test_genai_client_streaming_json_extra_close_brace():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = '} ```json\n{"a": 1}\n```'
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        res = llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)
        assert text in res


@patch("utils.time.sleep")
def test_genai_client_streaming_loop_detection_retry(mock_sleep):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = "abcde12345" * 15
        bad_chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason=None)])
        good_chunk = MagicMock(text="success", candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.side_effect = [[bad_chunk], [good_chunk]]

        res = llm.call_with_retry("model", "prompt", retries=2)
        assert res == "success"


def test_genai_client_streaming_json_loop_detection():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        text = '{"a": 1}{"a": 1}{"a": 1}{"a": 1}'
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason=None)])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]
        with pytest.raises(RuntimeError, match="Repetitive JSON"):
            llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)


def test_genai_client_sync_with_tools():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        mock_response = MagicMock(text="sync result", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        res = llm.call_with_retry("model", "prompt", tools=[dummy_tool])
        assert res == "sync result"
        llm.client.models.generate_content.assert_called_once()


def test_genai_client_sync_loop_detection():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        mock_response = MagicMock(text="abcde12345" * 15, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        with pytest.raises(RuntimeError, match="Repetition loop"):
            llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)


def test_genai_client_unrecoverable_error():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.side_effect = ValueError("Fatal Error")
        with pytest.raises(ValueError, match="Fatal Error"):
            llm.call_with_retry("model", "prompt", retries=3)


def test_genai_client_unknown_runtime_error():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.side_effect = RuntimeError("Unknown error")
        with pytest.raises(RuntimeError, match="Unknown error"):
            llm.call_with_retry("model", "prompt", retries=1)


def test_genai_client_api_error_400():
    from google.genai.errors import APIError

    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        error = APIError("Bad Request", {})
        error.code = 400
        llm.client.models.count_tokens.side_effect = error
        with pytest.raises(APIError):
            llm.call_with_retry("model", "prompt", retries=3)


def test_workspace_get_path(tmp_path):
    from utils import Workspace

    ws = Workspace(str(tmp_path))
    assert ws.get_path("test.txt") == os.path.join(str(tmp_path), "test.txt")


def test_get_prompt_existing():
    from utils import get_prompt

    with patch("mlflow.genai.load_prompt") as mock_load:
        mock_prompt = MagicMock()
        mock_prompt.to_single_brace_format.return_value = "template"
        mock_load.return_value = mock_prompt
        assert get_prompt("test_prompt", "template") == "template"


def test_get_prompt_changed():
    from utils import get_prompt

    with patch("mlflow.genai.load_prompt") as mock_load, patch("mlflow.genai.register_prompt") as mock_register:
        mock_prompt = MagicMock()
        mock_prompt.to_single_brace_format.return_value = "old_template"
        mock_load.return_value = mock_prompt

        mock_new_prompt = MagicMock()
        mock_new_prompt.to_single_brace_format.return_value = "new_template"
        mock_register.return_value = mock_new_prompt

        assert get_prompt("test_prompt", "new_template") == "new_template"


def test_get_prompt_not_found():
    from utils import get_prompt

    with (
        patch("mlflow.genai.load_prompt", side_effect=Exception("not found")),
        patch("mlflow.genai.register_prompt") as mock_register,
    ):
        mock_prompt = MagicMock()
        mock_prompt.to_single_brace_format.return_value = "template"
        mock_register.return_value = mock_prompt
        assert get_prompt("test_prompt", "template") == "template"


def test_get_prompt_register_error():
    from utils import get_prompt

    with (
        patch("mlflow.genai.load_prompt", side_effect=Exception("not found")),
        patch("mlflow.genai.register_prompt", side_effect=Exception("register error")),
    ):
        assert get_prompt("test_prompt", "template") == "template"


def test_utils_file_memory_saver_load_exception(tmp_path):
    from unittest.mock import patch

    from utils import FileMemorySaver

    checkpoint_path = str(tmp_path / "test_cp.pkl")
    with open(checkpoint_path, "w") as f:
        f.write("dummy")

    with patch("pickle.load", side_effect=Exception("load error")):
        saver = FileMemorySaver(checkpoint_path)
        assert getattr(saver, "storage", {}) == {}


def test_utils_genai_client_no_api_keys():
    from unittest.mock import patch

    from utils import GenAIClient

    with patch.dict("os.environ", {}, clear=True), patch("utils.genai.Client"):
        client = GenAIClient()
        assert client is not None


def test_utils_genai_client_sync_debug_print():
    from unittest.mock import MagicMock, patch

    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        llm.client.models.generate_content.return_value = MagicMock(text="debug output", candidates=[])

        dummy_tool = MagicMock()

        llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)


def test_utils_genai_client_streaming_debug_newline():
    from unittest.mock import MagicMock, patch

    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="chunk", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        llm.call_with_retry("model", "prompt", retries=1)


def test_utils_genai_client_elapsed_time_zero():
    from unittest.mock import MagicMock, patch

    from utils import GenAIClient

    with patch("utils.genai.Client"), patch("utils.time.time", return_value=0.0):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="chunk", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        llm.call_with_retry("model", "prompt", retries=1)


def test_utils_genai_client_empty_full_text():
    from unittest.mock import MagicMock, patch

    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        llm.client.models.generate_content_stream.return_value = []
        llm.call_with_retry("model", "prompt", retries=1)


def test_utils_genai_client_max_tokens_finish_reason():
    from unittest.mock import MagicMock, patch

    import pytest

    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="chunk", candidates=[MagicMock(finish_reason="MAX_TOKENS")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        with pytest.raises(RuntimeError, match="max_output_tokens was reached"):
            llm.call_with_retry("model", "prompt", retries=1)


def test_utils_legacy_facade_functions():
    from unittest.mock import patch

    from utils import call_llm_with_reasoning, read_artifact

    with (
        patch("utils._default_llm.generate_with_reasoning", return_value="gencode"),
        patch("utils._default_workspace.read_artifact", return_value="readart"),
    ):
        assert call_llm_with_reasoning("prompt") == "gencode"
        assert read_artifact("file") == "readart"


def test_conftest_fixtures_used(mock_workspace, mock_genai_client, mock_mlflow):
    assert mock_workspace is not None
    assert mock_genai_client is not None
    assert mock_mlflow is not None


def test_genai_client_thinking_level():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="thoughtful", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        res = llm.call_with_retry("gemma-model", "prompt", thinking_level="high", retries=1)
        assert res == "thoughtful"


def test_genai_client_streaming_dots_newline(capsys):
    import uuid

    import config
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        with patch.object(config, "VERBOSE", True):
            llm = GenAIClient(debug_mode=False)
            llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
            long_str = "".join(uuid.uuid4().hex for _ in range(130))
            chunk = MagicMock(text=long_str, candidates=[MagicMock(finish_reason="STOP")])
            llm.client.models.generate_content_stream.return_value = [chunk]
            res = llm.call_with_retry("model", "prompt", retries=1)
            assert len(res) >= 4000
            assert "." * 80 + "\n" in capsys.readouterr().out


def test_genai_client_streaming_non_verbose(capsys):
    import config
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        with patch.object(config, "VERBOSE", False):
            llm = GenAIClient(debug_mode=False)
            llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
            chunk = MagicMock(text="response chunk", candidates=[MagicMock(finish_reason="STOP")])
            llm.client.models.generate_content_stream.return_value = [chunk]
            res = llm.call_with_retry("model", "prompt", retries=1)
            assert res == "response chunk"
            captured = capsys.readouterr().out
            assert "⏳ Generating..." in captured
            assert "Done!" in captured


def test_genai_client_tools_non_verbose(capsys):
    import config
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        with patch.object(config, "VERBOSE", False):
            llm = GenAIClient(debug_mode=False)
            llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
            mock_resp = MagicMock()
            mock_resp.text = "tool response"
            mock_resp.candidates = [MagicMock(finish_reason="STOP")]
            llm.client.models.generate_content.return_value = mock_resp

            def dummy_tool():
                pass

            res = llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)
            assert res == "tool response"
            captured = capsys.readouterr().out
            assert "⏳ Generating (with tools)..." in captured
            assert "Done!" in captured


def test_stream_with_timeout_trigger():
    import pytest

    from utils import stream_with_timeout

    def slow_generator():
        import time

        yield "chunk1"
        time.sleep(0.5)
        yield "chunk2"

    stream = slow_generator()
    # If timeout_sec is very small (e.g. 0.1), it should raise TimeoutError on the second chunk
    wrapped = stream_with_timeout(stream, timeout_sec=0.1)

    # First chunk is retrieved immediately
    assert next(wrapped) == "chunk1"

    # Second chunk takes 0.5s, so the 0.1s timeout should trigger
    with pytest.raises(TimeoutError) as exc_info:
        next(wrapped)
    assert "Streaming request timed out" in str(exc_info.value)


def test_genai_client_retry_on_timeout():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        # Simulate timeout on first call, success on second
        # Using a custom generator that times out or raises TimeoutError
        def timeout_stream():
            raise TimeoutError("Simulated read timeout")
            yield

        good_chunk = MagicMock(text="success_after_timeout", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [timeout_stream(), [good_chunk]]

        with patch("utils.time.sleep") as mock_sleep:
            res = llm.call_with_retry("model", "prompt", retries=2, delay=0)
            assert res == "success_after_timeout"
            mock_sleep.assert_called_once()


def test_stream_with_timeout_exception():
    import pytest

    from utils import stream_with_timeout

    def error_generator():
        yield "chunk1"
        raise ValueError("Stream failed")

    stream = error_generator()
    wrapped = stream_with_timeout(stream, timeout_sec=1.0)
    assert next(wrapped) == "chunk1"
    with pytest.raises(ValueError) as exc_info:
        next(wrapped)
    assert "Stream failed" in str(exc_info.value)


def test_progress_spinner_clear_message(capsys):
    from utils import ProgressSpinner

    spinner = ProgressSpinner("msg")
    spinner.start()
    spinner.stop("finished")
    captured = capsys.readouterr().out
    assert "finished" in captured


def test_genai_client_zero_retries():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        res = llm.call_with_retry("model", "prompt", retries=0)
        assert res == ""


def test_genai_client_init_debug_mode_with_models():
    from unittest.mock import MagicMock, patch

    from utils import GenAIClient

    with patch("utils.genai.Client") as MockClient:
        mock_model_1 = MagicMock()
        mock_model_1.name = "model-with-action"
        mock_model_1.supported_actions = ["generateContent"]

        mock_model_2 = MagicMock()
        mock_model_2.name = "model-without-action"
        mock_model_2.supported_actions = []

        mock_model_3 = MagicMock()
        mock_model_3.name = "model-no-attr"
        del mock_model_3.supported_actions

        mock_client_instance = MockClient.return_value
        mock_client_instance.models.list.return_value = [mock_model_1, mock_model_2, mock_model_3]

        GenAIClient(debug_mode=True)


def test_genai_client_init_normal_mode_missing_models():
    from unittest.mock import MagicMock, patch

    import config
    from utils import GenAIClient

    with patch("utils.genai.Client") as MockClient:
        mock_model_1 = MagicMock()
        mock_model_1.name = "unrelated-model"

        mock_client_instance = MockClient.return_value
        mock_client_instance.models.list.return_value = [mock_model_1]

        with patch.object(config, "VERBOSE", True):
            GenAIClient(debug_mode=False)


def test_genai_client_init_exception_on_list_models():
    from unittest.mock import patch

    import config
    from utils import GenAIClient

    with patch("utils.genai.Client") as MockClient:
        mock_client_instance = MockClient.return_value
        mock_client_instance.models.list.side_effect = Exception("API Error")

        with patch.object(config, "VERBOSE", True):
            GenAIClient(debug_mode=False)


def test_genai_client_verbose_coverage_various():
    from unittest.mock import MagicMock, patch

    from google.genai.errors import APIError

    import config
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        with patch("utils.types.GenerateContentConfig"):
            llm = GenAIClient(debug_mode=True)
            llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

            # 1. tools specified
            mock_tools = [lambda x: x]
            mock_response = MagicMock()
            mock_response.text = "output text"
            mock_response.candidates = [MagicMock(finish_reason="STOP")]
            llm.client.models.generate_content.return_value = mock_response
            with patch.object(config, "VERBOSE", True):
                res = llm.call_with_retry("model", "prompt", tools=mock_tools)
                assert res == "output text"

            # 2. debug_mode=True & streaming chunks
            chunk = MagicMock(text="stream chunk", candidates=[MagicMock(finish_reason="STOP")])
            llm.client.models.generate_content_stream.return_value = [chunk]
            with patch.object(config, "VERBOSE", True):
                res = llm.call_with_retry("model", "prompt", retries=1)
                assert res == "stream chunk"

            # 3. response_schema with invalid JSON output
            chunk_invalid_json = MagicMock(text="not json", candidates=[MagicMock(finish_reason="STOP")])
            llm.client.models.generate_content_stream.return_value = [chunk_invalid_json]
            with patch.object(config, "VERBOSE", True):
                try:
                    llm.call_with_retry("model", "prompt", response_schema=MagicMock(), retries=1)
                except RuntimeError as e:
                    assert "Invalid JSON" in str(e)

            # 4. Empty full_text
            chunk_empty = MagicMock(text="", candidates=[MagicMock(finish_reason="STOP")])
            llm.client.models.generate_content_stream.return_value = [chunk_empty]
            with patch.object(config, "VERBOSE", True):
                res = llm.call_with_retry("model", "prompt", retries=1)
                assert res == ""

            # 5. MAX_TOKENS finish reason
            chunk_max_tokens = MagicMock(text="some text", candidates=[MagicMock(finish_reason="MAX_TOKENS_REACHED")])
            llm.client.models.generate_content_stream.return_value = [chunk_max_tokens]
            with patch.object(config, "VERBOSE", True):
                try:
                    llm.call_with_retry("model", "prompt", retries=1)
                except RuntimeError as e:
                    assert "max_output_tokens" in str(e)

            # 6. API Error (recoverable) then success
            llm.client.models.generate_content_stream.side_effect = [
                APIError(503, {"error": "overload"}),
                [MagicMock(text="success", candidates=[MagicMock(finish_reason="STOP")])],
            ]
            with patch.object(config, "VERBOSE", True):
                # mock time.sleep to run fast
                with patch("time.sleep"):
                    res = llm.call_with_retry("model", "prompt", retries=2, delay=0)
                    assert res == "success"


def test_progress_spinner_tty(capsys):
    import time

    from utils import ProgressSpinner

    with patch("sys.stdout.isatty", return_value=True):
        spinner = ProgressSpinner("msg")
        assert spinner.is_tty is True
        spinner.start()
        time.sleep(0.15)
        spinner.stop("finished")
    captured = capsys.readouterr().out
    assert "msg" in captured
    assert "finished" in captured


def test_call_with_retry_tty(capsys):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="hello", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        with patch("sys.stdout.isatty", return_value=True):
            with patch("utils.config.VERBOSE", False):
                res = llm.call_with_retry("model", "prompt", retries=1)
                assert res == "hello"


def test_call_with_retry_thinking_parts(capsys):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        # Create a mock chunk that has a thinking part
        part1 = MagicMock()
        part1.thought = True
        part1.text = "Thinking about math... "

        part2 = MagicMock()
        part2.thought = False
        part2.text = "The answer is 42"

        candidate = MagicMock()
        candidate.content.parts = [part1, part2]
        candidate.finish_reason = "STOP"

        chunk1 = MagicMock(text="", candidates=[candidate])
        chunk2 = MagicMock(text="", candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.return_value = [chunk1, chunk2]
        with patch("utils.config.VERBOSE", False):
            with patch("sys.stdout.isatty", return_value=True):
                res = llm.call_with_retry("model", "prompt", retries=1)
                assert res == "The answer is 42"


def test_call_with_retry_thinking_non_tty_large(capsys):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        part = MagicMock()
        part.thought = True
        part.text = "".join(f"word{i:05d} " for i in range(1500))

        candidate = MagicMock()
        candidate.content.parts = [part]
        candidate.finish_reason = None

        chunk1 = MagicMock(text="", candidates=[candidate])
        chunk2 = MagicMock(text="done", candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.return_value = [chunk1, chunk2]
        with patch("utils.config.VERBOSE", False):
            with patch("sys.stdout.isatty", return_value=False):
                res = llm.call_with_retry("model", "prompt", retries=1)
                assert res == "done"


def test_call_with_retry_large_generation_non_tty(capsys):
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        large_text = "".join(f"word{i:05d} " for i in range(150))
        chunk1 = MagicMock(text=large_text, candidates=[MagicMock(finish_reason=None)])
        chunk2 = MagicMock(text="b", candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.return_value = [chunk1, chunk2]
        with patch("utils.config.VERBOSE", False):
            with patch("sys.stdout.isatty", return_value=False):
                res = llm.call_with_retry("model", "prompt", retries=1)
                assert res == large_text + "b"


def test_call_with_retry_connection_error():
    from utils import GenAIClient

    class MockConnectionError(Exception):
        pass

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        # Raise connection error first, then return successful chunk
        def err_stream():
            raise MockConnectionError("Failed to connect to host")
            yield

        good_chunk = MagicMock(text="recovered", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [err_stream(), [good_chunk]]

        with patch("utils.time.sleep") as mock_sleep:
            res = llm.call_with_retry("model", "prompt", retries=2, delay=0)
            assert res == "recovered"
            mock_sleep.assert_called_once()


def test_apply_search_replace_blocks():
    import textwrap

    import pytest

    from utils import apply_search_replace_blocks

    # 1. Happy path - single replacement
    original = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    return a - b"
    response = textwrap.dedent("""
    We will modify the sub function.
    <<<<<<< SEARCH
    def sub(a, b):
        return a - b
    =======
    def sub(a, b):
        # Subtract b from a
        return a - b
    >>>>>>> REPLACE
    """)
    expected = "def add(a, b):\n    return a + b\n\ndef sub(a, b):\n    # Subtract b from a\n    return a - b"
    assert apply_search_replace_blocks(original, response) == expected

    # 2. Multiple replacements
    response_multi = textwrap.dedent("""
    <<<<<<< SEARCH
    def add(a, b):
        return a + b
    =======
    def add(a, b):
        return a + b + 0
    >>>>>>> REPLACE

    <<<<<<< SEARCH
    def sub(a, b):
        return a - b
    =======
    def sub(a, b):
        return a - b - 0
    >>>>>>> REPLACE
    """)
    expected_multi = "def add(a, b):\n    return a + b + 0\n\ndef sub(a, b):\n    return a - b - 0"
    assert apply_search_replace_blocks(original, response_multi) == expected_multi

    # 3. No blocks found
    with pytest.raises(ValueError, match="No Search/Replace blocks found"):
        apply_search_replace_blocks(original, "This has no search block.")

    # 4. Target block not found
    response_not_found = textwrap.dedent("""
    <<<<<<< SEARCH
    def mul(a, b):
        return a * b
    =======
    def mul(a, b):
        return a * b * 1
    >>>>>>> REPLACE
    """)
    with pytest.raises(ValueError, match="Search/Replace Block 1 not found"):
        apply_search_replace_blocks(original, response_not_found)

    # 5. Ambiguous target block
    ambiguous_original = "val = 1\nval = 1"
    response_ambiguous = textwrap.dedent("""
    <<<<<<< SEARCH
    val = 1
    =======
    val = 2
    >>>>>>> REPLACE
    """)
    with pytest.raises(ValueError, match="matches multiple times"):
        apply_search_replace_blocks(ambiguous_original, response_ambiguous)

    # 6. Carriage returns handling
    original_cr = "def test():\r\n    return 42"
    response_cr = (
        "<<<<<<< SEARCH\r\ndef test():\r\n    return 42\r\n=======\r\ndef test():\r\n    return 100\r\n>>>>>>> REPLACE"
    )
    assert "return 100" in apply_search_replace_blocks(original_cr, response_cr)

    # 7. Flexible indentation alignment
    original_indent = "class Foo:\n    def bar(self):\n        val = 1\n        return val"
    response_indent = textwrap.dedent("""
    <<<<<<< SEARCH
    def bar(self):
        val = 1
    =======
    def bar(self):
        # set val to 2
        val = 2
    >>>>>>> REPLACE
    """)
    expected_indent = "class Foo:\n    def bar(self):\n        # set val to 2\n        val = 2\n        return val"
    assert apply_search_replace_blocks(original_indent, response_indent) == expected_indent

    # 8. Trailing periods/punctuation robustness
    original_punc = "def add(a, b):\n    return a + b"
    response_punc = textwrap.dedent("""
    <<<<<<< SEARCH
    def add(a, b):
        return a + b.
    =======
    def add(a, b):
        return a + b + 0
    >>>>>>> REPLACE
    """)
    expected_punc = "def add(a, b):\n    return a + b + 0"
    assert apply_search_replace_blocks(original_punc, response_punc) == expected_punc

    # 9. Fuzzy matching for minor typos
    original_typo = "def format(self):\n    obase = int(self.variables['obase'])\n    return obase"
    response_typo = textwrap.dedent("""
    <<<<<<< SEARCH
    def format(self):
        obase = int(self.variables['obbase'])
    =======
    def format(self):
        obase = int(self.variables['obase_target'])
    >>>>>>> REPLACE
    """)
    expected_typo = "def format(self):\n    obase = int(self.variables['obase_target'])\n    return obase"
    assert apply_search_replace_blocks(original_typo, response_typo) == expected_typo

    # 10. Empty response text
    with pytest.raises(ValueError, match="Response text is empty"):
        apply_search_replace_blocks(original_typo, "")

    # 11. Stripped newlines matching (single match vs multiple matches)
    original_newlines = "\nval = 1\n"
    response_newlines_single = textwrap.dedent("""
    <<<<<<< SEARCH


    val = 1


    =======
    val = 2
    >>>>>>> REPLACE
    """)
    assert apply_search_replace_blocks(original_newlines, response_newlines_single) == "\nval = 2\n"

    original_newlines_multi = "\nval = 1\nval = 1\n"
    response_newlines_multi = textwrap.dedent("""
    <<<<<<< SEARCH


    val = 1


    =======
    val = 2
    >>>>>>> REPLACE
    """)
    with pytest.raises(ValueError, match="matches multiple times"):
        apply_search_replace_blocks(original_newlines_multi, response_newlines_multi)

    # 12. Fuzzy matching multiple matches
    original_fuzzy_multi = "val = 1\nval = 1"
    response_fuzzy_multi = textwrap.dedent("""
    <<<<<<< SEARCH
    vval = 1
    =======
    val = 2
    >>>>>>> REPLACE
    """)
    with pytest.raises(ValueError, match="matches multiple times"):
        apply_search_replace_blocks(original_fuzzy_multi, response_fuzzy_multi)


def test_run_bc_command():
    from utils import run_bc_command

    with patch("utils.evaluate_math_expression") as mock_eval:
        mock_eval.return_value = "42"
        assert run_bc_command("1+1") == "42"
        mock_eval.assert_called_once_with("1+1")


def test_copy_dict_robust():
    from collections import defaultdict

    from utils import _copy_dict_robust

    # Test list handling
    assert _copy_dict_robust([1, 2, [3, 4]]) == [1, 2, [3, 4]]

    # Test defaultdict handling
    dd = defaultdict(int)
    dd["a"] = 1
    assert _copy_dict_robust(dd) == {"a": 1}

    # Test RuntimeError retry handling
    class BadDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.raised = 0

        def items(self):
            if self.raised < 2:
                self.raised += 1
                raise RuntimeError("Size changed")
            return super().items()

    bd = BadDict({"a": 1})
    assert _copy_dict_robust(bd) == {"a": 1}

    # Test RuntimeError retry failure (always fails and returns original)
    class AlwaysBadDict(dict):
        def items(self):
            raise RuntimeError("Always size changed")

    abd = AlwaysBadDict({"a": 1})
    assert _copy_dict_robust(abd) == abd


def test_genai_client_429_retry_handling():
    from utils import GenAIClient

    class MockAPIError(Exception):
        def __init__(self, message, code):
            self.code = code
            super().__init__(message)

    with patch("utils.APIError", MockAPIError), patch("utils.ClientError", MockAPIError):
        with patch("utils.genai.Client"):
            llm = GenAIClient(debug_mode=False)
            llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

            err_429_with_delay = MockAPIError("Resource exhausted. Please retry in 1.5s.", 429)
            err_429_no_delay = MockAPIError("Resource exhausted.", 429)
            good_chunk = MagicMock(text="success", candidates=[MagicMock(finish_reason="STOP")])

            # Test case 1: 429 with suggested delay
            llm.client.models.generate_content_stream.side_effect = [err_429_with_delay, [good_chunk]]
            with patch("utils.time.sleep") as mock_sleep, patch("utils.config.VERBOSE", False):
                res = llm.call_with_retry("model", "prompt", retries=2, delay=2)
                assert res == "success"
                mock_sleep.assert_called_once_with(15)  # max(1.5 + 5, 15) -> 15 -> int() -> 15

            # Test case 2: 429 without suggested delay (exponential backoff)
            llm.client.models.generate_content_stream.side_effect = [err_429_no_delay, [good_chunk]]
            with patch("utils.time.sleep") as mock_sleep, patch("utils.config.VERBOSE", False):
                res = llm.call_with_retry("model", "prompt", retries=2, delay=2)
                assert res == "success"
                mock_sleep.assert_called_once_with(30)  # max(2 * 2, 30) -> 30


def test_find_flexible_match_empty_search():
    from utils import _find_flexible_match

    assert _find_flexible_match("some code", "") == (None, None)


def test_find_flexible_match_fallback_comments_only():
    from utils import _find_flexible_match

    # Search block contains only comments and empty lines
    code = "line1\n\n# comment here\nline3"
    search = "\n# comment here"
    # This will trigger fallback matching, and compare clean_search containing empty lines
    # causing lines_similar to get called with empty strings.
    assert _find_flexible_match(code, search) == ("\n# comment here", 1)


def test_find_flexible_match_fallback_no_match():
    from utils import _find_flexible_match

    code = "line1\nline2"
    search = "# non existent comment"
    assert _find_flexible_match(code, search) == (None, 0)


def test_adjust_indentation_corner_cases():
    from utils import _adjust_indentation

    # original_matched_block is empty (should return 0 diff indent)
    assert _adjust_indentation("replace", "search", "") == "replace"

    # diff < 0 (strip indentation)
    # search_block has indent 4, original_matched_block has indent 2 -> diff = 2 - 4 = -2
    # replace_block has line with indent 4 -> strip_len = min(2, 4) = 2 -> output has indent 2
    assert _adjust_indentation("    replace\n\n", "    search", "  match") == "  replace\n"


def test_genai_client_streaming_with_parts():
    from utils import GenAIClient

    with patch("utils.genai.Client"):
        llm = GenAIClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        # chunk1 contains only thinking process (spinner stays running)
        part_thought = MagicMock(thought=True, text="thinking process")
        part_empty = MagicMock(thought=False, text="")  # triggers line 369
        candidate1 = MagicMock()
        candidate1.content.parts = [part_thought, part_empty]
        chunk1 = MagicMock(candidates=[candidate1])

        # chunk2 contains no parts and no text (triggers line 439)
        chunk2 = MagicMock(text="", candidates=[])

        # chunk3 contains the final output text (stops spinner)
        part_text = MagicMock(thought=False, text="json output")
        candidate3 = MagicMock()
        candidate3.content.parts = [part_text]
        candidate3.finish_reason = "STOP"
        chunk3 = MagicMock(candidates=[candidate3])

        llm.client.models.generate_content_stream.return_value = [chunk1, chunk2, chunk3]

        # Call call_with_retry with thinking level to trigger thinking config and parts mapping
        with patch("utils.config.VERBOSE", False):
            res = llm.call_with_retry("gemma-model", "prompt", thinking_level="high", retries=1)
            assert res == "json output"
