# mypy: ignore-errors
import os
from unittest.mock import MagicMock, patch

import pytest

from utils import (
    FileMemorySaver,
    Workspace,
    add_line_numbers,
    extract_code,
    extract_json,
    llm_gencode,
    run_bc_command,
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


def test_add_line_numbers():
    code = "def foo():\n    pass"
    expected = "   1 | def foo():\n   2 |     pass"
    assert add_line_numbers(code) == expected
    assert add_line_numbers("") == ""


@patch("utils.subprocess.run")
def test_run_bc_command_success(mock_run):
    mock_run.return_value = MagicMock(returncode=0, stdout="3.1415\n", stderr="")
    result = run_bc_command("scale=4; 22/7")
    assert result == "3.1415"
    mock_run.assert_called_once()


@patch("utils.subprocess.run")
def test_run_bc_command_error(mock_run):
    mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="parse error\n")
    result = run_bc_command("invalid math")
    assert result == "Error: parse error"


def test_run_bc_command_timeout():
    import subprocess

    with patch("utils.subprocess.run", side_effect=subprocess.TimeoutExpired(cmd="bc", timeout=5)):
        result = run_bc_command("1+1")
        assert "timed out after 5" in result


def test_run_bc_command_exception():
    with patch("utils.subprocess.run", side_effect=Exception("mocked error")):
        result = run_bc_command("1+1")
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


def test_llm_client_generate_methods():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        with patch.object(llm, "call_with_retry", side_effect=["code_res", "doc_res"]) as mock_call:
            assert llm.generate_code("prompt") == "code_res"
            assert llm.generate_doc("prompt") == "doc_res"
            assert mock_call.call_count == 2


@patch("utils._default_llm.generate_code")
def test_llm_gencode_facade(mock_generate_code):
    mock_generate_code.return_value = "test_code"
    assert llm_gencode("prompt") == "test_code"
    mock_generate_code.assert_called_once_with("prompt")


@patch("utils._default_workspace.save_artifact")
def test_save_artifact_facade(mock_save_artifact):
    mock_save_artifact.return_value = "path"
    assert save_artifact("file", "content") == "path"
    mock_save_artifact.assert_called_once_with("file", "content")


@patch("utils._default_llm.generate_doc")
def test_llm_gendoc_facade(mock_generate_doc):
    from utils import llm_gendoc

    mock_generate_doc.return_value = "doc_res"
    assert llm_gendoc("prompt") == "doc_res"


def test_llm_client_no_api_keys():
    from utils import LLMClient

    with patch.dict("os.environ", {}, clear=True), patch("utils.genai.Client"):
        client = LLMClient(debug_mode=False)
        assert client is not None


def test_llm_client_init_exception():
    from utils import LLMClient

    with patch("utils.genai.Client") as MockClient:
        mock_client = MockClient.return_value
        mock_client.models.list.side_effect = Exception("list error")
        LLMClient(debug_mode=False)  # Should catch and print error


def test_llm_client_retry_logic():
    from google.genai.errors import APIError

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)

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


def test_llm_client_retry_logic_json_error():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
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
def test_llm_client_streaming_loop_numeric_exceed(mock_sleep):
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
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
def test_llm_client_streaming_loop_numeric_allowed(mock_sleep):
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = "012345678901234" * 9
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        res = llm.call_with_retry("model", "prompt", retries=1)
        assert text in res


def test_llm_client_streaming_malformed_brackets():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = '}}}{"a": 1}'
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)


def test_llm_client_sync_loop_numeric_allowed():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        text = "01234567890123" * 9
        mock_response = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        res = llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)
        assert text in res


def test_llm_client_sync_debug_mode():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        mock_response = MagicMock(text="test sync debug", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        res = llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)
        assert res == "test sync debug"


def test_llm_client_streaming_json_object_incomplete():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="}{", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]

        with pytest.raises(RuntimeError, match="Invalid JSON"):
            llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)


def test_llm_client_streaming_debug_mode():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="debug output", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        res = llm.call_with_retry("model", "prompt", retries=1)
        assert res == "debug output"


def test_llm_client_streaming_long_normal_text():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        long_str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" * 2
        chunk = MagicMock(text=long_str, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        res = llm.call_with_retry("model", "prompt", retries=1)
        assert long_str in res


def test_llm_client_streaming_json_extra_close_brace():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = '} ```json\n{"a": 1}\n```'
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]

        res = llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)
        assert text in res


@patch("utils.time.sleep")
def test_llm_client_streaming_loop_detection_retry(mock_sleep):
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        text = "abcde12345" * 15
        bad_chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason=None)])
        good_chunk = MagicMock(text="success", candidates=[MagicMock(finish_reason="STOP")])

        llm.client.models.generate_content_stream.side_effect = [[bad_chunk], [good_chunk]]

        res = llm.call_with_retry("model", "prompt", retries=2)
        assert res == "success"


def test_llm_client_streaming_json_loop_detection():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        text = '{"a": 1}{"a": 1}{"a": 1}{"a": 1}'
        chunk = MagicMock(text=text, candidates=[MagicMock(finish_reason=None)])
        llm.client.models.generate_content_stream.side_effect = [[chunk]]
        with pytest.raises(RuntimeError, match="Repetitive JSON"):
            llm.call_with_retry("model", "prompt", response_schema={"type": "object"}, retries=1)


def test_llm_client_sync_with_tools():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        mock_response = MagicMock(text="sync result", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        res = llm.call_with_retry("model", "prompt", tools=[dummy_tool])
        assert res == "sync result"
        llm.client.models.generate_content.assert_called_once()


def test_llm_client_sync_loop_detection():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)

        mock_response = MagicMock(text="abcde12345" * 15, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content.return_value = mock_response

        dummy_tool = MagicMock()

        with pytest.raises(RuntimeError, match="Repetition loop"):
            llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)


def test_llm_client_unrecoverable_error():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.side_effect = ValueError("Fatal Error")
        with pytest.raises(ValueError, match="Fatal Error"):
            llm.call_with_retry("model", "prompt", retries=3)


def test_llm_client_unknown_runtime_error():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.side_effect = RuntimeError("Unknown error")
        with pytest.raises(RuntimeError, match="Unknown error"):
            llm.call_with_retry("model", "prompt", retries=1)


def test_llm_client_api_error_400():
    from google.genai.errors import APIError

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
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


def test_utils_llm_client_no_api_keys():
    from unittest.mock import patch

    from utils import LLMClient

    with patch.dict("os.environ", {}, clear=True), patch("utils.genai.Client"):
        client = LLMClient()
        assert client is not None


def test_utils_llm_client_sync_debug_print():
    from unittest.mock import MagicMock, patch

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        llm.client.models.generate_content.return_value = MagicMock(text="debug output", candidates=[])

        dummy_tool = MagicMock()

        llm.call_with_retry("model", "prompt", tools=[dummy_tool], retries=1)


def test_utils_llm_client_streaming_debug_newline():
    from unittest.mock import MagicMock, patch

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=True)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="chunk", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        llm.call_with_retry("model", "prompt", retries=1)


def test_utils_llm_client_elapsed_time_zero():
    from unittest.mock import MagicMock, patch

    from utils import LLMClient

    with patch("utils.genai.Client"), patch("utils.time.time", side_effect=[0.0, 0.0]):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="chunk", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        llm.call_with_retry("model", "prompt", retries=1)


def test_utils_llm_client_empty_full_text():
    from unittest.mock import MagicMock, patch

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        llm.client.models.generate_content_stream.return_value = []
        llm.call_with_retry("model", "prompt", retries=1)


def test_utils_llm_client_max_tokens_finish_reason():
    from unittest.mock import MagicMock, patch

    import pytest

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="chunk", candidates=[MagicMock(finish_reason="MAX_TOKENS")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        with pytest.raises(RuntimeError, match="max_output_tokens was reached"):
            llm.call_with_retry("model", "prompt", retries=1)


def test_utils_legacy_facade_functions():
    from unittest.mock import patch

    from utils import llm_gencode, read_artifact

    with (
        patch("utils._default_llm.generate_code", return_value="gencode"),
        patch("utils._default_workspace.read_artifact", return_value="readart"),
    ):
        assert llm_gencode("prompt") == "gencode"
        assert read_artifact("file") == "readart"


def test_conftest_fixtures_used(mock_workspace, mock_llm_client, mock_mlflow):
    assert mock_workspace is not None
    assert mock_llm_client is not None
    assert mock_mlflow is not None


def test_llm_client_thinking_level():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        chunk = MagicMock(text="thoughtful", candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        res = llm.call_with_retry("model", "prompt", thinking_level="high", retries=1)
        assert res == "thoughtful"


def test_llm_client_streaming_dots_newline(capsys):
    import uuid

    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        llm.client.models.count_tokens.return_value = MagicMock(total_tokens=10)
        long_str = "".join(uuid.uuid4().hex for _ in range(130))
        chunk = MagicMock(text=long_str, candidates=[MagicMock(finish_reason="STOP")])
        llm.client.models.generate_content_stream.return_value = [chunk]
        res = llm.call_with_retry("model", "prompt", retries=1)
        assert len(res) >= 4000
        assert "." * 80 + "\n" in capsys.readouterr().out


def test_llm_client_zero_retries():
    from utils import LLMClient

    with patch("utils.genai.Client"):
        llm = LLMClient(debug_mode=False)
        res = llm.call_with_retry("model", "prompt", retries=0)
        assert res == ""
