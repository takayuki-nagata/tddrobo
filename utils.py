import json
import os
import pickle
import re
import subprocess
import sys
import threading
import time
from typing import Any

import mlflow
from google import genai
from google.genai import types
from google.genai.errors import APIError, ClientError
from langgraph.checkpoint.memory import MemorySaver

import config

ARTIFACTS_DIR = config.ARTIFACTS_DIR
MODEL_PRIMARY = config.MODEL_PRIMARY
MODEL_SECONDARY = config.MODEL_SECONDARY
TOOL_TIMEOUT_SEC = config.TOOL_TIMEOUT_SEC
LLM_SEED = config.LLM_SEED
LLM_TOP_K = config.LLM_TOP_K
LLM_TOP_P = config.LLM_TOP_P
LLM_MAX_OUTPUT_TOKENS = config.LLM_MAX_OUTPUT_TOKENS


# --- Tool Definitions ---
def evaluate_math_expression(expression: str) -> str:
    """
    A calculator tool that evaluates a mathematical expression and returns the result.
    Use this tool to perform or verify complex calculations instead of guessing the results.

    Args:
        expression: The mathematical expression to evaluate (e.g., '10/3' or standard math formula).
    """
    if config.VERBOSE:
        print(f"\n[Tool Called] 🛠️ Evaluating math expression:\n{expression.strip()}")
    try:
        # We use 'bc -l' as the high-precision math backend.
        # bc requires a trailing NEWLINE for expression parsing.
        # The -l option enables the standard math library.
        result = subprocess.run(
            ["bc", "-l"], input=expression + "\n", capture_output=True, text=True, timeout=TOOL_TIMEOUT_SEC
        )
        if result.returncode == 0:
            stderr_strip = result.stderr.strip()
            if stderr_strip:
                error = f"Error: {stderr_strip}"
                if config.VERBOSE:
                    print(f"[Tool Result] ⚠️ {error}")
                return error
            output = result.stdout.strip()
            if config.VERBOSE:
                print(f"[Tool Result] 📥 Output: {output}")
            return output
        else:
            error = f"Error: {result.stderr.strip()}"
            if config.VERBOSE:
                print(f"[Tool Result] ⚠️ {error}")
            return error
    except subprocess.TimeoutExpired as e:
        return f"Error: Command timed out after {e.timeout} seconds."
    except Exception as e:
        return f"Exception occurred while running math evaluation: {str(e)}"


# For backwards compatibility or aliases
def run_bc_command(expression: str) -> str:
    return evaluate_math_expression(expression)


def _copy_dict_robust(d):
    for _ in range(5):
        try:
            if isinstance(d, dict):
                return {k: _copy_dict_robust(v) for k, v in list(d.items())}
            elif isinstance(d, list):
                return [_copy_dict_robust(item) for item in list(d)]
            else:
                return d
        except RuntimeError:
            time.sleep(0.01)
            continue
    return d


# --- State Persistence ---
class FileMemorySaver(MemorySaver):
    """
    A LangGraph MemorySaver that persists state to a local file using Pickle.
    Provides checkpointing capabilities for the workflow.
    """

    def __init__(self, file_path: str):
        super().__init__()
        self.file_path = file_path
        self.lock = threading.Lock()
        self._load()

    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "rb") as f:
                    data = pickle.load(f)

                    # Restore all fetched states to avoid breaking existing structure
                    for k, v in data.items():
                        if hasattr(self, k):
                            attr = getattr(self, k)
                            if type(attr).__name__ == "defaultdict":
                                attr.update(v)
                            else:
                                setattr(self, k, v)
                        else:
                            setattr(self, k, v)
            except Exception as e:
                print(f"⚠️ Failed to load checkpoint: {e}")

    def _save(self):
        with self.lock:
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            temp_path = self.file_path + ".tmp"
            try:
                dump_data = {}
                for k, v in list(self.__dict__.items()):
                    if k in ["lock", "file_path", "serde"]:
                        continue
                    # Convert defaultdict safely to standard dict to prevent Pickle errors and handle size changes
                    val_to_save = _copy_dict_robust(v)

                    # Test pickling to safely exclude un-serializable internal functions or handlers
                    try:
                        pickle.dumps(val_to_save)
                        dump_data[k] = val_to_save
                    except Exception:
                        continue
                with open(temp_path, "wb") as f:
                    pickle.dump(dump_data, f)
                os.replace(temp_path, self.file_path)
            except Exception as e:
                print(f"⚠️ Failed to save checkpoint: {e}")

    def put(self, *args, **kwargs):
        res = super().put(*args, **kwargs)
        self._save()
        return res

    def put_writes(self, *args, **kwargs):
        res = super().put_writes(*args, **kwargs)
        self._save()
        return res


class ProgressSpinner:
    """A thread-safe spinner that runs in the background to show progress during blocking operations."""

    def __init__(self, message: str = "⏳ Thinking..."):
        self.message = message
        self.spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.idx = 0
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.is_tty = sys.stdout.isatty()

    def start(self):
        if self.is_tty:
            self.stop_event.clear()
            self.thread = threading.Thread(target=self._spin)
            self.thread.daemon = True
            self.thread.start()
        else:
            print(f"{self.message}", flush=True)

    def _spin(self):
        while not self.stop_event.is_set():
            char = self.spinner[self.idx % len(self.spinner)]
            self.idx += 1
            print(f"\r{self.message} {char}", end="", flush=True)
            self.stop_event.wait(0.1)

    def stop(self, clear_message: str = ""):
        if self.is_tty:
            self.stop_event.set()
            if self.thread:
                self.thread.join(timeout=1.0)

            # Clear the spinner line first
            out_clear = "\r" + " " * 80 + "\r"
            if clear_message:
                out_clear += f"{clear_message}"
            print(out_clear, end="", flush=True)
        else:
            if clear_message:
                print(f"{clear_message}", flush=True)


def stream_with_timeout(stream, timeout_sec=300):
    """Wraps a generator stream with a background thread to enforce a read timeout on chunk yield operations."""
    import queue

    q: queue.Queue[Any] = queue.Queue()
    done = object()

    def worker():
        try:
            for chunk in stream:
                q.put(chunk)
        except Exception as e:
            q.put(e)
        finally:
            q.put(done)

    t = threading.Thread(target=worker)
    t.daemon = True
    t.start()

    while True:
        try:
            item = q.get(timeout=timeout_sec)
            if item is done:
                break
            if isinstance(item, Exception):
                raise item
            yield item
        except queue.Empty:
            raise TimeoutError(f"Streaming request timed out (no chunks received for {timeout_sec}s)")


# --- Core Services ---
class GenAIClient:
    """Encapsulates google.genai API connections and generation logic (tuned for Gemma 4)."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        if not os.environ.get("GOOGLE_API_KEY"):
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key
            else:
                print("Warning: GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

        self.client = genai.Client(http_options={"timeout": 900000})
        try:
            if self.debug_mode:
                print("List of models that support generateContent:\n")
                for m in self.client.models.list():
                    if (
                        hasattr(m, "supported_actions")
                        and m.supported_actions
                        and "generateContent" in m.supported_actions
                    ):
                        print(m.name)
            else:
                available_models = [m.name for m in self.client.models.list() if m.name]
                for target_model in (MODEL_PRIMARY, MODEL_SECONDARY):
                    if not any(target_model in name for name in available_models):
                        if config.VERBOSE:
                            print(f"Warning: Model '{target_model}' might not be available.")
        except Exception as e:
            if config.VERBOSE:
                print(f"Warning: Could not fetch models list: {e}")
        if config.VERBOSE:
            print("\nReady to use google.genai")

    def _check_repetition(self, text: str, is_streaming: bool = False) -> None:
        """Checks for repetition loops in the generated text."""
        if len(text) <= 100:
            return

        check_text = text[-16000:] if is_streaming else text
        normalized_text = re.sub(r"\s+", " ", check_text)
        loop_match_long = re.search(r"(.{15,}?)\1{3,}", normalized_text)
        loop_match_short = re.search(r"(.{3,}?)\1{14,}", normalized_text)
        for loop_match in filter(None, [loop_match_long, loop_match_short]):
            repeated_str = loop_match.group(1).strip()
            repeat_count = len(loop_match.group(0)) // len(loop_match.group(1))
            is_numeric = bool(re.fullmatch(r"[\d\s.,_+*/=-]+", repeated_str))
            if is_numeric:
                if (len(loop_match.group(1)) >= 15 and repeat_count < 10) or (
                    len(loop_match.group(1)) < 15 and repeat_count < 50
                ):
                    continue
            if len(set(repeated_str)) > 2 or re.search(r"[a-zA-Z0-9]", repeated_str):
                context = " during streaming" if is_streaming else ""
                print(f"\n❌ Error: Repetition loop detected{context}. Pattern: {repeated_str[:100]!r}")
                raise RuntimeError(f"Repetition loop detected in generated text. Pattern: {repeated_str[:100]!r}")

    def _check_json_repetition(
        self,
        full_text: str,
        start_idx: int,
        brace_count: int,
        current_obj_start: int,
        json_objects_seen: dict[str, int],
    ) -> tuple[int, int]:
        """Checks for repetitive JSON objects during streaming."""
        for i in range(start_idx, len(full_text)):
            char = full_text[i]
            if char == "{":
                if brace_count == 0:
                    current_obj_start = i
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0 and current_obj_start != -1:
                    obj_str = full_text[current_obj_start : i + 1]
                    normalized_obj = re.sub(r"\s+", "", obj_str)
                    json_objects_seen[normalized_obj] = json_objects_seen.get(normalized_obj, 0) + 1
                    if json_objects_seen[normalized_obj] >= 4:
                        print(
                            "\n❌ Error: Repeated JSON object detected during streaming. "
                            f"Object: {normalized_obj[:100]!r}"
                        )
                        raise RuntimeError(f"Repetitive JSON object detected. Object: {normalized_obj[:100]!r}")
                    current_obj_start = -1
                elif brace_count < 0:
                    brace_count = 0
                    current_obj_start = -1
        return brace_count, current_obj_start

    def _process_response_stream(
        self,
        model_name: str,
        prompt: str,
        response_stream: Any,
        response_schema: Any,
        start_time: float,
    ) -> tuple[str, Any]:
        """Reads chunks from the LLM response stream, displaying real-time or periodic progress logging."""
        full_text = ""
        dots_printed = 0
        json_objects_seen: dict[str, int] = {}
        brace_count = 0
        current_obj_start = -1
        finish_reason = None

        spinner = None
        if not config.VERBOSE:
            spinner = ProgressSpinner("⏳ Thinking...")
            spinner.start()

        spinner_stopped = False
        gen_spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        spinner_idx = 0
        thinking_text = ""
        last_logged_thinking_len = 0
        last_logged_gen_len = 0
        last_logged_time = start_time

        try:
            for chunk in stream_with_timeout(response_stream, timeout_sec=900):
                if chunk.candidates and chunk.candidates[0].finish_reason:
                    finish_reason = chunk.candidates[0].finish_reason

                parts = []
                if chunk.candidates and len(chunk.candidates) > 0 and chunk.candidates[0].content:
                    raw_parts = chunk.candidates[0].content.parts
                    if isinstance(raw_parts, list) and len(raw_parts) > 0:
                        parts = raw_parts

                if not parts and chunk.text:

                    class SimulatedPart:
                        def __init__(self, text: str):
                            self.text = text
                            self.thought = False

                    parts = [SimulatedPart(chunk.text)]

                for p in parts:
                    p_text = getattr(p, "text", "") or ""
                    if not p_text:
                        continue
                    if getattr(p, "thought", None) is True:
                        thinking_text += p_text
                        self._check_repetition(thinking_text, is_streaming=True)
                        if spinner:
                            spinner.message = f"⏳ Thinking... ({len(thinking_text):,} chars)"
                            if not sys.stdout.isatty():
                                current_time = time.time()
                                size_diff = len(thinking_text) - last_logged_thinking_len
                                time_diff = current_time - last_logged_time
                                if size_diff >= 1000 or time_diff >= 5:
                                    spinner_char = gen_spinner[spinner_idx % len(gen_spinner)]
                                    spinner_idx += 1
                                    elapsed = current_time - start_time
                                    msg = (
                                        f"⏳ Thinking... {spinner_char} "
                                        f"({len(thinking_text):,} chars, elapsed: {elapsed:.1f}s)"
                                    )
                                    print(msg, flush=True)
                                    last_logged_thinking_len = len(thinking_text)
                                    last_logged_time = current_time
                    else:
                        if spinner and not spinner_stopped:
                            spinner.stop()
                            spinner_stopped = True

                        prev_len = len(full_text)
                        full_text += p_text

                        if self.debug_mode and config.VERBOSE:
                            print(p_text, end="", flush=True)
                        elif not self.debug_mode:
                            dots_to_print = len(full_text) // 50 - dots_printed
                            for _ in range(dots_to_print):
                                if config.VERBOSE:
                                    print(".", end="", flush=True)
                                dots_printed += 1
                                if dots_printed % 80 == 0 and config.VERBOSE:
                                    print()

                            if not config.VERBOSE:
                                spinner_char = gen_spinner[spinner_idx % len(gen_spinner)]
                                spinner_idx += 1
                                gen_msg = f"\r⏳ Generating... {spinner_char} ({len(full_text):,} chars)"

                                if sys.stdout.isatty():
                                    print(gen_msg, end="", flush=True)
                                else:
                                    current_time = time.time()
                                    size_diff = len(full_text) - last_logged_gen_len
                                    time_diff = current_time - last_logged_time
                                    if size_diff >= 1000 or time_diff >= 5:
                                        elapsed = current_time - start_time
                                        msg = (
                                            f"⏳ Generating... {spinner_char} "
                                            f"({len(full_text):,} chars, elapsed: {elapsed:.1f}s)"
                                        )
                                        print(msg, flush=True)
                                        last_logged_gen_len = len(full_text)
                                        last_logged_time = current_time

                        self._check_repetition(full_text, is_streaming=True)

                        if response_schema:
                            brace_count, current_obj_start = self._check_json_repetition(
                                full_text, prev_len, brace_count, current_obj_start, json_objects_seen
                            )

                if not parts and not spinner_stopped:
                    if spinner:
                        spinner.message = f"⏳ Thinking... ({len(thinking_text):,} chars)"
        finally:
            if spinner and not spinner_stopped:
                spinner.stop()

        elapsed_time = time.time() - start_time
        if self.debug_mode and config.VERBOSE:
            print()
        elif not config.VERBOSE:
            if sys.stdout.isatty():
                print(
                    f"\r⏳ Generating... Done! ({len(full_text):,} chars in {elapsed_time:.1f}s)    ",
                    flush=True,
                )
            else:
                print(
                    f"⏳ Generating... Done! ({len(full_text):,} chars in {elapsed_time:.1f}s)",
                    flush=True,
                )

        return full_text, finish_reason

    @mlflow.trace(name="call_with_retry")
    def call_with_retry(
        self,
        model_name: str,
        prompt: str,
        retries: int = 30,
        delay: int = 5,
        response_schema: Any = None,
        thinking_level: str | None = None,
        tools: list[Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """
        Call the LLM API with built-in retry logic, temperature scaling, and loop detection.

        Args:
            model_name (str): The specific model to use.
            prompt (str): The prompt text to send.
            retries (int): Maximum number of retry attempts.
            delay (int): Delay in seconds between retries.
            response_schema (Optional[Type[BaseModel]]): Pydantic schema for structured output.
            thinking_level (Optional[str]): Thinking configuration for reasoning models.
            tools (Optional[list]): List of callable tools for the model to use.
            temperature (float): Base temperature for generation (will increase on retries).

        Returns:
            str: The generated text from the LLM.
        """
        temp_boost = 0.0
        for attempt in range(retries):
            current_temp = min(1.0, temperature + temp_boost)
            try:
                count_response = self.client.models.count_tokens(model=model_name, contents=prompt)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                if config.VERBOSE:
                    print(f"\nPrompt tokens: {count_response.total_tokens:,}")

                config_kwargs: dict[str, Any] = {
                    "temperature": current_temp,
                    "seed": LLM_SEED,
                    "top_k": LLM_TOP_K,
                    "top_p": LLM_TOP_P,
                    "max_output_tokens": LLM_MAX_OUTPUT_TOKENS,
                }
                if response_schema:
                    config_kwargs["response_mime_type"] = "application/json"
                    config_kwargs["response_schema"] = response_schema
                if thinking_level and ("gemma" in model_name.lower() or "thinking" in model_name.lower()):
                    config_kwargs["thinking_config"] = types.ThinkingConfig(
                        thinking_level=thinking_level.upper(),  # type: ignore[arg-type]
                        include_thoughts=True,
                    )
                if tools:
                    config_kwargs["tools"] = tools
                genai_config = types.GenerateContentConfig(**config_kwargs)

                finish_reason = None

                if tools:
                    if config.VERBOSE:
                        print(f"[{timestamp}] ⏳ Generating (synchronous, temp={current_temp:.1f})...", flush=True)
                        start_time = time.time()
                        response: Any = self.client.models.generate_content(
                            model=model_name, contents=prompt, config=genai_config
                        )
                    else:
                        spinner = ProgressSpinner("⏳ Thinking (with tools)...")
                        spinner.start()
                        start_time = time.time()
                        try:
                            response = self.client.models.generate_content(
                                model=model_name, contents=prompt, config=genai_config
                            )
                        finally:
                            spinner.stop()
                    elapsed_time = time.time() - start_time
                    full_text = response.text or ""
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason = response.candidates[0].finish_reason

                    if self.debug_mode and full_text and config.VERBOSE:
                        print(full_text)

                    if not config.VERBOSE:
                        print(
                            f"⏳ Generating (with tools)... Done! ({len(full_text):,} chars in {elapsed_time:.1f}s)",
                            flush=True,
                        )

                    self._check_repetition(full_text, is_streaming=False)
                else:
                    if config.VERBOSE:
                        print(f"[{timestamp}] ⏳ Generating (streaming, temp={current_temp:.1f})...", flush=True)
                    start_time = time.time()
                    response_stream = self.client.models.generate_content_stream(
                        model=model_name, contents=prompt, config=genai_config
                    )
                    full_text, finish_reason = self._process_response_stream(
                        model_name, prompt, response_stream, response_schema, start_time
                    )
                    elapsed_time = time.time() - start_time
                if full_text:
                    count_resp: Any = self.client.models.count_tokens(model=model_name, contents=full_text)
                    out_tokens = count_resp.total_tokens or 0
                    tps = out_tokens / elapsed_time if elapsed_time > 0 else 0
                    if config.VERBOSE:
                        print(
                            f"\n[Generated length: {len(full_text):,} chars / "
                            f"{elapsed_time:.1f} seconds ({tps:.1f} tokens/s)]"
                        )
                    if response_schema:
                        try:
                            json.loads(extract_json(full_text))
                        except json.JSONDecodeError as e:
                            if config.VERBOSE:
                                print(f"\n❌ Error: Invalid JSON output detected. ({e})")
                            raise RuntimeError("Invalid JSON output.")
                else:
                    if config.VERBOSE:
                        print(f"\n[Generated length: 0 chars / {elapsed_time:.1f} seconds]")

                if finish_reason and "MAX_TOKENS" in str(finish_reason):
                    if config.VERBOSE:
                        print("\n❌ Error: Generation stopped because max_output_tokens was reached.")
                    raise RuntimeError("Generation stopped because max_output_tokens was reached.")

                return full_text or ""
            except Exception as e:
                is_retryable = False
                is_degeneration = False
                if isinstance(e, RuntimeError) and (
                    "max_output_tokens" in str(e)
                    or "Repetition loop" in str(e)
                    or "Invalid JSON" in str(e)
                    or "timed out" in str(e).lower()
                ):
                    is_retryable = True
                    is_degeneration = True
                elif isinstance(e, APIError) and hasattr(e, "code") and e.code >= 500:
                    is_retryable = True
                elif isinstance(e, (APIError, ClientError)) and hasattr(e, "code") and e.code == 429:
                    is_retryable = True
                    # Parse the API's suggested retry delay from the error message
                    import re as _re

                    retry_match = _re.search(r"retry in (\d+(?:\.\d+)?)s", str(e), _re.IGNORECASE)
                    if retry_match:
                        suggested_delay = float(retry_match.group(1))
                        delay = int(max(suggested_delay + 5, 15))  # Add 5s buffer
                    else:
                        delay = max(delay * 2, 30)  # Exponential backoff
                elif (
                    isinstance(e, TimeoutError) or "timeout" in type(e).__name__.lower() or "timeout" in str(e).lower()
                ):
                    is_retryable = True
                elif "connect" in type(e).__name__.lower() or "connection" in str(e).lower():
                    is_retryable = True

                if is_retryable and attempt < retries - 1:
                    if is_degeneration:
                        temp_boost += 0.1
                    next_temp = min(1.0, temperature + temp_boost)
                    print(
                        f"\n⚠️ Recoverable error ({str(e)}). Retrying in {delay} seconds... "
                        f"(Next temp: {next_temp:.1f}, Attempt: {attempt + 1}/{retries})"
                    )
                    time.sleep(delay)
                else:
                    raise
        return ""

    def generate_with_reasoning(
        self,
        prompt: str,
        response_schema: Any = None,
        tools: list[Any] | None = None,
        thinking_level: str | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Helper method to generate text using the primary model with advanced reasoning."""
        return self.call_with_retry(
            MODEL_PRIMARY,
            prompt,
            response_schema=response_schema,
            thinking_level=thinking_level or "HIGH",
            tools=tools,
            temperature=temperature,
        )

    def generate_standard(
        self,
        prompt: str,
        response_schema: Any = None,
        tools: list[Any] | None = None,
        temperature: float = 0.0,
    ) -> str:
        """Helper method to generate text using the secondary model for standard tasks."""
        return self.call_with_retry(
            MODEL_SECONDARY,
            prompt,
            response_schema=response_schema,
            thinking_level="MINIMAL",
            tools=tools,
            temperature=temperature,
        )


class Workspace:
    """Encapsulates file system operations for the workflow."""

    def __init__(self, base_dir: str):
        self._base_dir = base_dir

    @property
    def base_dir(self) -> str:
        # Dynamically resolve using config.ARTIFACTS_DIR unless a custom directory was requested
        if self._base_dir == "artifacts" or self._base_dir == config.ARTIFACTS_DIR:
            return config.ARTIFACTS_DIR
        return self._base_dir

    def save_artifact(self, filename: str, content: str) -> str:
        """Saves content to a file in the workspace directory and returns the path."""
        path = os.path.join(self.base_dir, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    def read_artifact(self, filename: str) -> str:
        """Reads content from a file in the workspace directory."""
        path = os.path.join(self.base_dir, filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def get_path(self, filename: str) -> str:
        """Returns the full path for a given filename in the workspace."""
        return os.path.join(self.base_dir, filename)


# --- Global Default Instances (Facade Pattern) ---
_default_llm = GenAIClient(debug_mode=config.DEBUG_MODE)
_default_workspace = Workspace("artifacts")


# --- Legacy Wrapper Functions ---
def call_llm_with_reasoning(*args, **kwargs):
    return _default_llm.generate_with_reasoning(*args, **kwargs)


def call_llm_standard(*args, **kwargs):
    return _default_llm.generate_standard(*args, **kwargs)


def save_artifact(filename: str, content: str) -> str:
    return _default_workspace.save_artifact(filename, content)


def read_artifact(filename: str) -> str:
    return _default_workspace.read_artifact(filename)


# --- Helper Functions ---


@mlflow.trace(name="get_prompt")
def get_prompt(name: str, default_template: str) -> str:
    """Fetch prompt template from MLflow registry or register if it doesn't exist."""
    import mlflow.genai

    mlflow_template = re.sub(r"(?<!\{)\{([a-zA-Z_]\w*)\}(?!\})", r"{{\1}}", default_template)

    try:
        # If already registered in MLflow, get the latest version
        prompt = mlflow.genai.load_prompt(f"prompts:/{name}@latest")

        # If the prompt in the code has changed, register it as a new version to MLflow
        if prompt.to_single_brace_format() != default_template:
            print(f"🔄 Prompt '{name}' has changed. Registering a new version to MLflow.")
            prompt = mlflow.genai.register_prompt(name=name, template=mlflow_template)

        return prompt.to_single_brace_format()
    except Exception:
        # If not registered, convert Python's {var} to MLflow's {{var}} and register
        try:
            prompt = mlflow.genai.register_prompt(name=name, template=mlflow_template)
            return prompt.to_single_brace_format()
        except Exception as e:
            print(f"⚠️ Could not register prompt '{name}' to MLflow: {e}")
            return default_template


def extract_json(text: str | None) -> str:
    """Extracts JSON from a markdown-formatted response."""
    if not text:
        return ""

    # 1. Try standard markdown code block extraction
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    # 2. Try slicing from the first '{' or '[' to the last '}' or ']'
    text_stripped = text.strip()
    start_candidates = [text_stripped.find("{"), text_stripped.find("[")]
    start_candidates = [idx for idx in start_candidates if idx != -1]

    end_candidates = [text_stripped.rfind("}"), text_stripped.rfind("]")]
    end_candidates = [idx for idx in end_candidates if idx != -1]

    if start_candidates and end_candidates:
        first_char_idx = min(start_candidates)
        last_char_idx = max(end_candidates)
        if last_char_idx > first_char_idx:
            return text_stripped[first_char_idx : last_char_idx + 1].strip()

    return text_stripped


def extract_code(text: str | None) -> str:
    """Extracts Python code from a markdown-formatted response."""
    if not text:
        return ""
    match = re.search(r"```python[ \t]*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def add_line_numbers(code: str | None) -> str:
    """Adds line numbers to a given code string for better LLM analysis."""
    if not code:
        return ""
    lines = code.splitlines()
    return "\n".join(f"{i + 1:4d} | {line}" for i, line in enumerate(lines))


def _find_flexible_match(current_code: str, search_block: str):
    search_lines = search_block.splitlines()
    if not search_lines:
        return None, None

    code_lines = current_code.splitlines()

    def is_comment_or_blank(line):
        stripped = line.strip()
        return not stripped or stripped.startswith("#") or (stripped.startswith("'''") or stripped.startswith('"""'))

    # Build mapping for search block: (cleaned_line, original_index)
    search_mapped = []
    for idx, line in enumerate(search_lines):
        if not is_comment_or_blank(line):
            search_mapped.append((line.strip().rstrip(".,"), idx))

    # Build mapping for current_code: (cleaned_line, original_index)
    code_mapped = []
    for idx, line in enumerate(code_lines):
        if not is_comment_or_blank(line):
            code_mapped.append((line.strip().rstrip(".,"), idx))

    def clean_line(line):
        return line.strip().rstrip(".,")

    import difflib

    def lines_similar(line1, line2):
        if line1 == line2:
            return True
        if not line1 or not line2:
            return False
        return difflib.SequenceMatcher(None, line1, line2).ratio() >= 0.90

    # Fallback to exact line count matching if search block is entirely comments or blank lines
    if not search_mapped:
        search_len = len(search_lines)
        clean_search = [clean_line(line) for line in search_lines]

        fallback_matches = []
        for i in range(len(code_lines) - search_len + 1):
            fallback_window = code_lines[i : i + search_len]
            clean_window = [clean_line(line) for line in fallback_window]
            match = True
            for s_line, w_line in zip(clean_search, clean_window):
                if not lines_similar(s_line, w_line):
                    match = False
                    break
            if match:
                fallback_matches.append(i)
        if len(fallback_matches) == 1:
            match_start_idx = fallback_matches[0]
            original_matched_block = "\n".join(code_lines[match_start_idx : match_start_idx + search_len])
            return original_matched_block, match_start_idx
        return None, len(fallback_matches)

    # Match the cleaned lines using a sliding window
    search_len = len(search_mapped)
    mapped_matches = []

    for i in range(len(code_mapped) - search_len + 1):
        mapped_window = code_mapped[i : i + search_len]
        match = True
        for (s_line, _), (w_line, _) in zip(search_mapped, mapped_window):
            if not lines_similar(s_line, w_line):
                match = False
                break
        if match:
            start_orig_idx = mapped_window[0][1]
            end_orig_idx = mapped_window[-1][1]
            mapped_matches.append((start_orig_idx, end_orig_idx))

    if len(mapped_matches) == 1:
        start_idx, end_idx = mapped_matches[0]
        original_matched_block = "\n".join(code_lines[start_idx : end_idx + 1])
        return original_matched_block, start_idx

    return None, len(mapped_matches)


def _adjust_indentation(replace_block: str, search_block: str, original_matched_block: str) -> str:
    def get_indent(block):
        for line in block.splitlines():
            if line.strip():
                return len(line) - len(line.lstrip())
        return 0

    orig_indent = get_indent(original_matched_block)
    search_indent = get_indent(search_block)
    diff = orig_indent - search_indent

    if diff == 0:
        return replace_block

    adjusted_lines = []
    for line in replace_block.splitlines():
        if not line.strip():
            adjusted_lines.append("")
        else:
            if diff > 0:
                adjusted_lines.append(" " * diff + line)
            else:
                strip_len = min(abs(diff), len(line) - len(line.lstrip()))
                adjusted_lines.append(line[strip_len:])
    return "\n".join(adjusted_lines)


def apply_search_replace_blocks(original_code: str, response_text: str) -> str:
    """
    Parses Search/Replace blocks from response_text and applies them to original_code.
    If no blocks are found, or if applying them fails due to mismatch/ambiguity,
    raises ValueError to trigger full code extraction fallback.
    """
    if not response_text:
        raise ValueError("Response text is empty.")

    # Normalize line endings to \n for robust matching
    original_code_norm = original_code.replace("\r\n", "\n")
    response_text_norm = response_text.replace("\r\n", "\n")

    # Match: <<<<<<< SEARCH\n<search_content>\n=======\n<replace_content>\n>>>>>>> REPLACE
    pattern = r"[ \t]*<<<<<<<\s*SEARCH\s*\n(.*?)\n[ \t]*=======\n(.*?)\n[ \t]*>>>>>>>\s*REPLACE"
    blocks = re.findall(pattern, response_text_norm, re.DOTALL)

    if not blocks:
        raise ValueError("No Search/Replace blocks found in response.")

    current_code = original_code_norm
    for idx, (search_block, replace_block) in enumerate(blocks):
        # We try to apply the block. First, check if exact match exists.
        count = current_code.count(search_block)
        if count == 1:
            current_code = current_code.replace(search_block, replace_block, 1)
        elif count > 1:
            raise ValueError(
                f"Search/Replace Block {idx + 1} matches multiple times ({count}) in the file. "
                f"It must be unique. Target SEARCH block:\n{search_block}"
            )
        else:
            # Let's try to match with stripped carriage returns or minor trailing spaces on each line.
            stripped_search = search_block.strip("\n")
            stripped_replace = replace_block.strip("\n")

            # Count with stripped newlines
            stripped_count = current_code.count(stripped_search) if stripped_search else 0
            if stripped_count == 1:
                current_code = current_code.replace(stripped_search, stripped_replace, 1)
            elif stripped_count > 1:
                raise ValueError(
                    f"Search/Replace Block {idx + 1} matches multiple times ({stripped_count}) in the file. "
                    f"It must be unique. Target SEARCH block:\n{search_block}"
                )
            elif stripped_search.endswith(".") and current_code.count(stripped_search[:-1]) == 1:
                current_code = current_code.replace(stripped_search[:-1], stripped_replace, 1)
            else:
                # Try flexible matching
                matched_orig, match_count = _find_flexible_match(current_code, search_block)
                if matched_orig is not None:
                    adjusted_replace = _adjust_indentation(replace_block, search_block, matched_orig)
                    current_code = current_code.replace(matched_orig, adjusted_replace, 1)
                else:
                    if match_count and match_count > 1:
                        raise ValueError(
                            f"Search/Replace Block {idx + 1} matches multiple times ({match_count}) in the file. "
                            f"It must be unique. Target SEARCH block:\n{search_block}"
                        )
                    else:
                        raise ValueError(
                            f"Search/Replace Block {idx + 1} not found in the target file. "
                            f"Please make sure the SEARCH block matches the file content exactly.\n"
                            f"Target SEARCH block that failed to match:\n{search_block}"
                        )

    return current_code
