import json
import os
import pickle
import re
import subprocess
import threading
import time
from typing import Any

import mlflow
from google import genai
from google.genai import types
from google.genai.errors import APIError
from langgraph.checkpoint.memory import MemorySaver

import config

ARTIFACTS_DIR = config.ARTIFACTS_DIR
MODEL_GENCODE = config.MODEL_GENCODE
MODEL_GENDOC = config.MODEL_GENDOC
TOOL_TIMEOUT_SEC = config.TOOL_TIMEOUT_SEC
LLM_SEED = config.LLM_SEED
LLM_TOP_K = config.LLM_TOP_K
LLM_TOP_P = config.LLM_TOP_P
LLM_MAX_OUTPUT_TOKENS = config.LLM_MAX_OUTPUT_TOKENS


# --- Tool Definitions ---
def run_bc_command(expression: str) -> str:
    """
    A calculator tool that executes a mathematical expression and returns the result.
    Use this tool to perform or verify complex calculations instead of guessing the results.

    Args:
        expression: The mathematical expression to evaluate (e.g., 'scale=10; 10/3' or 's(1.0)').
    """
    print(f"\n[Tool Called] 🛠️ Executing bc command with expression:\n{expression.strip()}")
    try:
        # bc requires a trailing NEWLINE for expression parsing.
        # The -l option enables the standard math library.
        result = subprocess.run(
            ["bc", "-l"], input=expression + "\n", capture_output=True, text=True, timeout=TOOL_TIMEOUT_SEC
        )
        if result.returncode == 0:
            output = result.stdout.strip()
            print(f"[Tool Result] 📥 Output: {output}")
            return output
        else:
            error = f"Error: {result.stderr.strip()}"
            print(f"[Tool Result] ⚠️ {error}")
            return error
    except subprocess.TimeoutExpired as e:
        return f"Error: Command timed out after {e.timeout} seconds."
    except Exception as e:
        return f"Exception occurred while running bc: {str(e)}"


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
                for k, v in self.__dict__.items():
                    if k in ["lock", "file_path", "serde"]:
                        continue
                    # Convert defaultdict to standard dict to prevent Pickle errors caused by lambda
                    if type(v).__name__ == "defaultdict":
                        val_to_save = dict(v)
                    else:
                        val_to_save = v

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


# --- Core Services ---
class LLMClient:
    """Encapsulates LLM API connections and generation logic."""

    def __init__(self, debug_mode: bool = False):
        self.debug_mode = debug_mode
        if not os.environ.get("GOOGLE_API_KEY"):
            gemini_key = os.environ.get("GEMINI_API_KEY")
            if gemini_key:
                os.environ["GOOGLE_API_KEY"] = gemini_key
            else:
                print("Warning: GOOGLE_API_KEY or GEMINI_API_KEY is not set.")

        self.client = genai.Client()
        print("List of models that support generateContent:\n")
        try:
            for m in self.client.models.list():
                if hasattr(m, "supported_actions") and m.supported_actions and "generateContent" in m.supported_actions:
                    print(m.name)
        except Exception as e:
            print(f"Could not fetch models list: {e}")
        print("\nReady to use google.genai")

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
                print(f"\nPrompt tokens: {count_response.total_tokens:,}")
                start_time = time.time()

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
                if thinking_level:
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)  # type: ignore[arg-type]
                if tools:
                    config_kwargs["tools"] = tools
                genai_config = types.GenerateContentConfig(**config_kwargs)

                finish_reason = None

                if tools:
                    print(f"[{timestamp}] ⏳ Generating (synchronous, temp={current_temp:.1f})...", flush=True)
                    response: Any = self.client.models.generate_content(
                        model=model_name, contents=prompt, config=genai_config
                    )
                    full_text = response.text or ""
                    if response.candidates and response.candidates[0].finish_reason:
                        finish_reason = response.candidates[0].finish_reason

                    if self.debug_mode and full_text:
                        print(full_text)

                    if len(full_text) > 100:
                        normalized_text = re.sub(r"\s+", " ", full_text)
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
                                print(f"\n❌ Error: Repetition loop detected. Pattern: {repeated_str[:100]!r}")
                                raise RuntimeError(
                                    f"Repetition loop detected in generated text. Pattern: {repeated_str[:100]!r}"
                                )
                else:
                    print(f"[{timestamp}] ⏳ Generating (streaming, temp={current_temp:.1f})...", flush=True)
                    response_stream: Any = self.client.models.generate_content_stream(
                        model=model_name, contents=prompt, config=genai_config
                    )
                    full_text = ""
                    dots_printed = 0
                    json_objects_seen: dict[str, int] = {}
                    brace_count = 0
                    current_obj_start = -1

                    for chunk in response_stream:
                        if chunk.candidates and chunk.candidates[0].finish_reason:
                            finish_reason = chunk.candidates[0].finish_reason
                        if chunk.text:
                            prev_len = len(full_text)
                            full_text += chunk.text
                            if self.debug_mode:
                                print(chunk.text, end="", flush=True)
                            else:
                                dots_to_print = len(full_text) // 50 - dots_printed
                                for _ in range(dots_to_print):
                                    print(".", end="", flush=True)
                                    dots_printed += 1
                                    if dots_printed % 80 == 0:
                                        print()

                            if len(full_text) > 100:
                                check_text = full_text[-2000:]
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
                                        print(
                                            "\n❌ Error: Repetition loop detected during streaming. "
                                            f"Pattern: {repeated_str[:100]!r}"
                                        )
                                        raise RuntimeError(
                                            "Repetition loop detected in generated text. "
                                            f"Pattern: {repeated_str[:100]!r}"
                                        )

                            if response_schema:
                                for i in range(prev_len, len(full_text)):
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
                                            json_objects_seen[normalized_obj] = (
                                                json_objects_seen.get(normalized_obj, 0) + 1
                                            )
                                            if json_objects_seen[normalized_obj] >= 4:
                                                print(
                                                    "\n❌ Error: Repeated JSON object detected during streaming. "
                                                    f"Object: {normalized_obj[:100]!r}"
                                                )
                                                raise RuntimeError(
                                                    f"Repetitive JSON object detected. Object: {normalized_obj[:100]!r}"
                                                )
                                            current_obj_start = -1
                                        elif brace_count < 0:
                                            brace_count = 0
                                            current_obj_start = -1

                    if self.debug_mode:
                        print()
                elapsed_time = time.time() - start_time
                if full_text:
                    count_resp: Any = self.client.models.count_tokens(model=model_name, contents=full_text)
                    out_tokens = count_resp.total_tokens or 0
                    tps = out_tokens / elapsed_time if elapsed_time > 0 else 0
                    print(
                        f"\n[Generated length: {len(full_text):,} chars / "
                        f"{elapsed_time:.1f} seconds ({tps:.1f} tokens/s)]"
                    )
                    if response_schema:
                        try:
                            json.loads(extract_json(full_text))
                        except json.JSONDecodeError as e:
                            print(f"\n❌ Error: Invalid JSON output detected. ({e})")
                            raise RuntimeError("Invalid JSON output.")
                else:
                    print(f"\n[Generated length: 0 chars / {elapsed_time:.1f} seconds]")

                if finish_reason and "MAX_TOKENS" in str(finish_reason):
                    print("\n❌ Error: Generation stopped because max_output_tokens was reached.")
                    raise RuntimeError("Generation stopped because max_output_tokens was reached.")

                return full_text or ""
            except Exception as e:
                is_retryable = False
                is_degeneration = False
                if isinstance(e, RuntimeError) and (
                    "max_output_tokens" in str(e) or "Repetition loop" in str(e) or "Invalid JSON" in str(e)
                ):
                    is_retryable = True
                    is_degeneration = True
                elif isinstance(e, APIError) and hasattr(e, "code") and e.code >= 500:
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

    def generate_code(
        self,
        prompt: str,
        response_schema: Any = None,
        tools: list[Any] | None = None,
        thinking_level: str | None = "high",
        temperature: float = 0.0,
    ) -> str:
        """Helper method to generate code using the predefined code generation model."""
        return self.call_with_retry(
            MODEL_GENCODE,
            prompt,
            response_schema=response_schema,
            thinking_level=thinking_level,
            tools=tools,
            temperature=temperature,
        )

    def generate_doc(self, prompt: str, response_schema: Any = None, temperature: float = 0.0) -> str:
        """Helper method to generate documents using the predefined document generation model."""
        return self.call_with_retry(MODEL_GENDOC, prompt, response_schema=response_schema, temperature=temperature)


class Workspace:
    """Encapsulates file system operations for the workflow."""

    def __init__(self, base_dir: str):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_artifact(self, filename: str, content: str) -> str:
        """Saves content to a file in the workspace directory and returns the path."""
        path = os.path.join(self.base_dir, filename)
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
_default_llm = LLMClient(debug_mode=config.DEBUG_MODE)
_default_workspace = Workspace(ARTIFACTS_DIR)


# --- Legacy Wrapper Functions ---
def llm_gencode(*args, **kwargs):
    return _default_llm.generate_code(*args, **kwargs)


def llm_gendoc(*args, **kwargs):
    return _default_llm.generate_doc(*args, **kwargs)


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
    match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text.strip()


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
