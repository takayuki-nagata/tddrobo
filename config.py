import os
import time
import typing

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Project Goals & Defaults ---
GOAL = os.environ.get("TDD_GOAL", "")
SPEC_URL = os.environ.get("TDD_SPEC_URL", "")
DEFAULT_IMPL_NAME = os.environ.get("TDD_DEFAULT_IMPL_NAME", "impl.py")
DEFAULT_TEST_NAME = os.environ.get("TDD_DEFAULT_TEST_NAME", "test_impl.py")
DOMAIN_TIPS = os.environ.get("TDD_DOMAIN_TIPS", "")
DEFAULT_PYTHON_TIPS = (
    "\n# Python Runtime Environment:\n"
    "- The execution environment is Python 3.14.3.\n"
    "- Ensure all syntax, standard library modules, and built-in functions "
    "in your code are fully compatible with Python 3.14.\n"
    "- Do NOT use any deprecated or removed features, modules, classes, or "
    "attributes from the Python standard library. Check modern alternatives "
    "for any standard library APIs that were removed or changed in recent "
    "Python versions.\n"
    "\n# Debugging & Error Diagnostics Tip:\n"
    "- When implementing a runner/evaluator/interpreter's core entrypoint function "
    "(such as `execute()` or `run()`), do NOT silently ignore or hide internal "
    "exceptions (like `RuntimeError` or `ZeroDivisionError`) behind generic return values (like `None`).\n"
    "- While the specification may require returning a fallback value (e.g. `None` or empty results) "
    "when an error occurs, you MUST print the exception details and stack trace to `sys.stderr` "
    "(using `traceback.print_exc()`) before returning. This allows the testing framework to capture "
    "the detailed trace of parsing or execution failures and diagnostic reports, preventing debugging "
    "loops.\n"
)

# --- Agent Iterations & Coverage Goals ---
MAX_ITERATIONS = int(os.environ.get("TDD_MAX_ITERATIONS", 150))
MAX_TEST_PLAN_ITERATIONS = int(os.environ.get("TDD_MAX_TEST_PLAN_ITERATIONS", 10))
MAX_TEST_ITERATIONS = int(os.environ.get("TDD_MAX_TEST_ITERATIONS", 10))
MAX_REFACTOR_ITERATIONS = int(os.environ.get("TDD_MAX_REFACTOR_ITERATIONS", 5))
MAX_STAGNANT_ITERATIONS = int(os.environ.get("TDD_MAX_STAGNANT_ITERATIONS", 3))
LOOP_DETECTION_THRESHOLD = int(os.environ.get("TDD_LOOP_DETECTION_THRESHOLD", 8))
MAX_SYNTAX_ERROR_ITERATIONS = int(os.environ.get("TDD_MAX_SYNTAX_ERROR_ITERATIONS", 3))
TARGET_TEST_PLAN_COVERAGE = int(os.environ.get("TDD_TARGET_TEST_PLAN_COVERAGE", 95))
TARGET_TEST_COVERAGE = int(os.environ.get("TDD_TARGET_TEST_COVERAGE", 95))
TARGET_DESIGN_QUALITY = int(os.environ.get("TDD_TARGET_DESIGN_QUALITY", 92))


# --- LLM Settings ---
MODEL_PRIMARY = os.environ.get("MODEL_PRIMARY", "gemma-4-31b-it")
MODEL_SECONDARY = os.environ.get("MODEL_SECONDARY", "gemma-4-26b-a4b-it")
LLM_SEED = int(os.environ.get("LLM_SEED", 123))
LLM_TOP_K = int(os.environ.get("LLM_TOP_K", 40))
LLM_TOP_P = float(os.environ.get("LLM_TOP_P", 0.95))
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", 8192))
LLM_RETRIES = int(os.environ.get("LLM_RETRIES", 30))
LLM_RETRY_DELAY = int(os.environ.get("LLM_RETRY_DELAY", 5))
LLM_FALLBACK_THRESHOLD = int(os.environ.get("LLM_FALLBACK_THRESHOLD", 5))
LLM_STRUCTURED_RETRIES = int(os.environ.get("LLM_STRUCTURED_RETRIES", 5))

# --- Timeouts (Seconds) ---
FETCH_TIMEOUT_SEC = int(os.environ.get("TDD_FETCH_TIMEOUT_SEC", 10))
SYNTAX_CHECK_TIMEOUT_SEC = int(os.environ.get("TDD_SYNTAX_CHECK_TIMEOUT_SEC", 10))
TEST_EXECUTION_TIMEOUT_SEC = int(os.environ.get("TDD_TEST_EXECUTION_TIMEOUT_SEC", 30))
TOOL_TIMEOUT_SEC = int(os.environ.get("TDD_TOOL_TIMEOUT_SEC", 5))

# --- Linter & Testing Settings ---
FLAKE8_SELECT = os.environ.get("TDD_FLAKE8_SELECT", "E999,F821,F822")
PYTEST_MAXFAIL = int(os.environ.get("TDD_PYTEST_MAXFAIL", 0))
TDD_ROBO_DEBUG = os.environ.get("TDD_ROBO_DEBUG", "1")


# --- System Settings ---
RUN_NAME = os.environ.get("TDD_RUN_NAME", "tdd_agent_run")
SESSION_ID = os.environ.get("TDD_SESSION_ID", f"tdd_session_{time.strftime('%Y%m%d_%H%M%S')}")
TDD_BASE_ARTIFACTS_DIR = os.environ.get("TDD_ARTIFACTS_DIR", "artifacts")
ARTIFACTS_DIR = os.path.join(TDD_BASE_ARTIFACTS_DIR, SESSION_ID)
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
VERBOSE = os.environ.get("TDD_VERBOSE", "false").lower() == "true"

ORACLE_VERIFIER: typing.Optional[typing.Callable[..., str]] = None
