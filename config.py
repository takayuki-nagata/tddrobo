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

# --- Agent Iterations & Coverage Goals ---
MAX_ITERATIONS = int(os.environ.get("TDD_MAX_ITERATIONS", 150))
MAX_TEST_PLAN_ITERATIONS = int(os.environ.get("TDD_MAX_TEST_PLAN_ITERATIONS", 10))
MAX_TEST_ITERATIONS = int(os.environ.get("TDD_MAX_TEST_ITERATIONS", 10))
TARGET_TEST_PLAN_COVERAGE = int(os.environ.get("TDD_TARGET_TEST_PLAN_COVERAGE", 95))
TARGET_TEST_COVERAGE = int(os.environ.get("TDD_TARGET_TEST_COVERAGE", 95))

# --- LLM Settings ---
MODEL_PRIMARY = os.environ.get("MODEL_PRIMARY", "gemma-4-31b-it")
MODEL_SECONDARY = os.environ.get("MODEL_SECONDARY", "gemma-4-26b-a4b-it")
LLM_SEED = int(os.environ.get("LLM_SEED", 123))
LLM_TOP_K = int(os.environ.get("LLM_TOP_K", 40))
LLM_TOP_P = float(os.environ.get("LLM_TOP_P", 0.95))
LLM_MAX_OUTPUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", 32768))

# --- Timeouts (Seconds) ---
FETCH_TIMEOUT_SEC = int(os.environ.get("TDD_FETCH_TIMEOUT_SEC", 10))
SYNTAX_CHECK_TIMEOUT_SEC = int(os.environ.get("TDD_SYNTAX_CHECK_TIMEOUT_SEC", 10))
TEST_EXECUTION_TIMEOUT_SEC = int(os.environ.get("TDD_TEST_EXECUTION_TIMEOUT_SEC", 30))
TOOL_TIMEOUT_SEC = int(os.environ.get("TDD_TOOL_TIMEOUT_SEC", 5))

# --- Linter & Testing Settings ---
FLAKE8_SELECT = os.environ.get("TDD_FLAKE8_SELECT", "E999,F821,F822")
PYTEST_MAXFAIL = int(os.environ.get("TDD_PYTEST_MAXFAIL", 0))


# --- System Settings ---
RUN_NAME = os.environ.get("TDD_RUN_NAME", "tdd_agent_run")
SESSION_ID = os.environ.get("TDD_SESSION_ID", f"tdd_session_{time.strftime('%Y%m%d_%H%M%S')}")
TDD_BASE_ARTIFACTS_DIR = os.environ.get("TDD_ARTIFACTS_DIR", "artifacts")
ARTIFACTS_DIR = os.path.join(TDD_BASE_ARTIFACTS_DIR, SESSION_ID)
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"
VERBOSE = os.environ.get("TDD_VERBOSE", "false").lower() == "true"

ORACLE_VERIFIER: typing.Optional[typing.Callable[[str], str]] = None
