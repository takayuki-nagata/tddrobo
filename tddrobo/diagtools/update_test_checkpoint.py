import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from tddrobo.agent import TDDAgent
from tddrobo.utils import FileMemorySaver

checkpoint_path = "artifacts/bc_clone_session/checkpoint.pkl"
test_file_path = "artifacts/bc_clone_session/test_py_bc_req012_integration.py"

if not os.path.exists(test_file_path):
    print("Error: test file not found")
    sys.exit(1)

with open(test_file_path, "r", encoding="utf-8") as f:
    slim_test_code = f.read()

memory_saver = FileMemorySaver(checkpoint_path)
tdd_agent = TDDAgent(checkpointer=memory_saver)
config_opts = {"configurable": {"thread_id": "bc_clone_session", "recursion_limit": 150}}

state_obj = tdd_agent.app.get_state(config_opts)
state = state_obj.values
print("Before update:")
print(f"  integration_tests_code length: {len(state.get('integration_tests_code', ''))}")
print(f"  tests_code length: {len(state.get('tests_code', ''))}")

tdd_agent.app.update_state(config_opts, {"integration_tests_code": slim_test_code, "tests_code": slim_test_code})

state_obj_new = tdd_agent.app.get_state(config_opts)
state_new = state_obj_new.values
print("After update:")
print(f"  integration_tests_code length: {len(state_new.get('integration_tests_code', ''))}")
print(f"  tests_code length: {len(state_new.get('tests_code', ''))}")
print("Checkpoint updated successfully!")
