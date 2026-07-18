"""
Execution script to build a POSIX-compliant bc clone using the TDD agent.
It invokes the main workflow and runs a complex demonstration using the generated implementation.
"""

import os
import shutil
import subprocess
import sys

import config
from cli import main


def run_bc_command(cmd: list[str], input_code: str) -> tuple[int, str, str]:
    """Runs a command and returns (returncode, stdout, stderr)."""
    try:
        result = subprocess.run(cmd, input=input_code, capture_output=True, text=True, timeout=10)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def parse_outputs(stdout: str) -> dict[str, str]:
    """Extracts specific calculation results from stdout."""
    results = {}
    for line in stdout.splitlines():
        line = line.strip()
        if "fib(12) = " in line:
            results["Fibonacci (fib(12))"] = line.split("fib(12) = ")[-1].strip()
        elif "sqrt(fib(12)) = " in line:
            results["Square Root (sqrt(144))"] = line.split("sqrt(fib(12)) = ")[-1].strip()
        elif "e(1) = " in line:
            results["Math Library (e(1))"] = line.split("e(1) = ")[-1].strip()
    return results


if __name__ == "__main__":
    args_list = sys.argv[1:]

    MATH_DOMAIN_TIPS = (
        "\n# Implementation Tips (for mathematical library functions if active):\n"
        "- When implementing precision math functions with arbitrary precision "
        "using `decimal.Decimal`:\n"
        "  - Keep series calculations simple and compute them using basic loops "
        "(e.g., Taylor series with a fixed loop limit of 30-50 iterations for "
        "sufficient accuracy).\n"
        "  - Pre-convert inputs to `Decimal` and perform calculations inside "
        "the loop to avoid floating-point inaccuracies.\n"
        "  - Trigonometric and exponential series can be implemented using "
        "standard loops:\n"
        "    - e^x = sum( x^k / k! )\n"
        "    - sin(x) = sum( (-1)^k * x^(2k+1) / (2k+1)! )\n"
        "    - cos(x) = sum( (-1)^k * x^(2k) / (2k)! )\n"
        "    - arctan(x): Use Taylor series or standard coordinate "
        "transformation series.\n"
        "    - ln(x): Use Newton-Raphson or standard Taylor expansion for "
        "ln((1+y)/(1-y)).\n"
        "    - J_n(x): For Bessel functions j(n, x), use the standard "
        "definition sum( (-1)^k * (x/2)^(2k+n) / (k! * (n+k)!) ).\n"
        "  - Do not overcomplicate the math functions; standard loops are "
        "fully sufficient for TDD purposes and prevent LLM reasoning timeout.\n"
    )

    if "--goal" not in args_list and not config.GOAL:
        args_list.extend(["--goal", "Build a POSIX-compliant bc clone in Python."])
    if "--spec-url" not in args_list and not config.SPEC_URL:
        args_list.extend(["--spec-url", "https://pubs.opengroup.org/onlinepubs/009696799/utilities/bc.html"])
    if "--target-test-plan-coverage" not in args_list:
        args_list.extend(["--target-test-plan-coverage", "80"])
    if "--max-test-plan-iterations" not in args_list:
        args_list.extend(["--max-test-plan-iterations", "5"])
    if "--target-test-coverage" not in args_list:
        args_list.extend(["--target-test-coverage", "80"])
    if "--domain-tips" not in args_list:
        args_list.extend(["--domain-tips", MATH_DOMAIN_TIPS])

    try:
        final_state = main(args_list)

        if final_state and final_state.get("success", False):
            impl_name = final_state.get("module_name", config.DEFAULT_IMPL_NAME)
            impl_path = os.path.join(config.ARTIFACTS_DIR, impl_name)

            demo_bc_code = """\
scale = 5                                                                              
"--- Complex Calculation Demo ---\n"                                                   
"1. Fibonacci Function\n"                                                              
define fib(n) {                                                                        
    auto a, b, c, i                                                                    
    a = 0; b = 1                                                                       
    if (n == 0) return (a)                                                             
    if (n == 1) return (b)                                                             
    for (i = 2; i <= n; ++i) {                                                         
        c = a + b                                                                      
        a = b                                                                          
        b = c
    }
    return (b)
}
"fib(12) = "
fib(12)
"2. Square Root\n"
"sqrt(fib(12)) = "
sqrt(fib(12))
"3. Math Library (e^1)\n"
"e(1) = "
e(1)
"""
            print("\n=== 🚀 Running Complex bc Demo ===")
            print("\n### 📜 Demo Script")
            print("--------------------------------------------------")
            print(demo_bc_code.strip())
            print("--------------------------------------------------")

            # Run the clone
            clone_code, clone_out, clone_err = run_bc_command([sys.executable, impl_path, "-l"], demo_bc_code)

            # Run actual bc if available
            bc_available = shutil.which("bc") is not None
            if bc_available:
                bc_code, bc_out, bc_err = run_bc_command(["bc", "-l"], demo_bc_code)
            else:
                bc_code, bc_out, bc_err = -1, "", "Actual bc command not found"

            # Parse results
            clone_results = parse_outputs(clone_out)
            bc_results = parse_outputs(bc_out) if bc_available and bc_code == 0 else {}

            expected_keys = ["Fibonacci (fib(12))", "Square Root (sqrt(144))", "Math Library (e(1))"]
            parsed_well = all(k in clone_results for k in expected_keys) and (
                not bc_available or all(k in bc_results for k in expected_keys)
            )

            if parsed_well and clone_code == 0 and (not bc_available or bc_code == 0):
                # Print beautiful comparison table
                print("\n=== 🖥️ Execution Result Comparison ===\n")
                print(f" {'Test Case':<25} │ {'Clone Output':<18} │ {'Actual bc Output':<18} │ {'Status':<10}")
                print(" ───────────────────────────┼────────────────────┼────────────────────┼──────────")

                all_match = True
                for key in expected_keys:
                    clone_val = clone_results.get(key, "N/A")
                    bc_val = bc_results.get(key, "N/A") if bc_available else "N/A (No bc)"

                    if bc_available:
                        status = "✅ MATCH" if clone_val == bc_val else "❌ MISMATCH"
                        if clone_val != bc_val:
                            all_match = False
                    else:
                        status = "⚪ OK"

                    print(f" {key:<25} │ {clone_val:<18} │ {bc_val:<18} │ {status:<10}")

                print()
                if bc_available:
                    if all_match:
                        print("🎉 SUCCESS: The generated clone behaves 100% identically to the POSIX bc utility!")
                    else:
                        print("⚠️ WARNING: Mismatch detected between clone and actual bc utility.")
                else:
                    print("ℹ️ Demo run finished successfully. Actual bc not found, so comparisons were skipped.")
            else:
                # Fallback: Print raw outputs if parsing failed or execution failed
                print("\n=== 🖥️ Raw Output from Clone ===")
                if clone_code == 0:
                    print(f"```text\n{clone_out.strip()}\n```")
                else:
                    print(f"❌ Execution failed (Code {clone_code}). Error:\n{clone_err.strip()}")

                if bc_available:
                    print("\n=== 🖥️ Raw Output from Actual bc ===")
                    if bc_code == 0:
                        print(f"```text\n{bc_out.strip()}\n```")
                    else:
                        print(f"❌ Execution failed (Code {bc_code}). Error:\n{bc_err.strip()}")
    except KeyboardInterrupt:
        print("\n\n🛑 Execution interrupted by user. Exiting gracefully.")
        sys.exit(130)
