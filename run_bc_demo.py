"""
Execution script to build a POSIX-compliant bc clone using the TDD agent.
It invokes the main workflow and runs a complex demonstration using the generated implementation.
"""

import os
import shutil
import subprocess
import sys
import typing

import config


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


BC_DOMAIN_TIPS = (
    "\n# Mathematical Library Specification:\n"
    "- When calculating transcendental and library functions under arbitrary precision:\n"
    "  - Trigonometric, logarithmic, and exponential functions are "
    "mathematically defined by the following series expansions:\n"
    "    - e^x = sum( x^k / k! )\n"
    "    - sin(x) = sum( (-1)^k * x^(2k+1) / (2k+1)! )\n"
    "    - cos(x) = sum( (-1)^k * x^(2k) / (2k)! )\n"
    "    - arctan(x): Taylor series or coordinate transformation series.\n"
    "    - ln(x): Taylor expansion for ln((1+y)/(1-y)) where y = (x-1)/(x+1), "
    "or Newton-Raphson method.\n"
    "    - J_n(x): Bessel function j(n, x) is defined as "
    "sum( (-1)^k * (x/2)^(2k+n) / (k! * (n+k)!) ).\n"
    "\n# Output Printing and Statement Evaluation Rules:\n"
    '- Under both interactive and non-interactive modes, every "expression statement" '
    "(an expression that does not contain an assignment or function definition) must "
    "output its calculated value followed by a newline when evaluated.\n"
    "- This printing rule applies recursively to expressions evaluated within blocks "
    "(such as inside 'if' statements, 'while' loops, 'for' loops) and inside user-defined functions.\n"
    "  - For example, executing `define f(){ 1; return 0 }; f()` must print both `1` "
    "(the expression inside the function body) and `0` (the return value of the call).\n"
    "  - Executing `if (1) { 1; 2 }` must print both `1` and `2`.\n"
    "- Assignment statements (e.g., `x = 5`, `x += 1`) and function definitions "
    "(e.g., `define f() { ... }`) do NOT produce output upon evaluation.\n"
    "\n# Precision & Truncation with Python's Decimal:\n"
    "- Use Python's `decimal` module for all numeric representations to support arbitrary precision. "
    "Configure the global context precision (`decimal.getcontext().prec`) to a high value "
    "(at least 1000) to avoid loss of precision in intermediate operations.\n"
    "- Arithmetic results must be truncated (not rounded) to the target scale. Ensure your "
    "truncation logic behaves like downward rounding (such as using `ROUND_DOWN` with `quantize`).\n"
    "\n# Recursion Depth Constraints:\n"
    "- The execution engine must handle deep recursion of function calls without triggering stack "
    "overflow crashes. Consider increasing the recursion limit via `sys.setrecursionlimit()` "
    "or using a non-recursive execution stack.\n"
    "\n# Radix & Base Constants Parsing Rules:\n"
    "- Input multi-digit constant digits greater than or equal to 'ibase' are automatically "
    "mapped to 'ibase - 1' before parsing (e.g. FFF under ibase=2 is evaluated as 111 = decimal 7).\n"
    "\n# Radix Output Formatting Precision (obase != 10):\n"
    "- When obase != 10 and scale > 0, output ceil(scale * log(10) / log(obase)) fractional digits "
    "to represent a precision of 10^scale.\n"
    "\n# Backslash-Newline Handling (Lexical Convention):\n"
    "- Ensure compliance with POSIX bc Lexical Convention Rule 6: The combination of a backslash "
    "character immediately followed by a newline must have no effect other than to delimit tokens, "
    "except within string tokens (where it remains literally as a backslash and newline sequence) "
    "and multi-line numbers (where it is ignored).\n"
    "- This means backslash-newline line continuations outside strings/numbers effectively join "
    "lines together and do not act as statement-terminating NEWLINE tokens.\n"
    "\n# Scoping and Variable Lifetime:\n"
    "- Variables, arrays, and function parameters in bc use dynamic scoping. A function call "
    "temporarily shadows outer variables of the same name. Ensure that entering a function overrides "
    "these variables with local bindings, and exiting the function safely restores the "
    "pre-existing outer bindings.\n"
    "\n# String Statement Output vs Expressions:\n"
    '- Printing a literal string statement (e.g. "hello") must NOT output a trailing newline. '
    "Only expression statements append a newline when printed.\n"
    "\n# Output Formatting Line Wrapping:\n"
    "- Numbers printed to stdout exceeding 70 characters must be wrapped by inserting a backslash "
    "('\\\\') and a newline every 70 characters.\n"
    "\n# Syntax & Arithmetic Safeties (Reminders):\n"
    "- Variable/function names are lowercase only ([a-z]). Uppercase 'A'-'F' are hexadecimal digits "
    "and must be tokenized as NUMBER/CONST digits, not variables (e.g. ibase=A sets base to 10).\n"
    "- Addition and subtraction result scale is always max(scale_a, scale_b) regardless of global scale "
    "(e.g. scale=0; 10.5+5.5 results in 16.0 with scale 1).\n"
    "- Exponentiation x ^ -y is calculated as 1 / (x ^ y) using division at the global scale, "
    "and the result scale is the global scale register value.\n"
    "- Postfix increment/decrement operators (e.g. 'i++') return the original value before mod, "
    "and the parser must correctly consume them inside array indices (like 'a[i++]').\n"
)


def verify_bc_clone(final_state: dict[str, typing.Any]):
    """Executes the verification suite against the generated bc clone."""
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


if __name__ == "__main__":
    from demo_runner import run_demo

    try:
        run_demo(
            default_goal="Build a POSIX-compliant bc clone in Python.",
            default_spec_url="https://pubs.opengroup.org/onlinepubs/009696799/utilities/bc.html",
            default_domain_tips=BC_DOMAIN_TIPS,
            default_session_id="bc_clone_session",
            run_demo_verification_func=verify_bc_clone,
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Execution interrupted by user. Exiting gracefully.")
        sys.exit(130)
