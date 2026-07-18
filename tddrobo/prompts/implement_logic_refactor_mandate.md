## 🧹 STRUCTURAL REFACTORING MANDATE
The codebase has entered an implementation deadlock, and a structural refactoring has been planned in the Architectural Audit Report below. 
You MUST perform a major structural rewrite of the flawed components as specified in the report.

Key Constraints:
1. **Enforce the Planned Restructuring**: Restructure class designs, data representations, interfaces, or internal execution flows exactly as specified in the `refactoring_plan` section of the audit report.
2. **Discard Legacy Shortcuts**: Do NOT attempt to preserve legacy structures or use local shortcuts if the audit report demands structural changes. Replace flawed functions or rewrite class variables completely if required.
3. **Preserve Invariants & Edge Cases**: You must strictly preserve all low-level behaviors, boundary handlings, state variables, and API signatures identified in the "safeties_and_invariants" section of the audit report. Do NOT discard existing working logic for edge cases during refactoring.

Output Constraint:
Output only your changes using the strict Search/Replace block format:
<<<<<<< SEARCH
[Exact lines of existing code to modify]
=======
[New drop-in replacement code]
>>>>>>> REPLACE
