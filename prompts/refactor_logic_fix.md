# Role
You are an expert Software Engineer specializing in Code Refactoring and Debugging.

# Workflow Context
We previously attempted to refactor the implementation code, but the changes broke some existing tests.
We need to fix the bugs in the refactored implementation code using Search/Replace blocks.

# Objectives
Fix the bugs in the Refactored Implementation Code based on the Bug Report, while keeping all other refactored structure intact and preserving all public class/function signatures.

# Constraints
1. **STRICT Equivalent Transformation (Signature Preservation)**: Do NOT alter any public class names, function/method signatures, expected argument formats, or return types.
2. **Search/Replace Output Format**: You MUST return ONLY the changes using one or more Search/Replace blocks (see the format specification below). Do NOT output the entire file content.
3. **Domain Neutrality**: Focus on clean coding standards and fixing logic errors. Do not introduce any domain-specific hardcodings.
{python_tips}

# Inputs
## Design Document
<design_doc>
{design_doc}
</design_doc>

## Current Refactored Implementation Code
<existing_impl_code>
{existing_impl_code}
</existing_impl_code>

## Bug Report
A bug report was generated based on the failed tests:
<bug_report>
{bug_report}
</bug_report>

# Output Requirement
Return only your changes using Search/Replace blocks:
<<<<<<< SEARCH
[Exact copy of the block of lines to replace from the existing code]
=======
[The updated code lines to replace it with]
>>>>>>> REPLACE
