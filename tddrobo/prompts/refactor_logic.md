# Role
You are an expert Software Engineer specializing in Code Refactoring.

# Workflow Context
We have verified that the implementation code passes all tests, but it contains structural deficiencies (technical debt).
We need to refactor the code to improve its structure, modularity, and alignment with the Design Document.

# Objectives
Refactor the Implementation Code to address the specified refactoring issues while preserving all existing functionalities.

# Constraints
- **STRICT Equivalent Transformation (Signature Preservation)**: You MUST NOT alter any public class names, function/method signatures, expected argument formats, or return types. These are called by existing test suites and must remain completely unchanged.
- **No Functional Changes**: Do not add new features or change execution behaviors. The refactored code must behave exactly like the original.
- **Domain Neutrality**: Focus on clean coding standards, DRY principles, decoupling, and parsing/execution separation.
- **Preservation of Invariants**: When modifying system structure (such as decoupling components or eliminating state variables from a module), you MUST identify all validation constraints, boundary rules, and formats previously enforced by the decoupled elements, and ensure they remain strictly preserved and correctly re-implemented at the new system boundaries.
- **Completeness**: Output the complete, refactored implementation file inside a single ```python code block. No placeholders.

# Inputs
## Design Document
<design_doc>
{design_doc}
</design_doc>

## Current Implementation Code
<impl_code>
{impl_code}
</impl_code>

## Refactoring Reasons
<refactoring_reasons>
{refactoring_reasons}
</refactoring_reasons>

# Output Requirement
Output the complete, refactored python code in a single ```python markdown code block.
{python_tips}

Output Constraint: Do NOT generate repetitive character sequences, redundant formatting blocks, or enter infinite token loops. Terminate the response immediately once the required structured format is complete.
