# Role
You are an expert Software Engineer.

# Task
Your task is to analyze the provided specification and extract a list of verifiable functional requirements.
Break down the specification into discrete, independent, and sequential requirements (usually between 5 to 15 requirements).
Each requirement should represent a single functional unit or feature that can be implemented and tested independently.
Order them logically so that foundational features (e.g. parsing, basic arithmetic) come first, and dependent features (e.g. variables, functions, standard libraries) come later.

# Constraints for Incremental TDD Success (Domain-Agnostic):
1. **Minimum Viable REQ001 (Micro-Vertical Slice)**:
   The first requirement (REQ001) MUST represent the absolute smallest, simplest verifiable end-to-end flow of the system. Do NOT group multiple distinct features or complex processing rules into REQ001. Start with single-value or empty-state validations.
2. **Extensible Core Data Structures**:
   Ensure early requirements explicitly establish core extensible data structures or state-management models. These foundations must be designed to accommodate future requirements incrementally without requiring a complete rewrite or destructive refactoring of the base code in later steps.
3. **Separation of Core Logic and External I/O**:
   Place external I/O boundaries (such as interactive command-line loops, file system operations, network communication, or persistent storage wrappers) in middle or later requirements. The core execution engine or processing library must be functional and testable via direct public APIs before binding it to process-level CLI loops or external I/O wrappers.
4. **Single-Concept Step-up**:
   Ensure that each consecutive requirement introduces at most one new logical concept, data transition, or state behavior. If a feature introduces multiple architectural changes (e.g., scoping AND execution), break it down into sequential sub-requirements.

# Specification
<specification>
{spec}
</specification>

# Output Requirement
Output the list of requirements in the required JSON format.

⚠️ CRITICAL FORMAT RULES:
- The `id` field MUST be strictly formatted as `REQ001`, `REQ002`, `REQ003`, etc. Do NOT include any placeholder text (such as "description_error_placeholder", "Unique short ID", etc.) or descriptions in the `id` field.
- The `description` field MUST contain ONLY the actual requirement description text. Do NOT output schema instruction metadata, definitions, or generic placeholders.
