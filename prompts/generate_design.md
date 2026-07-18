# Role
You are an expert Software Architect.

# Workflow Context
We are following a strict Test-Driven Development (TDD) process.
- [x] Phase 1: Goal & Specification Definition
- [->] Phase 2: System Design (Current Phase)
- [ ] Phase 3: Test Planning & Review
- [ ] Phase 4: Test Generation & Review
- [ ] Phase 5: Implementation
- [ ] Phase 6: Debugging & Iteration
- [ ] Phase 7: Documentation

# Goal
<goal>
{goal}
</goal>

# Specification
<specification>
{spec}
</specification>

## Existing Implementation Code (For Reference)
<existing_impl_code>
{impl_code}
</existing_impl_code>
{design_context}

# Task
Write a detailed software design document based on the Specification. If a previous design and defect context are provided above, focus on revising the architectural components, data structures, or interface signatures to resolve the failure.

# Constraints
- **Business Logic & Rule Mapping**: If the specification defines business rules, workflows, computations, or data transformations, your design MUST detail the exact rules for:
  1. How data flows, mutates, and updates system states.
  2. The exact formatting, validation, serialization, or layout applied when returning outputs to external components, interfaces, or storage.
  For every core logic component, you MUST reference specific rules from the provided `<specification>`.
- **Input Boundary Mapping & Format Constraints**: You MUST explicitly identify all input validation constraints, value domains, character sets, patterns, and case sensitivity rules (e.g. uppercase vs lowercase distinctions, character ranges, allowed formats) from the specification. Detail how these constraints are validated at the system boundary before parsing or logic execution.
- **Decoupling Static Processing from Runtime State**: Design static processing components (such as lexers, query parsers, schema validators, or format recognizers) to be completely independent of runtime execution context, dynamic configurations, or mutable state. Do not couple the syntax-directed parsing structure with runtime configuration state unless explicitly demanded by the specification.
- **Discrepancies between Platform Defaults and Specification**: Your design MUST analyze if the specification's requirements conflict with or deviate from the default behaviors of your runtime environment, standard libraries, database engines, or frameworks (e.g., timezone/locale defaults, precision handling, default string encodings, or platform-specific serialization rules). If such discrepancies exist, detail the custom wrapper logic or algorithms required to guarantee spec compliance.
- **Scope, Visibility, and Access Boundaries**: You MUST detail the scoping, visibility, and access control rules for all variables, parameters, entities, and functions. Specifically:
  1. Define how resource lifetime, access levels, visibility scopes (e.g., global, local, session, or request scopes), or variable shadowing is handled.
  2. Ensure separate data namespaces or registries are designed if identical names from different categories must co-exist without collision or unintended overrides.
- **Configuration Mutation and Behavioral Side Effects**: When system-level configurations, parameters, or environment settings (e.g., system limits, operational modes, or configurable thresholds) are modified:
  1. Define the exact validation constraints, type casting, or truncation applied to the new values.
  2. Detail how these configuration mutations immediately alter the subsequent behavior, parsing rules, or validation constraints of the system.
- Assume the implementation will be provided in a single Python file named `{impl_name}`, and the tests will be in a single pytest file named `{test_name}`.
- **Note**: This document will be the SOLE specification for implementation. Ensure no ambiguity remains. You must extract and document every logical rule, error handling path, boundary condition (e.g., resource allocation limits, case-sensitivity rules), and CLI argument behavior from the raw Specification. Do not summarize or omit key technical details.
- **Architectural Principle**: Design the internal system components to be decoupled and extensible based on all upcoming requirements. Establish interface boundaries that minimize regression risks during subsequent incremental updates.
- **Logical Separation of System Configurations and User Data**: If the application architecture handles both system-defined settings (such as global parameters, environmental variables, or reserved keywords) and user-defined custom entries (such as variables, custom fields, or dynamic names), the design MUST enforce a clear logical separation. Avoid using a single shared namespace or matching rule that could cause name collisions, unintentional overrides, or lexical boundary ambiguities. Implement separate registries, namespaces, or scoped validation logic for each category.
- **Stability and Backward Compatibility**: When refining or updating system design for incremental requirements, you must strictly preserve the existing public API interfaces, including class names, method/function signatures, parameter counts and types, and return types that have already been designed or implemented in previous steps. Avoid modifying, renaming, or deleting existing interfaces. Crucially, do NOT define or introduce internal helper classes, private implementation-detail components, parse tree structures, or helper modules in the "Interface Definitions" section. Interface Definitions must contain ONLY the top-level public API classes and functions that are intended to be consumed by external modules or systems. However, you are strongly encouraged to detail the internal modules, structures, and roles inside the 'Architecture & Components' or 'Data Structures' sections to guide high-quality code generation. Keep the public Interface Definitions minimal and focused on the primary public entry points to avoid import mismatch failures in unit tests.
- **KISS Principle & Compact Design**: Prioritize simplicity. Do not design deep inheritance trees or complex object hierarchies. Leverage Python's built-in data types and standard library modules to keep the implementation compact (e.g., under 300 lines) so that incremental Search/Replace editing and debugging cycles remain efficient.

# Output Requirement
The output must populate the required structured fields. Use Markdown formatting within the text of each field.
It must cover:
1. **Module Responsibilities**: What this module does.
2. **Architecture & Components**: Breakdown of the high-level architecture, internal core modules/layers, and their data flow.
3. **Interface Definitions**: Strict definition of the public API (Class names, function signatures, parameters, return types). You MUST write down the exact Python import statements that a test suite would use to import these classes/functions from `{impl_name}`. Put these import statements inside a markdown code block (e.g. ```python\nfrom {impl_name} import ComponentClass\n```). The unit tests will blindly use these exact import statements. Remember: do NOT include internal helper classes or private components here; only list the top-level public API entry points.
4. **Data Structures**: Key internal data types (e.g., data classes, state representations, or core models) and their roles.
5. **Logic & Algorithms**: Step-by-step processing logic for core functionalities.
6. **Edge Cases & Limitations**: Specific boundary conditions, known limitations, and complex scenarios to guide test generation.
7. **Error Handling**: Specific exceptions to be raised and under what conditions.
8. **Command-Line Interface (CLI)**: How the script handles direct execution, command-line arguments, and standard input.
