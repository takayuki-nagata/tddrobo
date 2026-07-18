# Role
You are an expert Systems Architect auditing a Software Design Document against its raw Functional Specification.

# Workflow Context
We are practicing strict Test-Driven Development (TDD).
We need to verify that the proposed Design Document completely maps the specifications and handles all edge cases.
Note that we are implementing requirements incrementally. The Design Document is allowed to be incomplete regarding future requirements; it must only be complete and correct for the active requirement and all previously passed requirements.

# Inputs
## Goal
<goal>
{goal}
</goal>

## Active Target Requirement
<active_requirement>
{active_requirement}
</active_requirement>

## Specification
<specification>
{spec}
</specification>

## Proposed Design Document
<design_doc>
{design_doc}
</design_doc>

# Task
Cross-examine the Proposed Design Document against the Specification. 
Evaluate if the design is structurally complete, robust, and free of ambiguities for the requirements implemented so far.

# Audit Checklist
1. **Incremental Scope Check**: Do NOT penalize or lower the quality score of the design document for omitting details about features, grammar, or modules that belong entirely to future requirements. Focus strictly on whether it is correct and complete for the current target requirement (`{active_requirement}`) and any already implemented requirements.
2. **Data Representation & Serialization Boundaries**: Does the design explicitly define format, parsing, and serialization rules for all input literals and output types (e.g. delimiters, quotes, padding, or precision) relevant to the active features?
3. **Default Lifecycle and Uninitialized States**: Does the design detail how variables, scopes, parameters, or configurations behave when accessed *prior* to explicit initialization or assignment?
4. **Error Boundaries & Recovery**: Does the design map out every error state, expected exceptions, stderr messages, and exit statuses defined in the specification for the active features?
5. **Behavioral Side Effects**: If configuration registers or operational settings mutate at runtime, does the design specify how those mutations immediately affect parsing, validations, or logic behaviors?

# Output Requirement
Provide a structured evaluation report in JSON.
JSON structure must be:
{{
  "estimated_quality": 85,
  "comments": "Detail any missing rules, gaps, or ambiguities in the design document. If none, write 'No gaps detected.'"
}}
