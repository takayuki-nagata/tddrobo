# Output Requirement

⚠️ CRITICAL TOKEN LIMIT: Your output MUST be under 8000 tokens. If you exceed this, your response will be truncated and the code will break. Keep your Search/Replace blocks SMALL and TARGETED.

- **If the existing implementation code is EMPTY**:
  Return the complete, valid Python code inside a single ```python block. Do not use placeholders.

- **If the existing implementation code is NOT EMPTY**:
  You MUST return ONLY the changes using one or more Search/Replace blocks.

  🚫 ABSOLUTELY FORBIDDEN: Do NOT output the entire file content. Do NOT rewrite functions that are working correctly. Only change the specific lines needed to fix the failing tests.

  ✅ CORRECT: 2-5 small Search/Replace blocks targeting specific functions or sections (total output ~2000-5000 chars)
  ❌ WRONG: One giant Search/Replace block replacing the whole file (output >10000 chars)

  Format for each Search/Replace block:
  <<<<<<< SEARCH
  [Exact copy of the block of lines to replace from the existing code, including correct spaces and indentation]
  =======
  [The updated code lines to replace it with]
  >>>>>>> REPLACE

  Example:
  <<<<<<< SEARCH
  def add(a, b):
      return a + b
  =======
  def add(a, b):
      # Add two numbers
      return a + b
  >>>>>>> REPLACE

  Guidelines:
  1. The SEARCH section must match the existing code *exactly*, character-for-character, including indentation, spaces, and blank lines.
  2. Do not combine multiple disjoint changes into a single block with a large amount of unchanged lines in between. Use multiple separate Search/Replace blocks.
  3. To add new functions or imports, replace a nearby boundary (e.g. an existing import or a blank line between existing functions).
  4. Ensure your edits maintain valid syntax and correct indentation.
  5. Each Search/Replace block should be SMALL (under 30 lines in both SEARCH and REPLACE sections). If you need to change more, split into multiple blocks.
  6. NEVER include unchanged code in a Search/Replace block just for context, EXCEPT when it is necessary to make the SEARCH block unique (as described in rule 7). Otherwise, only include lines that are changing or immediately adjacent.
  7. Each SEARCH block MUST match exactly one location in the target file. If the lines you wish to replace appear in multiple places, you MUST include unique surrounding lines (such as the containing function signature, unique comments, or specific neighboring variable assignments) to make EACH SEARCH block unique.
     🚫 ABSOLUTELY FORBIDDEN: Do NOT output multiple identical SEARCH blocks in a single response, expecting them to be applied sequentially to different occurrences. Each SEARCH block must be distinct and uniquely identifiable. This is a valid exception to rule 6.
