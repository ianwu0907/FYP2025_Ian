"""
Transformation Generator Module (Redesigned)

Architecture:
  1. CODE RECIPE BOOK: For each irregularity type in the taxonomy,
     provide a concrete pandas code pattern that addresses it.
     These are NOT templates to fill — they are reference examples
     the LLM adapts to the specific data.

  2. SINGLE CODE GENERATION CALL: One LLM call that receives:
     - The target schema (from SchemaEstimator)
     - Detected irregularities + their code recipes
     - Full source data
     - A complete few-shot example of a transform function
     The LLM writes a `def transform(df)` function.

  3. EXECUTE → VALIDATE → FEEDBACK LOOP: Run the code, check
     output against schema expectations, regenerate with error
     context if validation fails.

Design principles:
  - The old Strategy stage is removed. Weak models produced vague
    strategies that led to bad code. Instead, the code recipes
    provide concrete implementation guidance directly.
  - Few-shot: a complete working transform function is shown.
  - Full data is passed — no truncation.
  - Feedback loop provides the actual error/output for targeted fixes.
  - wide-format strategy (melt vs loop) is decided in Python, not by LLM.
"""

import json
import logging
import os
import re
import traceback
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from openai import OpenAI

from ..detector.irregularity_detector import get_code_guidance_for

logger = logging.getLogger(__name__)


# ============================================================================
# Code Recipe Book
# ============================================================================
# Each recipe is a concrete pandas code pattern that handles one
# irregularity type. The LLM sees these as reference material and
# adapts them to the specific spreadsheet.
#
# Recipes use generic variable names (df, result, etc.) so the LLM
# can see the PATTERN, not a hard-coded solution.
# ============================================================================

CODE_RECIPES = {
    "METADATA_ROWS": "# Skip metadata rows in your loop using `if i < {data_start_row}: continue`",

    "MULTI_LEVEL_HEADER": '''
# RECIPE: MULTI_LEVEL_HEADER
# 1. Forward-fill the header rows horizontally:
# headers = df.iloc[{header_row_indices}].ffill(axis=1)
# 2. In your nested `for j` loop, use the BOTTOM-most row of `headers` for the specific category.
#    DO NOT try to extract row-level dimensions (like Year in column 0) from the `headers` DataFrame.
#    Row dimensions should always be extracted from `df.iloc[i, ...]`.
'''.strip(),

    # WIDE_FORMAT is handled dynamically by _get_relevant_recipes()
    # based on the _is_simple_wide() check — do not add a static recipe here.
    "WIDE_FORMAT": None,

    "BILINGUAL_ALTERNATING_ROWS": '''
# RECIPE: BILINGUAL_ALTERNATING_ROWS
# Detect the primary-language row by checking which row of each pair has
# numeric values in the value columns — do NOT rely on row-index parity alone,
# as the primary language may be at an even or odd index depending on the table.
#
# Recommended pattern:
#   rows_to_keep = []
#   for i in range(data_start_row, data_end_row + 1):
#       has_data = any(
#           pd.notna(df.iloc[i, j]) and str(df.iloc[i, j]).strip()
#           for j in value_col_indices   # derive from headers, not hardcoded
#       )
#       if has_data:
#           rows_to_keep.append(i)
#
# If the primary row contains Count data and the secondary row contains
# Percentage data, extract BOTH metrics simultaneously:
#    count_val = df.iloc[i, j]
#    pct_val   = df.iloc[i+1, j] if i+1 < len(df) else None
'''.strip(),

    "NESTED_COLUMN_GROUPS": '''
# RECIPE: NESTED_COLUMN_GROUPS — Build a column mapping with group structure
# After reading multi-level headers, create a structured mapping:
# column_map = {col_index: (group_value, sub_value)}
# Example for year x value_type:
#   {3: ("2011", "Number"), 4: ("2011", "%"), 5: ("2016", "Number"), ...}
# Example for marital_status x sex:
#   {3: ("Never married", "Male"), 4: ("Never married", "Female"), ...}
#
# Forward-fill group labels across columns (groups span multiple cols):
group_labels = {}
current_group = None
for j in range(len(df.columns)):
    val = df.iloc[group_header_row, j]
    if pd.notna(val) and str(val).strip():
        current_group = str(val).strip()
    if current_group:
        group_labels[j] = current_group

# Then unpivot using the mapping:
records = []
for _, row in result.iterrows():
    base = {dim_col: row[dim_col_idx] for dim_col, dim_col_idx in id_columns}
    for col_idx, (group_val, sub_val) in column_map.items():
        record = {**base, group_dim_name: group_val, sub_dim_name: sub_val}
        record[value_col_name] = row.iloc[col_idx]
        records.append(record)
result = pd.DataFrame(records)
''',

    "INLINE_BILINGUAL": '''
# RECIPE: INLINE_BILINGUAL — Split cell content by detected separator
# CRITICAL: Do NOT assume "\\n" is the separator.
# First inspect the actual bilingual cells from the EVIDENCE to find the
# real separator. Common separators: "\\n", "/", "(", or a space between
# CJK and Latin characters.
#
# sep = "<separator found in the EVIDENCE for this spreadsheet>"
#
def split_bilingual(val, sep):
    """Split a bilingual cell value into (primary_lang, secondary_lang) parts."""
    if pd.isna(val):
        return val, val
    s = str(val)
    if sep in s:
        parts = s.split(sep, 1)
        return parts[0].strip(), parts[1].strip()
    return s.strip(), s.strip()

# Apply to header cells and data cells as needed.
# Pass the detected separator: split_bilingual(cell_value, sep)
''',

    "SECTION_HEADER_ROWS": '''
# RECIPE: SECTION_HEADER_ROWS — Forward-fill section labels
# Identify section header rows (label in first col, numeric cols all empty)
# Create a new column and forward-fill
result["section"] = None
for i in range(len(result)):
    row = result.iloc[i]
    if pd.notna(row.iloc[label_col_idx]) and all(
        pd.isna(row.iloc[j]) or str(row.iloc[j]).strip() == ""
        for j in numeric_col_indices
    ):
        result.iloc[i, result.columns.get_loc("section")] = row.iloc[label_col_idx]
# Forward fill the section column
result["section"] = result["section"].fillna(method="ffill")
# Drop the section header rows themselves (they have no numeric data)
result = result.dropna(subset=[result.columns[j] for j in numeric_col_indices], how="all")
''',

    "IMPLICIT_AGGREGATION_ROWS": '''
# RECIPE: IMPLICIT_AGGREGATION_ROWS — Remove redundant aggregation rows
#
# *** DO NOT hardcode a keyword list. ***
# The schema's EXCLUDE_ROWS field and the detection EVIDENCE already identify
# which specific labels and category values must be removed. Use those directly.
#
# How to build the exclusion logic for each form:
#
# Form 1 — Semantic hierarchy aggregation (parent rows with no keyword signal):
#   Read the detection EVIDENCE. It will name the specific parent-level labels.
#   semantic_agg_labels = ["<parent label from EVIDENCE>", ...]
#   result = result[~result.iloc[:, label_col_idx].isin(semantic_agg_labels)]
#
# Form 2 — Cross-group aggregation (entire coarser category group):
#   The data has a CATEGORY column whose values distinguish coarse vs. granular
#   groups (e.g. col 3 = "Type of Abuse" vs "Type of Abuse and Sex").
#   Filter on that CATEGORY column using the exact coarser value from EXCLUDE_ROWS.
#
#   coarser_categories = ["<exact coarser category value from EXCLUDE_ROWS>"]
#   result = result[~result.iloc[:, category_col_idx].isin(coarser_categories)]
#
#   TWO CRITICAL RULES:
#   1. Filter on the CATEGORY column (the one that distinguishes group granularity),
#      NOT on the item/label column (the one with specific item names like abuse types).
#      Filtering the item column removes valid granular rows that share the same
#      item name across different category groups.
#   2. Use .isin() for EXACT match — NEVER str.contains().
#      Coarser category names are often substrings of finer ones
#      (e.g. "Type of Abuse" is contained in "Type of Abuse and Sex"),
#      so str.contains() silently deletes valid granular rows.
''',

    "AGGREGATE_COLUMNS": '''
# RECIPE: AGGREGATE_COLUMNS — Drop aggregate columns before reshaping
# Read the schema EXCLUDE_COLUMNS field to find which column indices to drop.
# These are typically the last column in each repeating group or standalone
# summary columns. Use the indices from the schema, not a keyword scan.
agg_col_indices = [indices_of_aggregate_columns]  # from schema EXCLUDE_COLUMNS
result = result.drop(columns=[result.columns[i] for i in agg_col_indices
                               if i < len(result.columns)], errors="ignore")
''',

    "EMBEDDED_DIMENSION_IN_COLUMN": '''
# RECIPE: EMBEDDED_DIMENSION_IN_COLUMN — Conditional split
# CRITICAL: Do NOT hardcode a delimiter. Read the EVIDENCE field from the
# detected irregularity to find the exact delimiter for THIS spreadsheet.
# Common examples: " - ", " – ", " / ", ":", "_" — always verify from the data.
#
# delimiter = "<the delimiter identified in the EVIDENCE>"  ← set from actual data
#
def split_embedded(val, delimiter):
    if pd.isna(val):
        return val, None
    s = str(val).strip()
    if delimiter in s:
        parts = s.split(delimiter, 1)
        return parts[0].strip(), parts[1].strip()
    return s, None

# Apply to the compound column (replace compound_col_idx with the actual index):
result[["primary_dim", "secondary_dim"]] = result.iloc[:, compound_col_idx].apply(
    lambda x: pd.Series(split_embedded(x, delimiter))
)
''',

    "SPARSE_ROW_FILL": '''
# RECIPE: SPARSE_ROW_FILL
# For columns where a value like 'Year' is written once and implies the same for rows below:
# CRITICAL: Replace empty strings with NaN before forward-filling!
# df.iloc[:, col_idx] = df.iloc[:, col_idx].replace(r'^\\s*$', np.nan, regex=True).ffill()
'''.strip(),
}


# ============================================================================
# Wide-format recipes — selected at runtime by _get_relevant_recipes()
# based on _is_simple_wide(). Never shown together; LLM always receives
# exactly one path with no conditional branching to evaluate.
# ============================================================================

_WIDE_FORMAT_MELT_RECIPE = '''
# RECIPE: WIDE_FORMAT (simple) — use pd.melt()
# Applicable because WIDE_FORMAT is the only structural irregularity.
# Wide format column headers may be numeric (years) OR non-numeric
# (income brackets like "<$10k", age groups like "0-14", region names, etc.).
# Treat all header values as strings when passing to value_vars.
#
# Step 1: Slice to data region and ensure string column names for melt
result = df.iloc[{data_start_row}:{data_end_row} + 1].copy()
result.columns = [str(c) for c in result.columns]
result = result.dropna(how="all")
#
# Step 2: Identify id_vars and value_vars from the HEADERS shown above
#   id_col_indices    = [...]   # columns that stay as dimensions
#   value_col_indices = [...]   # columns to unpivot
#   dim_col_name      = "..."   # name for the new dimension column (from schema)
#   value_col_name    = "..."   # name for the value column (from schema)
#
id_vars    = [result.columns[i] for i in id_col_indices]
value_vars = [result.columns[i] for i in value_col_indices]
result = pd.melt(result,
                 id_vars=id_vars,
                 value_vars=value_vars,
                 var_name=dim_col_name,
                 value_name=value_col_name)
# Step 3: Rename columns to exactly match the target schema
result = result.rename(columns={{...}})
# Step 4: Drop rows where the value column is null (unpivoted empty cells)
result = result.dropna(subset=[value_col_name])
'''.strip()

_WIDE_FORMAT_LOOP_RECIPE = '''
# RECIPE: WIDE_FORMAT (complex) — use record-collection loop
# pd.melt() is NOT suitable here because co-occurring irregularities
# (MULTI_LEVEL_HEADER, NESTED_COLUMN_GROUPS, or BILINGUAL_ALTERNATING_ROWS)
# require irregular row handling that melt cannot perform.
# Wide format column headers may be numeric OR non-numeric strings —
# read the actual header row values to build your column mapping.
# Do NOT assume headers are numeric; convert them to str before mapping.
# See MULTI_LEVEL_HEADER / NESTED_COLUMN_GROUPS recipes for the loop pattern.
'''.strip()


# ============================================================================
# TransformationGenerator
# ============================================================================

class TransformationGenerator:
    """
    Generates transformation code using a code-recipe-guided approach:

    1. Build prompt with code recipes for detected irregularities
    2. Single LLM call generates a `def transform(df)` function
    3. Execute → Validate → Feedback loop

    Wide-format strategy (melt vs loop) is decided deterministically in
    Python via _is_simple_wide() before any LLM call is made. The LLM
    always receives a single, unambiguous instruction — never a branch.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        base_url = os.getenv("OPENAI_BASE_URL")
        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        self.client = OpenAI(**kw)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_completion_tokens = config.get("max_completion_tokens", 6000)
        self.max_retries = config.get("max_retries", 5)
        self._last_execution_log = []  # captured print output from last execution
        self._simple_wide = False      # set during _generate_code; reused in _regenerate_code

    # ==================================================================
    # Public API
    # ==================================================================

    def generate_and_execute(self,
                             encoded_data: Dict[str, Any],
                             detection_result: Dict[str, Any],
                             schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and execute transformation code.

        Args:
            encoded_data:     From SpreadsheetEncoder
            detection_result: From IrregularityDetector
            schema:           From SchemaEstimator

        Returns:
            Dict with: normalized_df, transformation_code,
            validation_result, attempts
        """
        logger.info("=" * 60)
        logger.info("TRANSFORMATION GENERATOR")
        logger.info("=" * 60)

        df = encoded_data["dataframe"]
        labels = detection_result.get("labels", [])
        physical = detection_result.get("physical", {})

        # Stage 1: Generate code
        logger.info("\n--- Stage 1: Code Generation ---")
        code = self._generate_code(
            df, physical, detection_result, schema, labels
        )
        logger.info(f"Code generated ({len(code)} chars)")

        # Stage 2: Execute and validate with retry loop
        logger.info("\n--- Stage 2: Execution & Validation ---")

        for attempt in range(self.max_retries):
            logger.info(f"\nAttempt {attempt + 1}/{self.max_retries}")
            self._last_execution_log = []  # reset before each attempt

            try:
                result_df = self._execute_code(code, df)
                logger.info(f"Code executed. Result shape: {result_df.shape}")

                validation = self._validate_result(result_df, schema)

                if validation["is_valid"]:
                    logger.info("✓ Validation PASSED")
                    return {
                        "normalized_df": result_df,
                        "transformation_code": code,
                        "transformation_strategy": {"approach": "code_recipe_guided"},
                        "validation_result": validation,
                        "attempts": attempt + 1,
                    }
                else:
                    logger.warning(f"✗ Validation FAILED: {validation['errors']}")
                    if attempt < self.max_retries - 1:
                        feedback = self._build_feedback(
                            validation, result_df, df, schema
                        )
                        code = self._regenerate_code(
                            df, physical, detection_result, schema,
                            labels, code, feedback
                        )

                    else:
                        logger.error("Max retries reached")
                        return {
                            "normalized_df": result_df,
                            "transformation_code": code,
                            "transformation_strategy": {"approach": "code_recipe_guided"},
                            "validation_result": validation,
                            "attempts": attempt + 1,
                        }

            except Exception as e:
                logger.error(f"Execution error: {e}")
                if attempt < self.max_retries - 1:
                    feedback = {
                        "error_type": "EXECUTION_ERROR",
                        "error_message": f"{type(e).__name__}: {str(e)}",
                        "error_trace": traceback.format_exc(),
                        "execution_log": self._last_execution_log,
                    }
                    code = self._regenerate_code(
                        df, physical, detection_result, schema,
                        labels, code, feedback
                    )
                else:
                    logger.error(f"Max retries reached with execution error. Falling back to raw data.")
                    return {
                        "normalized_df": df.copy(),
                        "transformation_code": code,
                        "transformation_strategy": {"approach": "failed_execution"},
                        "validation_result": {
                            "is_valid": False,
                            "errors": [f"Final execution attempt crashed: {type(e).__name__} - {str(e)}"],
                            "warnings": []
                        },
                        "attempts": attempt + 1,
                    }

        raise RuntimeError("Transformation failed after all retries")

    # ==================================================================
    # Wide-format complexity check (deterministic — no LLM judgment)
    # ==================================================================

    @staticmethod
    def _is_simple_wide(labels: List[str]) -> bool:
        """
        Returns True when the table is wide-format but has no co-occurring
        structural irregularities that prevent pd.melt() from being used.

        This check is a pure set operation on detected labels.
        It is intentionally performed in Python before any LLM call so
        that the LLM always receives a single unambiguous instruction
        rather than a conditional branch to evaluate itself.
        """
        if "WIDE_FORMAT" not in labels:
            return False
        blocking = {"MULTI_LEVEL_HEADER", "NESTED_COLUMN_GROUPS", "BILINGUAL_ALTERNATING_ROWS", "EMBEDDED_DIMENSION_IN_COLUMN"}
        return blocking.isdisjoint(labels)

    # ==================================================================
    # Code Generation
    # ==================================================================

    def _generate_code(self, df: pd.DataFrame,
                       physical: Dict[str, Any],
                       detection_result: Dict[str, Any],
                       schema: Dict[str, Any],
                       labels: List[str]) -> str:
        """Generate transformation code with code recipes as guidance."""

        # Decide wide-format strategy in Python — store for reuse in regeneration
        self._simple_wide = self._is_simple_wide(labels)
        if self._simple_wide:
            logger.info("  Wide-format strategy: pd.melt() "
                        "(WIDE_FORMAT only, no blocking irregularities)")
        elif "WIDE_FORMAT" in labels:
            logger.info("  Wide-format strategy: record-collection loop "
                        "(blocking irregularities present: "
                        f"{[l for l in labels if l in ('MULTI_LEVEL_HEADER','NESTED_COLUMN_GROUPS','BILINGUAL_ALTERNATING_ROWS')]})")

        prompt = self._build_code_prompt(
            df, physical, detection_result, schema, labels, self._simple_wide
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt(self._simple_wide)},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self.max_completion_tokens,
            )
            code = resp.choices[0].message.content.strip()
            code = self._extract_code(code)
            logger.debug(f"Generated code:\n{code[:800]}...")
            return code

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            raise

    def _system_prompt(self, simple_wide: bool = False) -> str:
        common_header = (
            "You are an expert Python/pandas programmer. You write a "
            "`def transform(df)` function that transforms a messy "
            "spreadsheet DataFrame into STRICT TIDY DATA format.\n\n"

            "*** TARGET PARADIGM: TIDY DATA ***\n"
            "1. Each variable forms a column.\n"
            "2. Each observation forms a row.\n"
            "3. Each type of observational unit forms a table "
            "(drop Total/Average/Summary rows).\n\n"

            "RULES:\n"
            "1. Use only pandas, numpy, and standard library.\n"
            "2. Use .iloc for positional access (column INDICES, not names). "
            "The input df has numeric column names (0, 1, 2...).\n"
            "3. NEVER use `raise`, `assert`, or explicit row count checks "
            "(e.g., `if len(records) < expected`).\n"
            "4. Output ONLY the Python code. No markdown. No explanation.\n\n"
        )

        if simple_wide:
            mandatory = (
                "*** MANDATORY PATTERN: pd.melt() ***\n"
                "This table has WIDE_FORMAT with no blocking irregularities. "
                "Use pd.melt() to unpivot — do NOT use a manual record loop.\n"
                "```python\n"
                "import pandas as pd\n"
                "import numpy as np\n\n"
                "def transform(df):\n"
                "    print(f\"Input shape: {df.shape}\")\n\n"
                "    result = df.iloc[data_start_row:data_end_row + 1].copy()\n"
                "    result.columns = [str(c) for c in result.columns]\n"
                "    result = result.dropna(how='all')\n\n"
                "    result = pd.melt(result,\n"
                "                     id_vars=[...],       # dimension columns to keep\n"
                "                     value_vars=[...],    # wide columns to unpivot\n"
                "                     var_name='...',      # new dimension column name\n"
                "                     value_name='...')    # value column name\n"
                "    result = result.rename(columns={...}) # rename to exact schema names\n"
                "    result = result.dropna(subset=['...']) # drop rows with no value\n"
                "    print(f\"Final output: {result.shape}\")\n"
                "    return result\n"
                "```\n"
            )
        else:
            mandatory = (
                "*** MANDATORY PATTERN: record-collection loop ***\n"
                "This table has co-occurring structural irregularities. "
                "Use a record-collection loop — do NOT use pd.melt().\n"
                "```python\n"
                "import pandas as pd\n"
                "import numpy as np\n\n"
                "def transform(df):\n"
                "    print(f\"Input shape: {df.shape}\")\n\n"
                "    # 1. Clean and forward-fill sparse dimension columns on the raw df BEFORE the loop if needed.\n"
                "    # 2. Isolate multi-level headers and forward-fill horizontally if needed.\n\n"
                "    records = []\n"
                "    for i in range(len(df)):\n"
                "        # Skip rows to exclude\n"
                "        \n"
                "        # Extract row-level dimensions\n"
                "        \n"
                "        # Extract values (use nested `for j in range(...)` for wide columns)\n"
                "        # CRITICAL: dict keys MUST EXACTLY MATCH the target schema column names!\n"
                "        # records.append({\"schema_col_1\": val1, \"schema_col_2\": val2})\n\n"
                "    target_cols = ['schema_col_1', 'schema_col_2']  # REPLACE with exact schema columns\n"
                "    result = pd.DataFrame(records, columns=target_cols)\n"
                "    print(f\"Final output: {result.shape}\")\n"
                "    return result\n"
                "```\n"
            )

        return common_header + mandatory
    # 新增方法
    def _format_header_lineage(self, df: pd.DataFrame,
                               physical: Dict[str, Any],
                               labels: List[str]) -> str:
        """
        When NESTED_COLUMN_GROUPS is present, reconstruct the ancestor
        path for each data column by forward-filling header rows.
        This makes implicit hierarchy explicit for the LLM.
        """
        if "NESTED_COLUMN_GROUPS" not in labels and "MULTI_LEVEL_HEADER" not in labels:
            return "(no nested groups detected)"

        start = physical["data_start_row"]
        if start < 2:
            return "(single header row, no lineage needed)"

        # Forward-fill each header row horizontally (mimics merged cell semantics)
        header_rows = []
        for i in range(start):
            row = []
            last_val = None
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    last_val = str(val).strip()
                row.append(last_val)
            header_rows.append(row)

        # Build lineage string per column
        lines = []
        for j in range(len(df.columns)):
            path_parts = []
            for row in header_rows:
                v = row[j]
                if v and (not path_parts or v != path_parts[-1]):
                    path_parts.append(v)
            if path_parts:
                lines.append(f"  Col {j}: {' > '.join(path_parts)}")

        return "\n".join(lines) if lines else "(unable to reconstruct lineage)"
    def _build_code_prompt(self, df: pd.DataFrame,
                           physical: Dict[str, Any],
                           detection_result: Dict[str, Any],
                           schema: Dict[str, Any],
                           labels: List[str],
                           simple_wide: bool = False) -> str:

        lineage_text = self._format_header_lineage(df, physical, labels)
        recipes_text = self._get_relevant_recipes(labels, physical, simple_wide)
        schema_text = self._format_schema(schema)
        headers_text = self._format_headers(df, physical)
        data_text = self._format_data(df, physical)
        irregularity_text = self._format_irregularities(
            detection_result.get("irregularities", [])
        )
        col_types = physical.get("column_dtype_profile", {})
        target_cols = [c["name"] for c in schema.get("target_columns", [])]
        est_rows = schema.get("expected_output", {}).get(
            "row_count_estimate", "unknown"
        )

        # Choose few-shot example to match the selected strategy
        few_shot = self._few_shot_melt() if simple_wide else self._few_shot_loop()

        return f"""Write a `def transform(df):` function to convert this spreadsheet to tidy format.

=== TARGET SCHEMA ===
{schema_text}

=== DETECTED IRREGULARITIES ===
{irregularity_text}

=== CODE RECIPES (adapt these patterns to the specific data) ===
{recipes_text}

=== SOURCE DATA ===
Shape: {df.shape[0]} rows × {df.shape[1]} columns
Column types: {col_types}

HEADERS:
{headers_text}

COLUMN LINEAGE (semantic path for each data column):
{lineage_text}

DATA (full data region):
{data_text}

=== REQUIREMENTS ===
1. def transform(df): → returns DataFrame
2. Use .iloc for column access (positional indices)
3. Output columns MUST be exactly: {target_cols}
4. Expected output: ~{est_rows} rows
5. Add print() after each step
6. Adapt the code recipes above to THIS specific data

{few_shot}

Now write the transform function for THIS spreadsheet. Output ONLY code."""

    # ==================================================================
    # Few-shot examples (one per strategy)
    # ==================================================================

    @staticmethod
    def _few_shot_melt() -> str:
        return """\
=== FEW-SHOT EXAMPLE ===
⚠️  WARNING: The values below (row indices, column indices, column names)
are SPECIFIC TO THIS EXAMPLE spreadsheet. For YOUR spreadsheet, derive
all such values from the HEADERS, PHYSICAL FEATURES, and SOURCE DATA above.
Do NOT copy any number or string literal from this example into your code.

This example had: WIDE_FORMAT only (no blocking irregularities).
Example physical features: data_start_row=1, data_end_row=10,
id columns at indices 0-1, value columns at indices 2-4 (years 2019/2020/2021).

```python
import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    # 1 = data_start_row, 11 = data_end_row+1 — READ FROM PHYSICAL FEATURES FOR YOUR DATA
    result = df.iloc[1:11].copy()
    result.columns = [str(c) for c in result.columns]
    result = result.dropna(how="all")
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Melt wide → long
    # id_vars / value_vars derived from HEADERS above — EXAMPLE-SPECIFIC indices
    id_vars    = [result.columns[0], result.columns[1]]   # cols 0,1 = dimensions
    value_vars = [result.columns[2], result.columns[3], result.columns[4]]  # cols 2-4 = years
    result = pd.melt(result,
                     id_vars=id_vars,
                     value_vars=value_vars,
                     var_name="year",       # EXAMPLE-SPECIFIC: use YOUR schema dim name
                     value_name="revenue")  # EXAMPLE-SPECIFIC: use YOUR schema value name
    result = result.dropna(subset=["revenue"])

    # Step 4: Rename to exact schema column names — EXAMPLE-SPECIFIC names below
    result = result.rename(columns={result.columns[0]: "region",
                                    result.columns[1]: "product"})
    result["year"] = result["year"].astype(str)
    result["revenue"] = pd.to_numeric(result["revenue"], errors="coerce")

    # Final column order must match schema exactly — EXAMPLE-SPECIFIC
    result = result[["region", "product", "year", "revenue"]]
    print(f"Final output: {result.shape}")
    return result
```"""

    @staticmethod
    def _few_shot_loop() -> str:
        return """\
=== FEW-SHOT EXAMPLE ===
⚠️  WARNING: The values below (row indices, column indices, year strings, column names)
are SPECIFIC TO THIS EXAMPLE spreadsheet. For YOUR spreadsheet, derive
all such values from the HEADERS, PHYSICAL FEATURES, and SOURCE DATA above.
Do NOT copy any number or string literal from this example into your code.

This example had: WIDE_FORMAT + BILINGUAL_ALTERNATING_ROWS + IMPLICIT_AGGREGATION_ROWS
Example physical features: data_start_row=6, data_end_row=56,
value columns at indices 3-8 (3 year groups × 2 value types), label at col 0.

```python
import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    # 6 = data_start_row, 57 = data_end_row+1 — READ FROM PHYSICAL FEATURES FOR YOUR DATA
    result = df.iloc[6:57].copy()
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {result.shape}")

    # Step 3: Identify bilingual row pairs — keep only rows with numeric data
    # Detection: check which rows have values in the value columns.
    # [3, 4, 5, 6, 7, 8] are the value column indices FOR THIS EXAMPLE — derive yours from headers.
    rows_to_keep = []
    for i in range(len(result)):
        has_data = any(
            pd.notna(result.iloc[i, j]) and str(result.iloc[i, j]).strip()
            for j in [3, 4, 5, 6, 7, 8]   # EXAMPLE-SPECIFIC: replace with actual value col indices
        )
        if has_data:
            rows_to_keep.append(i)
    result_cn = result.iloc[rows_to_keep].copy().reset_index(drop=True)

    # Get secondary-language labels from the row immediately after each primary row
    # col 0 = label column FOR THIS EXAMPLE — replace with the actual label column index
    en_labels = []
    for i in rows_to_keep:
        if i + 1 < len(result):
            en_labels.append(str(result.iloc[i + 1, 0]).strip())  # 0 = label col, EXAMPLE-SPECIFIC
        else:
            en_labels.append("")
    result_cn["label_en"] = en_labels
    print(f"After bilingual merge: {result_cn.shape}")

    # Step 4: Unpivot wide columns into long format using record loop
    # Column mapping below is EXAMPLE-SPECIFIC (years 2011/2016/2021, col indices 3-8).
    # For your data: read the header rows to build the equivalent mapping.
    records = []
    for i in range(len(result_cn)):
        row = result_cn.iloc[i]
        # "ethnicity_cn"/"ethnicity_en" are EXAMPLE schema columns — use YOUR schema column names
        base = {"ethnicity_cn": str(row.iloc[0]).strip(),   # col 0 = label, EXAMPLE-SPECIFIC
                "ethnicity_en": row["label_en"]}
        # ("2011", 3, 4) means: year value, count col index, pct col index — EXAMPLE-SPECIFIC
        for year, count_col, pct_col in [("2011", 3, 4), ("2016", 5, 6), ("2021", 7, 8)]:
            record = {**base, "year": year}
            record["count"] = pd.to_numeric(row.iloc[count_col], errors="coerce")
            record["percentage"] = pd.to_numeric(row.iloc[pct_col], errors="coerce")
            records.append(record)

    # Column list below is EXAMPLE-SPECIFIC — replace with YOUR exact schema column names
    output = pd.DataFrame(records)
    output = output[["ethnicity_cn", "ethnicity_en", "year", "count", "percentage"]]
    print(f"Final output: {output.shape}")
    return output
```"""

    # ==================================================================
    # Recipe assembly
    # ==================================================================

    def _get_relevant_recipes(self, labels: List[str],
                              physical: Dict[str, Any],
                              simple_wide: bool = False) -> str:
        """Assemble code recipes for detected irregularity labels."""
        lines = []
        start_r = physical.get("data_start_row", 0)
        end_r = physical.get("data_end_row", "len(df)-1")

        for label in labels:
            # WIDE_FORMAT is handled separately based on simple_wide flag
            if label == "WIDE_FORMAT":
                lines.append("--- WIDE_FORMAT ---")
                if simple_wide:
                    recipe = _WIDE_FORMAT_MELT_RECIPE.replace(
                        "{data_start_row}", str(start_r)
                    ).replace(
                        "{data_end_row}", str(end_r)
                    )
                    lines.append(recipe)
                else:
                    lines.append(_WIDE_FORMAT_LOOP_RECIPE)
                lines.append("")
                continue

            recipe = CODE_RECIPES.get(label)
            if recipe is None:
                continue

            # Substitute known physical values
            recipe = recipe.replace("{data_start_row}", str(start_r))
            recipe = recipe.replace("{data_end_row}", str(end_r))
            recipe = recipe.replace(
                "{header_row_indices}",
                str(list(range(start_r)))
            )
            lines.append(f"--- {label} ---")
            lines.append(recipe.strip())
            lines.append("")

        # Universal best practice — wording adapts to chosen strategy
        lines.append("--- UNIVERSAL BEST PRACTICE (Apply to all tables) ---")
        if simple_wide:
            lines.append(
                "# SAFE TYPE HANDLING\n"
                "# After pd.melt(), convert the value column with "
                "pd.to_numeric(..., errors='coerce').\n"
                "# ALWAYS check pd.notna(x) before .strip() or .split()."
            )
        else:
            lines.append(
                f"# THE GOLDEN EXTRACTION LOOP\n"
                f"# Loop from {start_r} to {end_r}.\n"
                f"# Extract valid observations into `records.append({{...}})`.\n"
                f"# Ensure the dictionary keys exactly match the final schema columns.\n"
                f"\n"
                f"# SAFE STRING & TYPE HANDLING\n"
                f"# ALWAYS check pd.notna(x) before .strip() or .split()."
            )
        lines.append("")

        return "\n".join(lines) if lines else "(no specific recipes)"

    # ==================================================================
    # Code regeneration with feedback
    # ==================================================================
    def _get_structure_comprehension(self, df: pd.DataFrame,
                                     physical: Dict[str, Any],
                                     labels: List[str]) -> str:
        """
        TreeThinker-style pre-step: ask LLM to explicitly describe
        the header hierarchy before attempting code fix.
        One small focused call, output is injected into retry prompt.
        """
        start = physical["data_start_row"]
        header_preview = []
        for i in range(start):
            row_vals = []
            for j in range(len(df.columns)):
                v = df.iloc[i, j]
                if pd.notna(v) and str(v).strip():
                    row_vals.append(f"[{j}]='{str(v).strip()[:40]}'")
            header_preview.append(f"Row {i}: {', '.join(row_vals)}")

        prompt = (
                "Look at these header rows from a spreadsheet:\n\n"
                + "\n".join(header_preview)
                + f"\n\ndata_start_row = {start}\n\n"
                  "Answer ONLY these two questions:\n"
                  "1. For each data column index (columns >= first non-dimension column), "
                  "write its full semantic path by combining all header levels. "
                  "Example: 'Col 3: Year=2019 > ValueType=Number'\n"
                  "2. Which column indices are row-level dimensions (left side, "
                  "they label each row, NOT values to unpivot)?\n"
                  "Be concrete. List every column."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=600,
            )
            return resp.choices[0].message.content.strip()
        except Exception:
            return "(structure comprehension unavailable)"
    def _regenerate_code(self, df: pd.DataFrame,
                         physical: Dict[str, Any],
                         detection_result: Dict[str, Any],
                         schema: Dict[str, Any],
                         labels: List[str],
                         previous_code: str,
                         feedback: Dict[str, Any]) -> str:
        structural_labels = {"NESTED_COLUMN_GROUPS", "MULTI_LEVEL_HEADER", "WIDE_FORMAT"}
        if structural_labels & set(labels):
            comprehension = self._get_structure_comprehension(df, physical, labels)
            # 把理解结果注入到 feedback 里，下面的 prompt 会用到
            feedback["structure_comprehension"] = comprehension
        """Regenerate code using detailed error feedback, data snapshots, and CoT."""
        logger.info("Regenerating code with feedback...")

        target_cols = [c["name"] for c in schema.get("target_columns", [])]
        est_rows = schema.get("expected_output", {}).get("row_count_estimate", "?")

        prompt = f"""The previous transformation code failed. Fix it based on the feedback below.
    
## PREVIOUS CODE
```python
{previous_code}
```

## ERROR FEEDBACK
"""
        if feedback.get("error_type") == "EXECUTION_ERROR":
            null_counts = df.isna().sum().to_dict()
            dtypes = df.dtypes.astype(str).to_dict()

            prompt += f"""Type: EXECUTION ERROR
Message: {feedback['error_message']}
Traceback:
{feedback.get('error_trace', '')}

### DATA SNAPSHOT AT INPUT
To help you debug, here is the state of the initial DataFrame:
Null counts per column: {null_counts}
Data types: {dtypes}

Fix the Python error so the code runs without crashing.
"""
        else:
            prompt += "Type: VALIDATION FAILURE\n\n"
            for issue in feedback.get("issues", []):
                prompt += f"Issue: {issue['description']}\n"
                if issue.get("fix_hint"):
                    prompt += f"Diagnosis:\n{issue['fix_hint']}\n"
                prompt += "\n"

            exec_log = feedback.get("execution_log", [])
            if exec_log:
                prompt += "## EXECUTION LOG (print output from previous run)\n"
                log_lines = exec_log[-30:]
                for line in log_lines:
                    if len(line) > 200:
                        line = line[:200] + "..."
                    prompt += f"  {line}\n"
                prompt += "\n"

            prompt += f"""## CURRENT OUTPUT
Shape: {feedback.get('result_shape')}
Columns: {feedback.get('result_columns')}
Sample (first 5 rows):
{feedback.get('result_sample', 'N/A')}
"""
        if feedback.get("structure_comprehension"):
            prompt += f"""
        ## STRUCTURAL RE-ANALYSIS
        Before fixing, here is the correct interpretation of this table's header hierarchy.
        Use this as ground truth when writing your column_map — do NOT infer column meanings from the raw data again.
        
        {feedback['structure_comprehension']}
        
        """
        prompt += f"""## REQUIREMENTS
1. Analyze the failure and write a brief ## DIAGNOSIS.
2. Write a ## STEP-BY-STEP PLAN on how you will fix the code.
3. Output the corrected Python code inside a single ```python block.
4. Output columns MUST be exactly: {target_cols}
5. Expected approximately {est_rows} rows.
6. def transform(df): → returns DataFrame
7. Use pd.to_numeric(..., errors='coerce') for safe type conversions to avoid NaN integer errors.

Output the diagnosis, plan, and the Python code block."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    # Reuse the same strategy decision from _generate_code
                    {"role": "system", "content": self._system_prompt(self._simple_wide)},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self.max_completion_tokens,
            )
            code = resp.choices[0].message.content.strip()
            return self._extract_code(code)
        except Exception as e:
            logger.error(f"Code regeneration failed: {e}")
            return previous_code

    # ==================================================================
    # Execution
    # ==================================================================

    def _execute_code(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """
        Execute the transform function safely.
        Captures all print() output for diagnostic analysis.
        """
        captured_output = []

        def capturing_print(*args, **kwargs):
            """Capture print output for diagnostics while still printing."""
            text = " ".join(str(a) for a in args)
            captured_output.append(text)
            print(*args, **kwargs)  # still print to console

        exec_globals = {
            "pd": pd,
            "np": np,
            "df": df.copy(),
            "print": capturing_print,
        }

        exec(code, exec_globals)

        if "transform" not in exec_globals:
            raise ValueError("Code must define a 'transform' function")

        result = exec_globals["transform"](df.copy())

        # Store captured output for feedback analysis
        self._last_execution_log = captured_output

        return result

    # ==================================================================
    # Validation
    # ==================================================================

    def _validate_result(self, result_df: pd.DataFrame,
                         schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the transformation output against schema expectations."""
        expected_count = schema.get("expected_output", {}).get("row_count_estimate", 0)

        errors = []
        warnings = []

        # Check 1: Not empty
        if result_df.empty:
            errors.append("Result DataFrame is empty")

        # Check 2: Expected columns
        expected_cols = [c["name"] for c in schema.get("target_columns", [])]
        actual_cols = list(result_df.columns)
        missing = set(expected_cols) - set(actual_cols)
        extra = set(actual_cols) - set(expected_cols)
        if missing:
            errors.append(f"Missing columns: {missing}")
        if extra:
            warnings.append(f"Extra columns: {extra}")

        # Check 3: Row count
        actual_count = len(result_df)
        if actual_count == 0:
            errors.append("Row count is 0 (Result DataFrame is empty). Your extraction loop failed to capture any data.")
        elif actual_count > 50000:
            warnings.append(f"Row count is extremely high ({actual_count} rows). Verify your nested loops.")

        # Check 4: Null check (Enhanced)
        if not result_df.empty:
            for col in result_df.columns:
                null_pct = result_df[col].isnull().mean() * 100

                if null_pct == 100:
                    errors.append(f"Column '{col}' is 100% null. Check your column renaming, unpivot/melt, or merge logic. Do NOT use `raise ValueError` to handle this, fix the dataframe manipulation.")
                elif null_pct > 80:
                    errors.append(f"Column '{col}' is {null_pct:.0f}% null. The extraction logic is likely misaligned.")
                elif null_pct > 50:
                    warnings.append(f"Column '{col}' is {null_pct:.0f}% null")

        # Check 5: Semantic sample validation
        sample_checks = self._validate_samples(result_df, schema)
        for check in sample_checks:
            if not check.get("passed", True):
                errors.append(
                    f"Sample validation failed: {check.get('description')}"
                )

        return {
            "is_valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "actual_shape": result_df.shape,
            "expected_shape": (expected_count, len(expected_cols)),
        }

    def _validate_samples(self, result_df: pd.DataFrame,
                          schema: Dict[str, Any]) -> List[Dict]:
        """Validate against schema-provided sample rows and provide closest match on failure."""
        checks = []
        for sample in schema.get("validation_samples", []):
            expected = sample.get("expected_row", {})
            if not expected:
                continue

            check = {
                "description": sample.get("description", "Sample check"),
                "expected": expected,
                "passed": False,
            }

            try:
                conditions = pd.Series([True] * len(result_df))
                for col, val in expected.items():
                    if col not in result_df.columns:
                        continue
                    if pd.isna(val):
                        conditions &= result_df[col].isna()
                    elif isinstance(val, (int, float)):
                        conditions &= (
                                (result_df[col] == val) |
                                (pd.to_numeric(result_df[col], errors="coerce")
                                 .sub(val).abs().lt(0.01))
                        )
                    else:
                        conditions &= (result_df[col].astype(str) == str(val))

                if conditions.any():
                    check["passed"] = True
                else:
                    dim_cols = [k for k, v in expected.items() if isinstance(v, str) and k in result_df.columns]
                    if dim_cols:
                        query_parts = [f"`{k}` == '{expected[k]}'" for k in dim_cols]
                        if query_parts:
                            closest_match = result_df.query(" and ".join(query_parts))
                            if not closest_match.empty:
                                check["error"] = f"Row not found. Closest match produced:\n{closest_match.head(1).to_dict('records')}"
                            else:
                                check["error"] = f"Row completely missing. Could not find dimensions: {expected}"
            except Exception as e:
                check["error"] = str(e)

            checks.append(check)
        return checks

    # ==================================================================
    # Feedback builder
    # ==================================================================

    def _build_feedback(self, validation: Dict[str, Any],
                        result_df: pd.DataFrame,
                        source_df: pd.DataFrame,
                        schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build structured feedback for code regeneration.

        Key improvement: when row count is wrong, parse the execution
        log to find WHICH step caused the biggest row loss, and show
        samples of what was dropped.
        """
        issues = []
        expected_cols = [c["name"] for c in schema.get("target_columns", [])]
        expected_count = schema.get("expected_output", {}).get(
            "row_count_estimate", 0
        )

        for err in validation.get("errors", []):
            issue = {"description": err}
            if "Missing columns" in err:
                issue["fix_hint"] = (
                    f"Ensure output has EXACTLY these columns: "
                    f"{expected_cols}. Check column renaming at the end."
                )
            elif "Row count" in err:
                step_analysis = self._analyze_row_loss(
                    source_df, result_df, expected_count
                )
                issue["fix_hint"] = step_analysis
            elif "Sample validation" in err:
                issue["fix_hint"] = (
                    "A specific expected row was not found in the output. "
                    "Check column mapping and value extraction logic."
                )
            issues.append(issue)

        sample_str = ""
        if not result_df.empty:
            sample_str = result_df.head(5).to_string()

        return {
            "error_type": "VALIDATION_FAILURE",
            "issues": issues,
            "result_shape": result_df.shape,
            "result_columns": list(result_df.columns),
            "result_sample": sample_str,
            "execution_log": self._last_execution_log,
        }

    def _analyze_row_loss(self, source_df: pd.DataFrame,
                          result_df: pd.DataFrame,
                          expected_count: int) -> str:
        """
        Parse captured execution log to find which step caused the
        biggest row loss. Returns a diagnostic string for the LLM.
        """
        lines = []

        shape_history = []
        import re as _re
        for log_line in self._last_execution_log:
            m = _re.search(r"(\((\d+),\s*\d+\))", log_line)
            if m:
                rows = int(m.group(2))
                desc = log_line[:log_line.find(m.group(1))].strip()
                desc = desc.rstrip(":").strip()
                shape_history.append((desc, rows))

        if shape_history:
            lines.append("STEP-BY-STEP ROW COUNT FROM EXECUTION LOG:")
            prev_rows = None
            biggest_drop_step = None
            biggest_drop_amount = 0

            for desc, rows in shape_history:
                drop = ""
                if prev_rows is not None and rows < prev_rows:
                    lost = prev_rows - rows
                    drop = f"  ← LOST {lost} rows"
                    if lost > biggest_drop_amount:
                        biggest_drop_amount = lost
                        biggest_drop_step = desc
                lines.append(f"  {desc}: {rows} rows{drop}")
                prev_rows = rows

            if biggest_drop_step and biggest_drop_amount > 0:
                lines.append(f"\n*** BIGGEST ROW LOSS: '{biggest_drop_step}' "
                             f"dropped {biggest_drop_amount} rows ***")
                lines.append(
                    f"This step is likely the bug. The rows it removed "
                    f"were probably VALID data rows, not rows that should "
                    f"be filtered out."
                )
                lines.append(
                    f"IMPORTANT: Do NOT drop rows just because a column "
                    f"value looks unusual (e.g., '不適用', 'N/A', 'None'). "
                    f"These may be legitimate values meaning 'not applicable' "
                    f"for certain years/categories. Only drop rows that are "
                    f"TRUE aggregation rows or metadata rows."
                )
        else:
            lines.append(
                "Could not parse step-by-step row counts from execution log."
            )

        lines.append(f"\nExpected ~{expected_count} rows, got {len(result_df)}.")

        return "\n".join(lines)

    # ==================================================================
    # Formatters
    # ==================================================================

    def _format_schema(self, schema: Dict[str, Any]) -> str:
        lines = []
        obs = schema.get("observation_unit", {})
        lines.append(f"Observation unit: {obs.get('description', 'Unknown')}")

        lines.append("\nTarget columns:")
        for col in schema.get("target_columns", []):
            marker = "[DIM]" if col.get("is_dimension") else "[VAL]"
            lines.append(
                f"  {marker} {col['name']} ({col.get('data_type', 'string')})"
                f": {col.get('description', '')}"
            )
            if col.get("source"):
                lines.append(f"       source: {col['source']}")

        exp = schema.get("expected_output", {})
        lines.append(
            f"\nExpected: ~{exp.get('row_count_estimate', '?')} rows "
            f"({exp.get('row_count_formula', '')})"
        )

        excl = schema.get("exclusions", {})
        er = excl.get("exclude_rows", {}).get("description", "")
        ec = excl.get("exclude_columns", {}).get("description", "")
        if er:
            lines.append(f"Exclude rows: {er}")
        if ec:
            lines.append(f"Exclude columns: {ec}")

        samples = schema.get("validation_samples", [])
        if samples:
            lines.append("\nValidation samples:")
            for s in samples:
                lines.append(
                    f"  {s.get('description', '')}: {s.get('expected_row', {})}"
                )

        return "\n".join(lines)

    def _format_headers(self, df: pd.DataFrame,
                        physical: Dict[str, Any]) -> str:
        start = physical["data_start_row"]
        # 新增：拿到左侧 dimension 列的列索引范围
        left_dim_cols = set(range(physical.get("left_header_cols_num", 1)))

        lines = []
        for i in range(start):
            parts = []
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    s = str(val).strip().replace("\n", "\\n")
                    if len(s) > 60:
                        s = s[:60] + "..."
                    # ← 核心改动：加 tag
                    tag = "[ROW_DIM]" if j in left_dim_cols else "[COL_HEADER]"
                    parts.append(f'[{j}]{tag}="{s}"')
            if parts:
                lines.append(f"  Row {i}: {', '.join(parts)}")
            else:
                lines.append(f"  Row {i}: (blank)")
        return "\n".join(lines) if lines else "(no header rows)"

    def _format_data(self, df: pd.DataFrame,
                     physical: Dict[str, Any]) -> str:
        """Full data — no truncation."""
        sr = physical.get("data_start_row", 0)
        er = physical.get("data_end_row", len(df) - 1)
        lines = []
        for i in range(sr, er + 1):
            parts = []
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    s = str(val).strip().replace("\n", "\\n")
                    if len(s) > 50:
                        s = s[:50] + "..."
                    parts.append(f'[{j}]="{s}"')
            if parts:
                lines.append(f"  Row {i}: {', '.join(parts)}")
            else:
                lines.append(f"  Row {i}: (blank)")
        lines.append(f"  (Total: {er - sr + 1} rows)")
        return "\n".join(lines)

    def _format_irregularities(self, irregularities: List[Dict]) -> str:
        lines = []
        for ir in irregularities:
            lines.append(f"- {ir['label']}: {ir.get('evidence', '')}")
            if ir.get("details"):
                lines.append(f"  Details: {ir['details']}")
        return "\n".join(lines) if lines else "(none)"

    # ==================================================================
    # Utilities
    # ==================================================================

    def _extract_code(self, text: str) -> str:
        """Extract Python code from LLM response safely, ignoring CoT text."""
        text = text.strip()

        matches = re.findall(r"```python\n(.*?)\n```", text, re.DOTALL)
        if matches:
            return matches[-1].strip()

        matches_generic = re.findall(r"```\n(.*?)\n```", text, re.DOTALL)
        if matches_generic:
            return matches_generic[-1].strip()

        return text