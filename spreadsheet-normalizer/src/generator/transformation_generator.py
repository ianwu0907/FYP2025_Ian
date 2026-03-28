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

    "METADATA_ROWS": '''
# RECIPE: METADATA_ROWS — Slice to data region only
data_start_row = {data_start_row}  # from physical features
data_end_row = {data_end_row}
result = df.iloc[data_start_row:data_end_row + 1].copy()
result = result.reset_index(drop=True)
''',

    "MULTI_LEVEL_HEADER": '''
# RECIPE: MULTI_LEVEL_HEADER — Read header rows to build column semantics
# Do NOT rely on df.columns — read raw cell values from header rows
header_row_indices = {header_row_indices}  # e.g., [3, 4, 5]
# Build a mapping: column_index -> semantic meaning
col_semantics = {{}}
for j in range(len(df.columns)):
    parts = []
    for hr in header_row_indices:
        val = df.iloc[hr, j]
        if pd.notna(val) and str(val).strip():
            parts.append(str(val).strip())
    col_semantics[j] = " | ".join(parts) if parts else f"col_{{j}}"
# Now col_semantics[3] might be "2011 | Number", col_semantics[4] might be "2011 | %"
''',

    "NESTED_COLUMN_GROUPS": '''
# RECIPE: NESTED_COLUMN_GROUPS — Build a column mapping with group structure
# After reading multi-level headers, create a structured mapping:
# column_map = {{col_index: (group_value, sub_value)}}
# Example for year x value_type:
#   {{3: ("2011", "Number"), 4: ("2011", "%"), 5: ("2016", "Number"), ...}}
# Example for marital_status x sex:
#   {{3: ("Never married", "Male"), 4: ("Never married", "Female"), ...}}
#
# Forward-fill group labels across columns (groups span multiple cols):
group_labels = {{}}
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
    base = {{dim_col: row[dim_col_idx] for dim_col, dim_col_idx in id_columns}}
    for col_idx, (group_val, sub_val) in column_map.items():
        record = {{**base, group_dim_name: group_val, sub_dim_name: sub_val}}
        record[value_col_name] = row.iloc[col_idx]
        records.append(record)
result = pd.DataFrame(records)
''',

    "WIDE_FORMAT": '''
# RECIPE: WIDE_FORMAT — Unpivot value columns into rows
# Option A: Simple melt (when columns are a single dimension)
id_cols = [col_indices_to_keep]  # columns that stay as-is
value_cols = [col_indices_to_unpivot]  # columns that become rows
result = pd.melt(result, id_vars=id_cols, value_vars=value_cols,
                 var_name="new_dimension_name", value_name="value_column_name")
 
# Option B: Manual iteration (when columns encode multiple dimensions)
# See NESTED_COLUMN_GROUPS recipe above
''',

    "BILINGUAL_ALTERNATING_ROWS": '''
# RECIPE: BILINGUAL_ALTERNATING_ROWS — Merge row pairs
# Chinese rows (even indices in data region) have numeric data
# English rows (odd indices) have English labels only
merged_rows = []
data_rows = result.values.tolist()
i = 0
while i < len(data_rows) - 1:
    cn_row = data_rows[i]      # Chinese label + numeric data
    en_row = data_rows[i + 1]  # English label, no numeric data
    # Check if en_row is indeed a label-only row (all numeric cols empty)
    en_has_data = any(pd.notna(en_row[j]) and str(en_row[j]).strip()
                      for j in numeric_col_indices)
    if not en_has_data:
        # Merge: take label from both, data from Chinese row
        merged = {{"label_cn": cn_row[label_col], "label_en": en_row[label_col]}}
        for j in numeric_col_indices:
            merged[col_name_for(j)] = cn_row[j]
        merged_rows.append(merged)
        i += 2
    else:
        # Not a bilingual pair — treat as regular row
        merged_rows.append(build_row(cn_row))
        i += 1
result = pd.DataFrame(merged_rows)
''',

    "INLINE_BILINGUAL": '''
# RECIPE: INLINE_BILINGUAL — Split cell content by newline
# Cells like "種族\\nEthnicity" → split into Chinese and English parts
def split_bilingual(val):
    """Split a bilingual cell value into (chinese, english) parts."""
    if pd.isna(val):
        return val, val
    s = str(val)
    if "\\n" in s:
        parts = s.split("\\n", 1)
        return parts[0].strip(), parts[1].strip()
    return s.strip(), s.strip()
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
# Apply ALL relevant forms:
 
# Form 1: Keyword-based total/subtotal rows
agg_keywords = ["total", "subtotal", "overall", "all ", "sum",
                 "合計", "總計", "小計", "总计", "合计"]
def is_keyword_agg(val):
    if pd.isna(val):
        return False
    return any(kw in str(val).strip().lower() for kw in agg_keywords)
result = result[~result[label_col].apply(is_keyword_agg)]
 
# Form 2: Semantic hierarchy aggregation
# Some rows are parent-level aggregates of child rows, identifiable
# only by domain meaning (no keyword or delimiter signal).
# Example: "Asian (other than Chinese)" sums Filipino + Indonesian + ...
# Use the specific labels identified in the detection EVIDENCE:
#   semantic_agg_labels = ["label_1", "label_2", ...]
#   result = result[~result[label_col].isin(semantic_agg_labels)]
# *** Build this list from the detection EVIDENCE ***
 
# Note: Cross-group aggregation (where an entire category group is a
# coarser version of another) should be handled by removing the coarser
# category values. Use the EXCLUDE_ROWS from the schema to identify
# which category values to drop.
''',

    "AGGREGATE_COLUMNS": '''
# RECIPE: AGGREGATE_COLUMNS — Drop aggregate columns before reshaping
# Identify columns whose headers contain Total/Overall/合計/Median
# These are typically the last column in each repeating group
# or standalone summary columns
agg_col_indices = [indices_of_aggregate_columns]  # from schema exclusions
result = result.drop(columns=[result.columns[i] for i in agg_col_indices
                               if i < len(result.columns)], errors="ignore")
''',

    "EMBEDDED_DIMENSION_IN_COLUMN": '''
# RECIPE: EMBEDDED_DIMENSION_IN_COLUMN — Conditional split
delimiter = " - "  # detected delimiter
def split_embedded(val):
    if pd.isna(val):
        return val, None
    s = str(val).strip()
    if delimiter in s:
        parts = s.split(delimiter, 1)
        return parts[0].strip(), parts[1].strip()
    return s, None
 
result[["primary_dim", "secondary_dim"]] = result["compound_col"].apply(
    lambda x: pd.Series(split_embedded(x))
)
''',

    "SPARSE_ROW_FILL": '''
# RECIPE: SPARSE_ROW_FILL — Forward-fill sparse dimension column
# A dimension column (e.g., Year) has values only in the first row
# of each group; blanks below carry forward the same value.
# Apply EARLY, before any row filtering or reshaping.
result[sparse_col_name] = result[sparse_col_name].fillna(method="ffill")
''',
}


# ============================================================================
# TransformationGenerator
# ============================================================================

class TransformationGenerator:
    """
    Generates transformation code using a code-recipe-guided approach:

    1. Build prompt with code recipes for detected irregularities
    2. Single LLM call generates a `def transform(df)` function
    3. Execute → Validate → Feedback loop
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
                        "error_message": str(e),
                        "error_trace": traceback.format_exc(),
                    }
                    code = self._regenerate_code(
                        df, physical, detection_result, schema,
                        labels, code, feedback
                    )
                else:
                    raise

        raise RuntimeError("Transformation failed after all retries")

    # ==================================================================
    # Code Generation
    # ==================================================================

    def _generate_code(self, df: pd.DataFrame,
                       physical: Dict[str, Any],
                       detection_result: Dict[str, Any],
                       schema: Dict[str, Any],
                       labels: List[str]) -> str:
        """Generate transformation code with code recipes as guidance."""

        prompt = self._build_code_prompt(
            df, physical, detection_result, schema, labels
        )

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
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

    def _system_prompt(self) -> str:
        return (
            "You are an expert Python/pandas programmer. You write a "
            "`def transform(df)` function that transforms a messy "
            "spreadsheet DataFrame into tidy format.\n\n"

            "RULES:\n"
            "1. Function signature: `def transform(df):` taking and "
            "returning a DataFrame.\n"
            "2. Use only pandas, numpy, and standard library.\n"
            "3. Use .iloc for positional access (column INDICES, not names). "
            "The input df has auto-generated column names that are unreliable.\n"
            "4. Add print() statements after each major step showing "
            "the shape and a sample.\n"
            "5. Handle edge cases: missing values, type conversion.\n"
            "6. Return the final DataFrame with EXACT column names "
            "matching the target schema.\n\n"

            "Output ONLY the Python code. No markdown. No explanation."
        )

    def _build_code_prompt(self, df: pd.DataFrame,
                           physical: Dict[str, Any],
                           detection_result: Dict[str, Any],
                           schema: Dict[str, Any],
                           labels: List[str]) -> str:

        recipes_text = self._get_relevant_recipes(labels, physical)
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

DATA (full data region):
{data_text}

=== REQUIREMENTS ===
1. def transform(df): → returns DataFrame
2. Use .iloc for column access (positional indices)
3. Output columns MUST be exactly: {target_cols}
4. Expected output: ~{est_rows} rows
5. Add print() after each step
6. Adapt the code recipes above to THIS specific data

=== FEW-SHOT EXAMPLE ===
Here is a COMPLETE transform function for a different spreadsheet that had
WIDE_FORMAT + BILINGUAL_ALTERNATING_ROWS + IMPLICIT_AGGREGATION_ROWS:

```python
import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {{df.shape}}")

    # Step 1: Slice to data region
    result = df.iloc[6:57].copy()
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {{result.shape}}")

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {{result.shape}}")

    # Step 3: Identify bilingual row pairs — keep only Chinese rows (have data)
    rows_to_keep = []
    for i in range(len(result)):
        # Check if this row has numeric data in value columns
        has_data = False
        for j in [3, 4, 5, 6, 7, 8]:
            val = result.iloc[i, j]
            if pd.notna(val) and str(val).strip():
                has_data = True
                break
        if has_data:
            rows_to_keep.append(i)
    result_cn = result.iloc[rows_to_keep].copy().reset_index(drop=True)

    # Get English labels from the row after each Chinese row
    en_labels = []
    for i in rows_to_keep:
        if i + 1 < len(result):
            en_labels.append(str(result.iloc[i + 1, 0]).strip())
        else:
            en_labels.append("")
    result_cn["label_en"] = en_labels
    print(f"After bilingual merge: {{result_cn.shape}}")

    # Step 4: Remove aggregation rows
    agg_kw = ["total", "all ", "overall", "合計", "總計"]
    mask = result_cn.iloc[:, 0].apply(
        lambda x: not any(k in str(x).lower() for k in agg_kw) if pd.notna(x) else True
    )
    result_cn = result_cn[mask].reset_index(drop=True)
    print(f"After removing aggregation rows: {{result_cn.shape}}")

    # Step 5: Unpivot year groups
    # Column mapping: {{3: ("2011", "count"), 4: ("2011", "pct"),
    #                  5: ("2016", "count"), 6: ("2016", "pct"),
    #                  7: ("2021", "count"), 8: ("2021", "pct")}}
    records = []
    for i in range(len(result_cn)):
        row = result_cn.iloc[i]
        base = {{"ethnicity_cn": str(row.iloc[0]).strip(),
                "ethnicity_en": row["label_en"]}}
        for year, count_col, pct_col in [("2011", 3, 4), ("2016", 5, 6), ("2021", 7, 8)]:
            record = {{**base, "year": year}}
            record["count"] = pd.to_numeric(row.iloc[count_col], errors="coerce")
            record["percentage"] = pd.to_numeric(row.iloc[pct_col], errors="coerce")
            records.append(record)

    output = pd.DataFrame(records)
    output = output[["ethnicity_cn", "ethnicity_en", "year", "count", "percentage"]]
    print(f"Final output: {{output.shape}}")
    return output
```

Now write the transform function for THIS spreadsheet. Output ONLY code."""

    # ==================================================================
    # Recipe assembly
    # ==================================================================

    def _get_relevant_recipes(self, labels: List[str],
                              physical: Dict[str, Any]) -> str:
        """Assemble code recipes for detected irregularity labels."""
        lines = []
        for label in labels:
            if label in CODE_RECIPES:
                recipe = CODE_RECIPES[label]
                # Substitute known physical values
                recipe = recipe.replace(
                    "{data_start_row}",
                    str(physical.get("data_start_row", 0))
                )
                recipe = recipe.replace(
                    "{data_end_row}",
                    str(physical.get("data_end_row", "len(df)-1"))
                )
                recipe = recipe.replace(
                    "{header_row_indices}",
                    str(list(range(physical.get("data_start_row", 0))))
                )
                lines.append(f"--- {label} ---")
                lines.append(recipe.strip())
                lines.append("")

        return "\n".join(lines) if lines else "(no specific recipes)"

    # ==================================================================
    # Code regeneration with feedback
    # ==================================================================

    def _regenerate_code(self, df: pd.DataFrame,
                         physical: Dict[str, Any],
                         detection_result: Dict[str, Any],
                         schema: Dict[str, Any],
                         labels: List[str],
                         previous_code: str,
                         feedback: Dict[str, Any]) -> str:
        """Regenerate code using detailed error feedback."""
        logger.info("Regenerating code with feedback...")

        target_cols = [c["name"] for c in schema.get("target_columns", [])]
        est_rows = schema.get("expected_output", {}).get(
            "row_count_estimate", "?"
        )

        prompt = f"""The previous transformation code failed. Fix it based on the feedback below.

## PREVIOUS CODE
```python
{previous_code}
```

## ERROR FEEDBACK
"""
        if feedback.get("error_type") == "EXECUTION_ERROR":
            prompt += f"""Type: EXECUTION ERROR
Message: {feedback['error_message']}
Traceback:
{feedback.get('error_trace', '')}

Fix the Python error so the code runs without crashing.
"""
        else:
            # Validation failure — include rich diagnostics
            prompt += f"Type: VALIDATION FAILURE\n\n"

            for issue in feedback.get("issues", []):
                prompt += f"Issue: {issue['description']}\n"
                if issue.get("fix_hint"):
                    prompt += f"Diagnosis:\n{issue['fix_hint']}\n"
                prompt += "\n"

            # Include execution log so LLM can see step-by-step
            exec_log = feedback.get("execution_log", [])
            if exec_log:
                prompt += "## EXECUTION LOG (print output from previous run)\n"
                # Show last 30 lines to stay within token budget
                log_lines = exec_log[-30:]
                for line in log_lines:
                    # Truncate very long lines
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

        prompt += f"""## REQUIREMENTS
1. Fix ALL issues described above
2. Output columns MUST be exactly: {target_cols}
3. Expected approximately {est_rows} rows
4. def transform(df): → returns DataFrame
5. Do NOT drop rows unless they are TRUE aggregation/total rows or metadata
6. Values like '不適用', 'N/A', 'not applicable' are legitimate dimension values — do NOT treat them as missing

Output ONLY the corrected Python code."""

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._system_prompt()},
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
        expected_count = schema.get("expected_output", {}).get(
            "row_count_estimate", 0
        )
        actual_count = len(result_df)
        if expected_count > 0:
            ratio = actual_count / expected_count
            if ratio < 0.3:
                errors.append(
                    f"Row count too low: {actual_count} vs expected "
                    f"~{expected_count}"
                )
            elif ratio > 3.0:
                warnings.append(
                    f"Row count much higher than expected: {actual_count} "
                    f"vs ~{expected_count}"
                )

        # Check 4: Null check
        if not result_df.empty:
            for col in result_df.columns:
                null_pct = result_df[col].isnull().mean() * 100
                if null_pct > 50:
                    warnings.append(
                        f"Column '{col}' is {null_pct:.0f}% null"
                    )

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
        """Validate against schema-provided sample rows."""
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
                # Analyze the execution log to find where rows were lost
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

        # Result sample for debugging
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

        # Parse shape changes from execution log
        shape_history = []
        import re as _re
        for log_line in self._last_execution_log:
            # Look for patterns like "After ...: (1128, 8)" or shape: (1128, 8)
            m = _re.search(r"(\((\d+),\s*\d+\))", log_line)
            if m:
                rows = int(m.group(2))
                # Extract the step description
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
        start = physical.get("data_start_row", 0)
        lines = []
        for i in range(start):
            parts = []
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    s = str(val).strip().replace("\n", "\\n")
                    if len(s) > 60:
                        s = s[:60] + "..."
                    parts.append(f'[{j}]="{s}"')
            if parts:
                lines.append(f"  Row {i}: {', '.join(parts)}")
        return "\n".join(lines) if lines else "  (no headers)"

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
        """Extract Python code from LLM response."""
        text = text.strip()
        if "```python" in text:
            start = text.find("```python") + 9
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        return text