"""
Schema Estimator Module (Redesigned)

Consumes:
  - encoded_data  (DataFrame + metadata from SpreadsheetEncoder)
  - detection_result  (from IrregularityDetector: physical features,
    irregularity labels with evidence)

Produces:
  - A target tidy schema dict compatible with TransformationGenerator

Architecture:
  1.  Assemble context: full data + irregularity labels + per-label
      schema guidance from the taxonomy.
  2.  Single focused LLM call with few-shot examples.
      Output format is simple structured text (NOT JSON) to ensure
      stability on weak/open-source models.
  3.  Deterministic parsing + assembly into the schema dict.

Design principles:
  - ONE LLM call, ONE concern: "design the tidy schema".
  - Schema guidance for each irregularity is injected automatically
    from the taxonomy — the LLM does not need to figure out how to
    handle each irregularity from scratch.
  - Output format is line-based text with fixed prefixes. Even if
    the model deviates slightly, the parser can recover.
  - Full data is passed — no truncation.
"""

import json
import logging
import os
import re
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from openai import OpenAI

from ..detector.irregularity_detector import get_schema_guidance_for

logger = logging.getLogger(__name__)


class SchemaEstimator:
    """
    Estimates the ideal tidy schema using one focused LLM call
    informed by detected irregularities and their handling guidance.
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
        self.max_tokens = config.get("max_completion_tokens", 2500)

    # ==================================================================
    # Public API
    # ==================================================================

    def estimate_schema(self,
                        encoded_data: Dict[str, Any],
                        detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate the target tidy schema.

        Args:
            encoded_data:     From SpreadsheetEncoder (has 'dataframe', 'metadata')
            detection_result: From IrregularityDetector (has 'physical',
                              'irregularities', 'labels')

        Returns:
            Schema dict compatible with TransformationGenerator.
        """
        df = encoded_data["dataframe"]
        physical = detection_result["physical"]
        irregularities = detection_result["irregularities"]
        labels = detection_result["labels"]

        logger.info("Estimating tidy schema (single focused LLM call)...")
        logger.info(f"  Irregularities to handle: {labels}")

        # Build prompt with guidance and full data
        prompt = self._build_prompt(df, physical, irregularities, labels)
        system = self._system_prompt()

        # LLM call
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_completion_tokens=self.max_tokens,
                timeout=120,
            )
            text = resp.choices[0].message.content.strip()
            logger.debug(f"Schema LLM response:\n{text}")

            schema = self._parse_and_assemble(text, df, physical,
                                              detection_result)

        except Exception as e:
            logger.error(f"Schema estimation LLM call failed: {e}")
            schema = self._fallback_schema(df, physical, detection_result)

        # Attach pass-through metadata for downstream
        schema["source_metadata"] = encoded_data.get("metadata", {})
        schema["detection_result"] = detection_result

        # Log summary
        col_names = [c["name"] for c in schema.get("target_columns", [])]
        est = schema.get("expected_output", {}).get("row_count_estimate", "?")
        logger.info(f"  Target columns: {col_names}")
        logger.info(f"  Expected rows:  {est}")

        return schema

    # ==================================================================
    # Prompt construction
    # ==================================================================

    def _system_prompt(self) -> str:
        return (
            "You are a data architect. Your goal is to transform messy "
            "spreadsheets into TIDY DATA format.\n\n"

            "*** TIDY DATA PRINCIPLES (Hadley Wickham) YOU MUST ENFORCE ***:\n"
            "1. Each VARIABLE forms a column.\n"
            "2. Each OBSERVATION forms a row.\n"
            "3. Each type of OBSERVATIONAL UNIT forms a table.\n\n"

            "Given a messy spreadsheet, its detected structural "
            "irregularities, and handling guidance for each irregularity, "
            "design the ideal tidy output schema. Answer in the EXACT "
            "text format shown in the examples. Be precise about column "
            "names and row estimates."
        )
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
    def _build_prompt(self, df: pd.DataFrame,
                      physical: Dict[str, Any],
                      irregularities: List[Dict],
                      labels: List[str]) -> str:

        guidance_text = get_schema_guidance_for(labels)
        irregularity_text = self._format_irregularities(irregularities)
        headers_text = self._format_headers(df, physical)
        data_text = self._format_data(df, physical)
        lineage_text = self._format_header_lineage(df, physical, labels)

        return f"""Design the tidy output schema for this spreadsheet.

=== OUTPUT FORMAT (follow exactly) ===

OBSERVATION: <what one row in the tidy output represents>

TARGET_COLUMNS:
- <name> (<type>, <role>): <description> | source: <where this comes from>

ROW_ESTIMATE: <formula> = <integer>
NOTE: ROW_ESTIMATE MUST end with "= <a specific integer>". Do NOT write
only a description. If exact count is unknown, multiply your best estimates
of each dimension and write the result as an integer. Example:
"20 years × 8 types × 2 sexes = 320" — always conclude with "= <integer>".

EXCLUDE_ROWS: Drop rows where <Column Name> IN (<"Exact", "Values", "To", "Drop">). Be extremely specific.
EXCLUDE_COLUMNS: <which columns to remove and why>

SAMPLE_ROW: <col1>=<val1>, <col2>=<val2>, ...

Rules for TARGET_COLUMNS:
  - <type> is one of: string, integer, float
  - <role> is one of: dimension, value
  - One column per line, each starting with "- "
  - Use snake_case for all column names
  - Dimensions go first, then values
  - If a source column contains values that will be filtered out, that column must still be
    included as a dimension — do not drop the column just because
    some of its values are excluded. The column encodes real
    variation across the rows that are kept.
  - CRITICAL: Do NOT include a target column whose values can only be extracted from rows that are listed in EXCLUDE_ROWS
=== YOUR TASK ===

IRREGULARITIES:
{irregularity_text}

GUIDANCE:
{guidance_text}

PHYSICAL FEATURES:
  Data region: rows {physical['data_start_row']} to {physical['data_end_row']} ({physical['data_rows']} rows)
  ACTUAL SOURCE COLUMNS (these are the ONLY columns that exist in the data — do NOT invent target columns that have no source here):
  {self._format_col_names(physical)}
  Column types: {physical['column_dtype_profile']}

HEADERS:
{headers_text}

COLUMN LINEAGE (semantic path for each data column):
{lineage_text}

DATA (full data region):
{data_text}

Now design the tidy schema following the EXACT format above."""

    # ==================================================================
    # Formatters
    # ==================================================================

    def _format_col_names(self, physical: Dict[str, Any]) -> str:
        names = physical.get("actual_column_names", [])
        if not names:
            return "(unnamed columns)"
        return ", ".join(
            f'[{j}]="{name}"' for j, name in enumerate(names)
            if name and not name.startswith("Unnamed")
        ) or "(unnamed columns)"

    def _format_irregularities(self, irregularities: List[Dict]) -> str:
        lines = []
        for ir in irregularities:
            lines.append(f"  {ir['label']}: {ir.get('evidence', '')}")
            if ir.get("details"):
                lines.append(f"    Details: {ir['details']}")
        return "\n".join(lines) if lines else "  (none detected)"

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
        """Full data region — no truncation."""
        sr = physical["data_start_row"]
        er = physical["data_end_row"]
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

    # ==================================================================
    # Response parsing + schema assembly
    # ==================================================================

    def _parse_and_assemble(self, text: str,
                            df: pd.DataFrame,
                            physical: Dict[str, Any],
                            detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse the structured-text LLM response and assemble into
        the schema dict expected by TransformationGenerator.
        """
        observation = ""
        target_columns = []
        row_formula = ""
        row_estimate = 0
        exclude_rows = ""
        exclude_cols = ""
        sample_row = {}

        # State machine for multi-line TARGET_COLUMNS section
        in_columns_section = False

        for line in text.split("\n"):
            raw = line.strip()
            if not raw:
                continue
            upper = raw.upper()

            # --- OBSERVATION ---
            if upper.startswith("OBSERVATION:"):
                observation = raw.split(":", 1)[1].strip()
                in_columns_section = False

            # --- TARGET_COLUMNS section start ---
            elif upper.startswith("TARGET_COLUMNS"):
                in_columns_section = True

            # --- Column definition line ---
            elif in_columns_section and raw.startswith("- "):
                col = self._parse_column_line(raw)
                if col:
                    target_columns.append(col)

            # --- ROW_ESTIMATE ---
            elif upper.startswith("ROW_ESTIMATE:"):
                in_columns_section = False
                val = raw.split(":", 1)[1].strip()
                row_formula = val
                # Take the last "= <pure integer>" match, skipping "= 20 years" style
                eq_matches = re.findall(r"=\s*(\d+)(?!\s*[a-zA-Z\u4e00-\u9fff])", val)
                if eq_matches:
                    row_estimate = int(eq_matches[-1])
                else:
                    # Fallback: last number in the string
                    nums = re.findall(r"\d+", val)
                    if nums:
                        row_estimate = int(nums[-1])

            # --- EXCLUDE_ROWS ---
            elif upper.startswith("EXCLUDE_ROW"):
                in_columns_section = False
                exclude_rows = raw.split(":", 1)[1].strip()

            # --- EXCLUDE_COLUMNS ---
            elif upper.startswith("EXCLUDE_COL"):
                in_columns_section = False
                exclude_cols = raw.split(":", 1)[1].strip()

            # --- SAMPLE_ROW ---
            elif upper.startswith("SAMPLE_ROW:"):
                in_columns_section = False
                sample_str = raw.split(":", 1)[1].strip()
                sample_row = self._parse_sample_row(sample_str)

            # --- Unrecognized line inside columns section ---
            elif in_columns_section:
                # Stop columns section if this looks like a new section
                if re.match(r"^[A-Z_]+:", raw):
                    in_columns_section = False

        # Build schema dict
        schema = {
            "observation_unit": {
                "description": observation,
                "dimensions": [c["name"] for c in target_columns
                               if c.get("is_dimension")],
                "example": observation,
            },
            "target_columns": target_columns,
            "expected_output": {
                "row_count_formula": row_formula,
                "row_count_estimate": row_estimate,
                "column_count": len(target_columns),
            },
            "exclusions": {
                "exclude_rows": {
                    "description": exclude_rows,
                    "criteria": [exclude_rows] if exclude_rows else [],
                },
                "exclude_columns": {
                    "description": exclude_cols,
                    "criteria": [exclude_cols] if exclude_cols else [],
                },
            },
            "handling_special_cases": {},
            "validation_samples": [],
            "schema_reasoning": observation,
            "expected_output_columns": [c["name"] for c in target_columns],
        }

        # Build validation sample if parsed
        if sample_row:
            schema["validation_samples"].append({
                "description": "LLM-provided sample row",
                "expected_row": sample_row,
            })

        return schema

    def _parse_column_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a column definition line like:
          - year (string, dimension): Year of sales | source: column headers

        Handles multiple levels of format degradation:
          Level 1: full format with type, role, description, source
          Level 2: name (type, role) only
          Level 3: name (role) only
          Level 4: just a name
        """
        line = line.lstrip("- ").strip()

        # Level 1: full format
        #   name (type, role): description | source: ...
        m = re.match(
            r"(\w+)\s*"                   # column name
            r"\(([^,)]+),?\s*([^)]*)\)"   # (type, role)
            r"\s*:?\s*(.*)",              # : description ...
            line
        )
        if m:
            name = m.group(1).strip()
            dtype = m.group(2).strip().lower()
            role = m.group(3).strip().lower()
            rest = m.group(4).strip()

            is_dim = "dim" in role
            description = rest.split("|")[0].strip() if "|" in rest else rest
            source = ""
            if "|" in rest:
                source_part = rest.split("|")[1].strip()
                # Remove "source:" prefix if present
                source = re.sub(r"^source:\s*", "", source_part, flags=re.I)

            # Normalize data type
            if dtype in ("str", "string", "text"):
                dtype = "string"
            elif dtype in ("int", "integer"):
                dtype = "integer"
            elif dtype in ("float", "number", "numeric", "decimal", "double"):
                dtype = "float"

            return {
                "name": name,
                "data_type": dtype,
                "description": description,
                "is_dimension": is_dim,
                "nullable": not is_dim,
                "source": source,
            }

        # Level 2: name (type, role) — no description
        m2 = re.match(r"(\w+)\s*\(([^)]+)\)", line)
        if m2:
            name = m2.group(1).strip()
            inner = m2.group(2).strip().lower()
            is_dim = "dim" in inner
            dtype = "string" if is_dim else "float"
            return {
                "name": name,
                "data_type": dtype,
                "description": "",
                "is_dimension": is_dim,
                "nullable": not is_dim,
                "source": "",
            }

        # Level 3: just a word (name)
        m3 = re.match(r"(\w+)", line)
        if m3:
            return {
                "name": m3.group(1).strip(),
                "data_type": "string",
                "description": "",
                "is_dimension": False,
                "nullable": True,
                "source": "",
            }

        return None

    def _parse_sample_row(self, s: str) -> Dict[str, Any]:
        """
        Parse: col1=val1, col2=val2, ...
        Handles commas inside values by splitting on ", key=" pattern.
        """
        result = {}
        for part in re.split(r",\s*(?=\w+=)", s):
            part = part.strip()
            if "=" in part:
                key, val = part.split("=", 1)
                key = key.strip()
                val = val.strip()

                if val.upper() in ("NULL", "NONE", "NAN", "N/A"):
                    result[key] = None  # ← 加这个
                continue
                # Try numeric conversion
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass  # keep as string
                result[key] = val
        return result

    # ==================================================================
    # Fallback schema
    # ==================================================================

    def _fallback_schema(self, df: pd.DataFrame,
                         physical: Dict[str, Any],
                         detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal pass-through schema when LLM call fails entirely.
        """
        sr = physical["data_start_row"]
        er = physical["data_end_row"]
        col_types = physical["column_dtype_profile"]

        target_columns = []
        for j in range(len(df.columns)):
            ct = col_types.get(j, "empty")
            if ct == "empty":
                continue
            target_columns.append({
                "name": f"column_{j}",
                "data_type": "float" if ct == "numeric" else "string",
                "description": f"Column {j}",
                "is_dimension": ct == "text",
                "nullable": True,
                "source": f"Column {j}",
            })

        return {
            "observation_unit": {
                "description": "One data row from source",
                "dimensions": [],
                "example": "",
            },
            "target_columns": target_columns,
            "expected_output": {
                "row_count_formula": "same as source data rows",
                "row_count_estimate": er - sr + 1,
                "column_count": len(target_columns),
            },
            "exclusions": {
                "exclude_rows": {"description": "None", "criteria": []},
                "exclude_columns": {"description": "None", "criteria": []},
            },
            "handling_special_cases": {},
            "validation_samples": [],
            "schema_reasoning": "Fallback: LLM call failed",
            "expected_output_columns": [c["name"] for c in target_columns],
        }