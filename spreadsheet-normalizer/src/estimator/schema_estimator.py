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

            "TIDY DATA PRINCIPLES (Hadley Wickham):\n"
            "1. Each VARIABLE forms a column.\n"
            "2. Each OBSERVATION forms a row.\n"
            "3. Each type of OBSERVATIONAL UNIT forms a table.\n\n"

            "This means:\n"
            "- If column headers contain VALUES (years, categories, age "
            "groups), those must become a new variable column, and the "
            "table must be reshaped from wide to long.\n"
            "- If a single row mixes multiple observations (e.g., male "
            "and female counts side by side), they must be split into "
            "separate rows.\n"
            "- Aggregated/total rows and columns should be excluded "
            "because they violate the principle that each row is one "
            "atomic observation.\n\n"

            "Given a messy spreadsheet, its detected structural "
            "irregularities, and handling guidance for each irregularity, "
            "design the ideal tidy output schema. Answer in the EXACT "
            "text format shown in the examples. Be precise about column "
            "names and row estimates."
        )

    def _build_prompt(self, df: pd.DataFrame,
                      physical: Dict[str, Any],
                      irregularities: List[Dict],
                      labels: List[str]) -> str:

        guidance_text = get_schema_guidance_for(labels)
        irregularity_text = self._format_irregularities(irregularities)
        headers_text = self._format_headers(df, physical)
        data_text = self._format_data(df, physical)

        return f"""Design the tidy output schema for this spreadsheet.

=== OUTPUT FORMAT (follow exactly) ===

OBSERVATION: <what one row in the tidy output represents>

TARGET_COLUMNS:
- <name> (<type>, <role>): <description> | source: <where this comes from>

ROW_ESTIMATE: <formula> = <number>

EXCLUDE_ROWS: <which rows to remove and why>
EXCLUDE_COLUMNS: <which columns to remove and why>

SAMPLE_ROW: <col1>=<val1>, <col2>=<val2>, ...

Rules for TARGET_COLUMNS:
  - <type> is one of: string, integer, float
  - <role> is one of: dimension, value
  - One column per line, each starting with "- "
  - Use snake_case for all column names
  - Dimensions go first, then values

=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
IRREGULARITIES:
  METADATA_ROWS: Rows 0-1 contain titles
  MULTI_LEVEL_HEADER: Row 3 has years (2019, 2020), Row 4 has Number/%
  NESTED_COLUMN_GROUPS: 2019→[col3,col4], 2020→[col5,col6], each with Number/%
  WIDE_FORMAT: Year × value-type encoded as columns
  IMPLICIT_AGGREGATION_ROWS: Row 20 "Total" sums all regions
  AGGREGATE_COLUMNS: Col 7 "Total" sums across years
GUIDANCE:
  [MULTI_LEVEL_HEADER] Read all header rows together to understand full column meaning.
  [NESTED_COLUMN_GROUPS] Each nesting level becomes a dimension. Identify what each level represents.
  [WIDE_FORMAT] Values in column headers become a new dimension column.
  [IMPLICIT_AGGREGATION_ROWS] Exclude aggregation rows.
  [AGGREGATE_COLUMNS] Exclude aggregate columns.
HEADERS:
  Row 0: [0]="Sales Report 2019-2020"
  Row 3: [3]="2019", [5]="2020"
  Row 4: [0]="Region", [1]="Product", [3]="Revenue", [4]="Units", [5]="Revenue", [6]="Units", [7]="Total Rev"
DATA:
  Row 5: [0]="North", [1]="Widget A", [3]="15000", [4]="120", [5]="18000", [6]="140", [7]="33000"
  Row 6: [0]="North", [1]="Widget B", [3]="8000", [4]="65", [5]="9500", [6]="80", [7]="17500"
  Row 20: [0]="Total", [3]="50000", [4]="400", [5]="60000", [6]="500", [7]="110000"
SCHEMA:
OBSERVATION: One product's sales metric in one region for one year

TARGET_COLUMNS:
- region (string, dimension): Geographic region | source: column 0
- product (string, dimension): Product name | source: column 1
- year (string, dimension): Year of sales | source: column headers (2019, 2020)
- revenue (float, value): Sales revenue | source: columns 3,5 (under each year)
- units (integer, value): Units sold | source: columns 4,6 (under each year)

ROW_ESTIMATE: 14 detail rows × 2 years = 28

EXCLUDE_ROWS: Row 20 "Total" (aggregation row)
EXCLUDE_COLUMNS: Column 7 "Total Rev" (aggregate column, sums across years)

SAMPLE_ROW: region=North, product=Widget A, year=2019, revenue=15000, units=120

--- Example 2 ---
IRREGULARITIES:
  METADATA_ROWS: Rows 0-1 contain table title in Chinese and English
  MULTI_LEVEL_HEADER: Row 3 has marital status groups, Row 4 has Male/Female/Overall
  NESTED_COLUMN_GROUPS: 3 marital-status groups × 3 sex sub-columns each
  WIDE_FORMAT: Marital status × sex encoded as column headers
  BILINGUAL_ALTERNATING_ROWS: Chinese rows have data, English rows are label-only
  IMPLICIT_AGGREGATION_ROWS: Rows with "All ethnic minorities" are totals
  AGGREGATE_COLUMNS: "Overall" sub-columns and "Total" group are aggregates
GUIDANCE:
  [NESTED_COLUMN_GROUPS] Each nesting level becomes a separate dimension.
  [WIDE_FORMAT] Values in column headers become a new dimension column.
  [BILINGUAL_ALTERNATING_ROWS] Pairs represent a SINGLE observation. One row per pair, with separate language columns.
  [AGGREGATE_COLUMNS] Exclude aggregate columns.
HEADERS:
  Row 3: [3]="Never married", [6]="Married", [9]="Widowed/divorced", [12]="Total"
  Row 4: [0]="Ethnicity", [3]="Male", [4]="Female", [5]="Overall", [6]="Male", [7]="Female", [8]="Overall", [9]="Male", [10]="Female", [11]="Overall", [12]="Male", [13]="Female", [14]="Overall"
DATA:
  Row 7: [1]="亞洲人（非華人）", [3]="23", [4]="35.6", [5]="34.1", [6]="73.9", ...
  Row 8: [1]="Asian (other than Chinese)", (no numeric data)
  Row 9: [1]="菲律賓人", [3]="26.1", [4]="36.4", ...
  Row 10: [1]="Filipino", (no numeric data)
SCHEMA:
OBSERVATION: One ethnicity's marital-status percentage for one sex

TARGET_COLUMNS:
- ethnicity_cn (string, dimension): Ethnicity name in Chinese | source: column 1 (Chinese rows)
- ethnicity_en (string, dimension): Ethnicity name in English | source: column 1 (English rows)
- marital_status (string, dimension): Marital status category | source: column group headers (Never married, Married, Widowed/divorced)
- sex (string, dimension): Sex category | source: sub-column headers (Male, Female)
- percentage (float, value): Percentage distribution | source: value cells under each group×sex combination

ROW_ESTIMATE: 17 ethnicities × 3 marital statuses × 2 sexes = 102

EXCLUDE_ROWS: "All ethnic minorities" rows (aggregation), English-only rows (merged into Chinese rows as ethnicity_en)
EXCLUDE_COLUMNS: "Overall" sub-columns within each group (col 5,8,11), entire "Total" group (col 12,13,14) — all are aggregates

SAMPLE_ROW: ethnicity_cn=亞洲人（非華人）, ethnicity_en=Asian (other than Chinese), marital_status=Never married, sex=Male, percentage=23

--- Example 3 ---
IRREGULARITIES:
  INLINE_BILINGUAL: Chinese and English labels in adjacent columns (col 3/4, col 5/6)
  EMBEDDED_DIMENSION_IN_COLUMN: Column 5/6 values like "Physical abuse - Male" encode abuse type + sex with " - "
  IMPLICIT_AGGREGATION_ROWS: Category "Type of Abuse" is a coarser aggregation of "Type of Abuse and Sex" — the former has "Physical abuse = 390" while the latter has "Physical abuse - Male = 200" + "Physical abuse - Female = 190"
  SPARSE_ROW_FILL: Year in column 0 written once per block, forward-fill needed
GUIDANCE:
  [EMBEDDED_DIMENSION_IN_COLUMN] Split compound labels into separate dimension columns. Rows without delimiter get NULL for secondary dimension.
  [IMPLICIT_AGGREGATION_ROWS] Exclude the coarser category group entirely. Keep only the most granular observations.
  [INLINE_BILINGUAL] Decide which language to keep, or split into two columns.
  [SPARSE_ROW_FILL] Forward-fill before any other processing.
HEADERS:
  Row 0: [0]="Year", [1]="Report Status CN", [2]="Report Status EN", [3]="Category CN", [4]="Category EN", [5]="Item CN", [6]="Item EN", [7]="Count"
DATA:
  Row 1: [0]="2005", [1]="不適用", [3]="虐待長者性質", [4]="Type of Abuse", [5]="身體虐待", [6]="Physical abuse", [7]="390"
  Row 2: [0]="2005", [3]="虐待長者性質", [5]="精神虐待", [6]="Psychological abuse", [7]="26"
  Row 9: [0]="2005", [3]="虐待長者性質及受虐長者性別", [4]="Type of Abuse and Sex", [5]="身體虐待 - 男性", [6]="Physical abuse - Male", [7]="200"
  Row 10: [0]="2005", [5]="精神虐待 - 男性", [6]="Psychological abuse - Male", [7]="15"
  Row 17: [0]="2005", [5]="身體虐待 - 女性", [6]="Physical abuse - Female", [7]="190"
  Row 25: [0]="2005", [3]="施虐者與受虐長者關係", [4]="Abuser Relationship", [5]="子", [6]="Son", [7]="57"
  Row 37: [0]="2005", [3]="受虐長者居住地區", [4]="Residential District", [5]="中西區", [6]="Central and Western", [7]="19"
SCHEMA:
OBSERVATION: One case count for a specific year, report status, indicator group, item category, and optional sex breakdown

TARGET_COLUMNS:
- year (integer, dimension): Reporting year | source: column 0 (forward-filled)
- report_status_cn (string, dimension): Report status in Chinese | source: column 1 (forward-filled)
- report_status_en (string, dimension): Report status in English | source: column 2 (forward-filled)
- indicator_group_cn (string, dimension): Indicator group in Chinese | source: column 3 (forward-filled)
- indicator_group_en (string, dimension): Indicator group in English | source: column 4 (forward-filled)
- item_cn (string, dimension): Item category in Chinese | source: column 5, primary part before " - " if delimiter present
- item_en (string, dimension): Item category in English | source: column 6, primary part before " - " if delimiter present
- sex (string, dimension): Sex extracted from embedded delimiter | source: columns 5/6, secondary part after " - "; NULL for rows without delimiter
- count (integer, value): Number of cases | source: column 7

ROW_ESTIMATE: 17 years × 3 non-redundant indicator groups × ~13 items average = 663

EXCLUDE_ROWS: All rows where indicator_group is the coarser "虐待長者性質"/"Type of Abuse" — these are aggregates of the finer "虐待長者性質及受虐長者性別"/"Type of Abuse and Sex" group which fully decomposes the same items by sex
EXCLUDE_COLUM

=== YOUR TASK ===

IRREGULARITIES:
{irregularity_text}

GUIDANCE:
{guidance_text}

PHYSICAL FEATURES:
  Data region: rows {physical['data_start_row']} to {physical['data_end_row']} ({physical['data_rows']} rows)
  Column types: {physical['column_dtype_profile']}

HEADERS:
{headers_text}

DATA (full data region):
{data_text}

Now design the tidy schema following the EXACT format above."""

    # ==================================================================
    # Formatters
    # ==================================================================

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
            else:
                lines.append(f"  Row {i}: (blank)")
        return "\n".join(lines) if lines else "  (no header rows)"

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
                # Try to extract the number after "="
                eq_match = re.search(r"=\s*(\d+)", val)
                if eq_match:
                    row_estimate = int(eq_match.group(1))
                else:
                    # Try to find any number
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