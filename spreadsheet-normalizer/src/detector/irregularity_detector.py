"""
Irregularity Detector Module

Diagnoses structural irregularities in a spreadsheet WITHOUT prescribing
solutions.  Downstream modules (SchemaEstimator, TransformationGenerator)
consume the detected labels and look up the corresponding handling guidance.

Architecture
------------
1.  PhysicalFeatureExtractor  – pure deterministic; extracts objective facts
    about the spreadsheet (header depth, column data types, blank rows, etc.).
    NO semantic judgment.

2.  IrregularityDetector  – a single focused LLM call that receives physical
    features + full data, and outputs which irregularities (from a predefined
    taxonomy) are present, together with evidence.

Taxonomy
--------
The IRREGULARITY_TAXONOMY dict below is the single source of truth.
Each entry contains:
  - description :  what this irregularity looks like (used in the LLM prompt)
  - schema_guidance :  hints for SchemaEstimator
  - code_guidance :  hints for TransformationGenerator
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
from openai import OpenAI

logger = logging.getLogger(__name__)


# ============================================================================
# Irregularity Taxonomy  (single source of truth)
# ============================================================================

IRREGULARITY_TAXONOMY = {

    # --- Header irregularities ---

    "MULTI_LEVEL_HEADER": {
        "description":
            "Column headers span two or more rows, forming a hierarchy. "
            "Upper rows contain group labels (e.g., years, categories) "
            "and lower rows contain sub-labels (e.g., Number, %). "
            "Some header cells may be merged across multiple columns.",
        "schema_guidance":
            "The header hierarchy encodes dimensions. Each level may "
            "introduce a new dimension in the tidy output. Read all "
            "header rows together to understand the full column meaning.",
        "code_guidance":
            "Do NOT use pd.read_excel header= directly. Manually "
            "construct column semantics by reading each header row. "
            "Combine parent + child labels to form composite column "
            "identifiers before reshaping.",
    },

    "NESTED_COLUMN_GROUPS": {
        "description":
            "Value columns are organized into repeating nested groups, "
            "where a top-level category spans several sub-columns. "
            "For example: three marital-status groups each containing "
            "Male/Female/Overall sub-columns, or three year groups "
            "each containing Count/% sub-columns.",
        "schema_guidance":
            "Each nesting level typically becomes a separate dimension "
            "in the tidy output. Identify what each level represents. "
            "Aggregate sub-columns (e.g., Overall, Total within a "
            "group) should usually be excluded.",
        "code_guidance":
            "Build a column mapping: {col_index: (level1_value, "
            "level2_value, ...)}. Use this mapping to melt/unpivot "
            "into multiple dimension columns in one step. Iterate "
            "over groups rather than hard-coding column indices.",
    },

    # --- Layout irregularities ---

    "WIDE_FORMAT": {
        "description":
            "Column headers contain values that should be a dimension "
            "in the tidy output. Common examples: years as columns, "
            "age groups as columns, geographic regions as columns. "
            "The table needs to be reshaped from wide to long.",
        "schema_guidance":
            "The values encoded in column headers become a new "
            "dimension column. Each row in the wide table generates "
            "multiple rows in the tidy output (one per column-group).",
        "code_guidance":
            "Use pd.melt() or manual iteration to unpivot. Identify "
            "which columns are ID columns (to keep) and which are "
            "value columns (to unpivot). The column header becomes "
            "the value of the new dimension column.",
    },

    # --- Row irregularities ---

    "BILINGUAL_ALTERNATING_ROWS": {
        "description":
            "Data rows alternate between two languages. Typically "
            "row N has labels in language A (with numeric data) and "
            "row N+1 has the same label in language B (with or "
            "without duplicate numeric data).",
        "schema_guidance":
            "Bilingual pairs represent a SINGLE observation. The "
            "tidy output should have one row per observation, "
            "optionally with separate columns for each language "
            "label (e.g., name_cn, name_en). Do NOT double-count.",
        "code_guidance":
            "Pair up consecutive rows. Take numeric data from "
            "whichever row has it (usually the first of the pair). "
            "Extract the label from both rows into separate language "
            "columns. Then drop the duplicate row.",
    },

    "INLINE_BILINGUAL": {
        "description":
            "Individual cells contain text in two languages within "
            "the same cell, often separated by a newline (\\n). "
            "Example: '種族\\nEthnicity'. This may appear in headers, "
            "data cells, or both.",
        "schema_guidance":
            "The multilingual content in a single cell is a display "
            "convention. Decide which language to keep, or split "
            "into two columns (one per language).",
        "code_guidance":
            "Split cell values on '\\n' and take the desired language "
            "part. Apply to both header cells and data cells as "
            "needed.",
    },

    "SECTION_HEADER_ROWS": {
        "description":
            "Non-data rows within the data region act as group "
            "headers. They typically have a label in the first "
            "column and empty numeric columns. The label applies "
            "to all subsequent rows until the next section header.",
        "schema_guidance":
            "Section headers encode an additional dimension. The "
            "tidy output needs a new column whose value is "
            "forward-filled from these section header rows.",
        "code_guidance":
            "Identify section header rows (label present, numeric "
            "columns empty). Create a new column, forward-fill the "
            "section value, then drop the section header rows "
            "themselves.",
    },

    "IMPLICIT_AGGREGATION_ROWS": {
        "description":
            "Some rows in the data are redundant because their "
            "values can be derived (typically summed) from other "
            "rows. This includes:\n"
            "(a) Keyword-based totals: rows with labels like Total, "
            "Overall, 合計 that sum the detail rows in their group.\n"
            "(b) Cross-group aggregation: an entire category group "
            "is a coarser version of another group. For example, "
            "group 'Type of Abuse' has 'Physical abuse = 390', "
            "while group 'Type of Abuse and Sex' has "
            "'Physical abuse - Male = 200' and "
            "'Physical abuse - Female = 190' (200+190=390). The "
            "coarser group is entirely redundant.\n"
            "(c) Semantic hierarchy: a row is a parent-level "
            "aggregate of child rows below it, with no keyword or "
            "delimiter signal. For example, 'Asian (other than "
            "Chinese) = 365611' is the sum of 'Filipino', "
            "'Indonesian', etc. This can only be identified by "
            "understanding the domain meaning of the labels.",
        "schema_guidance":
            "All forms of aggregation rows should be EXCLUDED from "
            "the tidy output because they are redundant. Keep only "
            "the MOST GRANULAR observations. For cross-group "
            "aggregation, exclude the entire coarser category group. "
            "For semantic hierarchy, exclude the parent-level rows.",
        "code_guidance":
            "Filter out aggregation rows BEFORE any reshaping. "
            "For keyword-based: filter by label keywords (Total, "
            "Overall, 合計, etc.). "
            "For cross-group: identify which category values "
            "represent the coarser group and drop all rows with "
            "those category values. "
            "For semantic hierarchy: use the specific parent-level "
            "labels identified in the detection EVIDENCE to build "
            "an exclusion list. This cannot be done generically — "
            "the code must use the specific labels from detection.",
    },

    # --- Column irregularities ---

    "AGGREGATE_COLUMNS": {
        "description":
            "Some columns contain totals or overall summaries that "
            "aggregate other columns in the same row. For example: "
            "an 'Overall' column that combines Male + Female, a "
            "'Total' column that sums across age groups, or a "
            "'Median' column alongside distribution breakdowns.",
        "schema_guidance":
            "Aggregate columns should be EXCLUDED from the tidy "
            "output. They are redundant and can be recomputed. "
            "Identify them by their header text (Total, Overall, "
            "合計, Median, etc.) or by their position (typically "
            "the last column in a repeating group).",
        "code_guidance":
            "Identify and drop aggregate columns before reshaping. "
            "Use the column indices from the schema exclusions. "
            "These are typically the last column in each repeating "
            "group or standalone summary columns.",
    },

    "EMBEDDED_DIMENSION_IN_COLUMN": {
        "description":
            "A label column encodes multiple dimensions using a "
            "delimiter within cell values. For example: "
            "'Physical abuse - Male' encodes both abuse_type and "
            "gender. Not all rows may contain the delimiter — rows "
            "without it represent a different category or an "
            "aggregate.",
        "schema_guidance":
            "Split the compound label into separate dimension "
            "columns. Rows without the delimiter should get "
            "NULL/None for the secondary dimension (unless they "
            "are aggregation rows that should be excluded).",
        "code_guidance":
            "Use str.split(delimiter) conditionally. For rows "
            "containing the delimiter, split into primary and "
            "secondary dimensions. For rows without the delimiter, "
            "keep the original value as the primary dimension and "
            "set the secondary dimension to None.",
    },

    # --- Content irregularities ---

    "METADATA_ROWS": {
        "description":
            "The spreadsheet contains non-data rows such as titles, "
            "subtitles, source citations, footnotes, or index "
            "references. These are typically at the top or bottom "
            "of the sheet and do not follow the data structure.",
        "schema_guidance":
            "Metadata rows should be excluded entirely. They do "
            "not contribute to the tidy output.",
        "code_guidance":
            "Slice the DataFrame to only include the data region. "
            "Use the data_start_row and data_end_row from physical "
            "feature extraction.",
    },

    "SPARSE_ROW_FILL": {
        "description":
            "A dimension column has values only in certain rows, "
            "with the understanding that blank cells carry forward "
            "the last non-blank value. Common with Year or Category "
            "columns where the value is written once at the start "
            "of a block and left blank for subsequent rows in that "
            "block.",
        "schema_guidance":
            "The sparse column is a real dimension. Its values "
            "must be forward-filled before any other processing.",
        "code_guidance":
            "Apply df[col].fillna(method='ffill') or equivalent "
            "to the sparse dimension column early in the pipeline, "
            "before any row filtering or reshaping.",
    },
}

# Quick lookup: label → schema guidance, label → code guidance
SCHEMA_GUIDANCE = {k: v["schema_guidance"] for k, v in IRREGULARITY_TAXONOMY.items()}
CODE_GUIDANCE = {k: v["code_guidance"] for k, v in IRREGULARITY_TAXONOMY.items()}


# ============================================================================
# Physical Feature Extractor (deterministic — NO semantic judgment)
# ============================================================================

class PhysicalFeatureExtractor:
    """
    Extracts objective, physical facts about a DataFrame.
    Every output is a statement about what IS in the data,
    never what it MEANS.
    """

    def __init__(self, config: dict = None):
        config = config or {}
        self.min_numeric_ratio = config.get("min_numeric_ratio", 0.3)

    def extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        header_depth, data_start = self._find_data_start(df)
        data_end = self._find_data_end(df, data_start)
        col_dtypes = self._column_dtype_profile(df, data_start, data_end)
        blank_rows = self._find_blank_rows(df, data_start, data_end)

        return {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "header_depth": header_depth,
            "data_start_row": data_start,
            "data_end_row": data_end,
            "data_rows": data_end - data_start + 1,
            "column_dtype_profile": col_dtypes,
            "blank_rows_in_data": blank_rows,
        }

    def _find_data_start(self, df: pd.DataFrame) -> tuple:
        """Find where numeric data begins. Returns (header_depth, data_start_row)."""
        for i in range(len(df)):
            row = df.iloc[i]
            non_empty = row.dropna()
            non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
            if len(non_empty) == 0:
                continue
            numeric_count = sum(1 for v in non_empty if self._is_numeric(v))

            # Check if all numerics are year-like (header, not data)
            year_count = 0
            for v in non_empty:
                if self._is_numeric(v):
                    try:
                        n = float(str(v).strip().replace(",", ""))
                        if 1900 <= n <= 2100 and n == int(n):
                            year_count += 1
                    except (ValueError, OverflowError):
                        pass

            is_year_only_row = numeric_count > 0 and year_count == numeric_count
            ratio = numeric_count / len(non_empty)

            if not is_year_only_row and ratio >= self.min_numeric_ratio and numeric_count >= 2:
                return i, i  # header_depth = rows before this
        return 1, 1

    def _find_data_end(self, df: pd.DataFrame, start: int) -> int:
        if start >= len(df):
            return max(len(df) - 1, 0)
        for i in range(len(df) - 1, start - 1, -1):
            row = df.iloc[i]
            non_empty = row.dropna()
            non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
            if len(non_empty) >= 2:
                return i
        return start

    def _column_dtype_profile(self, df: pd.DataFrame,
                              start: int, end: int) -> Dict[int, str]:
        """For each column, report whether it's predominantly numeric, text, or empty."""
        profile = {}
        for j in range(len(df.columns)):
            col = df.iloc[start:end + 1, j]
            non_empty = col.dropna()
            non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
            if len(non_empty) == 0:
                profile[j] = "empty"
                continue
            num_count = sum(1 for v in non_empty if self._is_numeric(v))
            ratio = num_count / len(non_empty)
            profile[j] = "numeric" if ratio >= 0.7 else ("text" if ratio <= 0.2 else "mixed")
        return profile

    def _find_blank_rows(self, df: pd.DataFrame,
                         start: int, end: int) -> List[int]:
        blanks = []
        for i in range(start, end + 1):
            row = df.iloc[i]
            non_empty = row.dropna()
            non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
            if len(non_empty) == 0:
                blanks.append(i)
        return blanks

    @staticmethod
    def _is_numeric(val) -> bool:
        if isinstance(val, (int, float)):
            try:
                return val == val  # False for NaN
            except Exception:
                return True
        if isinstance(val, str):
            s = val.strip().replace(",", "").replace(" ", "").replace("%", "")
            if not s:
                return False
            if s.startswith("(") and s.endswith(")"):
                s = s[1:-1]
            try:
                float(s)
                return True
            except ValueError:
                return False
        return False


# ============================================================================
# Irregularity Detector (LLM-based)
# ============================================================================

class IrregularityDetector:
    """
    Uses a single LLM call to detect which irregularities from the
    taxonomy are present in the spreadsheet.

    Output: list of {label, evidence, details} dicts.
    """

    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        base_url = os.getenv("OPENAI_BASE_URL")
        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url
        self.client = OpenAI(**kw)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.max_tokens = config.get("max_completion_tokens", 2000)
        self.feature_extractor = PhysicalFeatureExtractor(config)

    def detect(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run detection pipeline.

        Returns:
            {
                "physical": { ... physical features ... },
                "irregularities": [
                    {"label": "WIDE_FORMAT", "evidence": "...", "details": "..."},
                    ...
                ],
                "labels": ["WIDE_FORMAT", ...],   # convenience list
            }
        """
        df = encoded_data["dataframe"]

        logger.info("Extracting physical features...")
        physical = self.feature_extractor.extract(df)
        self._log_physical(physical)

        logger.info("Detecting irregularities (LLM call)...")
        irregularities = self._detect_irregularities(df, physical)
        labels = [ir["label"] for ir in irregularities]
        logger.info(f"Detected {len(irregularities)} irregularities: {labels}")

        return {
            "physical": physical,
            "irregularities": irregularities,
            "labels": labels,
        }

    # ---- LLM call ----

    def _detect_irregularities(self, df: pd.DataFrame,
                               physical: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = self._build_prompt(df, physical)
        system = self._system_prompt()

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
            logger.debug(f"LLM irregularity response:\n{text}")
            return self._parse_response(text)
        except Exception as e:
            logger.error(f"Irregularity detection LLM call failed: {e}")
            return []

    def _system_prompt(self) -> str:
        return (
            "You are a spreadsheet structure analyst. "
            "Given a spreadsheet's physical features and data, identify "
            "which structural irregularities are present from a predefined list. "
            "Output ONLY the irregularities you detect, in the exact format shown."
        )

    def _build_prompt(self, df: pd.DataFrame,
                      physical: Dict[str, Any]) -> str:
        taxonomy_text = self._format_taxonomy()
        headers_text = self._format_all_header_rows(df, physical)
        data_text = self._format_data_rows(df, physical)
        physical_text = self._format_physical(physical)

        return f"""Examine this spreadsheet and identify which structural irregularities are present.

=== IRREGULARITY TYPES (only use labels from this list) ===

{taxonomy_text}

=== FEW-SHOT EXAMPLES ===

--- Example 1 ---
HEADERS:
  Row 0: [0]="表3.1", [2]="2011年...按種族劃分..."
  Row 1: [0]="Table 3.1", [2]="Ethnic minorities by ethnicity..."
  Row 3: [3]="2011", [5]="2016", [7]="2021"
  Row 4: [0]="種族\\nEthnicity", [3]="數目\\nNumber", [4]="百分比\\n%", [5]="數目\\nNumber", [6]="百分比\\n%"
DATA:
  Row 6: [0]="亞洲人（非華人）", [3]="365611", [4]="81", [5]="457188", [6]="78.2"
  Row 7: [0]="Asian (other than Chinese)", (no numeric data)
  Row 8: [0]="菲律賓人", [3]="180000", [4]="39.9"
  Row 9: [0]="Filipino", (no numeric data)
  ...
  Row 41: [0]="All ethnic minorities[3]", [3]="451700", [4]="100"
DETECTED:
IRREGULARITY: METADATA_ROWS
EVIDENCE: Rows 0-1 contain table title and number, not data
DETAILS: Title in Chinese (row 0) and English (row 1)

IRREGULARITY: MULTI_LEVEL_HEADER
EVIDENCE: Row 3 has year labels (2011, 2016, 2021), Row 4 has sub-labels (Number, %). Together they form a 2-level header.
DETAILS: Level 1 = year (2011, 2016, 2021), Level 2 = value type (Number, %)

IRREGULARITY: INLINE_BILINGUAL
EVIDENCE: Header cell [0]="種族\\nEthnicity" contains Chinese and English separated by newline
DETAILS: Pattern appears in multiple header cells

IRREGULARITY: WIDE_FORMAT
EVIDENCE: Columns 3-8 encode year (2011, 2016, 2021) × value type (Number, %) as column headers
DETAILS: 3 year groups × 2 value types = 6 value columns that should be unpivoted

IRREGULARITY: NESTED_COLUMN_GROUPS
EVIDENCE: Each year (2011, 2016, 2021) has two sub-columns (Number, %). Columns: 2011→[3,4], 2016→[5,6], 2021→[7,8]
DETAILS: 2 nesting levels: year × value_type

IRREGULARITY: BILINGUAL_ALTERNATING_ROWS
EVIDENCE: Row 6 "亞洲人（非華人）" has data, Row 7 "Asian (other than Chinese)" has no numeric values. Pattern repeats for all ethnicities.
DETAILS: Chinese rows contain numeric data, English rows are label-only

IRREGULARITY: IMPLICIT_AGGREGATION_ROWS
EVIDENCE: Row 41 "All ethnic minorities[3]" contains totals (451700, 100%)
DETAILS: Summary of all ethnicity detail rows above

--- Example 2 ---
HEADERS:
  Row 0: [0]="Case Type", [3]="2019", [4]="2020", [5]="2021"
DATA:
  Row 1: [0]="Physical abuse - Male", [3]="120", [4]="135", [5]="142"
  Row 2: [0]="Physical abuse - Female", [3]="89", [4]="95"
  Row 3: [0]="Sexual abuse", [3]="45", [4]="52", [5]="58"
DETECTED:
IRREGULARITY: WIDE_FORMAT
EVIDENCE: Columns 3-5 have years (2019, 2020, 2021) as headers
DETAILS: Year dimension encoded as columns

IRREGULARITY: EMBEDDED_DIMENSION_IN_COLUMN
EVIDENCE: Column 0 values like "Physical abuse - Male" contain two dimensions separated by " - ". Not all rows have the delimiter (e.g., "Sexual abuse").
DETAILS: Primary dimension = case_type, secondary = gender, delimiter = " - "

HEADERS:
  Row 0: [0]="Year", [1]="Report Status", [3]="Category", [4]="Category EN", [5]="Item", [6]="Item EN", [7]="Count"
DATA:
  Row 1: [0]="2005", [1]="N/A", [3]="Type of Abuse", [4]="Type of Abuse", [5]="Physical abuse", [6]="Physical abuse", [7]="390"
  Row 2: [0]="2005", [1]="N/A", [3]="Type of Abuse", [5]="Psychological abuse", [7]="26"
  ...
  Row 10: [0]="2005", [1]="N/A", [3]="Type of Abuse and Sex", [5]="Physical abuse - Male", [7]="200"
  Row 11: [0]="2005", [1]="N/A", [3]="Type of Abuse and Sex", [5]="Physical abuse - Female", [7]="190"
  Row 12: [0]="2005", [1]="N/A", [3]="Type of Abuse and Sex", [5]="Psychological abuse - Male", [7]="15"
DETECTED:
IRREGULARITY: IMPLICIT_AGGREGATION_ROWS
EVIDENCE: Category "Type of Abuse" has "Physical abuse = 390", while category "Type of Abuse and Sex" has "Physical abuse - Male = 200" + "Physical abuse - Female = 190" = 390. The first group is a coarser aggregation of the second group along the sex dimension.
DETAILS: Cross-group aggregation — the "Type of Abuse" group is entirely redundant because every value can be derived by summing the male + female rows from "Type of Abuse and Sex"

IRREGULARITY: EMBEDDED_DIMENSION_IN_COLUMN
EVIDENCE: Column 5 values like "Physical abuse - Male" encode both abuse type and sex separated by " - "
DETAILS: Delimiter " - ", primary = abuse type, secondary = sex

--- Example 3 (distinguishing INLINE_BILINGUAL from BILINGUAL_ALTERNATING_ROWS) ---
HEADERS:
  Row 0: [0]="Year", [1]="Status CN", [2]="Status EN", [3]="Category CN", [4]="Category EN", [5]="Item CN", [6]="Item EN", [7]="Count"
DATA:
  Row 1: [0]="2005", [1]="不適用", [3]="虐待長者性質", [4]="Type of Abuse", [5]="身體虐待", [6]="Physical abuse", [7]="390"
  Row 2: [0]="2005", [1]="不適用", [3]="虐待長者性質", [4]="Type of Abuse", [5]="精神虐待", [6]="Psychological abuse", [7]="26"
  Row 3: [0]="2005", [1]="不適用", [3]="虐待長者性質", [4]="Type of Abuse", [5]="疏忽照顧", [6]="Neglect", [7]="3"
DETECTED:
IRREGULARITY: INLINE_BILINGUAL
EVIDENCE: Chinese and English labels appear side-by-side in paired columns: [3]="虐待長者性質" with [4]="Type of Abuse", [5]="身體虐待" with [6]="Physical abuse". Every row contains BOTH languages with numeric data.
DETAILS: This is NOT BILINGUAL_ALTERNATING_ROWS because every row has data — there are no label-only rows. The bilingual content is organized in adjacent column pairs, not alternating rows.

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
EXCLUDE_COLUMNS: None

SAMPLE_ROW: year=2005, report_status_cn=不適用, report_status_en=N/A, indicator_group_cn=虐待長者性質及受虐長者性別, indicator_group_en=Type of Abuse and Sex, item_cn=身體虐待, item_en=Physical abuse, sex=Male, count=200

=== MANDATORY CHECKLIST ===
Before finishing, verify EACH of the following. Report any that apply:
1. Do any label columns contain " - " or "/" separating two dimensions? → EMBEDDED_DIMENSION_IN_COLUMN
2. Are there category groups where one is a coarser aggregation of another? → IMPLICIT_AGGREGATION_ROWS
3. Are there rows with Total/Overall/合計 keywords? → IMPLICIT_AGGREGATION_ROWS
4. Do adjacent columns contain the same content in two languages? → INLINE_BILINGUAL
5. Do consecutive rows alternate between two languages? → BILINGUAL_ALTERNATING_ROWS
6. Are any dimension columns only filled at block starts? → SPARSE_ROW_FILL

=== YOUR TASK ===

PHYSICAL FEATURES:
{physical_text}

HEADERS:
{headers_text}

DATA (full data region):
{data_text}

Examine the data above and list ALL irregularities you detect.
For each one, use EXACTLY this format (three lines per irregularity):

IRREGULARITY: <label from the taxonomy>
EVIDENCE: <specific rows, columns, or cell values that show this>
DETAILS: <brief additional context>

Only report irregularities you have clear evidence for. Do not guess."""

    # ---- Formatters ----

    def _format_taxonomy(self) -> str:
        lines = []
        for label, info in IRREGULARITY_TAXONOMY.items():
            lines.append(f"- {label}: {info['description']}")
        return "\n".join(lines)

    def _format_all_header_rows(self, df: pd.DataFrame,
                                physical: Dict[str, Any]) -> str:
        """Show all rows before data_start_row as header context."""
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
            # show blank rows too for completeness
            else:
                lines.append(f"  Row {i}: (blank)")
        return "\n".join(lines) if lines else "  (no header rows)"

    def _format_data_rows(self, df: pd.DataFrame,
                          physical: Dict[str, Any]) -> str:
        """Format ALL data rows — no truncation for accuracy."""
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

        lines.append(f"  (Total: {er - sr + 1} rows in data region)")
        return "\n".join(lines)

    def _format_physical(self, physical: Dict[str, Any]) -> str:
        parts = [
            f"Shape: {physical['shape']['rows']} rows × {physical['shape']['cols']} cols",
            f"Header depth: {physical['header_depth']} rows before data",
            f"Data region: rows {physical['data_start_row']} to {physical['data_end_row']} ({physical['data_rows']} rows)",
            f"Column types: {physical['column_dtype_profile']}",
        ]
        if physical["blank_rows_in_data"]:
            parts.append(f"Blank rows within data: {physical['blank_rows_in_data']}")
        return "\n".join(parts)

    # ---- Response parsing ----

    def _parse_response(self, text: str) -> List[Dict[str, Any]]:
        """
        Parse blocks of:
            IRREGULARITY: LABEL
            EVIDENCE: ...
            DETAILS: ...
        """
        valid_labels = set(IRREGULARITY_TAXONOMY.keys())
        results = []
        current = None

        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue

            upper = line.upper()

            if upper.startswith("IRREGULARITY:"):
                # Save previous block if any
                if current and current["label"] in valid_labels:
                    results.append(current)
                label = line.split(":", 1)[1].strip().upper()
                # Normalize: remove any extra text after the label
                for vl in valid_labels:
                    if label.startswith(vl):
                        label = vl
                        break
                current = {"label": label, "evidence": "", "details": ""}

            elif upper.startswith("EVIDENCE:") and current:
                current["evidence"] = line.split(":", 1)[1].strip()

            elif upper.startswith("DETAILS:") and current:
                current["details"] = line.split(":", 1)[1].strip()

        # Don't forget the last block
        if current and current["label"] in valid_labels:
            results.append(current)

        return results

    def _log_physical(self, physical: Dict[str, Any]):
        logger.info(f"  Shape: {physical['shape']}")
        logger.info(f"  Data region: rows {physical['data_start_row']}"
                    f" to {physical['data_end_row']}"
                    f" ({physical['data_rows']} rows)")
        logger.info(f"  Column types: {physical['column_dtype_profile']}")
        if physical["blank_rows_in_data"]:
            logger.info(f"  Blank rows in data: {physical['blank_rows_in_data']}")


# ============================================================================
# Convenience: build guidance text for downstream modules
# ============================================================================

def get_schema_guidance_for(labels: List[str]) -> str:
    """
    Given a list of detected irregularity labels, assemble the
    relevant schema guidance into a text block for the SchemaEstimator
    prompt.
    """
    lines = []
    for label in labels:
        if label in SCHEMA_GUIDANCE:
            lines.append(f"[{label}] {SCHEMA_GUIDANCE[label]}")
    return "\n".join(lines) if lines else "No specific guidance."


def get_code_guidance_for(labels: List[str]) -> str:
    """
    Given a list of detected irregularity labels, assemble the
    relevant code generation guidance into a text block for the
    TransformationGenerator prompt.
    """
    lines = []
    for label in labels:
        if label in CODE_GUIDANCE:
            lines.append(f"[{label}] {CODE_GUIDANCE[label]}")
    return "\n".join(lines) if lines else "No specific guidance."