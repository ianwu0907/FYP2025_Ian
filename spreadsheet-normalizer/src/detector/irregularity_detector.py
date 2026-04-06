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
from typing import Dict, List, Any, Optional, Tuple

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
            "where a top-level category spans several sub-columns. ",
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
            "Column headers are VALUES of a variable rather than "
            "variable names. The table spreads observations across multiple columns that represent "
            "different levels of the same variable. The defining test: could "
            "the column headers be placed into a single new column "
            "without losing information? If yes, it is wide format. "
            "Structural signature: multiple adjacent value columns "
            "whose headers all belong to the same domain. Headers may be "
            "NUMERIC OR NON-NUMERIC (strings, ranges, coded labels). "
            "CRITICAL — do NOT flag WIDE_FORMAT if the dimension values "
            "appear in DATA CELLS of a single column (that is long/"
            "tidy format, or SPARSE_ROW_FILL). Wide format requires "
            "the variable to be spread ACROSS COLUMN HEADERS, not down "
            "rows in one column",
        "schema_guidance":
            "The values encoded in column headers become a new "
            "dimension column. Identify which left-side columns are "
            "ID/label columns (keep as-is) and which right-side "
            "columns are value columns with dimension-value headers "
            "(to unpivot). Each row in the wide table generates "
            "multiple rows in the tidy output (one per header value).",
        "code_guidance":
            "Choose the unpivot approach based on co-occurring "
            "irregularities (this decision is made automatically). "
            "If WIDE_FORMAT is the ONLY structural irregularity: "
            "use pd.melt() — identify id_vars and value_vars from the "
            "header row, set var_name to the new dimension column name. "
            "If MULTI_LEVEL_HEADER, NESTED_COLUMN_GROUPS, or "
            "BILINGUAL_ALTERNATING_ROWS also present: use a "
            "record-collection loop — build a column mapping from the "
            "header rows, then iterate rows and columns to append one "
            "record per value cell. Do NOT attempt to detect which "
            "approach to use — it is provided in the CODE RECIPES.",
    },

    # --- Row irregularities ---

    "BILINGUAL_ALTERNATING_ROWS": {
        "description": (
            "Data rows appear in pairs: the primary-language row contains both the label "
            "AND numeric values, while the immediately following secondary-language row "
            "contains ONLY a translated label and NO numeric values whatsoever. "
            "The secondary-language row is a pure label duplicate — it adds no data. "
            "PHYSICAL DISCRIMINATOR (check this first): if the candidate 'secondary' row "
            "has ANY non-empty numeric columns, it is NOT a bilingual duplicate row — "
            "it is an independent data row with its own dimension value. "
            "CRITICAL: Do NOT flag a dimension column whose values naturally cycle through "
            "two categories (e.g., 有/沒有, Yes/No, Male/Female, True/False) — those are "
            "normal data rows where every row has numeric values. "
            "CRITICAL: Do NOT flag this if bilingual labels are in SEPARATE ADJACENT COLUMNS "
            "(that is INLINE_BILINGUAL, not this irregularity)."
        ),
        "schema_guidance": (
            "The paired bilingual rows represent a SINGLE observation unit. The tidy output MUST "
            "combine them into one row. Create separate dimension columns for each language "
            "(e.g., 'name_cn', 'name_en'). If the secondary language row contains a different "
            "metric (e.g., Row 1 has Counts, Row 2 has Percentages), design the schema to capture "
            "BOTH metrics in the same single tidy row."
        ),
        "code_guidance":
            "Pair up consecutive rows. Take numeric data from "
            "whichever row has it (usually the first of the pair). "
            "Extract the label from both rows into separate language "
            "columns. Then drop the duplicate row.",
    },

    "INLINE_BILINGUAL": {
        "description":
            "Individual cells contain text in two languages within "
            "the same cell, separated by a detectable pattern. "
            "Common separators: newline (\\n), slash (/), parenthesis, "
            "or a space before a Latin character following CJK text. ",
        "schema_guidance":
            "The multilingual content in a single cell is a display "
            "convention. Decide which language to keep, or split "
            "into two columns (one per language).",
        "code_guidance":
            "First detect the separator used in this specific spreadsheet "
            "by inspecting the bilingual cells in the EVIDENCE "
            "(e.g. '\\n', '/', '(', or a space before a Latin word). "
            "Then split on that separator and take the desired language part. "
            "Do NOT assume '\\n' — always verify from the actual cell values. "
            "Apply to both header cells and data cells as needed.",
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
            "(a) Cross-group aggregation: a single category column "
            "contains two or more distinct category values, where one "
            "value represents a coarser aggregation of another. "
            "For example, column 3 has value 'Type of Abuse' whose "
            "rows have 'Physical abuse = 390', while another value "
            "'Type of Abuse and Sex' in the same column has rows "
            "'Physical abuse - Male = 200' and "
            "'Physical abuse - Female = 190' (200+190=390). The rows "
            "belonging to the coarser category value are entirely "
            "redundant and must be excluded.\n"
            "(b) Semantic hierarchy: a row is a parent-level "
            "aggregate of child rows below it, with no keyword or "
            "delimiter signal. For example, 'Asian (other than "
            "Chinese) = 365611' is the sum of 'Filipino', "
            "'Indonesian', etc. This can only be identified by "
            "understanding the domain meaning of the labels.",
        "schema_guidance":
            "All forms of aggregation rows that their values can be derived (typically summed) from other "
            "rows should be EXCLUDED from "
            "the tidy output. CRITICAL: You MUST explicitly list the EXACT "
            "string values of the coarser categories that need to be dropped. "
            "Do NOT write instructions to 'keep granular rows'—you must "
            "instruct the downstream module to explicitly DROP the coarser values.",
        "code_guidance":
            "For cross-group: identify the EXACT category value(s) "
            "in the category column that represent the coarser group. "
            "Use .isin([exact_value]) for filtering — NEVER "
            "str.contains(), because coarser category names are often "
            "substrings of finer ones and str.contains() "
            "will silently delete valid granular rows. "
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
            "output. They are redundant and can be recomputed. ",
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

    def extract(self, df: pd.DataFrame) -> Dict[str, Any]:
        header_depth, data_start = self._find_data_start(df)
        data_end = self._find_data_end(df, data_start)
        col_dtypes = self._column_dtype_profile(df, data_start, data_end)
        blank_rows = self._find_blank_rows(df, data_start, data_end)
        bilingual_candidate = self._detect_bilingual_candidate(
            df, data_start, data_end, col_dtypes
        )
        inline_bilingual_candidate = self._detect_inline_bilingual_candidate(
            df, data_start, data_end
        )

        actual_col_names = [str(c) for c in df.columns]
        left_dim_count = self._detect_left_dim_cols(col_dtypes)

        return {
            "shape": {"rows": len(df), "cols": len(df.columns)},
            "actual_column_names": actual_col_names,
            "header_depth": header_depth,
            "data_start_row": data_start,
            "data_end_row": data_end,
            "data_rows": data_end - data_start + 1,
            "column_dtype_profile": col_dtypes,
            "blank_rows_in_data": blank_rows,
            "bilingual_row_candidate": bilingual_candidate,
            "inline_bilingual_candidate": inline_bilingual_candidate,
            "left_header_cols_num": left_dim_count,
        }

    def _find_data_start(self, df: pd.DataFrame) -> Tuple[int, int]:
        """
        Deterministically locate the physical boundary between headers and data
        using structural signatures (Unnamed checks, density jump, type alignment).
        This strictly avoids LLM semantic guesswork or rigid column counts.

        Returns:
            header_depth (int): The number of rows belonging to the header.
            data_start_row (int): The index of the first row of actual data.
        """
        if df.empty:
            return 0, 0

        n_rows, n_cols = df.shape

        # =====================================================================
        # STEP 1: Fast-path (Pre-check)
        # =====================================================================
        # If pandas successfully parsed headers (no 'Unnamed' columns),
        # we completely trust the underlying engine. The real data starts at 0.
        named_cols = sum(1 for c in df.columns if not str(c).startswith("Unnamed"))
        if named_cols == n_cols and n_cols > 0:
            return 0, 0

        # =====================================================================
        # STEP 2: Transition-path (Search for structural jump)
        # =====================================================================
        # Pandas failed (Unnamed exists). The real header is likely buried inside `df`.
        # We look for the "Data Signature": a dense row that aligns with the
        # dominant data types of the bottom region of the table.

        row_fill_counts = df.notna().sum(axis=1)
        max_fill = row_fill_counts.max()

        if max_fill == 0:
            return 0, 0

        # 2a. Determine dominant type per column (using bottom half to avoid top-level noise)
        eval_start = min(5, n_rows // 2) if n_rows > 10 else 0
        eval_df = df.iloc[eval_start:]
        dom_types = {}

        for col in df.columns:
            non_empty = eval_df[col].dropna()
            if len(non_empty) == 0:
                dom_types[col] = 'empty'
                continue
            # Strip commas for numbers like "15,765"
            cleaned = non_empty.astype(str).str.replace(",", "", regex=False)
            num_mask = pd.to_numeric(cleaned, errors='coerce').notna()
            # If >50% parses as numeric, the column is numeric
            dom_types[col] = 'numeric' if num_mask.mean() > 0.5 else 'text'

        # 2b. Scan downwards to find the structural transition
        for i in range(n_rows):
            row = df.iloc[i]

            # Condition A: Density - must be part of the main table block (>= 80% of max fill)
            if row_fill_counts.iloc[i] < max_fill * 0.8:
                continue

            # Condition B: Type Alignment - does it match the expected data types?
            match_count = 0
            valid_cols = 0
            is_pure_string_row = True

            for col in df.columns:
                val = row[col]
                if pd.isna(val) or dom_types[col] == 'empty':
                    continue

                valid_cols += 1
                cleaned_val = str(val).strip().replace(",", "")
                is_num = pd.to_numeric(pd.Series([cleaned_val]), errors='coerce').notna().iloc[0]

                if is_num:
                    is_pure_string_row = False

                if dom_types[col] == 'numeric' and is_num:
                    match_count += 1
                elif dom_types[col] == 'text' and not is_num:
                    match_count += 1

            # --- Crucial Header Interception ---
            # If the row is 100% strings, it is highly likely the buried Header row.
            if is_pure_string_row:
                if 'numeric' in dom_types.values():
                    # Table has numeric columns, so a pure string row MUST be a header. Skip it.
                    continue
                else:
                    # Edge Case: The entire table is pure text.
                    # The first dense string row is the header, data starts at the next dense row.
                    if i + 1 < n_rows and row_fill_counts.iloc[i+1] >= max_fill * 0.8:
                        return i + 1, i + 1
                    return i, i

            alignment_rate = match_count / valid_cols if valid_cols > 0 else 0

            # If we hit a dense row that aligns with the table's dominant types (>70% match),
            # this is unambiguously the start of the data.
            if alignment_rate >= 0.7:
                return i, i

        # =====================================================================
        # STEP 3: Fallback (Desperation mode)
        # =====================================================================
        # If no clear jump is found, fallback to the first row that has any data.
        for i in range(n_rows):
            if row_fill_counts.iloc[i] > 0:
                return i, i

        return 0, 0

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
    def _detect_left_dim_cols(col_dtypes: Dict[int, str]) -> int:
        """
        Return the index of the first numeric column, used as the boundary
        for [ROW_DIM] / [COL_HEADER] tagging in header rows.

        This value is consumed ONLY by ``_format_headers`` in SchemaEstimator
        and TransformationGenerator, which tag header-row cells to help the
        LLM distinguish row-level dimension labels from column header values.
        It is NOT used in postfilter gates (those use raw numeric counts).

        Assumption: in multi-row header tables (header_depth ≥ 2) the
        leftmost columns are always text labels, so the first numeric column
        is a reliable boundary.  When header_depth = 0 there are no header
        rows to tag, so this value has no effect on LLM input regardless of
        what it returns.
        """
        for j in sorted(col_dtypes.keys()):
            if col_dtypes[j] == "numeric":
                return max(j, 1)
        return 1

    @staticmethod
    def _is_numeric(val) -> bool:
        # Native Python ints/floats and numpy scalar types (np.int64, np.float64, etc.)
        # numpy scalars are NOT subclasses of int/float in Python 3, so check both.
        if isinstance(val, (int, float, np.integer, np.floating)):
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

    def _detect_bilingual_candidate(
            self,
            df: pd.DataFrame,
            data_start: int,
            data_end: int,
            col_dtypes: Dict[int, str],
    ) -> bool:
        """
        Deterministic check: does a consistent pattern exist where one row
        has numeric values and the immediately following row has NONE?

        This is the physical precondition for BILINGUAL_ALTERNATING_ROWS.
        If False, the LLM is instructed not to flag that irregularity,
        preventing false positives from binary dimension columns (有/沒有,
        Male/Female, Yes/No) whose rows all contain numeric values.

        Uses only the first 40 data row pairs for efficiency.
        Requires >80% of sampled consecutive pairs to match the pattern.
        """
        numeric_cols = [j for j, t in col_dtypes.items() if t == "numeric"]
        if not numeric_cols:
            return False

        matched = 0
        total = 0
        limit = min(data_end, data_start + 39)

        for i in range(data_start, limit):
            if i + 1 > data_end:
                break
            row_a = df.iloc[i]
            row_b = df.iloc[i + 1]
            a_has_numeric = any(
                self._is_numeric(row_a.iloc[j])
                for j in numeric_cols
                if pd.notna(row_a.iloc[j])
            )
            b_has_numeric = any(
                self._is_numeric(row_b.iloc[j])
                for j in numeric_cols
                if pd.notna(row_b.iloc[j])
            )
            total += 1
            if a_has_numeric and not b_has_numeric:
                matched += 1

        if total == 0:
            return False
        return (matched / total) >= 0.8

    def _detect_inline_bilingual_candidate(
            self,
            df: pd.DataFrame,
            data_start: int,
            data_end: int,
    ) -> bool:
        """
        Deterministic check: do any text cells physically contain BOTH
        CJK characters AND Latin alphabetic characters within the same cell?

        This is the physical precondition for INLINE_BILINGUAL.
        Binary dimension columns (有/沒有, Male/Female) contain only one
        script and will return False, preventing misdetection.

        Scans the first 100 data rows. Returns True if at least 5% of
        non-numeric text cells contain both scripts.
        """
        def _has_cjk(s: str) -> bool:
            return any(
                '\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf'
                or '\uf900' <= c <= '\ufaff'
                for c in s
            )

        def _has_latin(s: str) -> bool:
            return any('a' <= c.lower() <= 'z' for c in s)

        mixed_count = 0
        total_text = 0
        limit = min(data_end + 1, data_start + 100)

        for i in range(data_start, limit):
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.isna(val):
                    continue
                s = str(val).strip()
                if not s:
                    continue
                # Skip purely numeric cells
                if self._is_numeric(val):
                    continue
                total_text += 1
                if _has_cjk(s) and _has_latin(s):
                    mixed_count += 1

        if total_text == 0:
            return False
        return (mixed_count / total_text) >= 0.05


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
        irregularities = self._postfilter_irregularities(irregularities, physical)
        labels = list(dict.fromkeys(ir["label"] for ir in irregularities))
        logger.info(f"Detected {len(irregularities)} irregularities: {labels}")

        return {
            "physical": physical,
            "irregularities": irregularities,
            "labels": labels,
        }

    # ---- LLM call ----
    def _postfilter_irregularities(
            self,
            irregularities: List[Dict[str, Any]],
            physical: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Deterministically remove detected labels that contradict physical
        features. This is a hard gate — it runs after the LLM call and
        cannot be overridden by LLM output.
        """
        filtered = []
        for ir in irregularities:
            label = ir["label"]

            if label == "BILINGUAL_ALTERNATING_ROWS" and \
                    not physical.get("bilingual_row_candidate", False):
                logger.info(
                    f"  [postfilter] Removed {label}: "
                    f"bilingual_row_candidate=False"
                )
                continue

            if label == "INLINE_BILINGUAL" and \
                    not physical.get("inline_bilingual_candidate", False):
                logger.info(
                    f"  [postfilter] Removed {label}: "
                    f"inline_bilingual_candidate=False"
                )
                continue

            # MULTI_LEVEL_HEADER and NESTED_COLUMN_GROUPS both require at
            # least two physical header rows.  header_depth < 2 means there
            # is zero or one header row — multi-row structure is impossible.
            # The LLM often misreads semantic hierarchies in column names or
            # data cells as structural multi-level headers when header_depth=0.
            if label in ("MULTI_LEVEL_HEADER", "NESTED_COLUMN_GROUPS") and \
                    physical.get("header_depth", 0) < 2:
                logger.info(
                    f"  [postfilter] Removed {label}: "
                    f"header_depth={physical.get('header_depth', 0)} < 2"
                )
                continue

            # AGGREGATE_COLUMNS requires at least three numeric columns:
            # one aggregate column plus at least two component columns that
            # it sums.  With fewer than three numeric columns in the table,
            # a "col A = col B + col C" column-aggregation structure is
            # structurally impossible.  This is a pure count — no assumption
            # is made about which columns are dimensions vs values.
            if label == "AGGREGATE_COLUMNS":
                total_numeric = sum(
                    1 for t in physical["column_dtype_profile"].values()
                    if t == "numeric"
                )
                if total_numeric < 3:
                    logger.info(
                        f"  [postfilter] Removed {label}: "
                        f"only {total_numeric} numeric column(s), "
                        f"need ≥ 3 for column-level aggregation"
                    )
                    continue

            filtered.append(ir)

        removed = len(irregularities) - len(filtered)
        if removed:
            logger.info(
                f"  [postfilter] Removed {removed} label(s) contradicting "
                f"physical features. Remaining: "
                f"{[ir['label'] for ir in filtered]}"
            )
        return filtered
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

=== MANDATORY CHECKLIST ===
Before finishing, verify EACH of the following. Report any that apply:
0. Look at ACTUAL COLUMN NAMES above. Ask: do these look like 
   VARIABLE NAMES (e.g. "Category", "Item", "Count", "年份/Year") 
   or DIMENSION VALUES (e.g. "2010", "2011", "Male", "Female")?
   - If VARIABLE NAMES → columns are properly labeled, do NOT flag WIDE_FORMAT
     unless you find a separate group of columns whose HEADERS are dimension values.
   - If DIMENSION VALUES → those columns are wide-format value columns → WIDE_FORMAT.
1. Do any label cells embed two dimensions in one value using a delimiter (e.g. " - ", " – ", "/", ":", "_", or any other separator)? If so, what is the exact delimiter? → EMBEDDED_DIMENSION_IN_COLUMN
2. Are there category groups where one is entirely a coarser aggregation of another? If yes, name the SPECIFIC COLUMN and the EXACT coarser category value(s) that must be excluded (e.g. "column 3 value '虐待長者性質' is a coarser version of '虐待長者性質及受虐長者性別'"). → IMPLICIT_AGGREGATION_ROWS
3. Is "inline_bilingual_candidate" True in the PHYSICAL FEATURES above? If NO, do NOT flag INLINE_BILINGUAL. If YES, confirm that cells physically contain both CJK and Latin text mixed within the same cell (not just two separate columns with different languages). → INLINE_BILINGUAL
4. Is "bilingual_row_candidate" True in the PHYSICAL FEATURES above? If NO, do NOT flag BILINGUAL_ALTERNATING_ROWS under any circumstances. If YES, confirm that the label-only rows are language translations of the preceding data rows (not an independent observation). → BILINGUAL_ALTERNATING_ROWS
5. Are any dimension columns only filled at block starts? → SPARSE_ROW_FILL
6. If MULTI_LEVEL_HEADER is present: do the top-level header values 
   repeat across multiple column groups (e.g., "Never married" spans 
   cols 3-5, "Married" spans cols 6-8, "Widowed" spans cols 9-11)? 
   If yes → NESTED_COLUMN_GROUPS. If each top-level value appears 
   only once → MULTI_LEVEL_HEADER only.
   
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
        col_names = physical.get("actual_column_names", [])
        col_names_str = ", ".join(
            f'[{j}]="{name}"' for j, name in enumerate(col_names)
            if name and not name.startswith("Unnamed")
        ) or "(unnamed columns)"

        parts = [
            f"Shape: {physical['shape']['rows']} rows × {physical['shape']['cols']} cols",
            f"ACTUAL COLUMN NAMES: {col_names_str}",
            f"Header depth: {physical['header_depth']} rows before data",
            f"Data region: rows {physical['data_start_row']} to {physical['data_end_row']} ({physical['data_rows']} rows)",
            f"Column types: {physical['column_dtype_profile']}",
            f"bilingual_row_candidate: {physical.get('bilingual_row_candidate', False)}",
            f"inline_bilingual_candidate: {physical.get('inline_bilingual_candidate', False)}",
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
        logger.info(f"  bilingual_row_candidate: {physical.get('bilingual_row_candidate', False)}")
        logger.info(f"  inline_bilingual_candidate: {physical.get('inline_bilingual_candidate', False)}")
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