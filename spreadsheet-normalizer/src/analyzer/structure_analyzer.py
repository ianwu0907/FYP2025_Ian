"""
Structure Analyzer Module — Decomposed Chain Architecture
=========================================================
Replaces a single monolithic LLM call with a chain of 4 focused micro-calls.

Each step asks ONE focused question, uses minimal context, and produces a
small, flat JSON. This dramatically improves reliability on non-reasoning models
(GPT-4o-mini, Qwen, etc.) while keeping the same public API.

Step 1 — Header & Semantics:   Which rows are headers? Where is the data region?
                                 What does this data represent?
Step 2 — Column Roles:          For each column, is it a dimension, value, aggregate,
                                 or metadata?
Step 3 — Row Patterns:          Bilingual pairs? Section markers? Total rows?
Step 4 — Implicit Aggregation:  (Conditional) Does a category column encode an extra
                                 dimension inside its values?
Synthesis — Pure Python:        Combine results → original output schema. No LLM call.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class StructureAnalyzer:
    """
    Analyzes spreadsheet structure using a chain of focused LLM micro-calls.

    Public API is identical to the original monolithic version — downstream
    modules receive the same result dict.
    """

    # Max tokens per micro-call (steps are simpler, so less is needed)
    _STEP_TOKEN_LIMITS = {
        "step1": 1500,
        "step2": 1200,
        "step3": 1200,
        "step4": 1200,
    }

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detect_implicit_aggregates = config.get("detect_implicit_aggregates", True)

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        # json_object mode: forces the model to output valid JSON (no markdown wrapper).
        # Supported by GPT-4o-mini and most Qwen2.5+ models via OpenAI-compatible APIs.
        # Set use_json_mode=false in config if your model does not support it.
        self.use_json_mode = config.get("use_json_mode", True)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API (unchanged)
    # ─────────────────────────────────────────────────────────────────────────

    def analyze(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the 4-step analysis chain and return the combined structure dict.
        Falls back to safe defaults if the chain fails.
        """
        logger.info("Starting 4-step structure analysis chain...")
        df: pd.DataFrame = encoded_data["dataframe"]

        try:
            # ── Step 1: Header detection + semantics + data region ──────────
            logger.info("Step 1/4 — Header & Semantics")
            step1 = self._step1_header_and_semantics(df)
            header_rows  = step1.get("header_structure", {}).get("header_rows", [0])
            data_start   = step1.get("data_region", {}).get("start_row", 1)
            data_end     = step1.get("data_region", {}).get("end_row", len(df) - 1)
            logger.info(f"  → header_rows={header_rows}, data=[{data_start}:{data_end}]")

            # ── Step 2: Column role classification ──────────────────────────
            logger.info("Step 2/4 — Column Roles")
            step2 = self._step2_column_roles(df, step1)
            col_patt = step2.get("column_patterns", {})
            logger.info(
                f"  → id_cols={[c['col_index'] for c in col_patt.get('id_columns', [])]}  "
                f"value_groups={len(col_patt.get('value_columns', []))}  "
                f"agg_cols={[c['col_index'] for c in col_patt.get('aggregate_columns', [])]}"
            )

            # ── Step 3: Row pattern detection ────────────────────────────────
            logger.info("Step 3/4 — Row Patterns")
            step3 = self._step3_row_patterns(df, step1, step2)
            row_patt = step3.get("row_patterns", {})
            logger.info(
                f"  → bilingual={row_patt.get('has_bilingual_rows')}  "
                f"section_markers={row_patt.get('has_section_markers')}  "
                f"total_rows={row_patt.get('has_total_rows')}"
            )

            # ── Step 4: Implicit aggregation (conditional) ───────────────────
            logger.info("Step 4/4 — Implicit Aggregation")
            step4 = self._step4_implicit_aggregation(df, step1, step2, step3)
            impl = step4.get("implicit_aggregation", {})
            if impl.get("has_implicit_aggregation"):
                d = impl.get("detection_details", {})
                logger.info(
                    f"  → DETECTED — col='{d.get('category_column')}', "
                    f"delimiter='{d.get('delimiter')}', "
                    f"extra_dim='{d.get('additional_dimension')}'"
                )
            else:
                logger.info("  → Not detected")

            # ── Synthesis: merge all steps → final schema ────────────────────
            result = self._synthesize(df, step1, step2, step3, step4)
            validated = self._validate_analysis(result, encoded_data)
            logger.info("Structure analysis chain complete.")
            return validated

        except Exception as exc:
            logger.error(f"Analysis chain failed: {exc}", exc_info=True)
            return self._get_default_analysis(encoded_data)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1 — Header detection, semantic understanding, data region
    # ─────────────────────────────────────────────────────────────────────────

    def _step1_header_and_semantics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Ask the LLM:
          1. What does this data represent?
          2. Which rows are headers?
          3. Where does the actual data start and end?

        Context provided: first 15 rows + last 5 rows + total shape.
        """
        rows_text = self._rows_preview(df, start=0, end=min(15, len(df)), label="FIRST ROWS")
        if len(df) > 20:
            rows_text += "\n...\n"
            rows_text += self._rows_preview(df, start=max(len(df)-5, 15), end=len(df), label="LAST ROWS")

        col_stats = self._column_stats(df)

        system_prompt = """\
You are a spreadsheet analyst. Given the first and last rows of a spreadsheet,
identify its semantic meaning, which rows are headers, and the data region.

DEFINITIONS
- Header row: contains column labels that describe what the data means, NOT actual data values.
  Headers are usually (but not always) at the top. Multiple consecutive header rows form multi-level headers.
- Data region: rows that contain actual observations/measurements.
- Footer rows: rows at the bottom with source notes, disclaimers, or blank separators — NOT real data.

RULES
1. A row is a header if its values are LABELS (describing columns), not measurements.
2. Rows containing only numeric values that look like real measurements are data rows.
3. Rows with "Source:", "Note:", "※", "* " at the start are footers — set end_row accordingly.
4. Empty rows between headers and data are NOT data rows; skip them when setting start_row.
5. If a spreadsheet has no visible footer, end_row = total_rows - 1.
6. column_header_mapping.value_columns lists columns whose HEADER VALUES are themselves data
   (e.g., years 2010/2015/2020, age bands "<15"/"15-24", regions "North"/"South").
   These columns will later be unpivoted into a single variable column.

OUTPUT — valid JSON only, no markdown:
{
  "semantic_understanding": {
    "data_description": "<one sentence: what this dataset measures>",
    "observation_unit": "<what one tidy row will represent>",
    "key_variables": ["<variable1>", "..."]
  },
  "header_structure": {
    "header_rows": [<int>, ...],
    "num_levels": <int>,
    "level_details": [
      {"level": <int>, "row_index": <int>, "semantic_meaning": "<str>", "values_found": ["<str>", "..."]}
    ],
    "column_header_mapping": {
      "description": "<how column headers relate to data>",
      "id_columns": [{"col_index": <int>, "semantic": "<role>"}],
      "value_columns": [{"col_indices": [<int>, ...], "header_values": ["<val>", "..."], "semantic": "<what the header values represent>"}]
    }
  },
  "data_region": {
    "start_row": <int>,
    "end_row": <int>,
    "notes": "<any caveats>"
  }
}"""

        user_prompt = f"""\
SPREADSHEET — {df.shape[0]} rows × {df.shape[1]} columns
Pandas column names (auto-assigned): {df.columns.tolist()}

{rows_text}

COLUMN STATISTICS (non-null counts, types, samples):
{col_stats}

────────────────────────────────────────────────────────
EXAMPLES — study these, then analyse the spreadsheet above.

EXAMPLE 1 — simple table, single header row
  Row 0: [0]='Country' [1]='Year' [2]='Exports_USD' [3]='Imports_USD'
  Row 1: [0]='Canada'  [1]=2019  [2]=450000       [3]=380000
  Row 2: [0]='Mexico'  [1]=2019  [2]=390000       [3]=420000
→ Row 0 is the only header. Data is rows 1-2. No footer.
Output excerpt:
{{
  "semantic_understanding": {{"data_description": "Trade statistics by country and year",
    "observation_unit": "one country in one year", "key_variables": ["country","year","exports","imports"]}},
  "header_structure": {{"header_rows": [0], "num_levels": 1,
    "level_details": [{{"level":0,"row_index":0,"semantic_meaning":"column labels","values_found":["Country","Year","Exports_USD","Imports_USD"]}}],
    "column_header_mapping": {{"description":"Single label row",
      "id_columns":[{{"col_index":0,"semantic":"country"}},{{"col_index":1,"semantic":"year"}}],
      "value_columns":[{{"col_indices":[2,3],"header_values":["Exports_USD","Imports_USD"],"semantic":"trade measurement type"}}]}}}},
  "data_region": {{"start_row":1,"end_row":2,"notes":"All rows after row 0 are data"}}
}}

EXAMPLE 2 — wide format, two-level headers, column headers are data values (quarters)
  Row 0: [0]=(empty)    [1]='Product'  [2]='Region A'         [4]='Region B'
  Row 1: [0]='Category' [1]=(empty)    [2]='Q1'  [3]='Q2'    [4]='Q1'  [5]='Q2'
  Row 2: [0]='Electronics' [1]='Laptop' [2]=120  [3]=145      [4]=98    [5]=110
  Row 3: [0]='Electronics' [1]='Phone'  [2]=340  [3]=310      [4]=280   [5]=295
  Row 8: [0]='Source: Sales DB 2023'
→ Rows 0-1 are headers. Data rows 2-7. Row 8 is a footer (starts with "Source:").
Output excerpt:
{{
  "semantic_understanding": {{"data_description":"Product sales by region and quarter",
    "observation_unit":"one product in one region for one quarter","key_variables":["category","product","region","quarter","sales"]}},
  "header_structure": {{"header_rows":[0,1],"num_levels":2,
    "level_details":[
      {{"level":0,"row_index":0,"semantic_meaning":"region grouping","values_found":["Region A","Region B"]}},
      {{"level":1,"row_index":1,"semantic_meaning":"quarter within region","values_found":["Category","Product","Q1","Q2"]}}
    ],
    "column_header_mapping":{{"description":"Two-level: region spans Q1/Q2",
      "id_columns":[{{"col_index":0,"semantic":"category"}},{{"col_index":1,"semantic":"product"}}],
      "value_columns":[{{"col_indices":[2,3,4,5],"header_values":["Region A Q1","Region A Q2","Region B Q1","Region B Q2"],"semantic":"region and quarter"}}]}}}},
  "data_region":{{"start_row":2,"end_row":7,"notes":"Row 8 is a footer note"}}
}}

EXAMPLE 3 — long format (already tidy), no unpivoting needed
  Row 0: [0]='PatientID' [1]='VisitDate' [2]='BloodPressure' [3]='HeartRate'
  Row 1: [0]='P001'  [1]='2023-01-10'  [2]=120  [3]=72
  Row 2: [0]='P001'  [1]='2023-06-15'  [2]=118  [3]=75
→ Single header. All columns are already tidy (ID + date + two measurements). No unpivoting.
Output excerpt:
{{
  "header_structure": {{"header_rows":[0],"num_levels":1,"level_details":[...],
    "column_header_mapping":{{"description":"All columns are direct variables — no unpivoting needed",
      "id_columns":[{{"col_index":0,"semantic":"patient id"}},{{"col_index":1,"semantic":"visit date"}}],
      "value_columns":[{{"col_indices":[2,3],"header_values":["BloodPressure","HeartRate"],"semantic":"measurement type"}}]}}}},
  "data_region":{{"start_row":1,"end_row":2,"notes":""}}
}}
────────────────────────────────────────────────────────

Now analyse the spreadsheet shown above.
Output ONLY the JSON — no markdown fences, no explanation."""

        default = {
            "semantic_understanding": {"data_description": "Unknown", "observation_unit": "Unknown", "key_variables": []},
            "header_structure": {"header_rows": [0], "num_levels": 1, "level_details": [], "column_header_mapping": {}},
            "data_region": {"start_row": 1, "end_row": len(df) - 1, "notes": "Default"},
        }
        return self._call_llm(system_prompt, user_prompt, "step1", default,
                              token_limit=self._STEP_TOKEN_LIMITS["step1"])

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2 — Column role classification
    # ─────────────────────────────────────────────────────────────────────────

    def _step2_column_roles(self, df: pd.DataFrame, step1: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ask the LLM to assign one role to every column:
          - dimension (id column, stays as-is in tidy output)
          - value_to_unpivot (column headers are data values; needs melt)
          - aggregate (total/subtotal column, should be excluded)
          - metadata (supplementary info that doesn't fit the observation unit)

        Context: header rows + first 15 data rows.
        """
        header_rows  = step1.get("header_structure", {}).get("header_rows", [0])
        data_start   = step1.get("data_region", {}).get("start_row", 1)
        data_end     = step1.get("data_region", {}).get("end_row", len(df) - 1)
        semantic_desc = step1.get("semantic_understanding", {}).get("data_description", "unknown")

        # Build context: header rows
        hdr_text = ""
        for r in header_rows:
            if r < len(df):
                hdr_text += self._rows_preview(df, start=r, end=r+1, label=f"HEADER ROW {r}")

        # Build context: first 15 data rows
        data_text = self._rows_preview(
            df, start=data_start, end=min(data_start + 15, data_end + 1),
            label="SAMPLE DATA ROWS"
        )

        col_stats = self._column_stats(df)

        system_prompt = """\
You classify every column of a spreadsheet into one of four roles, to determine
how the spreadsheet should be transformed into tidy format.

ROLE DEFINITIONS
- "dimension": A column that stays as a column in the tidy output. Holds identifying or
  categorical values (e.g., entity names, dates, category labels, IDs). Each row has its
  own distinct value.
- "value_to_unpivot": Multiple columns that all measure the SAME thing but for different
  instances of a variable — and crucially, their COLUMN HEADERS are themselves data values
  (e.g., years, age bands, regions, months). These columns must be melted (unpivoted) into
  a single "variable" column + a "value" column.
- "aggregate": A column whose values are computed totals or subtotals of other columns
  (e.g., a column named "Total", "Sum", "Grand Total", "合計"). Exclude from tidy output.
- "metadata": A supplementary column that does not describe the main observation unit
  (e.g., a rank column, a median mixed with counts, a notes column).

KEY RULE FOR value_to_unpivot
If multiple columns share the same measurement type AND their header labels are instances of
a variable (e.g., "2010" "2015" "2020" are all years; "0-4" "5-9" "10-14" are all age bands),
those columns should be classified together as value_to_unpivot.

GROUP value_to_unpivot columns that belong to the same variable together in ONE entry.
Different measurement types at the same granularity (e.g., count columns AND percentage columns
that correspond to the same year/age-group columns) should be listed as SEPARATE entries.

OUTPUT — valid JSON only, no markdown:
{
  "column_patterns": {
    "id_columns": [
      {"col_index": <int>, "name": "<column label or 'col_N'>", "notes": "<any special handling needed>"}
    ],
    "value_columns": [
      {
        "col_indices": [<int>, ...],
        "represents": "<what the header values represent, e.g. 'year', 'age_group', 'region'>",
        "header_row": <int or null>,
        "notes": "<e.g. 'headers are year numbers', 'two-level combined'>"
      }
    ],
    "aggregate_columns": [
      {"col_index": <int>, "type": "<total|subtotal|other>", "should_exclude": true}
    ],
    "metadata_columns": [
      {"col_index": <int>, "name": "<str>", "notes": "<str>"}
    ]
  }
}"""

        user_prompt = f"""\
DATASET SEMANTIC: {semantic_desc}
TOTAL SHAPE: {df.shape[0]} rows × {df.shape[1]} columns

{hdr_text}
{data_text}

COLUMN STATISTICS:
{col_stats}

────────────────────────────────────────────────────────
EXAMPLES

EXAMPLE 1 — Wide sales table (needs unpivot)
Header row 0: [0]='Store' [1]='Product' [2]='Total' [3]='Jan' [4]='Feb' [5]='Mar'
Data row:     [0]='NYC'   [1]='Laptop'  [2]=365    [3]=120  [4]=115  [5]=130
→ Col 0,1: dimension (store, product)
  Col 2: aggregate (Total = Jan+Feb+Mar)
  Col 3,4,5: value_to_unpivot — headers "Jan","Feb","Mar" are month values; all measure sales
Output:
{{
  "column_patterns": {{
    "id_columns": [{{"col_index":0,"name":"Store","notes":""}},{{"col_index":1,"name":"Product","notes":""}}],
    "value_columns": [{{"col_indices":[3,4,5],"represents":"month","header_row":0,"notes":"Jan/Feb/Mar are month instances"}}],
    "aggregate_columns": [{{"col_index":2,"type":"total","should_exclude":true}}],
    "metadata_columns": []
  }}
}}

EXAMPLE 2 — Demographic table with count + pct pairs (two value types, same granularity)
Header row 0: [0]='District' [1]='AgeGroup' [2]='Count_2010' [3]='Pct_2010' [4]='Count_2020' [5]='Pct_2020'
Data row:     [0]='Central'  [1]='0-14'     [2]=4500        [3]=18.2      [4]=3900        [5]=15.7
→ Col 0,1: dimension
  Col 2,4: value_to_unpivot (counts for each year) — represents "year"
  Col 3,5: value_to_unpivot (percentages for each year) — represents "year" but different metric
Output:
{{
  "column_patterns": {{
    "id_columns": [{{"col_index":0,"name":"District","notes":""}},{{"col_index":1,"name":"AgeGroup","notes":""}}],
    "value_columns": [
      {{"col_indices":[2,4],"represents":"year","header_row":0,"notes":"Count columns for 2010 and 2020"}},
      {{"col_indices":[3,5],"represents":"year","header_row":0,"notes":"Percentage columns for 2010 and 2020"}}
    ],
    "aggregate_columns": [],
    "metadata_columns": []
  }}
}}

EXAMPLE 3 — Already-tidy survey table (nothing to unpivot)
Header row 0: [0]='RespondentID' [1]='Age' [2]='Gender' [3]='Score' [4]='Rank'
Data row:     [0]='R001'         [1]=34    [2]='F'      [3]=87.5    [4]=12
→ Col 0,1,2: dimension; Col 3: value (single measurement, no unpivot); Col 4: metadata (rank)
Output:
{{
  "column_patterns": {{
    "id_columns": [{{"col_index":0,"name":"RespondentID","notes":""}},{{"col_index":1,"name":"Age","notes":""}},{{"col_index":2,"name":"Gender","notes":""}}],
    "value_columns": [{{"col_indices":[3],"represents":"measurement","header_row":0,"notes":"Single score column, no unpivot needed"}}],
    "aggregate_columns": [],
    "metadata_columns": [{{"col_index":4,"name":"Rank","notes":"derived rank, not a core observation"}}]
  }}
}}
────────────────────────────────────────────────────────

Now classify every column in the spreadsheet shown above.
Output ONLY the JSON — no markdown fences, no explanation."""

        default = {
            "column_patterns": {
                "id_columns": [],
                "value_columns": [],
                "aggregate_columns": [],
                "metadata_columns": [],
            }
        }
        return self._call_llm(system_prompt, user_prompt, "step2", default,
                              token_limit=self._STEP_TOKEN_LIMITS["step2"])

    # ─────────────────────────────────────────────────────────────────────────
    # Step 3 — Row pattern detection
    # ─────────────────────────────────────────────────────────────────────────

    def _step3_row_patterns(
            self,
            df: pd.DataFrame,
            step1: Dict[str, Any],
            step2: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Detect special row-level patterns:
          1. Bilingual row pairs (same entity in two languages, may carry different data)
          2. Section marker rows (a grouping value in one cell, rest empty)
          3. Total/aggregate rows (labelled "Total" or similar)

        Context: a 30-row spread sample from the data region.
        """
        data_start = step1.get("data_region", {}).get("start_row", 1)
        data_end   = step1.get("data_region", {}).get("end_row", len(df) - 1)
        semantic_desc = step1.get("semantic_understanding", {}).get("data_description", "unknown")

        # Sample: spread evenly across the data region (max 30 rows)
        sample_indices = self._spread_sample(data_start, data_end, n=30)
        sample_text = self._rows_at_indices(df, sample_indices)

        id_col_summary = [
            f"col {c['col_index']} '{c.get('name', '')}'"
            for c in step2.get("column_patterns", {}).get("id_columns", [])
        ]

        system_prompt = """\
You detect special row-level structural patterns in spreadsheets.

PATTERN 1 — BILINGUAL ROW PAIRS
Pairs of consecutive rows where each pair represents the same entity in two languages.
Signals: same numeric values in two adjacent rows, but text cells in opposite languages
(e.g., Chinese characters vs. Latin characters), OR text columns show translation pairs.
If detected: are the two rows purely translations of each other (SAME_DATA_TRANSLATION),
or do they carry different measurements (DIFFERENT_DATA_TYPES)?
Example of DIFFERENT_DATA_TYPES: one row has raw counts, the next has percentages.

PATTERN 2 — SECTION MARKER ROWS
A single row that marks the start of a data group. Its content: ONE meaningful value in ONE
column (e.g., a year, a region name, a category label) and all other cells empty or null.
The actual data rows immediately follow it. The marker value should be forward-filled.
Signals: a row where only one column has a value AND it is not a dimension that appears on
every data row.

PATTERN 3 — TOTAL / AGGREGATE ROWS
Rows that are computed summaries of surrounding data rows. Usually labelled "Total",
"Grand Total", "Sub-total", "合計", "小計" or similar in a dimension column, with numeric
values that are sums/averages of the rows above or below.
Do NOT confuse: a row labelled "Other" or "Unknown" is a real observation, not a total.

OUTPUT — valid JSON only, no markdown:
{
  "row_patterns": {
    "has_bilingual_rows": <true|false>,
    "bilingual_details": {
      "pattern": "<alternating|block|other>",
      "cn_rows": "<description of which rows are Chinese>",
      "en_rows": "<description of which rows are English>",
      "data_relationship": "<SAME_DATA_TRANSLATION|DIFFERENT_DATA_TYPES>",
      "if_different_types": {
        "cn_row_contains": "<e.g. counts>",
        "en_row_contains": "<e.g. percentages>"
      }
    },
    "has_section_markers": <true|false>,
    "section_marker_details": {
      "marker_column": <int>,
      "marker_rows": [<int>, ...],
      "marker_values": [<value>, ...],
      "semantic": "<what the marker values represent>"
    },
    "has_total_rows": <true|false>,
    "total_row_indices": [<int>, ...]
  }
}
If a pattern is absent, set its boolean to false and its detail field to {}.
"""

        user_prompt = f"""\
DATASET SEMANTIC: {semantic_desc}
IDENTIFIED DIMENSION COLUMNS: {id_col_summary}
DATA REGION: rows {data_start} to {data_end}

SAMPLE ROWS (spread across the data region):
{sample_text}

────────────────────────────────────────────────────────
EXAMPLES

EXAMPLE 1 — Section markers (year groups data rows)
Row 10: [0]=2018, [1]=(empty), [2]=(empty), [3]=(empty)
Row 11: [0]=(empty), [1]='Urban', [2]=1500, [3]=42.1
Row 12: [0]=(empty), [1]='Rural', [2]=820,  [3]=31.4
Row 13: [0]=2019, [1]=(empty), [2]=(empty), [3]=(empty)
Row 14: [0]=(empty), [1]='Urban', [2]=1620, [3]=43.5
→ Rows 10, 13 are section markers for years 2018, 2019. Marker column is 0.
Output:
{{
  "row_patterns": {{
    "has_bilingual_rows": false,
    "bilingual_details": {{}},
    "has_section_markers": true,
    "section_marker_details": {{"marker_column":0,"marker_rows":[10,13],"marker_values":[2018,2019],"semantic":"year"}},
    "has_total_rows": false,
    "total_row_indices": []
  }}
}}

EXAMPLE 2 — Bilingual alternating rows with DIFFERENT data types
Row 5:  [0]='Group A', [1]='子類型', [2]=1200, [3]=850
Row 6:  [0]='Group A', [1]='Sub-type', [2]=48.3, [3]=34.1
Row 7:  [0]='Group B', [1]='子類型', [2]=980,  [3]=420
Row 8:  [0]='Group B', [1]='Sub-type', [2]=39.5, [3]=17.2
→ Odd rows: Chinese label, large integers (counts). Even rows: English label, decimals (percentages).
Output:
{{
  "row_patterns": {{
    "has_bilingual_rows": true,
    "bilingual_details": {{
      "pattern": "alternating",
      "cn_rows": "rows at even offsets from data start (5, 7, ...)",
      "en_rows": "rows at odd offsets from data start (6, 8, ...)",
      "data_relationship": "DIFFERENT_DATA_TYPES",
      "if_different_types": {{"cn_row_contains": "counts", "en_row_contains": "percentages"}}
    }},
    "has_section_markers": false,
    "section_marker_details": {{}},
    "has_total_rows": false,
    "total_row_indices": []
  }}
}}

EXAMPLE 3 — Total rows mixed in
Row 20: [0]='Electronics', [1]='Laptop', [2]=450, [3]=...
Row 21: [0]='Electronics', [1]='Phone',  [2]=310, [3]=...
Row 22: [0]='Electronics', [1]='Total',  [2]=760, [3]=...   ← aggregate row
Row 23: [0]='Clothing',    [1]='Shirt',  [2]=190, [3]=...
Output:
{{
  "row_patterns": {{
    "has_bilingual_rows": false,
    "bilingual_details": {{}},
    "has_section_markers": false,
    "section_marker_details": {{}},
    "has_total_rows": true,
    "total_row_indices": [22]
  }}
}}
────────────────────────────────────────────────────────

Analyse the sample rows above. Only report a pattern if you see clear evidence of it.
Output ONLY the JSON — no markdown fences, no explanation."""

        default = {
            "row_patterns": {
                "has_bilingual_rows": False,
                "bilingual_details": {},
                "has_section_markers": False,
                "section_marker_details": {},
                "has_total_rows": False,
                "total_row_indices": [],
            }
        }
        return self._call_llm(system_prompt, user_prompt, "step3", default,
                              token_limit=self._STEP_TOKEN_LIMITS["step3"])

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4 — Implicit aggregation detection (conditional)
    # ─────────────────────────────────────────────────────────────────────────

    def _step4_implicit_aggregation(
            self,
            df: pd.DataFrame,
            step1: Dict[str, Any],
            step2: Dict[str, Any],
            step3: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Check whether any dimension column encodes an additional variable inside its values,
        using a delimiter to separate a base category from a sub-dimension.

        Only runs if:
          (a) detect_implicit_aggregates is True, AND
          (b) there is at least one string-type id_column with enough distinct values to
              make implicit aggregation plausible.

        Context: all unique values of each candidate column + 5 sample rows.
        """
        no_aggregation = {
            "implicit_aggregation": {
                "has_implicit_aggregation": False,
                "detection_details": {},
                "sample_detail_rows": [],
                "transformation_guidance": {},
            }
        }

        if not self.detect_implicit_aggregates:
            return no_aggregation

        # Find candidate columns: string-type id_columns
        id_cols = step2.get("column_patterns", {}).get("id_columns", [])
        data_start = step1.get("data_region", {}).get("start_row", 1)
        data_end   = step1.get("data_region", {}).get("end_row", len(df) - 1)
        data_df    = df.iloc[data_start:data_end + 1]

        candidates = []
        for col_info in id_cols:
            ci = col_info["col_index"]
            if ci >= len(df.columns):
                continue
            col_data = data_df.iloc[:, ci].dropna()
            # Only check string-type columns with >2 unique values.
            # dtype can be 'object' (classic pandas) or a StringDtype (pandas 2+).
            is_string_col = col_data.dtype == object or pd.api.types.is_string_dtype(col_data)
            if is_string_col and col_data.nunique() > 2:
                candidates.append((ci, col_info.get("name", f"col_{ci}"), col_data))

        if not candidates:
            return no_aggregation

        # Build prompt context for each candidate (max 3 candidates)
        blocks = []
        for ci, col_name, col_data in candidates[:3]:
            unique_vals = sorted(col_data.unique().tolist(), key=str)
            sample_rows = self._rows_at_indices(
                df,
                self._spread_sample(data_start, data_end, n=5),
                col_highlight=ci,
            )
            blocks.append(
                f"CANDIDATE COLUMN — index {ci}, label '{col_name}'\n"
                f"  Unique values ({len(unique_vals)} total):\n"
                + "\n".join(f"    {repr(v)}" for v in unique_vals[:60])
                + ("\n    ..." if len(unique_vals) > 60 else "")
                + f"\n\n  Sample rows:\n{sample_rows}"
            )
        candidates_text = "\n\n".join(blocks)

        system_prompt = """\
You detect a specific structural pattern called IMPLICIT AGGREGATION in spreadsheet data.

DEFINITION
Implicit aggregation occurs when a categorical column contains BOTH:
  • Summary values — represent a total or overall figure for a category
    (e.g. "Product A")
  • Detail values — represent breakdowns of that total by an additional dimension,
    with the extra dimension encoded INSIDE the value using a delimiter
    (e.g. "Product A - Online", "Product A - Retail")

The detail values are recognisable because: some value IS a prefix of another value,
separated by a consistent delimiter such as " - ", " / ", " : ", " > ".

DISTINGUISH FROM FALSE POSITIVES
• "/" in a compound noun that is NOT hierarchical (e.g. "Asia/Pacific" as a single region
  name) → NOT implicit aggregation.
• Two different categories where one happens to be shorter than the other but there is
  NO hierarchical relationship → NOT implicit aggregation.
• A column with only detail values (no summary counterpart) → NOT implicit aggregation.

GUIDANCE ON rows_to_exclude
Only list rows to exclude if the summary rows are TRUE AGGREGATES (the numbers are sums
of the detail rows). If the summary values represent a DIFFERENT, VALID observation
(e.g., "Neglect" vs "Neglect - Child" where "Neglect" without breakdown is its own
genuine category), then rows_to_exclude = "None".

OUTPUT — valid JSON only, no markdown:
{
  "implicit_aggregation": {
    "has_implicit_aggregation": <true|false>,
    "detection_details": {
      "category_column": "<column name or 'col_N'>",
      "summary_values": ["<val>", ...],
      "detail_values": ["<val>", ...],
      "additional_dimension": "<what the sub-part represents, e.g. 'gender', 'channel'>",
      "delimiter": "<delimiter string>",
      "reasoning": "<one or two sentences explaining the evidence>"
    },
    "sample_detail_rows": [
      {"row_index": <int>, "category_value": "<str>", "all_columns": {<col_name>: <value>, ...}}
    ],
    "transformation_guidance": {
      "rows_to_exclude": "<'None' unless summary rows are true aggregates>",
      "column_to_split": "<column name to split>",
      "expected_new_columns": ["<base_col_name>", "<extra_dim_col_name>"]
    }
  }
}
If NOT detected, return has_implicit_aggregation = false and leave other fields as empty objects/lists."""

        user_prompt = f"""\
Below are the candidate dimension columns with their unique values.
Determine whether any of them exhibit implicit aggregation.

{candidates_text}

────────────────────────────────────────────────────────
EXAMPLES

EXAMPLE 1 — IS implicit aggregation
Unique values of 'Incident Type':
  'Physical abuse'
  'Physical abuse - Male'
  'Physical abuse - Female'
  'Sexual abuse'
  'Sexual abuse - Male'
  'Sexual abuse - Female'
  'Neglect'
Evidence: 'Physical abuse' is a prefix of 'Physical abuse - Male' / '... Female'.
Delimiter ' - ' splits base type from gender. 'Neglect' has no sub-breakdown here.
Output:
{{
  "implicit_aggregation": {{
    "has_implicit_aggregation": true,
    "detection_details": {{
      "category_column": "Incident Type",
      "summary_values": ["Physical abuse", "Sexual abuse"],
      "detail_values": ["Physical abuse - Male","Physical abuse - Female","Sexual abuse - Male","Sexual abuse - Female"],
      "additional_dimension": "gender",
      "delimiter": " - ",
      "reasoning": "Base category values appear both alone (summary) and with ' - gender' suffix (detail). Numbers in summary rows equal sum of detail rows."
    }},
    "sample_detail_rows": [
      {{"row_index": 12, "category_value": "Physical abuse - Male", "all_columns": {{"Incident Type":"Physical abuse - Male","Count":34,"Year":2020}}}}
    ],
    "transformation_guidance": {{
      "rows_to_exclude": "None — 'Neglect' with no breakdown is its own valid category",
      "column_to_split": "Incident Type",
      "expected_new_columns": ["incident_base_type", "gender"]
    }}
  }}
}}

EXAMPLE 2 — NOT implicit aggregation (flat categories)
Unique values of 'Department':
  'Engineering', 'Marketing', 'Sales', 'Human Resources', 'Operations'
→ No value is a prefix of another. All are independent categories.
Output:
{{
  "implicit_aggregation": {{"has_implicit_aggregation": false, "detection_details": {{}}, "sample_detail_rows": [], "transformation_guidance": {{}}}}
}}

EXAMPLE 3 — NOT implicit aggregation (delimiter in name, not hierarchy)
Unique values of 'Trade Route':
  'Asia/Pacific', 'Europe/Middle East', 'Americas/Caribbean'
→ '/' is part of the region name itself; all values are at the same level; no value is a
  sub-breakdown of another.
Output:
{{
  "implicit_aggregation": {{"has_implicit_aggregation": false, "detection_details": {{}}, "sample_detail_rows": [], "transformation_guidance": {{}}}}
}}
────────────────────────────────────────────────────────

Analyse the candidate columns above.
Output ONLY the JSON — no markdown fences, no explanation."""

        result = self._call_llm(system_prompt, user_prompt, "step4", no_aggregation,
                                token_limit=self._STEP_TOKEN_LIMITS["step4"])

        # Normalise: wrap in "implicit_aggregation" if the model returned it at root level
        if "has_implicit_aggregation" in result and "implicit_aggregation" not in result:
            result = {"implicit_aggregation": result}

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # Synthesis — combine all step outputs → final schema (no LLM call)
    # ─────────────────────────────────────────────────────────────────────────

    def _synthesize(
            self,
            df: pd.DataFrame,
            step1: Dict[str, Any],
            step2: Dict[str, Any],
            step3: Dict[str, Any],
            step4: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pure Python: merge the 4 step results into the standard output schema.
        No LLM call — no ambiguity.
        """
        row_patt  = step3.get("row_patterns", {})
        impl_agg  = step4.get("implicit_aggregation", {})

        # ── Compile special_patterns list ─────────────────────────────────────
        special = []
        if row_patt.get("has_bilingual_rows"):
            d = row_patt.get("bilingual_details", {})
            special.append({
                "pattern_type": "bilingual_rows",
                "description": f"Bilingual {d.get('pattern','alternating')} row pairs — "
                               f"relationship: {d.get('data_relationship','unknown')}",
                "affected_rows_or_cols": f"pattern: {d.get('pattern')}",
            })
        if row_patt.get("has_section_markers"):
            d = row_patt.get("section_marker_details", {})
            special.append({
                "pattern_type": "section_markers",
                "description": f"Section marker rows encoding '{d.get('semantic','unknown')}' in "
                               f"column {d.get('marker_column')}",
                "affected_rows_or_cols": f"marker rows: {d.get('marker_rows', [])}",
            })
        if row_patt.get("has_total_rows"):
            special.append({
                "pattern_type": "total_rows",
                "description": "Aggregate/total rows mixed with detail rows",
                "affected_rows_or_cols": f"indices: {row_patt.get('total_row_indices', [])}",
            })
        if impl_agg.get("has_implicit_aggregation"):
            d = impl_agg.get("detection_details", {})
            special.append({
                "pattern_type": "implicit_aggregation",
                "description": f"Column '{d.get('category_column')}' encodes "
                               f"'{d.get('additional_dimension')}' via delimiter '{d.get('delimiter')}'",
                "affected_rows_or_cols": f"column: {d.get('category_column')}",
            })

        # ── Complexity heuristic ───────────────────────────────────────────────
        complexity_score = sum([
            row_patt.get("has_bilingual_rows", False),
            row_patt.get("has_section_markers", False),
            row_patt.get("has_total_rows", False),
            impl_agg.get("has_implicit_aggregation", False),
            len(step2.get("column_patterns", {}).get("value_columns", [])) > 1,
            ])
        complexity = "low" if complexity_score == 0 else ("medium" if complexity_score == 1 else "high")

        # ── Build transformation_notes ────────────────────────────────────────
        notes_parts = []
        col_patt = step2.get("column_patterns", {})
        for vg in col_patt.get("value_columns", []):
            notes_parts.append(
                f"Unpivot cols {vg['col_indices']} (represent '{vg.get('represents', 'unknown')}')"
            )
        if row_patt.get("has_bilingual_rows"):
            notes_parts.append("Handle bilingual row pairs")
        if row_patt.get("has_section_markers"):
            d = row_patt.get("section_marker_details", {})
            notes_parts.append(
                f"Forward-fill section marker in col {d.get('marker_column')} ('{d.get('semantic')}')"
            )
        if impl_agg.get("has_implicit_aggregation"):
            d = impl_agg.get("detection_details", {})
            notes_parts.append(
                f"Split col '{d.get('category_column')}' on '{d.get('delimiter')}' → "
                f"{impl_agg.get('transformation_guidance', {}).get('expected_new_columns', [])}"
            )
        transformation_notes = "; ".join(notes_parts) if notes_parts else "Standard tidy transformation"

        return {
            "semantic_understanding": step1.get("semantic_understanding", {}),
            "header_structure": step1.get("header_structure", {}),
            "data_region": step1.get("data_region", {}),
            "row_patterns": row_patt,
            "column_patterns": col_patt,
            "implicit_aggregation": impl_agg,
            "special_patterns": special,
            "transformation_complexity": complexity,
            "transformation_notes": transformation_notes,
        }

    # ─────────────────────────────────────────────────────────────────────────
    # LLM helper
    # ─────────────────────────────────────────────────────────────────────────

    def _call_llm(
            self,
            system_prompt: str,
            user_prompt: str,
            step_name: str,
            default: Dict[str, Any],
            token_limit: int = 1500,
    ) -> Dict[str, Any]:
        """
        Unified LLM caller.
          - Uses json_object response_format if use_json_mode=True (prevents markdown wrapping)
          - Falls back to regex JSON extraction if json_object mode is unavailable
          - Returns 'default' on any failure
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": token_limit,
        }
        if self.use_json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            raw = response.choices[0].message.content or ""
            logger.debug(f"[{step_name}] raw response ({len(raw)} chars): {raw[:400]}...")
            return self._parse_json(raw, step_name)
        except Exception as exc:
            logger.warning(f"[{step_name}] LLM call failed ({exc}); using default.")
            if self.use_json_mode:
                # Retry without json_object mode
                try:
                    kwargs.pop("response_format", None)
                    response = self.client.chat.completions.create(**kwargs)
                    raw = response.choices[0].message.content or ""
                    return self._parse_json(raw, step_name)
                except Exception as exc2:
                    logger.warning(f"[{step_name}] Retry also failed ({exc2}); using default.")
            return default

    def _parse_json(self, text: str, step_name: str) -> Dict[str, Any]:
        """Robustly parse JSON from LLM output."""
        text = text.strip()
        # Strip markdown code fences
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract the outermost {...} block
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            logger.error(f"[{step_name}] Could not parse JSON from response.")
            raise ValueError(f"[{step_name}] JSON parse failed")

    # ─────────────────────────────────────────────────────────────────────────
    # Data view helpers (context-limited, used per step)
    # ─────────────────────────────────────────────────────────────────────────

    def _rows_preview(
            self,
            df: pd.DataFrame,
            start: int,
            end: int,
            label: str = "ROWS",
            max_val_len: int = 50,
    ) -> str:
        """Render a slice of rows as readable text."""
        lines = [f"{label}:"]
        for i in range(start, min(end, len(df))):
            cells = []
            for j in range(len(df.columns)):
                v = df.iloc[i, j]
                if pd.notna(v) and str(v).strip():
                    cells.append(f"[{j}]={repr(str(v)[:max_val_len])}")
            lines.append(f"  Row {i}: " + (", ".join(cells) if cells else "(empty)"))
        return "\n".join(lines)

    def _rows_at_indices(
            self,
            df: pd.DataFrame,
            indices: List[int],
            col_highlight: Optional[int] = None,
            max_val_len: int = 50,
    ) -> str:
        """Render specific rows (for spread samples)."""
        lines = []
        for i in indices:
            if i >= len(df):
                continue
            cells = []
            for j in range(len(df.columns)):
                v = df.iloc[i, j]
                if pd.notna(v) and str(v).strip():
                    marker = "★" if j == col_highlight else ""
                    cells.append(f"{marker}[{j}]={repr(str(v)[:max_val_len])}")
            lines.append(f"  Row {i}: " + (", ".join(cells) if cells else "(empty)"))
        return "\n".join(lines)

    def _column_stats(self, df: pd.DataFrame) -> str:
        """Compact column statistics (non-null count, type, samples)."""
        lines = []
        for j, col in enumerate(df.columns):
            series = df.iloc[:, j].dropna()
            if series.empty:
                lines.append(f"  Col {j}: all empty")
                continue
            n_num = sum(1 for v in series if isinstance(v, (int, float)))
            n_str = len(series) - n_num
            has_cn = any(
                any("\u4e00" <= c <= "\u9fff" for c in str(v))
                for v in series.head(10)
            )
            samples = [repr(str(v)[:30]) for v in series.head(3).tolist()]
            lines.append(
                f"  Col {j}: {len(series)} non-null | "
                f"numeric={n_num} str={n_str} has_chinese={has_cn} | "
                f"samples=[{', '.join(samples)}]"
            )
        return "\n".join(lines)

    @staticmethod
    def _spread_sample(start: int, end: int, n: int = 30) -> List[int]:
        """Return up to n evenly-spaced row indices within [start, end]."""
        total = end - start + 1
        if total <= n:
            return list(range(start, end + 1))
        step = total / n
        return sorted(set(int(start + i * step) for i in range(n)))

    # ─────────────────────────────────────────────────────────────────────────
    # Validation + defaults (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_analysis(
            self,
            analysis: Dict[str, Any],
            encoded_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Fill missing fields with safe defaults; clamp row indices."""
        df: pd.DataFrame = encoded_data["dataframe"]

        defaults = {
            "semantic_understanding": {
                "data_description": "Unknown",
                "observation_unit": "Unknown",
                "key_variables": [],
            },
            "header_structure": {
                "header_rows": [0],
                "num_levels": 1,
                "level_details": [],
                "column_header_mapping": {},
            },
            "data_region": {
                "start_row": 1,
                "end_row": len(df) - 1,
                "notes": "",
            },
            "row_patterns": {
                "has_bilingual_rows": False,
                "has_section_markers": False,
                "has_total_rows": False,
            },
            "column_patterns": {
                "id_columns": [],
                "value_columns": [],
                "aggregate_columns": [],
                "metadata_columns": [],
            },
            "special_patterns": [],
            "transformation_complexity": "medium",
            "transformation_notes": "",
            "implicit_aggregation": {
                "has_implicit_aggregation": False,
                "detection_details": {},
                "sample_detail_rows": [],
                "transformation_guidance": {},
            },
        }

        for key, default_val in defaults.items():
            if key not in analysis:
                analysis[key] = default_val
            elif isinstance(default_val, dict):
                for sub_key, sub_default in default_val.items():
                    if sub_key not in analysis[key]:
                        analysis[key][sub_key] = sub_default

        # Clamp row indices
        dr = analysis.get("data_region", {})
        if dr.get("start_row", 0) < 0:
            dr["start_row"] = 0
        if dr.get("end_row", 0) >= len(df):
            dr["end_row"] = len(df) - 1

        return analysis

    def _get_default_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Safe fallback when the entire chain fails."""
        df: pd.DataFrame = encoded_data["dataframe"]
        return {
            "semantic_understanding": {
                "data_description": "Unable to analyse — using defaults",
                "observation_unit": "Unknown",
                "key_variables": list(df.columns),
            },
            "header_structure": {
                "header_rows": [0],
                "num_levels": 1,
                "level_details": [],
                "column_header_mapping": {},
            },
            "data_region": {
                "start_row": 1,
                "end_row": len(df) - 1,
                "notes": "Default analysis",
            },
            "row_patterns": {
                "has_bilingual_rows": False,
                "bilingual_details": {},
                "has_section_markers": False,
                "section_marker_details": {},
                "has_total_rows": False,
                "total_row_indices": [],
            },
            "column_patterns": {
                "id_columns": (
                    [{"col_index": 0, "name": str(df.columns[0]), "notes": ""}]
                    if len(df.columns) > 0 else []
                ),
                "value_columns": [],
                "aggregate_columns": [],
                "metadata_columns": [],
            },
            "special_patterns": [],
            "transformation_complexity": "unknown",
            "transformation_notes": "LLM chain failed — using minimal defaults",
            "implicit_aggregation": {
                "has_implicit_aggregation": False,
                "detection_details": {},
                "sample_detail_rows": [],
                "transformation_guidance": {},
            },
        }