"""
Structure Analyzer Module (Enhanced)
Analyzes table structure using LLM semantic reasoning to identify:
- Header structure (single/multi-level)
- Data regions
- Row patterns (bilingual, grouped by year, etc.)
- Column patterns (dimensions vs values)
- Implicit aggregation hierarchies
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
import pandas as pd
import re

logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """
    Analyzes spreadsheet structure using LLM semantic reasoning.

    Key principle: Let LLM understand the SEMANTIC structure, not pattern match.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the structure analyzer with configuration."""
        self.config = config
        self.detect_implicit_aggregates = config.get('detect_implicit_aggregates', True)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        # self.temperature = config.get('temperature', 0.1)
        self.max_completion_tokens = config.get('max_completion_tokens', 4000)

    def analyze(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure of the encoded spreadsheet using LLM semantic reasoning.

        Returns comprehensive structure analysis including:
        - semantic_understanding: What the data represents
        - header_structure: How headers are organized
        - data_region: Where data starts/ends
        - row_patterns: Patterns in row organization
        - column_patterns: Patterns in column organization
        - implicit_aggregation: Detected aggregation hierarchies (preserved feature)
        """
        logger.info("Analyzing table structure with LLM semantic reasoning...")

        df = encoded_data['dataframe']

        # LLM-based semantic structure analysis
        # LLM now handles ALL pattern detection including implicit aggregation
        llm_analysis = self._analyze_structure_with_llm(encoded_data)

        # Log implicit aggregation detection results from LLM
        implicit_agg = llm_analysis.get('implicit_aggregation', {})
        if implicit_agg.get('has_implicit_aggregation'):
            logger.info(f"LLM detected implicit aggregation:")
            logger.info(f"  Category column: {implicit_agg.get('detection_details', {}).get('category_column')}")
            logger.info(f"  Additional dimension: {implicit_agg.get('detection_details', {}).get('additional_dimension')}")
            logger.info(f"  Delimiter: {implicit_agg.get('detection_details', {}).get('delimiter')}")
        else:
            logger.info("No implicit aggregation detected by LLM")

        logger.info("Structure analysis complete")
        return llm_analysis

    def _analyze_structure_with_llm(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to perform semantic structure analysis."""
        prompt = self._create_structure_analysis_prompt(encoded_data)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                # temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )

            result_text = response.choices[0].message.content
            logger.debug(f"LLM Response: {result_text[:500]}...")

            # Parse JSON response
            analysis_result = self._parse_llm_response(result_text)

            # Validate and set defaults
            analysis_result = self._validate_analysis(analysis_result, encoded_data)

            return analysis_result

        except Exception as e:
            logger.error(f"Error in LLM structure analysis: {e}")
            return self._get_default_analysis(encoded_data)

    def _get_system_prompt(self) -> str:
        """System prompt for structure analysis."""
        return """You are an expert data analyst specializing in understanding spreadsheet structures.

Your task is to perform SEMANTIC analysis of spreadsheets - understanding what the data represents and how it's organized, not just mechanical pattern matching.

## TIDY DATA PRINCIPLES (Your Guide)
A tidy dataset has:
1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

Most spreadsheets are NOT tidy. Your job is to understand their structure so they can be transformed into tidy format.

## COMMON MESSY PATTERNS TO IDENTIFY
1. Column headers that are actually VALUES (e.g., years 2011, 2016, 2021 as column headers)
2. Multiple variables stored in one column
3. Variables stored in both rows AND columns
4. Multiple types of data interleaved (e.g., counts and percentages in alternating rows)
5. Multi-level headers spanning multiple rows
6. Bilingual content (Chinese/English pairs)
7. Section markers (e.g., year appearing once to mark a group of rows)
8. Implicit aggregation (totals mixed with breakdowns)

## IMPLICIT AGGREGATION - CRITICAL PATTERN
Implicit aggregation occurs when SUMMARY rows coexist with DETAIL rows in the same table.
Example: A category column might have values like:
  - "Physical abuse" (summary - total count)
  - "Physical abuse - Male" (detail - breakdown by gender)
  - "Physical abuse - Female" (detail - breakdown by gender)

The detail rows contain an ADDITIONAL DIMENSION (e.g., gender) that is encoded within the value itself,
often using a delimiter like " - " or "/".

When you detect this pattern:
1. Identify which column contains the category values
2. Identify which values are summaries vs details
3. Identify what additional dimension exists in detail rows
4. Provide sample data from detail rows so downstream can understand the pattern

Be thorough and precise. Output valid JSON only."""

    def _create_structure_analysis_prompt(self, encoded_data: Dict[str, Any]) -> str:
        """Create prompt for structure analysis."""
        df = encoded_data['dataframe']
        encoded_text = encoded_data.get('encoded_text', '')

        # Create detailed view of first 30 rows
        rows_preview = self._create_rows_preview(df, max_rows=30)

        # Get column statistics
        col_stats = self._get_column_statistics(df)

        prompt = f"""Analyze the semantic structure of this spreadsheet.

## SPREADSHEET OVERVIEW
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Column names from pandas: {df.columns.tolist()}

## DETAILED ROW CONTENTS (first 30 rows)
{rows_preview}

## COLUMN STATISTICS
{col_stats}

## ENCODED REPRESENTATION (SpreadsheetLLM format)
{encoded_text}

## YOUR ANALYSIS TASK

Analyze this spreadsheet and identify:

1. **Semantic Understanding**: What does this data represent? What is being measured?

2. **Header Structure**: 
   - Which rows contain headers?
   - Are there multi-level headers (headers spanning multiple rows)?
   - What does each header level represent?

3. **Data Region**:
   - Where does the actual data start (row index)?
   - Where does it end?
   - Are there any footer/note rows to exclude?

4. **Row Patterns**:
   - Are there bilingual rows (Chinese/English pairs)?
   - If bilingual, do both rows contain the SAME data (translation) or DIFFERENT data (e.g., counts vs percentages)?
   - Are there section markers (e.g., year appearing to mark a group)?
   - Are there total/subtotal rows mixed with detail rows?

5. **Column Patterns**:
   - Which columns are ID/dimension columns (should remain as columns in tidy format)?
   - Which columns contain values that should be unpivoted?
   - Are column headers actually data values (e.g., years, age groups)?

6. **Implicit Aggregation Detection** (CRITICAL):
   - Are there rows representing TOTALS/SUMMARIES alongside DETAIL breakdowns?
   - Look for category values where one is a substring of another (e.g., "Type A" vs "Type A - Male")
   - If detected:
     a. Which column contains these category values?
     b. Which values are summaries (shorter) vs details (longer, contain extra info)?
     c. What additional dimension is encoded in the detail values?
     d. What delimiter separates the parts (e.g., " - ", "/", ":")?
     e. Provide 3-5 sample values from DETAIL rows

7. **Other Special Patterns**:
   - Any merged cells or spanning headers?
   - Any other structural quirks?

## OUTPUT FORMAT (JSON)

{{
  "semantic_understanding": {{
    "data_description": "What this spreadsheet contains",
    "observation_unit": "What constitutes one observation in tidy format",
    "key_variables": ["List of variables/dimensions identified"]
  }},
  
  "header_structure": {{
    "header_rows": [list of row indices that are headers],
    "num_levels": 1,
    "level_details": [
      {{
        "level": 0,
        "row_index": 0,
        "semantic_meaning": "What this header level represents",
        "values_found": ["sample values"]
      }}
    ],
    "column_header_mapping": {{
      "description": "How column positions map to header values",
      "id_columns": [{{\"col_index\": 0, \"semantic\": \"year\"}}],
      "value_columns": [{{\"col_indices\": [3,4,5], \"header_values\": [\"<15\", \"15-24\", \"25-34\"], \"semantic\": \"age_group\"}}]
    }}
  }},
  
  "data_region": {{
    "start_row": 7,
    "end_row": 49,
    "notes": "Any notes about the data region"
  }},
  
  "row_patterns": {{
    "has_bilingual_rows": true,
    "bilingual_details": {{
      "pattern": "alternating",
      "cn_rows": "even indices starting from data_start",
      "en_rows": "odd indices",
      "data_relationship": "SAME_DATA_TRANSLATION | DIFFERENT_DATA_TYPES",
      "if_different_types": {{
        "cn_row_contains": "counts",
        "en_row_contains": "percentages"
      }}
    }},
    "has_section_markers": true,
    "section_marker_details": {{
      "marker_column": 0,
      "marker_rows": [7],
      "marker_values": [2011],
      "semantic": "year"
    }},
    "has_total_rows": false,
    "total_row_indices": []
  }},
  
  "column_patterns": {{
    "id_columns": [
      {{"col_index": 0, "name": "Year", "notes": "Section marker, needs forward fill"}},
      {{"col_index": 1, "name": "Ethnicity", "notes": "Bilingual"}}
    ],
    "value_columns": [
      {{"col_indices": [3,4,5,6,7,8,9], "represents": "age_group", "header_row": 4}}
    ],
    "aggregate_columns": [
      {{"col_index": 10, "type": "total", "should_exclude": true}}
    ],
    "metadata_columns": [
      {{"col_index": 11, "name": "median_age", "notes": "Only in CN rows"}}
    ]
  }},
  
  "implicit_aggregation": {{
    "has_implicit_aggregation": false,
    "detection_details": {{
      "category_column": "Name or index of column containing category values",
      "summary_values": ["List of category values that might appear to be summaries - but verify if they are truly aggregation rows or just a different category"],
      "detail_values": ["List of category values that contain additional dimensions (have delimiters)"],
      "additional_dimension": "What extra dimension is encoded in detail values (e.g., 'gender', 'age_group')",
      "delimiter": "The delimiter used to separate parts (e.g., ' - ', '/')",
      "reasoning": "Explain how you identified this pattern and whether the 'summary' values are true aggregations (Total, Sum) or just different categories"
    }},
    "sample_detail_rows": [
      {{
        "row_index": 10,
        "category_value": "Physical abuse - Male",
        "all_columns": {{"col_name": "value", "another_col": "value"}}
      }}
    ],
    "transformation_guidance": {{
      "rows_to_exclude": "CRITICAL: Specify 'None' if all rows are valid observations. Only specify rows to exclude if there are TRUE aggregation rows like 'Total', 'Sum', '合計'. Do NOT exclude rows just because they lack a delimiter - those may be different categories, not aggregations.",
      "column_to_split": "Which column contains combined values to split",
      "expected_new_columns": ["Suggested names for columns after splitting"]
    }}
  }},
  
  "special_patterns": [
    {{
      "pattern_type": "name_of_pattern",
      "description": "Detailed description",
      "affected_rows_or_cols": "specifics"
    }}
  ],
  
  "transformation_complexity": "low | medium | high",
  "transformation_notes": "Key challenges for transforming this to tidy format"
}}

Output ONLY the JSON object, no other text."""

        return prompt

    def _create_rows_preview(self, df: pd.DataFrame, max_rows: int = 30) -> str:
        """Create detailed preview of rows for LLM analysis."""
        lines = []
        for i in range(min(max_rows, len(df))):
            row_content = []
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    val_str = str(val)[:50]  # Truncate long values
                    row_content.append(f"[{j}]={repr(val_str)}")

            if row_content:
                lines.append(f"Row {i}: {', '.join(row_content)}")
            else:
                lines.append(f"Row {i}: (empty)")

        if len(df) > max_rows:
            lines.append(f"... ({len(df) - max_rows} more rows)")

        return "\n".join(lines)

    def _get_column_statistics(self, df: pd.DataFrame) -> str:
        """Get statistics for each column."""
        lines = []
        for j, col in enumerate(df.columns):
            non_null = df.iloc[:, j].dropna()
            if len(non_null) == 0:
                lines.append(f"Col {j}: All empty")
                continue

            # Check data types
            sample_vals = non_null.head(5).tolist()
            numeric_count = sum(1 for v in non_null if isinstance(v, (int, float)))
            string_count = len(non_null) - numeric_count

            # Check for Chinese characters
            has_chinese = any(
                any('\u4e00' <= c <= '\u9fff' for c in str(v))
                for v in non_null.head(10)
            )

            lines.append(
                f"Col {j}: {len(non_null)} non-null, "
                f"numeric={numeric_count}, string={string_count}, "
                f"has_chinese={has_chinese}, "
                f"samples={sample_vals[:3]}"
            )

        return "\n".join(lines)

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the LLM response and extract JSON."""
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            if response_text.startswith('json'):
                response_text = response_text[4:].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Try to find JSON object in text
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            raise ValueError("Could not extract valid JSON from LLM response")

    def _validate_analysis(self,
                           analysis: Dict[str, Any],
                           encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set defaults for analysis result."""
        df = encoded_data['dataframe']

        # Ensure required fields exist with defaults
        defaults = {
            'semantic_understanding': {
                'data_description': 'Unknown',
                'observation_unit': 'Unknown',
                'key_variables': []
            },
            'header_structure': {
                'header_rows': [0],
                'num_levels': 1,
                'level_details': [],
                'column_header_mapping': {}
            },
            'data_region': {
                'start_row': 1,
                'end_row': len(df) - 1,
                'notes': ''
            },
            'row_patterns': {
                'has_bilingual_rows': False,
                'has_section_markers': False,
                'has_total_rows': False
            },
            'column_patterns': {
                'id_columns': [],
                'value_columns': [],
                'aggregate_columns': [],
                'metadata_columns': []
            },
            'special_patterns': [],
            'transformation_complexity': 'medium',
            'transformation_notes': '',
            'implicit_aggregation': {
                'has_implicit_aggregation': False,
                'detection_details': {},
                'sample_detail_rows': [],
                'transformation_guidance': {}
            }
        }

        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
            elif isinstance(default_value, dict):
                for sub_key, sub_default in default_value.items():
                    if sub_key not in analysis[key]:
                        analysis[key][sub_key] = sub_default

        # Validate row indices are within bounds
        data_region = analysis.get('data_region', {})
        if data_region.get('start_row', 0) < 0:
            data_region['start_row'] = 0
        if data_region.get('end_row', 0) >= len(df):
            data_region['end_row'] = len(df) - 1

        return analysis

    def _get_default_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return default analysis if LLM fails."""
        df = encoded_data['dataframe']

        return {
            'semantic_understanding': {
                'data_description': 'Unable to analyze - using defaults',
                'observation_unit': 'Unknown',
                'key_variables': list(df.columns)
            },
            'header_structure': {
                'header_rows': [0],
                'num_levels': 1,
                'level_details': [],
                'column_header_mapping': {}
            },
            'data_region': {
                'start_row': 1,
                'end_row': len(df) - 1,
                'notes': 'Default analysis'
            },
            'row_patterns': {
                'has_bilingual_rows': False,
                'has_section_markers': False,
                'has_total_rows': False
            },
            'column_patterns': {
                'id_columns': [{'col_index': 0, 'name': str(df.columns[0])}] if len(df.columns) > 0 else [],
                'value_columns': [],
                'aggregate_columns': [],
                'metadata_columns': []
            },
            'special_patterns': [],
            'transformation_complexity': 'unknown',
            'transformation_notes': 'LLM analysis failed, using minimal defaults',
            'implicit_aggregation': {
                'has_implicit_aggregation': False,
                'detection_details': {},
                'sample_detail_rows': [],
                'transformation_guidance': {}
            }
        }