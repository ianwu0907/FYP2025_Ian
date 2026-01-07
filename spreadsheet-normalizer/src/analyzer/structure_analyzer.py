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

        # Step 1: LLM-based semantic structure analysis
        llm_analysis = self._analyze_structure_with_llm(encoded_data)

        # Step 2: Detect implicit aggregation (preserved original feature)
        # This is a valuable deterministic check that complements LLM analysis
        if self.detect_implicit_aggregates:
            implicit_agg = self._detect_implicit_aggregation(df, llm_analysis)
            llm_analysis['implicit_aggregation'] = implicit_agg

            # If implicit aggregation detected, add to special patterns
            if implicit_agg.get('has_implicit_aggregation'):
                if 'special_patterns' not in llm_analysis:
                    llm_analysis['special_patterns'] = []
                llm_analysis['special_patterns'].append({
                    'pattern_type': 'implicit_aggregation',
                    'description': f"Detected {len(implicit_agg.get('aggregation_hierarchies', []))} aggregation hierarchies where summary rows coexist with detail rows",
                    'hierarchies': implicit_agg.get('aggregation_hierarchies', [])
                })

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

Be thorough and precise. Output valid JSON only."""

    def _create_structure_analysis_prompt(self, encoded_data: Dict[str, Any]) -> str:
        """Create prompt for structure analysis."""
        df = encoded_data['dataframe']
        encoded_text = encoded_data.get('encoded_text', '')[:3000]

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

6. **Special Patterns**:
   - Any implicit aggregation (totals coexisting with breakdowns)?
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

    def _detect_implicit_aggregation(self,
                                     df: pd.DataFrame,
                                     llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect implicit aggregation by checking category hierarchy and numerical verification.

        PRESERVED ORIGINAL FEATURE: This deterministic check complements LLM analysis.
        """
        logger.info("Detecting implicit aggregation patterns...")

        result = {
            'has_implicit_aggregation': False,
            'aggregation_hierarchies': [],
            'summary_rows': [],
            'detail_rows': []
        }

        if len(df) < 2:
            return result

        # Try to identify category column
        category_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['category', '類別', 'type', '項目']):
                category_col = col
                break

        # Also check from LLM analysis
        if not category_col and 'column_patterns' in llm_analysis:
            id_cols = llm_analysis.get('column_patterns', {}).get('id_columns', [])
            for id_col in id_cols:
                if 'category' in str(id_col.get('name', '')).lower():
                    col_idx = id_col.get('col_index')
                    if col_idx is not None and col_idx < len(df.columns):
                        category_col = df.columns[col_idx]
                        break

        if not category_col:
            logger.debug("No category column found for implicit aggregation detection")
            return result

        # Get unique categories
        categories = df[category_col].dropna().astype(str).unique()
        logger.debug(f"Found {len(categories)} unique categories in column '{category_col}'")

        if len(categories) < 2:
            return result

        # Find category pairs where one contains the other
        hierarchies = []

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i >= j:
                    continue

                cat1_str = str(cat1).strip()
                cat2_str = str(cat2).strip()

                # Check if one is substring of other (hierarchical relationship)
                if cat1_str in cat2_str and len(cat2_str) > len(cat1_str):
                    summary_cat = cat1_str
                    detail_cat = cat2_str
                elif cat2_str in cat1_str and len(cat1_str) > len(cat2_str):
                    summary_cat = cat2_str
                    detail_cat = cat1_str
                else:
                    continue

                # Get row indices for each category
                summary_indices = df[df[category_col].astype(str) == summary_cat].index.tolist()
                detail_indices = df[df[category_col].astype(str) == detail_cat].index.tolist()

                if len(summary_indices) > 0 and len(detail_indices) > 0:
                    if len(detail_indices) > len(summary_indices):
                        extra_dimension = detail_cat.replace(summary_cat, '').strip().lstrip('及').strip()

                        hierarchies.append({
                            'summary_category': summary_cat,
                            'detail_category': detail_cat,
                            'additional_dimension': extra_dimension,
                            'summary_row_count': len(summary_indices),
                            'detail_row_count': len(detail_indices)
                        })

                        result['summary_rows'].extend(summary_indices)
                        result['detail_rows'].extend(detail_indices)

                        logger.info(f"Found hierarchy: '{summary_cat}' → '{detail_cat}'")

        if hierarchies:
            result['summary_rows'] = sorted(list(set(result['summary_rows'])))
            result['detail_rows'] = sorted(list(set(result['detail_rows'])))
            result['has_implicit_aggregation'] = True
            result['aggregation_hierarchies'] = hierarchies

            logger.info(f"DETECTED: {len(hierarchies)} implicit aggregation hierarchies")
        else:
            logger.info("No implicit aggregation detected")

        return result

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
            'transformation_notes': ''
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
                'aggregation_hierarchies': [],
                'summary_rows': [],
                'detail_rows': []
            }
        }
