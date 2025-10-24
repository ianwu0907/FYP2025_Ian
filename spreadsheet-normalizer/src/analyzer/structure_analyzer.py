"""
Structure Analyzer Module
Analyzes table structure to identify headers, data rows, metadata, and aggregation patterns
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
    Analyzes the structure of spreadsheet tables to identify:
    - Header rows
    - Data rows
    - Metadata rows
    - Aggregate rows (explicit and implicit)
    - Column groupings
    - Structure type
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the structure analyzer with configuration."""
        self.config = config
        self.detect_headers = config.get('detect_headers', True)
        self.detect_aggregates = config.get('detect_aggregates', True)
        self.detect_implicit_aggregates = config.get('detect_implicit_aggregates', True)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 3000)

    def analyze(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the structure of the encoded spreadsheet."""
        logger.info("Analyzing table structure with LLM...")

        df = encoded_data['dataframe']

        # Step 1: LLM-based structure analysis
        llm_analysis = self._analyze_with_llm(encoded_data)

        # Step 2: Detect implicit aggregation (numerical verification)
        if self.detect_implicit_aggregates:
            implicit_agg = self._detect_implicit_aggregation(df, llm_analysis)
            llm_analysis['implicit_aggregation'] = implicit_agg

        logger.info("Structure analysis complete")
        return llm_analysis

    def _analyze_with_llm(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to analyze table structure."""
        prompt = self._create_analysis_prompt(encoded_data)

        try:
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Parse response
            result_text = response.choices[0].message.content
            logger.debug(f"LLM Response: {result_text[:500]}...")

            # Extract JSON from response
            analysis_result = self._parse_llm_response(result_text)

            # Validate the analysis
            analysis_result = self._validate_analysis(analysis_result, encoded_data)

            return analysis_result

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            # Return default analysis if LLM fails
            return self._get_default_analysis(encoded_data)

    def _detect_implicit_aggregation(self,
                                     df: pd.DataFrame,
                                     llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect implicit aggregation by checking category hierarchy and numerical verification.
        """
        logger.info("Detecting implicit aggregation patterns...")

        result = {
            'has_implicit_aggregation': False,
            'aggregation_hierarchies': [],
            'summary_rows': [],
            'detail_rows': []
        }

        # Only proceed if we have enough data
        if len(df) < 2:
            return result

        # Try to identify category column (first match)
        category_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'category' in col_str or '類別' in col_str or 'category' in str(col):
                category_col = col
                break

        if not category_col:
            logger.debug("Cannot detect implicit aggregation: no category column found")
            logger.debug(f"Columns available: {df.columns.tolist()}")
            return result

        # Get unique categories
        categories = df[category_col].dropna().astype(str).unique()
        logger.debug(f"Found {len(categories)} unique categories")

        if len(categories) < 2:
            logger.debug("Cannot detect implicit aggregation: need at least 2 categories")
            return result

        # Find category pairs where one contains the other
        hierarchies = []

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i >= j:
                    continue

                cat1_str = str(cat1).strip()
                cat2_str = str(cat2).strip()

                # Check if cat1 is a substring of cat2 (cat2 is more detailed)
                if cat1_str in cat2_str and len(cat2_str) > len(cat1_str):
                    # cat1 is broader, cat2 is more detailed
                    summary_cat = cat1_str
                    detail_cat = cat2_str
                elif cat2_str in cat1_str and len(cat1_str) > len(cat2_str):
                    # cat2 is broader, cat1 is more detailed
                    summary_cat = cat2_str
                    detail_cat = cat1_str
                else:
                    continue

                # Found a potential hierarchy, now verify with numbers
                summary_indices = df[df[category_col].astype(str) == summary_cat].index.tolist()
                detail_indices = df[df[category_col].astype(str) == detail_cat].index.tolist()

                if len(summary_indices) > 0 and len(detail_indices) > 0:
                    # Try to find value column
                    value_col = None
                    for col in df.columns:
                        col_lower = str(col).lower()
                        if 'count' in col_lower or 'number' in col_lower or 'case' in col_lower or '數' in str(col):
                            value_col = col
                            break

                    if value_col:
                        # Simple check: if detail has more rows, likely a hierarchy
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

                            logger.info(f"Found hierarchy: '{summary_cat}' ({len(summary_indices)} rows) → '{detail_cat}' ({len(detail_indices)} rows)")

        if hierarchies:
            # Remove duplicates
            result['summary_rows'] = sorted(list(set(result['summary_rows'])))
            result['detail_rows'] = sorted(list(set(result['detail_rows'])))

            result['has_implicit_aggregation'] = True
            result['aggregation_hierarchies'] = hierarchies

            logger.info(f"DETECTED: {len(hierarchies)} implicit aggregation hierarchies")
            logger.info(f"Summary rows: {len(result['summary_rows'])} rows")
            logger.info(f"Detail rows: {len(result['detail_rows'])} rows")
        else:
            logger.info("No implicit aggregation detected")

        return result

    def _verify_aggregation_relationship(self,
                                         df: pd.DataFrame,
                                         category_col: str,
                                         value_col: str,
                                         item_col: Optional[str],
                                         summary_category: str,
                                         detail_category: str) -> Optional[Dict[str, Any]]:
        """
        Verify if summary_category rows are aggregates of detail_category rows.
        Returns hierarchy info if verified, None otherwise.
        """
        summary_df = df[df[category_col] == summary_category].copy()
        detail_df = df[df[category_col] == detail_category].copy()

        if len(summary_df) == 0 or len(detail_df) == 0:
            return None

        # If we have item columns, verify item by item
        if item_col:
            verified_items = []

            for item in summary_df[item_col].unique():
                summary_rows = summary_df[summary_df[item_col] == item]
                if len(summary_rows) == 0:
                    continue

                summary_value = pd.to_numeric(summary_rows[value_col], errors='coerce').sum()

                # Find matching detail rows
                # The detail item might have additional suffixes (e.g., "身體虐待 - 男性")
                item_base = str(item).split(' - ')[0]
                detail_matches = detail_df[
                    detail_df[item_col].astype(str).str.contains(item_base, na=False, regex=False)
                ]

                if len(detail_matches) == 0:
                    continue

                detail_value = pd.to_numeric(detail_matches[value_col], errors='coerce').sum()

                # Check if values match (allow small rounding error)
                if abs(summary_value - detail_value) < 1:
                    verified_items.append({
                        'item': str(item),
                        'summary_value': float(summary_value),
                        'detail_value': float(detail_value),
                        'detail_count': len(detail_matches)
                    })

            # If we verified at least half of the items, consider it a valid hierarchy
            if len(verified_items) >= len(summary_df) * 0.5:
                # Extract the additional dimension
                extra_dimension = detail_category.replace(summary_category, '').strip()
                extra_dimension = extra_dimension.lstrip('及').strip()

                return {
                    'summary_category': summary_category,
                    'detail_category': detail_category,
                    'additional_dimension': extra_dimension,
                    'verified_items': verified_items[:5],  # Only keep first 5 for logging
                    'total_verified': len(verified_items),
                    'verification_rate': len(verified_items) / len(summary_df)
                }

        return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for structure analysis."""
        return """You are an expert data analyst specializing in spreadsheet structure analysis.

Your task is to analyze the physical structure of tables and identify different types of rows:
- Header rows (column names)
- Data rows (actual data)
- Metadata rows (titles, dates, notes, descriptions)
- Aggregate rows (totals, subtotals, sums)
- Column groupings and hierarchies

Be precise and analytical. Output valid JSON only."""

    def _create_analysis_prompt(self, encoded_data: Dict[str, Any]) -> str:
        """Create the analysis prompt for the LLM."""
        metadata = encoded_data.get('metadata', {})
        encoded_text = encoded_data.get('encoded_text', '')[:1500]
        df = encoded_data['dataframe']

        # Get row and column counts from dataframe directly
        row_count = len(df)
        column_count = len(df.columns)
        column_names = df.columns.tolist()

        # Get sample of first and last few rows
        first_rows = df.head(10).to_string() if len(df) > 0 else "No data"

        prompt = f"""Analyze the structure of this spreadsheet table.

## Table Metadata:
- Total rows: {row_count}
- Total columns: {column_count}
- Column names: {column_names}

## Sample Data (first 10 rows):
{first_rows}

## Encoded Representation:
{encoded_text}

## Your Analysis Task:

1. **Identify Header Rows**: Which row(s) contain column headers?
   - Look for descriptive text, mixed languages, or field names
   - May be multiple rows for multi-level headers

2. **Identify Metadata Rows**: Which rows are NOT data (titles, dates, notes)?
   - Usually at the top
   - Contain descriptive text, dates, or documentation

3. **Identify Data Rows**: Which rows contain actual data?
   - Regular pattern of values
   - Numeric or categorical data

4. **Identify Aggregate Rows**: Which rows are totals/subtotals?
   - Look for keywords: "Total", "Sum", "小計", "總計", "合計"
   - Usually have special formatting or position

5. **Detect Implicit Aggregation**: 
   - Are there category fields with different levels of detail?
   - Example: "Type of Abuse" (broad) vs "Type of Abuse and Gender" (detailed)
   - If one category name contains another, it might indicate hierarchy
   - NOTE: You'll analyze semantically; numerical verification will be done separately

6. **Column Groupings**: Are columns organized into groups?
   - Merged headers
   - Parent-child relationships

7. **Structure Type**: What type of table is this?
   - "simple": Single header, straightforward rows
   - "multi_header": Multiple header rows
   - "pivot": Pivot table format
   - "hierarchical": Nested categories
   - "irregular": Unusual structure

## Output Format (JSON only):

{{
    "header_rows": [row_indices],
    "data_rows": ["start-end"],
    "metadata_rows": [row_indices],
    "aggregate_rows": [row_indices],
    "column_groups": [
        {{
            "parent_column": "column_name",
            "child_columns": ["col1", "col2", ...]
        }}
    ],
    "potential_category_hierarchy": [
        {{
            "broad_category": "category_name",
            "detailed_category": "category_name_with_more_dimensions",
            "reasoning": "why this might be hierarchical"
        }}
    ],
    "structure_type": "simple|multi_header|pivot|hierarchical|irregular",
    "confidence": 0.0-1.0,
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}}

**CRITICAL**: 
- Row indices are 0-based (first row is 0)
- Use ranges like "9-1319" for continuous data rows
- Be conservative with aggregate_rows (only mark if clearly labeled)
- For potential_category_hierarchy, look for semantic relationships in category field names

Output ONLY the JSON object."""

        return prompt

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
            result = json.loads(response_text)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            # Try to find JSON in the text
            json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            matches = re.findall(json_pattern, response_text, re.DOTALL)
            if matches:
                matches.sort(key=len, reverse=True)
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
            raise ValueError("Could not extract valid JSON from LLM response")

    def _validate_analysis(self,
                           analysis: Dict[str, Any],
                           encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enrich the analysis result."""
        df = encoded_data['dataframe']
        row_count = len(df)

        # Ensure required fields exist
        if 'header_rows' not in analysis:
            analysis['header_rows'] = []
        if 'data_rows' not in analysis:
            analysis['data_rows'] = []
        if 'metadata_rows' not in analysis:
            analysis['metadata_rows'] = []
        if 'aggregate_rows' not in analysis:
            analysis['aggregate_rows'] = []
        if 'column_groups' not in analysis:
            analysis['column_groups'] = []
        if 'structure_type' not in analysis:
            analysis['structure_type'] = 'simple'
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5
        if 'recommendations' not in analysis:
            analysis['recommendations'] = []
        if 'potential_category_hierarchy' not in analysis:
            analysis['potential_category_hierarchy'] = []

        # Validate row indices
        analysis['header_rows'] = [idx for idx in analysis['header_rows'] if 0 <= idx < row_count]
        analysis['metadata_rows'] = [idx for idx in analysis['metadata_rows'] if 0 <= idx < row_count]
        analysis['aggregate_rows'] = [idx for idx in analysis['aggregate_rows'] if 0 <= idx < row_count]

        # Ensure confidence is in valid range
        analysis['confidence'] = max(0.0, min(1.0, float(analysis['confidence'])))

        return analysis

    def _get_default_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a default analysis if LLM fails."""
        df = encoded_data['dataframe']
        row_count = len(df)

        # Simple heuristic: assume first row is header, rest is data
        return {
            'header_rows': [0] if row_count > 0 else [],
            'data_rows': ['1-' + str(row_count-1)] if row_count > 1 else [],
            'metadata_rows': [],
            'aggregate_rows': [],
            'column_groups': [],
            'potential_category_hierarchy': [],
            'structure_type': 'simple',
            'confidence': 0.3,
            'recommendations': ['Default analysis used due to LLM error']
        }