"""
Schema Estimator Module (Enhanced)
Uses LLM semantic reasoning to derive the ideal tidy schema.

Key principle: Let LLM reason freely about what the tidy representation should be,
based on tidy data principles and semantic understanding of the data.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
import re
import pandas as pd

logger = logging.getLogger(__name__)


class SchemaEstimator:
    """
    Estimates the target tidy schema using LLM semantic reasoning.

    NO hardcoded templates - LLM derives schema based on:
    1. Structure analysis results
    2. Tidy data principles
    3. Semantic understanding of what the data represents
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the schema estimator."""
        self.config = config

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        # self.temperature = config.get('temperature', 0.1)
        self.max_completion_tokens = config.get('max_completion_tokens', 4000)

    def estimate_schema(self,
                        encoded_data: Dict[str, Any],
                        structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate the ideal tidy schema using LLM reasoning.

        Returns:
            Schema specification including:
            - target_columns: List of column definitions
            - observation_unit: What each row represents
            - expected_row_count: Formula and estimate
            - data_relationships: How source maps to target
        """
        logger.info("Estimating tidy schema with LLM semantic reasoning...")

        prompt = self._create_schema_prompt(encoded_data, structure_analysis)

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

            schema_result = self._parse_llm_response(result_text)
            schema_result = self._validate_schema(schema_result, encoded_data, structure_analysis)

            # Store source metadata for transformation stage
            schema_result['source_metadata'] = encoded_data.get('metadata', {})
            schema_result['structure_analysis'] = structure_analysis

            logger.info(f"Schema estimation complete. Target columns: {[c['name'] for c in schema_result.get('target_columns', [])]}")
            return schema_result

        except Exception as e:
            logger.error(f"Error in schema estimation: {e}")
            return self._get_default_schema(encoded_data, structure_analysis)

    def _get_system_prompt(self) -> str:
        """System prompt for schema estimation."""
        return """You are an expert data architect specializing in data normalization and tidy data principles.

## TIDY DATA PRINCIPLES (Hadley Wickham)
1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

## YOUR EXPERTISE
You excel at looking at messy spreadsheet structures and determining:
- What the IDEAL tidy representation should be
- What columns the tidy table should have
- What each row should represent (the observation unit)
- How to handle special cases (bilingual data, aggregates, multi-level headers)

## KEY INSIGHT
"Tidy datasets are all alike, but every messy dataset is messy in its own way."
Your job is to find the tidy structure hidden within the messy spreadsheet.

## GUIDELINES
1. Column headers that are VALUES should become a single VARIABLE column
   - Years (2011, 2016, 2021) → year column
   - Age groups (<15, 15-24, ...) → age_group column
   
2. If data has MULTIPLE value types (counts AND percentages), decide:
   - Should they be separate columns (count, percentage)?
   - Or separate rows with a value_type column?
   - Usually separate columns is cleaner if they describe the same observation

3. For bilingual data:
   - Keep both languages as separate columns (name_cn, name_en)
   - Don't lose information

4. For implicit aggregation:
   - Totals should typically be EXCLUDED (they can be recomputed)
   - Keep only the most granular detail rows

5. Name columns clearly using snake_case

Output valid JSON only."""

    def _create_schema_prompt(self,
                              encoded_data: Dict[str, Any],
                              structure_analysis: Dict[str, Any]) -> str:
        """Create prompt for schema estimation."""
        df = encoded_data['dataframe']
        metadata = encoded_data.get('metadata', {})

        # Get sample data
        sample_data = self._get_sample_data(df, structure_analysis)

        # Format structure analysis for prompt
        structure_summary = self._summarize_structure_analysis(structure_analysis)

        prompt = f"""Based on the structure analysis, design the ideal tidy schema for this spreadsheet.

## STRUCTURE ANALYSIS SUMMARY
{structure_summary}

## SAMPLE DATA FROM SPREADSHEET
{sample_data}

## ORIGINAL COLUMN NAMES
{df.columns.tolist()}

## METADATA
- Total rows: {len(df)}
- Total columns: {len(df.columns)}

## YOUR TASK

Design the TARGET tidy schema. Think through:

1. **What is the observation unit?** 
   - What does ONE ROW in the tidy output represent?
   - e.g., "One ethnicity's population count for one age group in one year"

2. **What columns should the tidy table have?**
   - ID/dimension columns (categorical)
   - Value columns (numeric measurements)
   - Consider: Do you need separate columns for count vs percentage? Or a value_type column?

3. **How should bilingual content be handled?**
   - Separate columns for each language?
   - Which columns have bilingual data?

4. **What should be EXCLUDED?**
   - Total/aggregate rows?
   - Total/aggregate columns?
   - Metadata columns that don't fit the observation unit?

5. **Row count estimation**
   - How many rows should the tidy output have?
   - Formula: num_X × num_Y × num_Z = total

## OUTPUT FORMAT (JSON)

{{
  "observation_unit": {{
    "description": "What one row represents in plain language",
    "dimensions": ["list of dimensions that identify one observation"],
    "example": "One Filipino person's age group distribution in 2011"
  }},
  
  "target_columns": [
    {{
      "name": "column_name_snake_case",
      "data_type": "string | integer | float",
      "description": "What this column represents",
      "source": "Where this data comes from in the original spreadsheet",
      "is_dimension": true,
      "nullable": false
    }},
    {{
      "name": "another_column",
      "data_type": "float",
      "description": "Description",
      "source": "Source description",
      "is_dimension": false,
      "nullable": true
    }}
  ],
  
  "expected_output": {{
    "row_count_formula": "num_ethnicities × num_age_groups × num_years",
    "row_count_estimate": 147,
    "column_count": 7
  }},
  
  "exclusions": {{
    "exclude_rows": {{
      "description": "What rows to exclude",
      "criteria": ["total rows", "aggregate rows"],
      "row_indices_if_known": []
    }},
    "exclude_columns": {{
      "description": "What columns to exclude",
      "criteria": ["Total column", "Index column"],
      "col_indices_if_known": [10, 12]
    }}
  }},
  
  "handling_special_cases": {{
    "bilingual_handling": {{
      "strategy": "separate_columns | single_column | primary_only",
      "details": "How bilingual content will be handled"
    }},
    "multi_value_handling": {{
      "has_multiple_value_types": true,
      "value_types": ["count", "percentage"],
      "strategy": "separate_columns | separate_rows",
      "details": "How different value types (counts vs percentages) will be handled"
    }},
    "implicit_aggregation_handling": {{
      "has_aggregation": false,
      "strategy": "exclude_summary_rows | keep_all",
      "details": "How implicit aggregation will be handled"
    }}
  }},
  
  "validation_samples": [
    {{
      "description": "Spot check for first ethnicity, first age group",
      "expected_row": {{
        "year": 2011,
        "ethnicity_cn": "亞洲人（非華人）",
        "age_group": "<15",
        "count": 23984,
        "percentage": 6.6
      }},
      "source_location": "CN row 8 col 3 for count, EN row 9 col 3 for percentage"
    }}
  ],
  
  "schema_reasoning": "Explain your reasoning for this schema design"
}}

Think carefully about the tidy data principles and output only the JSON."""

        return prompt

    def _get_sample_data(self, df, structure_analysis: Dict[str, Any]) -> str:
        """Get relevant sample data for schema estimation."""
        lines = []

        # Get data region from structure analysis
        data_region = structure_analysis.get('data_region', {})
        start_row = data_region.get('start_row', 0)

        # Show header rows
        header_rows = structure_analysis.get('header_structure', {}).get('header_rows', [0])
        lines.append("HEADER ROWS:")
        for row_idx in header_rows:
            if row_idx < len(df):
                row_vals = []
                for j in range(len(df.columns)):
                    val = df.iloc[row_idx, j]
                    if pd.notna(val) and str(val).strip():
                        row_vals.append(f"[{j}]={str(val)[:30]}")
                lines.append(f"  Row {row_idx}: {', '.join(row_vals)}")

        # Show sample data rows
        lines.append("\nDATA ROWS (first 10):")
        for i in range(start_row, min(start_row + 10, len(df))):
            row_vals = []
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    row_vals.append(f"[{j}]={str(val)[:30]}")
            lines.append(f"  Row {i}: {', '.join(row_vals)}")

        return "\n".join(lines)

    def _summarize_structure_analysis(self, structure_analysis: Dict[str, Any]) -> str:
        """Summarize structure analysis for the prompt."""
        lines = []

        # Semantic understanding
        sem = structure_analysis.get('semantic_understanding', {})
        lines.append(f"DATA DESCRIPTION: {sem.get('data_description', 'Unknown')}")
        lines.append(f"KEY VARIABLES: {sem.get('key_variables', [])}")

        # Header structure
        header = structure_analysis.get('header_structure', {})
        lines.append(f"\nHEADER ROWS: {header.get('header_rows', [])}")
        lines.append(f"HEADER LEVELS: {header.get('num_levels', 1)}")

        # Data region
        data_region = structure_analysis.get('data_region', {})
        lines.append(f"\nDATA REGION: rows {data_region.get('start_row')} to {data_region.get('end_row')}")

        # Row patterns
        row_patterns = structure_analysis.get('row_patterns', {})
        if row_patterns.get('has_bilingual_rows'):
            bilingual = row_patterns.get('bilingual_details', {})
            lines.append(f"\nBILINGUAL ROWS: {bilingual.get('pattern', 'unknown')}")
            lines.append(f"  Data relationship: {bilingual.get('data_relationship', 'unknown')}")
            if bilingual.get('data_relationship') == 'DIFFERENT_DATA_TYPES':
                diff = bilingual.get('if_different_types', {})
                lines.append(f"  CN rows contain: {diff.get('cn_row_contains', 'unknown')}")
                lines.append(f"  EN rows contain: {diff.get('en_row_contains', 'unknown')}")

        if row_patterns.get('has_section_markers'):
            markers = row_patterns.get('section_marker_details', {})
            lines.append(f"\nSECTION MARKERS: {markers.get('marker_values', [])} (semantic: {markers.get('semantic', 'unknown')})")

        # Column patterns
        col_patterns = structure_analysis.get('column_patterns', {})
        lines.append(f"\nID COLUMNS: {col_patterns.get('id_columns', [])}")
        lines.append(f"VALUE COLUMNS: {col_patterns.get('value_columns', [])}")
        lines.append(f"AGGREGATE COLUMNS (to exclude): {col_patterns.get('aggregate_columns', [])}")

        # Special patterns
        special = structure_analysis.get('special_patterns', [])
        if special:
            lines.append("\nSPECIAL PATTERNS:")
            for pattern in special:
                lines.append(f"  - {pattern.get('pattern_type', 'unknown')}: {pattern.get('description', '')}")

        # Implicit aggregation - use LLM's semantic analysis results
        implicit_agg = structure_analysis.get('implicit_aggregation', {})
        if implicit_agg.get('has_implicit_aggregation'):
            lines.append(f"\n{'='*60}")
            lines.append(f"IMPLICIT AGGREGATION DETECTED (by LLM semantic analysis)")
            lines.append(f"{'='*60}")

            details = implicit_agg.get('detection_details', {})
            lines.append(f"\nDETECTION DETAILS:")
            lines.append(f"  Category column: {details.get('category_column', 'unknown')}")
            lines.append(f"  Summary values: {details.get('summary_values', [])[:3]}")
            lines.append(f"  Detail values: {details.get('detail_values', [])[:3]}")
            lines.append(f"  Additional dimension: {details.get('additional_dimension', 'unknown')}")
            lines.append(f"  Delimiter: '{details.get('delimiter', 'unknown')}'")
            lines.append(f"  LLM reasoning: {details.get('reasoning', '')[:200]}")

            # Sample data from detail rows
            samples = implicit_agg.get('sample_detail_rows', [])
            if samples:
                lines.append(f"\nSAMPLE DETAIL ROWS:")
                for sample in samples[:3]:
                    lines.append(f"  Row {sample.get('row_index', '?')}: {sample.get('all_columns', {})}")

            # Transformation guidance from LLM
            guidance = implicit_agg.get('transformation_guidance', {})
            if guidance:
                lines.append(f"\nTRANSFORMATION GUIDANCE (from LLM):")
                lines.append(f"  Rows to exclude: {guidance.get('rows_to_exclude', 'N/A')}")
                lines.append(f"  Column to split: {guidance.get('column_to_split', 'N/A')}")
                lines.append(f"  Expected new columns: {guidance.get('expected_new_columns', [])}")

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

    def _validate_schema(self,
                         schema: Dict[str, Any],
                         encoded_data: Dict[str, Any],
                         structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Validate schema and ensure all required fields exist."""

        # Ensure required fields
        if 'observation_unit' not in schema:
            schema['observation_unit'] = {
                'description': 'Unknown',
                'dimensions': [],
                'example': ''
            }

        if 'target_columns' not in schema:
            schema['target_columns'] = []

        if 'expected_output' not in schema:
            schema['expected_output'] = {
                'row_count_formula': 'unknown',
                'row_count_estimate': 0,
                'column_count': len(schema.get('target_columns', []))
            }

        if 'exclusions' not in schema:
            schema['exclusions'] = {
                'exclude_rows': {'description': '', 'criteria': []},
                'exclude_columns': {'description': '', 'criteria': []}
            }

        if 'handling_special_cases' not in schema:
            schema['handling_special_cases'] = {}

        if 'validation_samples' not in schema:
            schema['validation_samples'] = []

        # Generate expected_output_columns list for backward compatibility
        schema['expected_output_columns'] = [col['name'] for col in schema.get('target_columns', [])]

        return schema

    def _get_default_schema(self,
                            encoded_data: Dict[str, Any],
                            structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return default schema if LLM fails."""
        df = encoded_data['dataframe']

        # Create basic column schema from dataframe
        target_columns = []
        for i, col in enumerate(df.columns):
            target_columns.append({
                'name': f'column_{i}',
                'data_type': 'string',
                'description': f'Column {i}: {str(col)[:50]}',
                'source': f'Column {i}',
                'is_dimension': i == 0,
                'nullable': True
            })

        return {
            'observation_unit': {
                'description': 'One row from source',
                'dimensions': [],
                'example': ''
            },
            'target_columns': target_columns,
            'expected_output': {
                'row_count_formula': 'same as source',
                'row_count_estimate': len(df),
                'column_count': len(df.columns)
            },
            'exclusions': {
                'exclude_rows': {'description': 'None', 'criteria': []},
                'exclude_columns': {'description': 'None', 'criteria': []}
            },
            'handling_special_cases': {},
            'validation_samples': [],
            'schema_reasoning': 'Default schema due to LLM error - pass through',
            'expected_output_columns': [f'column_{i}' for i in range(len(df.columns))],
            'source_metadata': encoded_data.get('metadata', {}),
            'structure_analysis': structure_analysis
        }
