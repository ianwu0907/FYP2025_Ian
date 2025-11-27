"""
Schema Estimator Module (FINAL VERSION)
Uses complete metadata from encoder - NO redundant analysis
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
import re

logger = logging.getLogger(__name__)


class SchemaEstimator:
    """
    Estimates schemas using COMPLETE metadata from encoder.

    CRITICAL PRINCIPLE:
    This module does NOT analyze the dataframe directly.
    It relies entirely on encoded_data['metadata']['columns'] as the single source of truth.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the schema estimator with configuration."""
        self.config = config
        self.standardize_names = config.get('standardize_names', True)
        self.detect_types = config.get('detect_types', True)
        self.merge_similar_columns = config.get('merge_similar_columns', False)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4000)

    def estimate_schema(self,
                        encoded_data: Dict[str, Any],
                        structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate the standardized schema using metadata from encoder.

        IMPORTANT: All data distribution info comes from encoded_data['metadata']['columns']
        """
        logger.info("Estimating normalized schema with LLM...")

        # VERIFY: Check if we have enhanced metadata
        metadata = encoded_data.get('metadata', {})
        if 'columns' not in metadata:
            logger.error("Missing 'columns' in metadata! Encoder may not be enhanced.")
            raise ValueError(
                "encoded_data['metadata'] must contain 'columns' dict. "
                "Make sure you're using the enhanced encoder."
            )

        # Create schema estimation prompt with complete metadata
        prompt = self._create_schema_prompt(encoded_data, structure_analysis)

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
            schema_result = self._parse_llm_response(result_text)

            # Validate the schema
            schema_result = self._validate_schema(schema_result, encoded_data, structure_analysis)

            logger.info("Schema estimation complete")
            return schema_result

        except Exception as e:
            logger.error(f"Error in schema estimation: {e}")
            # Return default schema if LLM fails
            return self._get_default_schema(encoded_data, structure_analysis)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for schema estimation."""
        return """You are an expert data engineer specializing in database schema design and data normalization.

Your task is to design clean, normalized schemas for messy spreadsheet data by recognizing common table structure patterns and applying appropriate transformations.

**CRITICAL RULES:**
1. You will receive COMPLETE data distribution information including ALL unique values and frequencies
2. For boolean/categorical columns, you MUST use the ACTUAL values from the metadata, not guess
3. Generate transformation logic that handles ALL observed values, including edge cases like N/A
4. The metadata provided is the ONLY source of truth - do not assume any values not shown
5. Be precise, practical, and output valid JSON.
"""

    def _create_schema_prompt(self,
                              encoded_data: Dict[str, Any],
                              structure_analysis: Dict[str, Any]) -> str:
        """
        Create enhanced prompt using COMPLETE metadata from encoder.
        NO dataframe analysis here - all info from metadata.
        """
        metadata = encoded_data['metadata']
        columns_metadata = metadata.get('columns', {})

        # Build detailed column analysis from metadata
        column_analysis_text = self._build_column_analysis_from_metadata(columns_metadata)

        # Get shape info
        num_rows = metadata.get('num_rows', 0)
        column_names = metadata.get('column_names', [])

        prompt = f"""Design a normalized schema for this messy spreadsheet.

## CURRENT STRUCTURE:
- Total rows: {num_rows}
- Columns: {column_names}
- Structure type: {structure_analysis.get('structure_type', 'unknown')}

## COMPLETE COLUMN ANALYSIS (from encoding stage):
{column_analysis_text}

## CRITICAL INSTRUCTIONS FOR TYPE CONVERSION:

### For BOOLEAN columns:
You MUST examine the "Unique values" and "Value counts" above.
**DO NOT** guess values

### For CATEGORICAL columns:
- List ALL unique values shown in metadata
- Provide mapping if standardization is needed
- Note frequency distribution

### For COMPOSITE columns (with delimiters):
- Check "Potential delimiters" in metadata
- Verify split pattern with sample_split
- Create split_operations entries

## TABLE PATTERNS TO DETECT:

### Pattern 1: IMPLICIT AGGREGATION
Check structure_analysis for summary/detail rows:
{json.dumps(structure_analysis.get('implicit_aggregation', {}), indent=2)}

### Pattern 2: COMPOSITE COLUMNS  
Check each column's "potential_delimiters" in metadata above.
If delimiter percentage > 50%, likely needs splitting.

### Pattern 3: BILINGUAL COLUMNS
Check "has_bilingual_content" flag in metadata.
Keep both languages where appropriate.

## OUTPUT FORMAT:

{{
    "identified_patterns": [
        {{
            "pattern_name": "BOOLEAN_WITH_NA | COMPOSITE_COLUMNS | BILINGUAL | etc",
            "affected_columns": ["list columns"],
            "details": "specific characteristics from metadata",
            "solution": "transformation approach"
        }}
    ],
    "column_schema": [
        {{
            "original_column": "exact column name",
            "normalized_name": "snake_case_name",
            "data_type": "boolean_with_na | boolean | categorical | integer | numeric | string | date | identifier",
            "unique_values": ["copy", "from", "metadata"],
            "transformation_logic": "detailed description using ACTUAL metadata values",
            "operation": "keep | split | merge | drop"
        }}
    ],
    "split_operations": [
        {{
            "source_column": "column name",
            "split_pattern": "exact delimiter from metadata",
            "target_columns": ["new_col1", "new_col2"],
            "logic": "detailed split logic",
            "metadata_evidence": "cite frequency and sample_split from metadata"
        }}
    ],
    "type_conversions": [
        {{
            "column": "exact column name",
            "target_type": "boolean_with_na | categorical | etc",
            "conversion_logic": "DETAILED logic using EXACT values from metadata",
            "edge_cases": ["list edge cases from unique_values"],
            "value_mapping": {{
                "actual_value_1": mapped_value_1,
                "actual_value_2": mapped_value_2
            }}
        }}
    ],
    "normalization_plan": [
        "Step 1: Remove implicit aggregation rows if detected",
        "Step 2: Split composite columns with evidence from metadata",
        "Step 3: Convert types using ACTUAL unique_values from metadata",
        "..."
    ],
    "expected_output_columns": ["complete", "list", "of", "final", "columns"],
    "reasoning": "Explain decisions citing specific metadata (unique_values, value_counts, delimiters)"
}}

**MANDATORY VERIFICATION:**
✓ Have you used ACTUAL unique_values from metadata (not guessed)?
✓ For boolean columns, does value_mapping use metadata's exact values?
✓ For splits, did you cite the delimiter frequency from metadata?
✓ Does your reasoning reference specific metadata fields?

Output ONLY the JSON object.
"""

        return prompt

    def _build_column_analysis_from_metadata(self, columns_metadata: Dict[str, Any]) -> str:
        """
        Build column analysis text ENTIRELY from metadata.
        NO dataframe access here.
        """
        lines = []

        for col_name, col_meta in columns_metadata.items():
            lines.append(f"\n### Column: `{col_name}`")
            lines.append(f"- Data type: {col_meta.get('dtype', 'unknown')}")
            lines.append(f"- **Inferred type: {col_meta.get('inferred_type', 'unknown')}**")
            lines.append(f"- Unique count: {col_meta.get('unique_count', 0)}")
            lines.append(f"- Null percentage: {col_meta.get('null_percentage', 0):.2f}%")

            # Show unique values
            unique_values = col_meta.get('unique_values', [])
            if unique_values:
                if col_meta.get('unique_values_truncated'):
                    lines.append(f"- **Unique values (first 100):** {unique_values}")
                else:
                    lines.append(f"- **Unique values (complete):** {unique_values}")

            # Show value counts
            value_counts = col_meta.get('value_counts', {})
            if value_counts:
                lines.append(f"- **Value counts:**")
                for val, count in list(value_counts.items())[:10]:
                    lines.append(f"  - `{val}`: {count}")
                if len(value_counts) > 10:
                    lines.append(f"  - ... and {len(value_counts)-10} more")

            # Show sample values
            sample_values = col_meta.get('sample_values', [])
            if sample_values:
                lines.append(f"- Sample values: {sample_values[:5]}")

            # Show potential delimiters
            delimiters = col_meta.get('potential_delimiters', [])
            if delimiters:
                lines.append(f"- **Potential delimiters:**")
                for delim_info in delimiters[:3]:
                    lines.append(
                        f"  - `{delim_info['delimiter']}`: "
                        f"{delim_info['percentage']:.0f}% frequency, "
                        f"{'consistent' if delim_info.get('is_consistent') else 'inconsistent'} splits"
                    )
                    if delim_info.get('sample_split'):
                        lines.append(f"    Sample split: {delim_info['sample_split']}")

            # Show bilingual flag
            if col_meta.get('has_bilingual_content'):
                lines.append(f"- **Contains bilingual content** (Chinese + English)")

            # Show statistics for numeric columns
            stats = col_meta.get('statistics', {})
            if stats:
                lines.append(f"- Statistics: min={stats.get('min')}, max={stats.get('max')}, mean={stats.get('mean'):.2f}")

        return '\n'.join(lines)

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

    def _validate_schema(self,
                         schema: Dict[str, Any],
                         encoded_data: Dict[str, Any],
                         structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate schema and add metadata reference for downstream use.
        """
        # Ensure required fields exist
        if 'column_schema' not in schema:
            schema['column_schema'] = []

        if 'type_conversions' not in schema:
            schema['type_conversions'] = []

        if 'normalization_plan' not in schema:
            schema['normalization_plan'] = ['Apply default normalization']

        if 'expected_output_columns' not in schema:
            schema['expected_output_columns'] = [
                col['normalized_name'] for col in schema.get('column_schema', [])
                if 'normalized_name' in col
            ]

        if 'split_operations' not in schema:
            schema['split_operations'] = []

        if 'identified_patterns' not in schema:
            schema['identified_patterns'] = []

        # CRITICAL: Add reference to original metadata
        # This ensures downstream components can access complete metadata
        schema['source_metadata'] = encoded_data.get('metadata', {})

        # Validate column names
        for col_def in schema.get('column_schema', []):
            if 'normalized_name' in col_def:
                name = col_def['normalized_name']
                # Keep bilingual format like "年份/Year"
                if '/' not in name:
                    name = re.sub(r'[^a-zA-Z0-9_/]', '_', name)
                    name = re.sub(r'^[0-9]', '_', name)
                    name = re.sub(r'_+', '_', name)
                    name = name.strip('_')
                    col_def['normalized_name'] = name.lower()

        return schema

    def _get_default_schema(self,
                            encoded_data: Dict[str, Any],
                            structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return a default schema if LLM fails."""
        metadata = encoded_data.get('metadata', {})
        original_columns = metadata.get('column_names', [])

        # Create simple column schema
        column_schema = []
        for idx, col in enumerate(original_columns):
            col_name = str(col)
            normalized = re.sub(r'[^a-zA-Z0-9_/]', '_', col_name)
            normalized = re.sub(r'_+', '_', normalized).strip('_').lower()

            column_schema.append({
                'original_column': col_name,
                'operation': 'keep',
                'normalized_name': normalized if normalized else f'column_{idx}',
                'data_type': 'string',
                'description': f'Column {idx}',
                'transformation': 'direct_copy'
            })

        return {
            'identified_patterns': [],
            'column_schema': column_schema,
            'split_operations': [],
            'type_conversions': [],
            'normalization_plan': [
                'Keep all columns as-is',
                'Standardize column names to snake_case',
                'Remove special characters'
            ],
            'expected_output_columns': [col['normalized_name'] for col in column_schema],
            'reasoning': 'Default schema due to LLM error',
            'source_metadata': metadata
        }