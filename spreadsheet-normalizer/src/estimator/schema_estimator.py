"""
Schema Estimator Module
Uses LLM to propose standardized schemas for normalized tables
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
    Estimates and proposes standardized schemas for table normalization.
    Handles column naming, type inference, and semantic grouping.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the schema estimator with configuration."""
        self.config = config
        self.standardize_names = config.get('standardize_names', True)
        self.detect_types = config.get('detect_types', True)
        self.merge_similar_columns = config.get('merge_similar_columns', True)

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
        """Estimate the standardized schema for the normalized table."""
        logger.info("Estimating normalized schema with LLM...")

        # Create schema estimation prompt
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

Your task is to design a clean, normalized schema for messy spreadsheet data. The goal is to transform human-oriented tables into machine-readable, analysis-ready formats following these principles:

1. **Tidy Data Principles**:
   - Each variable forms a column
   - Each observation forms a row
   - Each type of observational unit forms a table

2. **Column Naming**:
   - Use clear, descriptive, snake_case names
   - Avoid special characters and spaces
   - Be consistent across similar columns
   - Prefer English names for standardization

3. **Data Types**:
   - Correctly identify: string, integer, float, date, boolean, categorical
   - Consider the semantic meaning, not just the format

4. **Normalization**:
   - Flatten multi-level headers into single-level columns
   - Separate embedded aggregates from observations
   - Remove or relocate metadata

Be precise, practical, and output valid JSON."""

    def _create_schema_prompt(self,
                              encoded_data: Dict[str, Any],
                              structure_analysis: Dict[str, Any]) -> str:
        """Create the schema estimation prompt for the LLM."""

        metadata = encoded_data['metadata']
        encoded_text = encoded_data['encoded_text'][:2000]

        prompt = f"""Based on the table structure analysis, design a normalized schema.

                ## Current Structure:
                - Original columns: {metadata['column_names']}
                - Structure type: {structure_analysis.get('structure_type', 'unknown')}
                - Header rows: {structure_analysis.get('header_rows', [])}
                - Column groups: {structure_analysis.get('column_groups', [])}
                
                ## Sample Data:
                {encoded_text}
                
                ## Structure Analysis:
                {json.dumps(structure_analysis.get('recommendations', []), indent=2)}
                
                ## Your Task:
                Design a normalized schema that:
                
                1. **Flattens multi-level headers** (if any):
                   - Combine hierarchical headers into single descriptive column names
                   - Example: "Type of Abuse" + "Male" → "abuse_type_male"
                
                2. **Standardizes column names**:
                   - Use snake_case
                   - Remove special characters
                   - Use English names primarily
                   - Be descriptive but concise
                
                3. **Identifies data types**:
                   - string, integer, float, date, boolean, categorical
                   - Consider semantic meaning
                
                4. **Handles duplicate information**:
                   - If same information in multiple languages, decide which to keep
                   - Merge redundant columns if appropriate
                
                5. **Plans for metadata**:
                   - Should metadata be in separate columns or removed?
                   - How to preserve important contextual information?
                
                ## Output Format:
                Provide a JSON object with this structure:
                {{
                    "column_schema": [
                        {{
                            "original_column": "name in original table",
                            "normalized_name": "standardized_name",
                            "data_type": "string|integer|float|date|boolean|categorical",
                            "description": "what this column represents",
                            "transformation": "how to derive this from original"
                        }}
                    ],
                    "normalization_plan": [
                        "step 1: ...",
                        "step 2: ...",
                        "step 3: ..."
                    ],
                    "metadata_handling": {{
                        "strategy": "remove|separate_columns|keep_as_is",
                        "metadata_columns": ["list of metadata to handle"],
                        "rationale": "why this approach"
                    }},
                    "expected_output_columns": ["final list of column names in normalized table"],
                    "reasoning": "brief explanation of design decisions"
                }}
                
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
            import re
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
        """Validate and enrich the schema result."""
        # Ensure required fields exist
        if 'column_schema' not in schema:
            schema['column_schema'] = []

        if 'normalization_plan' not in schema:
            schema['normalization_plan'] = ['Apply default normalization']

        if 'expected_output_columns' not in schema:
            schema['expected_output_columns'] = [
                col['normalized_name'] for col in schema.get('column_schema', [])
                if 'normalized_name' in col
            ]

        # Validate column names
        for col_def in schema.get('column_schema', []):
            if 'normalized_name' in col_def:
                name = col_def['normalized_name']
                name = re.sub(r'[^a-zA-Z0-9_]', '_', name)
                name = re.sub(r'^[0-9]', '_', name)
                name = re.sub(r'_+', '_', name)
                name = name.strip('_')
                col_def['normalized_name'] = name.lower()

        return schema

    def _get_default_schema(self,
                            encoded_data: Dict[str, Any],
                            structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return a default schema if LLM fails."""
        original_columns = encoded_data['metadata']['column_names']

        # Create simple column schema
        column_schema = []
        for idx, col in enumerate(original_columns):
            col_name = str(col)
            normalized = re.sub(r'[^a-zA-Z0-9_]', '_', col_name)
            normalized = re.sub(r'_+', '_', normalized).strip('_').lower()

            column_schema.append({
                'original_column': col_name,
                'normalized_name': normalized if normalized else f'column_{idx}',
                'data_type': 'string',
                'description': f'Column {idx}',
                'transformation': 'direct_copy'
            })

        return {
            'column_schema': column_schema,
            'normalization_plan': [
                'Keep all columns as-is',
                'Standardize column names to snake_case',
                'Remove special characters'
            ],
            'metadata_handling': {
                'strategy': 'keep_as_is',
                'metadata_columns': [],
                'rationale': 'Default fallback strategy'
            },
            'expected_output_columns': [col['normalized_name'] for col in column_schema],
            'reasoning': 'Default schema due to LLM error'
        }