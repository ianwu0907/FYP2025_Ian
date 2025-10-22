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

Your task is to design clean, normalized schemas for messy spreadsheet data by recognizing common table structure patterns and applying appropriate transformations.

Be precise, practical, and output valid JSON."""

    def _create_schema_prompt(self,
                              encoded_data: Dict[str, Any],
                              structure_analysis: Dict[str, Any]) -> str:
        """Create the schema estimation prompt for the LLM."""

        metadata = encoded_data['metadata']
        df = encoded_data['dataframe']

        # 获取样本数据
        sample_data = df.head(5).to_string() if len(df) > 0 else "No data"

        # 分析每一列的样本值，查找潜在的分隔符
        column_analysis = []
        for col in df.columns:
            non_null_values = df[col].dropna().astype(str).head(10).tolist()
            if non_null_values:
                # 检测分隔符
                delimiters_found = []
                for delimiter in [' - ', '-', ' / ', '/', '(', ')', ':', ',', '|']:
                    if any(delimiter in str(val) for val in non_null_values):
                        count = sum(1 for val in non_null_values if delimiter in str(val))
                        if count >= len(non_null_values) * 0.5:  # 至少50%的值包含这个分隔符
                            delimiters_found.append(f"{delimiter} (in {count}/{len(non_null_values)} values)")

                if delimiters_found:
                    column_analysis.append({
                        'column': col,
                        'potential_delimiters': delimiters_found,
                        'sample_values': non_null_values[:3]
                    })

        delimiter_analysis = ""
        if column_analysis:
            delimiter_analysis = "\n## AUTOMATIC DELIMITER DETECTION:\n"
            delimiter_analysis += "I detected these columns may need splitting:\n\n"
            for item in column_analysis:
                delimiter_analysis += f"Column: {item['column']}\n"
                delimiter_analysis += f"  Delimiters found: {item['potential_delimiters']}\n"
                delimiter_analysis += f"  Sample values: {item['sample_values']}\n\n"
            delimiter_analysis += "**ACTION REQUIRED**: For each column above, decide if it should be split.\n\n"

        prompt = f"""Design a normalized schema for this messy spreadsheet.
    
    {delimiter_analysis}
    
    ## Current Structure:
    - Columns: {metadata['column_names']}
    - Structure type: {structure_analysis.get('structure_type', 'unknown')}
    - Sample data:
    {sample_data}
    
    ## CRITICAL: COMPOSITE COLUMN DETECTION
    
    **STEP 1: Check EVERY column for composite values**
    
    For EACH column, examine the sample values above and ask:
    1. Do values contain consistent delimiters? (Look for: " - ", "-", "/", "()", ":", etc.)
    2. Do values have a pattern like "A - B", "A/B", "A (B)"?
    3. Does it make semantic sense to split this into multiple columns?
    
    **If YES to above**: This column MUST be split.
    
    **Example from your data**:
    If you see values like:
    - "Physical abuse - Male"
    - "Financial abuse - Female"
    - "Neglect - Male"
    
    This is a COMPOSITE COLUMN that MUST be split into:
    - Part 1: "Physical abuse", "Financial abuse", "Neglect" (abuse type)
    - Part 2: "Male", "Female", "Male" (gender)
    
    ## COMMON TABLE PROBLEMS & SOLUTIONS:
    
    ### Pattern 1: COMPOSITE COLUMNS (HIGHEST PRIORITY)
    **Problem**: One column contains multiple pieces of information
    **Detection**: 
    - Values contain delimiters: " - ", "-", "/", "()", ":", ",", "|"
    - Consistent pattern across rows
    - Semantic meaning suggests hierarchy
    
    **Examples**:
    - "Physical abuse - Male" → Type="Physical abuse", Gender="Male"  
    - "2023-Q1" → Year="2023", Quarter="Q1"
    
    **Solution**: MUST create split_operations entry
    ```
    {{
        "source_column": "item",
        "split_pattern": " - ",
        "target_columns": ["item1_type", "item2_gender", "item1_type_en", "item2_gender_en"],
        "logic": "Split on ' - ' to separate abuse type from gender, handle bilingual"
    }}
    ```
    
    ### Pattern 2: BILINGUAL COLUMNS
    Keep both languages. For column names, use format "Chinese/English".
    
    ### Pattern 3-6: [其他模式保持不变...]
    
    ## YOUR MANDATORY CHECKLIST:
    
    Before designing schema, YOU MUST:
    1. ✓ Check column "{df.columns[5] if len(df.columns) > 5 else 'item'}" - does it contain " - " delimiter?
    2. ✓ Check column "{df.columns[6] if len(df.columns) > 6 else 'Item'}" - does it contain " - " delimiter?
    3. ✓ For ANY column with consistent delimiters, ADD to split_operations
    4. ✓ Count: How many target columns will result from splits?
    5. ✓ Verify: expected_output_columns count matches your split plan
    
    ## Output Format:
    
    {{
        "identified_patterns": [
            {{
                "pattern_name": "COMPOSITE_COLUMNS",
                "affected_columns": ["list ALL columns that need splitting"],
                "details": "what delimiter, what pattern",
                "solution": "how to split"
            }}
        ],
        "column_schema": [...],
        "split_operations": [
            // YOU MUST FILL THIS if any composite columns detected
            {{
                "source_column": "name of column to split",
                "split_pattern": "exact delimiter",
                "target_columns": ["new_col1", "new_col2", "new_col3", "new_col4"],
                "logic": "detailed split logic including bilingual handling"
            }}
        ],
        "normalization_plan": [
            "Step 1: Split composite columns",
            "Step 2: ...",
            "..."
        ],
        "expected_output_columns": ["complete", "list", "of", "final", "columns"],
        "reasoning": "Explain your decisions, especially split decisions"
    }}
    
    **CRITICAL**: If you detect composite columns but don't add split_operations, the output will be WRONG.
    
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

        if 'split_operations' not in schema:
            schema['split_operations'] = []

        if 'identified_patterns' not in schema:
            schema['identified_patterns'] = []

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
        original_columns = encoded_data['metadata']['column_names']

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
            'normalization_plan': [
                'Keep all columns as-is',
                'Standardize column names to snake_case',
                'Remove special characters'
            ],
            'expected_output_columns': [col['normalized_name'] for col in column_schema],
            'reasoning': 'Default schema due to LLM error'
        }