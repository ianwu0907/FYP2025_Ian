"""
Transformation Generator Module
Generates and executes transformation code to normalize tables
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
import pandas as pd
import re

logger = logging.getLogger(__name__)


class TransformationGenerator:
    """
    Generates executable Python code to transform tables according to schema.
    Executes the code and validates the output.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the transformation generator with configuration."""
        self.config = config
        self.max_retries = config.get('max_retries', 3)
        self.validate_output = config.get('validate_output', True)
        self.preserve_data = config.get('preserve_data', True)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4000)

    def generate_and_execute(self,
                             encoded_data: Dict[str, Any],
                             structure_analysis: Dict[str, Any],
                             schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transformation code and execute it to produce normalized table."""
        logger.info("Generating transformation code with LLM...")

        df_original = encoded_data['dataframe']

        for attempt in range(self.max_retries):
            try:
                logger.info(f"Transformation attempt {attempt + 1}/{self.max_retries}")

                # Generate transformation code
                code = self._generate_code(encoded_data, structure_analysis, schema, attempt)

                # Execute the code
                normalized_df = self._execute_transformation(code, df_original)

                # Validate the result
                if self.validate_output:
                    validation = self._validate_output(df_original, normalized_df, schema)
                    if not validation['is_valid']:
                        logger.warning(f"Validation failed: {validation['errors']}")
                        if attempt < self.max_retries - 1:
                            continue
                else:
                    validation = {'is_valid': True, 'errors': [], 'warnings': []}

                # Success!
                result = {
                    'normalized_df': normalized_df,
                    'transformation_code': code,
                    'execution_log': f'Successfully transformed on attempt {attempt + 1}',
                    'validation_result': validation
                }

                logger.info("Transformation completed successfully")
                return result

            except Exception as e:
                logger.error(f"Transformation attempt {attempt + 1} failed: {e}")
                if attempt == self.max_retries - 1:
                    logger.error("All transformation attempts failed, using fallback")
                    return self._fallback_transformation(df_original, schema)

        return self._fallback_transformation(df_original, schema)

    def _generate_code(self,
                       encoded_data: Dict[str, Any],
                       structure_analysis: Dict[str, Any],
                       schema: Dict[str, Any],
                       attempt: int) -> str:
        """Generate Python transformation code using LLM."""
        prompt = self._create_code_generation_prompt(
            encoded_data, structure_analysis, schema, attempt
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            code_text = response.choices[0].message.content
            logger.debug(f"Generated code: {code_text[:500]}...")

            # Extract Python code from response
            code = self._extract_code(code_text)

            return code

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def _get_system_prompt(self) -> str:
        """Get the system prompt for code generation."""
        return """You are an expert Python programmer specializing in pandas data transformations.

Your task is to write clean, efficient Python code that transforms messy spreadsheet data into normalized tables.

**Requirements**:
1. Use pandas DataFrame operations
2. Handle edge cases (missing values, type conversions, etc.)
3. Follow the specified schema exactly
4. Preserve all data (no information loss)
5. Write defensive code with error handling
6. Include comments explaining key steps

**Code Structure**:
- Assume input DataFrame is called `df`
- Output must be a DataFrame called `result_df`
- Use standard pandas operations
- No external libraries except pandas and numpy

**Output Format**:
- Provide ONLY executable Python code
- No markdown, no explanations outside comments
- The code should be ready to run with exec()"""

    def _create_code_generation_prompt(self,
                                       encoded_data: Dict[str, Any],
                                       structure_analysis: Dict[str, Any],
                                       schema: Dict[str, Any],
                                       attempt: int) -> str:
        """Create the code generation prompt."""

        metadata = encoded_data['metadata']

        prompt = f"""Generate Python code to transform this spreadsheet data into a normalized format.

## Input Data Information:
- Shape: {metadata['num_rows']} rows × {metadata['num_cols']} columns
- Columns: {metadata['column_names']}
- Sample values: {json.dumps(metadata.get('sample_values', {}), indent=2)[:500]}

## Structure Analysis:
- Structure type: {structure_analysis.get('structure_type', 'unknown')}
- Header rows: {structure_analysis.get('header_rows', [])}
- Data rows: {structure_analysis.get('data_rows', 'unknown')}
- Metadata rows: {structure_analysis.get('metadata_rows', [])}
- Aggregate rows: {structure_analysis.get('aggregate_rows', [])}

## Target Schema:
{json.dumps(schema.get('column_schema', []), indent=2)[:1000]}

## Expected Output Columns:
{schema.get('expected_output_columns', [])}

## Normalization Plan:
{json.dumps(schema.get('normalization_plan', []), indent=2)}

## Your Task:
Write Python code that:

1. **Filters rows appropriately**:
   - Keep only data rows (exclude headers, metadata, aggregates)
   - Header rows: {structure_analysis.get('header_rows', [])}

2. **Transforms columns**:
   - Rename columns according to schema
   - Convert data types as specified
   - Handle missing/null values appropriately

3. **Applies transformations**:
   - Follow the normalization plan
   - Flatten hierarchical structures if needed
   - Merge or split columns as specified

4. **Validates output**:
   - Ensure all expected columns are present
   - Check for data integrity
   - Handle edge cases

## Code Template:
```python
import pandas as pd
import numpy as np

def transform(df):
    # Step 1: Filter to data rows only
    # TODO: Implement
    
    # Step 2: Select and rename columns
    # TODO: Implement
    
    # Step 3: Convert data types
    # TODO: Implement
    
    # Step 4: Clean and validate
    # TODO: Implement
    
    return result_df

# Execute transformation
result_df = transform(df)
```

## Important Notes:
- The input DataFrame is available as `df`
- Your output MUST be assigned to `result_df`
- Handle all edge cases
- Preserve data integrity
- Use try-except blocks for error-prone operations

{f'## Note: This is attempt {attempt + 1}. Previous attempts failed, please review and fix issues.' if attempt > 0 else ''}

Output ONLY the Python code, no explanations, no markdown."""

        return prompt

    def _extract_code(self, response_text: str) -> str:
        """Extract Python code from LLM response."""
        response_text = response_text.strip()

        # Remove markdown code blocks if present
        if '```python' in response_text:
            pattern = r'```python\s*(.*?)\s*```'
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                return matches[0].strip()
        elif '```' in response_text:
            pattern = r'```\s*(.*?)\s*```'
            matches = re.findall(pattern, response_text, re.DOTALL)
            if matches:
                return matches[0].strip()

        return response_text.strip()

    def _execute_transformation(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the transformation code safely."""
        logger.info("Executing transformation code...")

        # Prepare execution environment
        exec_globals = {
            'pd': pd,
            'np': __import__('numpy'),
            'df': df.copy(),
            'result_df': None
        }

        try:
            # Execute the code
            exec(code, exec_globals)

            # Get the result
            result_df = exec_globals.get('result_df')

            if result_df is None:
                raise ValueError("Transformation code did not produce 'result_df'")

            if not isinstance(result_df, pd.DataFrame):
                raise ValueError(f"result_df is not a DataFrame, got {type(result_df)}")

            logger.info(f"Transformation successful. Output shape: {result_df.shape}")
            return result_df

        except Exception as e:
            logger.error(f"Error executing transformation: {e}")
            raise

    def _validate_output(self,
                         df_original: pd.DataFrame,
                         df_normalized: pd.DataFrame,
                         schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the transformed output."""
        errors = []
        warnings = []

        # Check 1: Output is not empty
        if len(df_normalized) == 0:
            errors.append("Output DataFrame is empty")

        # Check 2: Has expected columns
        expected_cols = schema.get('expected_output_columns', [])
        actual_cols = df_normalized.columns.tolist()

        missing_cols = set(expected_cols) - set(actual_cols)
        if missing_cols:
            warnings.append(f"Missing expected columns: {missing_cols}")

        extra_cols = set(actual_cols) - set(expected_cols)
        if extra_cols:
            warnings.append(f"Extra columns in output: {extra_cols}")

        # Check 3: No excessive data loss
        if self.preserve_data:
            original_rows = len(df_original)
            output_rows = len(df_normalized)

            if output_rows < original_rows * 0.5:
                warnings.append(
                    f"Significant row reduction: {original_rows} → {output_rows}"
                )

        # Check 4: No all-null columns
        null_cols = df_normalized.columns[df_normalized.isnull().all()].tolist()
        if null_cols:
            warnings.append(f"Columns with all null values: {null_cols}")

        is_valid = len(errors) == 0

        return {
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings
        }

    def _fallback_transformation(self,
                                 df_original: pd.DataFrame,
                                 schema: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback transformation if LLM-generated code fails."""
        logger.warning("Using fallback transformation")

        df = df_original.copy()

        # Get column mapping from schema
        column_mapping = {}
        for col_def in schema.get('column_schema', []):
            if 'original_column' in col_def and 'normalized_name' in col_def:
                column_mapping[col_def['original_column']] = col_def['normalized_name']

        # Rename columns that exist
        existing_cols = {col: column_mapping.get(col, col) for col in df.columns if col in column_mapping}
        df = df.rename(columns=existing_cols)

        # Remove rows that are all null
        df = df.dropna(how='all')

        return {
            'normalized_df': df,
            'transformation_code': '# Fallback transformation\nresult_df = df.copy()',
            'execution_log': 'Used fallback transformation due to errors',
            'validation_result': {
                'is_valid': False,
                'errors': ['Used fallback transformation'],
                'warnings': ['Manual review recommended']
            }
        }