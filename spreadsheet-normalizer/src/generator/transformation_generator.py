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

Your task is to write clean, efficient Python code that transforms messy spreadsheet data into normalized tables using pattern-based approaches.

Write defensive code with proper error handling. Output ONLY executable Python code."""

    def _create_code_generation_prompt(self,
                                       encoded_data: Dict[str, Any],
                                       structure_analysis: Dict[str, Any],
                                       schema: Dict[str, Any],
                                       attempt: int) -> str:
        """Create the code generation prompt."""

        df = encoded_data['dataframe']
        sample_data = df.head(3).to_string()

        # Extract information from schema
        column_schema = schema.get('column_schema', [])
        split_operations = schema.get('split_operations', [])
        expected_cols = schema.get('expected_output_columns', [])
        patterns = schema.get('identified_patterns', [])

        # Get implicit aggregation information
        implicit_agg = structure_analysis.get('implicit_aggregation', {})
        has_implicit_agg = implicit_agg.get('has_implicit_aggregation', False)

        # Build aggregation handling instructions (generic)
        aggregation_handling = ""
        if has_implicit_agg:
            summary_rows = implicit_agg.get('summary_rows', [])
            detail_rows = implicit_agg.get('detail_rows', [])
            hierarchies = implicit_agg.get('aggregation_hierarchies', [])

            aggregation_handling = f"""
    ## CRITICAL: IMPLICIT AGGREGATION DETECTED
    
    Multiple granularity levels detected in the data. Some rows are aggregates (summaries) of other rows.
    
    Summary rows (indices): {summary_rows[:20]}... (total: {len(summary_rows)} rows)
    Detail rows (indices): {detail_rows[:20]}... (total: {len(detail_rows)} rows)
    
    Hierarchies: {json.dumps(hierarchies[:2], indent=2, ensure_ascii=False)}
    
    **ACTION REQUIRED**: Remove ONLY the summary rows to avoid double-counting:
    ```python
    summary_indices = {summary_rows}
    result = result.drop(index=summary_indices).reset_index(drop=True)
    ```
    
    **IMPORTANT**: This only removes rows from one category hierarchy. Other independent categories should remain untouched.
    """

        # Analyze bilingual column patterns from schema
        bilingual_cols_info = ""
        bilingual_pairs = []
        for col_def in column_schema:
            col_name = col_def.get('name', '')
            if '/' in col_name:  # Merged column name like "年份/Year"
                parts = col_name.split('/')
                if len(parts) == 2:
                    # Check if original data has these as separate columns
                    if parts[0] in df.columns and parts[1] in df.columns:
                        bilingual_pairs.append({
                            'merged_name': col_name,
                            'col1': parts[0],
                            'col2': parts[1]
                        })

        if bilingual_pairs:
            bilingual_cols_info = f"""
    ## DETECTED: Bilingual Column Pairs
    
    The schema suggests merged column names, but the original data has them as SEPARATE columns:
    {json.dumps(bilingual_pairs, indent=2, ensure_ascii=False)}
    
    **DECISION**: You should decide whether to:
    - Option A: Keep them separate (recommended if they contain different content)
    - Option B: Merge them into one column with "col1/col2" format
    
    If keeping separate, adjust the expected_output_columns accordingly.
    """

        prompt = f"""Generate Python transformation code based on the schema design.
    
    ## Current Data:
    - Shape: {df.shape}
    - Columns: {df.columns.tolist()}
    - Sample (first 3 rows):
    {sample_data}
    
    {aggregation_handling}
    
    {bilingual_cols_info}
    
    ## Identified Patterns:
    {json.dumps(patterns, indent=2, ensure_ascii=False)[:800]}
    
    ## Column Schema (target structure):
    {json.dumps(column_schema, indent=2, ensure_ascii=False)[:1500]}
    
    ## Split Operations (if any):
    {json.dumps(split_operations, indent=2, ensure_ascii=False)}
    
    ## Expected Output Columns:
    {expected_cols}
    
    ---
    
    ## GENERAL TRANSFORMATION PRINCIPLES:
    
    ### 1. Implicit Aggregation Handling
    - If aggregation detected above, remove summary rows FIRST
    - Use the exact indices provided
    - Only affects specific category hierarchies, not all data
    
    ### 2. Bilingual Column Handling
    - If original data has separate language columns (e.g., "类别" and "Category")
    - And schema suggests merged names (e.g., "类别/Category")
    - KEEP THEM SEPARATE unless there's strong reason to merge
    - Adjust expected_output_columns to match
    
    ### 3. Split Operations
    - Apply splits specified in split_operations
    - Handle cases where split doesn't produce expected parts (use fillna or conditionals)
    - Preserve original column naming style when creating new columns
    
    ### 4. Column Mapping
    - Map original columns to target schema
    - If schema specifies a type (e.g., integer, boolean), convert appropriately
    - Handle NaN/null values before type conversion
    
    ### 5. Edge Cases
    - Empty strings, NaN, None values
    - Rows with missing data
    - Unexpected split results (fewer or more parts than expected)
    
    ---
    
    ## CODE STRUCTURE TEMPLATE:
    ```python
    import pandas as pd
    import numpy as np
    
    def transform(df):
        result = df.copy()
        
        # STEP 0: Handle implicit aggregation (if detected)
        # Use the summary_indices provided above
        
        # STEP 1: Remove metadata rows (if any)
        # Check structure_analysis for metadata_rows
        
        # STEP 2: Apply split operations
        # For each split in split_operations:
        #   - Extract source_column, split_pattern, target_columns
        #   - Apply split
        #   - Handle missing parts
        
        # STEP 3: Handle bilingual columns
        # Decide whether to keep separate or merge
        # Based on actual data structure
        
        # STEP 4: Type conversions
        # Based on column_schema types
        
        # STEP 5: Column selection and ordering
        # Build final column list from expected_output_columns
        # Adjust if keeping bilingual columns separate
        # Ensure all columns exist (create empty if missing)
        
        return result
    
    # Execute transformation
    result_df = transform(df)
    ```
    
    ---
    
    ## CODE GENERATION RULES:
    
    1. **Be defensive**: Wrap risky operations in try-except
    2. **Check existence**: Verify columns exist before accessing
    3. **Handle nulls**: fillna() before operations that require non-null
    4. **Preserve intent**: If schema says "keep bilingual separate", do so
    5. **Complete code**: Must end with `result_df = transform(df)`
    6. **No truncation**: Write complete, executable code
    
    ## CRITICAL REMINDERS:
    
    - **Aggregation removal**: Only remove specific summary_indices, not entire categories
    - **Bilingual columns**: Respect the original data structure
    - **Split operations**: Follow the split_operations specification exactly
    - **Expected columns**: Adjust if needed based on decisions made
    - **Error handling**: Use try-except for operations that might fail
    
    {(f"## PREVIOUS ATTEMPT {attempt} FAILED"
      f"Review the errors and fix them. Common issues:"
      f"- Incorrect column selection removing too much data"
      f"- Creating columns that don't match expected_output_columns"
      f"- Not handling NaN/empty values in splits") if attempt > 0 else ""}
    
    ---
    
    **Output ONLY executable Python code. No explanations outside comments.**
    """

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

            if len(result_df) == 0:
                raise ValueError("Transformation produced empty DataFrame")

            logger.info(f"Transformation successful. Output shape: {result_df.shape}")
            return result_df

        except Exception as e:
            error_msg = f"Error during transformation: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Input shape: {df.shape}, columns: {df.columns.tolist()}")
            raise Exception(error_msg)

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

        # Check 3: No excessive data loss (allow reduction from summary removal)
        if self.preserve_data:
            original_rows = len(df_original)
            output_rows = len(df_normalized)

            # Allow up to 70% reduction (for summary row removal)
            if output_rows < original_rows * 0.3:
                warnings.append(
                    f"Significant row reduction: {original_rows} -> {output_rows} (may be due to summary removal)"
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