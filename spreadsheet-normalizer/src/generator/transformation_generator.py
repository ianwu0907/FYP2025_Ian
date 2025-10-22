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

        metadata = encoded_data['metadata']
        df = encoded_data['dataframe']

        sample_data = df.head(3).to_string()

        prompt = f"""Generate Python transformation code based on the schema design.

## Current Data:
- Shape: {df.shape}
- Columns: {df.columns.tolist()}
- Sample:
{sample_data}

## Identified Patterns:
{json.dumps(schema.get('identified_patterns', []), indent=2)}

## Target Schema:
{json.dumps(schema.get('column_schema', []), indent=2)[:1500]}

## Split Operations Required:
{json.dumps(schema.get('split_operations', []), indent=2)}

## Expected Output:
{schema.get('expected_output_columns', [])}

---

## CODE PATTERN LIBRARY:

### Pattern 1: Split Composite Column
# Example: "Physical abuse - Male" to ["Physical abuse", "Male"]

# Option A: Simple split
parts = df['column_name'].str.split(' - ', expand=True)
df['part1'] = parts[0].str.strip()
df['part2'] = parts[1].str.strip() if parts.shape[1] > 1 else ''

# Option B: Split with language detection
def split_bilingual_item(text):
    if pd.isna(text):
        return '', '', '', ''
    parts = str(text).split(' - ')
    if len(parts) >= 2:
        chinese_part = parts[0].strip()
        english_part = parts[1].strip()
        return chinese_part, '', english_part, ''
    return text, '', '', ''

df[['item1_cn', 'item2_cn', 'item1_en', 'item2_en']] = df['item'].apply(
    lambda x: pd.Series(split_bilingual_item(x))
)

# Option C: Regex for complex patterns
import re
df['category'] = df['item'].str.extract(r'^([^-]+)', expand=False).str.strip()
df['subcategory'] = df['item'].str.extract(r'-\s*(.+)$', expand=False).str.strip()

---

### Pattern 2: Handle Bilingual Content
# Keep bilingual in one column
df['year_bilingual'] = df['year_cn'] + '/' + df['year_en']

# Or separate into two columns
df['year_cn'] = df['year_bilingual'].str.split('/').str[0]
df['year_en'] = df['year_bilingual'].str.split('/').str[1]

# Or translate NaN/empty to bilingual
df['status_cn'] = df['status'].map({{
    True: 'Yes', False: 'No', None: 'N/A'
}})
df['status_en'] = df['status'].map({{
    True: 'Yes', False: 'No', None: 'N/A'
}})

---

### Pattern 3: Boolean to Categorical
# Map boolean to text
df['reported_cn'] = df['is_reported'].fillna(False).map({{
    True: 'Reported', 
    False: 'Not Reported',
}})
df['reported_en'] = df['is_reported'].fillna(False).map({{
    True: 'Reported',
    False: 'Not Reported',
}})

# Handle NaN/empty as special category
df['reported_cn'] = df['reported_cn'].fillna('N/A')
df['reported_en'] = df['reported_en'].fillna('N/A')

---

### Pattern 4: Column Reordering
# Reorder to match expected schema
expected_order = ['col1', 'col2', 'col3']
df = df[expected_order]

# Or use reindex with fill for missing
df = df.reindex(columns=expected_order, fill_value='')

---

### Pattern 5: Remove Metadata Rows
# Filter out metadata rows (identified in structure analysis)
metadata_rows = {structure_analysis.get('metadata_rows', [])}
if metadata_rows:
    max_meta_row = max(metadata_rows) if metadata_rows else -1
    df = df.iloc[max_meta_row + 1:].reset_index(drop=True)

# Or remove by condition
df = df[df['year'].notna() & (df['year'] != '')]

---

### Pattern 6: Type Conversion with Error Handling
# Safe numeric conversion
df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

# Safe string conversion
df['category'] = df['category'].astype(str).str.strip()

# Date parsing
df['date'] = pd.to_datetime(df['date'], errors='coerce')

---

## YOUR CODE GENERATION TASK:

Based on the schema design above, write transformation code that:

1. **Applies the split operations** specified in split_operations
2. **Handles bilingual columns** appropriately  
3. **Converts data types** as specified
4. **Reorders columns** to match expected_output_columns
5. **Handles edge cases** (NaN, empty strings, etc.)

## Code Template:

import pandas as pd
import numpy as np
import re

def transform(df):
    # Make a copy
    result = df.copy()
    
    # Step 1: Remove metadata rows
    # YOUR CODE based on structure_analysis
    
    # Step 2: Apply split operations
    # YOUR CODE based on split_operations from schema
    # Use the patterns from Pattern Library above
    
    # Step 3: Handle bilingual columns
    # YOUR CODE
    
    # Step 4: Type conversions
    # YOUR CODE
    
    # Step 5: Reorder columns to match expected output
    expected_cols = {schema.get('expected_output_columns', [])}
    
    # Step 6: Validation
    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in result.columns:
            result[col] = ''
    
    # Return only expected columns in correct order
    result = result[expected_cols]
    
    return result

# Execute
result_df = transform(df)

## CRITICAL REQUIREMENTS:

1. **Use try-except** for risky operations
2. **Handle NaN/None** explicitly  
3. **Validate intermediate steps** (check column counts, types)
4. **Match expected_output_columns EXACTLY**
5. **Document complex logic** with comments

{f"## PREVIOUS ATTEMPT FAILED - Review and fix issues" if attempt > 0 else ""}

Output ONLY executable Python code. No explanations outside comments."""

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

        # Check 3: No excessive data loss
        if self.preserve_data:
            original_rows = len(df_original)
            output_rows = len(df_normalized)

            if output_rows < original_rows * 0.5:
                warnings.append(
                    f"Significant row reduction: {original_rows} -> {output_rows}"
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