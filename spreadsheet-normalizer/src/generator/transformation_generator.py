"""
Transformation Generator Module (FINAL VERSION)
Uses unified metadata from encoder via schema - NO redundant analysis
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class TransformationGenerator:
    """
    Generates transformation code using metadata from encoder (via schema).

    CRITICAL PRINCIPLE:
    This module does NOT analyze the dataframe directly.
    It uses schema['source_metadata']['columns'] as the single source of truth.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the transformation generator."""
        self.config = config

        # 🔥 支持多种 LLM 提供商
        self.llm_provider = config.get('llm_provider', 'openai')  # 'openai', 'qwen', 'gemini', 'claude'
        
        if self.llm_provider == 'qwen':
            # 使用 Qwen (通义千问)
            api_key = os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = config.get('model', 'qwen-max')
            
        elif self.llm_provider == 'gemini':
            # 使用 Gemini (通过 API2D)
            api_key = os.getenv('API2D_API_KEY')
            if not api_key:
                raise ValueError("API2D_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv('GEMINI_BASE_URL', 'https://oa.api2d.net/v1')
            )
            self.model = config.get('model', 'gemini-2.0-flash-exp')
            
        elif self.llm_provider == 'claude':
            # 使用 Claude (通过 API2D)
            api_key = os.getenv('API2D_API_KEY')
            if not api_key:
                raise ValueError("API2D_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv('API2D_BASE_URL', 'https://oa.api2d.net/v1')
            )
            self.model = config.get('model', 'claude-3-5-sonnet-20241022')
            
        else:
            # 默认使用 OpenAI
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)
            self.model = config.get('model', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))

        # LLM settings
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4000)
        
        logger.info(f"Initialized TransformationGenerator with {self.llm_provider} ({self.model})")

        # Execution settings
        self.max_retries = config.get('max_retries', 3)

    def generate_and_execute(self,
                             encoded_data: Dict[str, Any],
                             structure_analysis: Dict[str, Any],
                             schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate transformation code and execute it."""
        logger.info("Generating transformation code with LLM...")

        # VERIFY: Check that schema has source_metadata
        if 'source_metadata' not in schema:
            logger.warning(
                "schema missing 'source_metadata'. "
                "Falling back to encoded_data['metadata']"
            )
            schema['source_metadata'] = encoded_data.get('metadata', {})

        # Generate the transformation code
        transformation_code = self._generate_code(encoded_data, structure_analysis, schema)

        logger.info("Executing transformation code...")

        # Execute with retries
        for attempt in range(self.max_retries):
            try:
                result_df = self._execute_code(transformation_code, encoded_data['dataframe'])

                # Validate the result
                validation_result = self._validate_result(result_df, schema, encoded_data)

                if validation_result['is_valid']:
                    logger.info("Transformation executed successfully")
                    return {
                        'normalized_df': result_df,
                        'transformation_code': transformation_code,
                        'validation_result': validation_result,
                        'attempts': attempt + 1
                    }
                else:
                    logger.warning(f"Validation failed on attempt {attempt + 1}")
                    if attempt < self.max_retries - 1:
                        # Regenerate code with feedback
                        transformation_code = self._regenerate_code(
                            encoded_data, structure_analysis, schema,
                            transformation_code, validation_result
                        )
                    else:
                        logger.error("Max retries reached, returning best effort result")
                        return {
                            'normalized_df': result_df,
                            'transformation_code': transformation_code,
                            'validation_result': validation_result,
                            'attempts': attempt + 1
                        }

            except Exception as e:
                logger.error(f"Execution failed on attempt {attempt + 1}: {e}")
                if attempt < self.max_retries - 1:
                    transformation_code = self._regenerate_code(
                        encoded_data, structure_analysis, schema,
                        transformation_code, {'error': str(e)}
                    )
                else:
                    raise

    def _generate_code(self,
                       encoded_data: Dict[str, Any],
                       structure_analysis: Dict[str, Any],
                       schema: Dict[str, Any]) -> str:
        """Generate the transformation code using LLM."""

        # Create the prompt with metadata from schema
        prompt = self._create_generation_prompt(encoded_data, structure_analysis, schema)

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

            code = response.choices[0].message.content

            # Extract code from markdown if present
            code = self._extract_code(code)

            logger.debug(f"Generated code:\n{code}")
            return code

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def _get_system_prompt(self) -> str:
        return """You are an expert Python programmer specializing in data transformation and pandas.
    
    Your task is to generate clean, efficient, and robust Python code that transforms messy spreadsheet data into normalized tables.
    
    CRITICAL RULES:
    1) Column metadata is provided, but unique values MAY be truncated/sample/top-k for large columns.
       - NEVER assume the unique_values list is complete.
       - Treat them as examples and write defensive conversion code.
    2) Always handle N/A / NaN / empty strings / whitespace / case variants.
    3) Do not hard-crash on unexpected input. Prefer returning None and emitting warnings via comments.
    4) The code must define a `transform(df)` function that returns the normalized DataFrame.
    5) Use only pandas/numpy/standard library. No network or file I/O.
    
    RECOMMENDED PRACTICES:
    - For boolean/categorical conversion: implement robust mapping:
      - normalize string: strip, lower, remove full-width spaces, unify punctuation
      - match by exact known examples AND fallback keywords/regex
      - unknown -> None
    - Keep original raw columns if needed for audit (optional).
    
    Output ONLY valid Python code.
    """


    def _create_generation_prompt(self,
                                  encoded_data: Dict[str, Any],
                                  structure_analysis: Dict[str, Any],
                                  schema: Dict[str, Any]) -> str:
        """
        Create prompt using UNIFIED metadata from schema.
        NO dataframe analysis here.
        """
        df = encoded_data['dataframe']
        source_metadata = schema.get('source_metadata', {})
        columns_metadata = source_metadata.get('columns', {})

        # Build type conversion guide from metadata
        type_conversion_guide = self._build_type_conversion_guide(schema, columns_metadata)

        prompt = f"""Generate a Python transformation function to normalize this table.

## INPUT DATA SHAPE:
- Rows: {len(df)}
- Columns: {list(df.columns)}

## IMPLICIT AGGREGATION:
{json.dumps(structure_analysis.get('implicit_aggregation', {}), indent=2)}

## TRANSFORMATION SCHEMA:
```json
{json.dumps({
            'column_schema': schema.get('column_schema', []),
            'split_operations': schema.get('split_operations', []),
            'type_conversions': schema.get('type_conversions', []),
            'expected_output_columns': schema.get('expected_output_columns', [])
        }, indent=2, ensure_ascii=False)}
```

## COMPLETE METADATA (from encoder):
{self._build_metadata_summary(columns_metadata)}

## TYPE CONVERSION GUIDE (with actual values):
{type_conversion_guide}

## CODE REQUIREMENTS:

1. **Function signature:**
```python
import pandas as pd
import numpy as np

def transform(df):
    \"\"\"Transform messy spreadsheet to normalized table\"\"\"
    result = df.copy()
    # ... transformation logic ...
    return result
```

2. **Handle implicit aggregation first:**
```python
# STEP 0: Remove summary rows if detected
summary_indices = {structure_analysis.get('implicit_aggregation', {}).get('summary_rows', [])}
if summary_indices:
    try:
        result = result.drop(index=summary_indices).reset_index(drop=True)
        print(f"Removed {{len(summary_indices)}} summary rows")
    except Exception as e:
        print(f"Warning: Could not remove summary rows: {{e}}")
```

3. **Column splitting (based on metadata delimiters):**
{self._generate_split_code_template(schema.get('split_operations', []), columns_metadata)}

4. **Type conversions (using ACTUAL metadata values):**
{self._generate_type_conversion_template(schema, columns_metadata)}

5. **Column renaming:**
```python
# STEP 3: Rename columns
rename_map = {{
    # Build from column_schema
}}
result.rename(columns=rename_map, inplace=True)
```

6. **Final column selection:**
```python
# STEP 4: Select and order final columns
expected_columns = {schema.get('expected_output_columns', [])}
result = result[expected_columns]
```

## CRITICAL REQUIREMENTS:
- Use try-except for each major transformation step
- Print informative messages for debugging
- Handle None/NaN appropriately
- Reference metadata in code comments (e.g., "# Based on metadata: unique_values=['有', '沒有']")
- Test defensive coding - don't assume data is clean

Generate ONLY the Python code, no markdown formatting, no explanations.
Start directly with import statements.
"""

        return prompt

    def _build_metadata_summary(self, columns_metadata: Dict[str, Any]) -> str:
        """Build concise metadata summary for prompt."""
        lines = []
        for col_name, col_meta in columns_metadata.items():
            lines.append(f"\n{col_name}:")
            lines.append(f"  Type: {col_meta.get('inferred_type')}")

            unique_vals = col_meta.get('unique_values', [])
            if unique_vals and len(unique_vals) <= 20:
                lines.append(f"  Unique values: {unique_vals}")

            value_counts = col_meta.get('value_counts', {})
            if value_counts and len(value_counts) <= 10:
                lines.append(f"  Value counts: {value_counts}")

        return '\n'.join(lines)

    def _build_type_conversion_guide(self,
                                     schema: Dict[str, Any],
                                     columns_metadata: Dict[str, Any]) -> str:
        """Build type conversion guide from schema and metadata."""
        lines = []

        type_conversions = schema.get('type_conversions', [])
        if not type_conversions:
            return "No type conversions specified."

        for conversion in type_conversions:
            col_name = conversion.get('column')
            target_type = conversion.get('target_type')
            value_mapping = conversion.get('value_mapping', {})

            lines.append(f"\n### Column: `{col_name}` → {target_type}")

            # Get actual unique values from metadata
            if col_name in columns_metadata:
                unique_vals = columns_metadata[col_name].get('unique_values', [])
                if unique_vals:
                    lines.append(f"**Metadata unique values:** {unique_vals}")

            # Show value mapping from schema
            if value_mapping:
                lines.append("**Value mapping (from schema):**")
                for orig, mapped in value_mapping.items():
                    lines.append(f"  `{orig}` → `{mapped}`")

            # Show conversion logic
            conversion_logic = conversion.get('conversion_logic', '')
            if conversion_logic:
                lines.append(f"**Logic:** {conversion_logic}")

        return '\n'.join(lines)

    def _generate_split_code_template(self,
                                      split_operations: List[Dict],
                                      columns_metadata: Dict[str, Any]) -> str:
        """Generate code template for column splitting."""
        if not split_operations:
            return "# No split operations required"

        lines = []
        lines.append("# STEP 1: Split composite columns")
        lines.append("")

        for op in split_operations:
            source = op['source_column']
            pattern = op['split_pattern']
            targets = op['target_columns']

            # Get metadata evidence
            evidence = ""
            if source in columns_metadata:
                delimiters = columns_metadata[source].get('potential_delimiters', [])
                for delim_info in delimiters:
                    if delim_info['delimiter'] == pattern:
                        evidence = f"  # Metadata: {delim_info['percentage']:.0f}% frequency, {delim_info.get('num_parts', '?')} parts"
                        break

            lines.append(f"# Split {source}{evidence}")
            lines.append(f"try:")
            lines.append(f"    split_result = result['{source}'].str.split('{pattern}', expand=True)")
            lines.append(f"    if split_result.shape[1] >= {len(targets)}:")
            for idx, target in enumerate(targets):
                lines.append(f"        result['{target}'] = split_result[{idx}]")
            lines.append(f"        print(f'Split {source} into {len(targets)} columns')")
            lines.append(f"    else:")
            lines.append(f"        print(f'Warning: Split of {source} produced {{split_result.shape[1]}} columns, expected {len(targets)}')")
            lines.append(f"except Exception as e:")
            lines.append(f"    print(f'Error splitting {source}: {{e}}')")
            lines.append("")

        return '\n'.join(lines)

    def _generate_type_conversion_template(self,
                                           schema: Dict[str, Any],
                                           columns_metadata: Dict[str, Any]) -> str:
        """Generate code template for type conversions."""
        type_conversions = schema.get('type_conversions', [])
        if not type_conversions:
            return "# STEP 2: No type conversions required"

        lines = []
        lines.append("# STEP 2: Type conversions")
        lines.append("")
        lines.append("# Define conversion helper functions")
        lines.append("")

        # Generate conversion functions
        for conversion in type_conversions:
            col_name = conversion.get('column')
            target_type = conversion.get('target_type')
            value_mapping = conversion.get('value_mapping', {})

            if target_type in ['boolean', 'boolean_with_na']:
                # Get actual unique values from metadata
                unique_vals = []
                if col_name in columns_metadata:
                    unique_vals = columns_metadata[col_name].get('unique_values', [])

                func_code = self._generate_boolean_converter(
                    col_name, value_mapping, unique_vals
                )
                lines.append(func_code)
                lines.append("")

        lines.append("# Apply conversions")
        for conversion in type_conversions:
            col_name = conversion.get('column')
            target_type = conversion.get('target_type')

            if target_type in ['boolean', 'boolean_with_na']:
                safe_name = col_name.replace(' ', '_').replace('/', '_').replace('（', '_').replace('）', '_')
                lines.append(f"try:")
                lines.append(f"    result['{col_name}'] = result['{col_name}'].apply(convert_{safe_name})")
                lines.append(f"    print(f'Converted {col_name} to boolean')")
                lines.append(f"except Exception as e:")
                lines.append(f"    print(f'Error converting {col_name}: {{e}}')")

        return '\n'.join(lines)

    def _generate_boolean_converter(self,
                                    col_name: str,
                                    value_mapping: Dict[str, Any],
                                    metadata_unique_vals: List[str]) -> str:
        """Generate a boolean converter function based on metadata."""

        # Create safe function name
        safe_name = col_name.replace(' ', '_').replace('/', '_').replace('（', '_').replace('）', '_')

        # Categorize values from mapping
        true_values = []
        false_values = []
        na_values = []

        for orig, mapped in value_mapping.items():
            orig_lower = orig.lower().strip()
            if mapped == True or mapped == 'True' or mapped == '1':
                true_values.append(orig_lower)
            elif mapped == False or mapped == 'False' or mapped == '0':
                false_values.append(orig_lower)
            elif mapped is None or mapped == 'None' or mapped == 'N/A':
                na_values.append(orig_lower)

        # Add common patterns not in mapping but in metadata
        common_true = ['有', 'yes', 'y', '是', 'true', '1', 't']
        common_false = ['沒有', '没有', 'no', 'n', '否', 'false', '0', 'f']
        common_na = ['不適用', '不适用', 'n/a', 'na', 'not applicable', 'none', 'null']

        # Expand lists with common patterns found in metadata
        metadata_lower = [v.lower().strip() for v in metadata_unique_vals]
        for val in common_true:
            if val in metadata_lower and val not in true_values:
                true_values.append(val)
        for val in common_false:
            if val in metadata_lower and val not in false_values:
                false_values.append(val)
        for val in common_na:
            if val in metadata_lower and val not in na_values:
                na_values.append(val)

        code = f"""def convert_{safe_name}(value):
    '''
    Convert {col_name} to boolean
    Based on metadata unique_values: {metadata_unique_vals}
    Value mapping: {value_mapping}
    '''
    if pd.isna(value):
        return None
    value_str = str(value).strip().lower()
    
    # True values
    if value_str in {true_values}:
        return True
    # False values
    elif value_str in {false_values}:
        return False
    # N/A values
    elif value_str in {na_values}:
        return None
    else:
        print(f"Warning: Unexpected value in {col_name}: '{{value}}'")
        return None"""

        return code

    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown or plain text."""
        text = text.strip()

        # Remove markdown code blocks
        if '```python' in text:
            start = text.find('```python') + 9
            end = text.find('```', start)
            text = text[start:end].strip()
        elif '```' in text:
            start = text.find('```') + 3
            end = text.find('```', start)
            text = text[start:end].strip()

        return text

    def _execute_code(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the transformation code safely with duplicate column handling."""
        
        def fix_duplicate_columns(dataframe, stage_name=""):
            """Fix duplicate column names by adding numeric suffixes."""
            if dataframe.columns.duplicated().any():
                duplicates = dataframe.columns[dataframe.columns.duplicated()].unique().tolist()
                logger.warning(f"[{stage_name}] Found duplicate column names: {duplicates}")
                
                # Create unique column names with numeric suffixes
                cols = pd.Series(dataframe.columns)
                for dup in cols[cols.duplicated()].unique():
                    dup_indices = cols[cols == dup].index
                    for i, idx in enumerate(dup_indices[1:], start=1):
                        cols[idx] = f"{dup}_{i}"
                
                dataframe.columns = cols.tolist()
                logger.info(f"[{stage_name}] Fixed columns to: {dataframe.columns.tolist()}")
            
            return dataframe
        
        # Fix duplicate columns in input DataFrame BEFORE any operations
        df = fix_duplicate_columns(df, "Input")
        
        # Create execution environment
        exec_globals = {
            'pd': pd,
            'np': np,
            'df': df.copy(),
            'print': print
        }

        # Execute the code
        exec(code, exec_globals)

        # Get the transform function
        if 'transform' not in exec_globals:
            raise ValueError("Generated code must define a 'transform' function")

        transform_func = exec_globals['transform']

        # Execute transformation
        result_df = transform_func(df.copy())
        
        # Fix duplicate columns in output DataFrame
        result_df = fix_duplicate_columns(result_df, "Output")

        return result_df
    def _validate_result(self,
                         result_df: pd.DataFrame,
                         schema: Dict[str, Any],
                         encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the transformation result."""
        errors = []
        warnings = []

        # Check if result is not empty
        if result_df.empty:
            errors.append("Result DataFrame is empty")

        # Check expected columns
        expected_cols = schema.get('expected_output_columns', [])
        actual_cols = list(result_df.columns)

        missing_cols = set(expected_cols) - set(actual_cols)
        extra_cols = set(actual_cols) - set(expected_cols)

        if missing_cols:
            errors.append(f"Missing expected columns: {missing_cols}")

        if extra_cols:
            warnings.append(f"Extra columns in result: {extra_cols}")

        # Check for excessive null values
        for col in result_df.columns:
            null_pct = result_df[col].isnull().sum() / len(result_df) * 100
            if null_pct > 90:
                warnings.append(f"Column '{col}' has {null_pct:.1f}% null values")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'original_rows': len(encoded_data['dataframe']),
            'result_rows': len(result_df),
            'columns_match': missing_cols == set() and extra_cols == set()
        }

    def _regenerate_code(self,
                         encoded_data: Dict[str, Any],
                         structure_analysis: Dict[str, Any],
                         schema: Dict[str, Any],
                         previous_code: str,
                         feedback: Dict[str, Any]) -> str:
        """Regenerate code with feedback from previous attempt."""
        logger.info("Regenerating code with feedback...")

        feedback_text = json.dumps(feedback, indent=2)

        prompt = f"""The previous transformation code failed. Please fix it.

## Previous Code:
```python
{previous_code}
```

## Feedback:
{feedback_text}

## Schema and Metadata:
Use the same schema and metadata as before.

Generate FIXED Python code that addresses the issues.
Output ONLY the code, no explanations.
"""

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

            code = response.choices[0].message.content
            code = self._extract_code(code)

            return code

        except Exception as e:
            logger.error(f"Error regenerating code: {e}")
            return previous_code