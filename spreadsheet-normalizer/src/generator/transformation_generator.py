"""
Transformation Generator Module (Enhanced)

Two-stage approach:
1. Strategy Generation: LLM generates free-form transformation strategy
2. Code Generation: LLM generates Python code based on strategy

Plus enhanced validation loop with semantic sampling.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import os
import pandas as pd
import numpy as np
import traceback

logger = logging.getLogger(__name__)


class TransformationGenerator:
    """
    Generates transformation code using two-stage LLM approach:

    Stage 1: Strategy Generation
    - LLM analyzes structure and schema
    - Outputs free-form transformation strategy
    - No predefined primitives - pure semantic reasoning

    Stage 2: Code Generation
    - LLM generates Python code based on strategy
    - Includes intermediate state expectations

    Stage 3: Validation & Iteration
    - Execute code
    - Validate output (structural + semantic sampling)
    - Feedback loop for correction
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the transformation generator."""
        self.config = config

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        # self.temperature = config.get('temperature', 0.1)
        self.max_completion_tokens = config.get('max_completion_tokens', 6000)

        # Execution settings
        self.max_retries = config.get('max_retries', 3)

    def generate_and_execute(self,
                             encoded_data: Dict[str, Any],
                             structure_analysis: Dict[str, Any],
                             schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate and execute transformation using two-stage approach.

        Returns:
            Dict containing:
            - normalized_df: The transformed DataFrame
            - transformation_code: The generated Python code
            - transformation_strategy: The strategy used
            - validation_result: Validation results
            - attempts: Number of attempts needed
        """
        logger.info("=" * 60)
        logger.info("TRANSFORMATION GENERATOR - Two Stage Approach")
        logger.info("=" * 60)

        # Stage 1: Generate transformation strategy
        logger.info("\n--- Stage 1: Strategy Generation ---")
        strategy = self._generate_strategy(encoded_data, structure_analysis, schema)
        logger.info(f"Strategy generated with {len(strategy.get('transformation_steps', []))} steps")

        # Stage 2: Generate code from strategy
        logger.info("\n--- Stage 2: Code Generation ---")
        transformation_code = self._generate_code_from_strategy(
            encoded_data, structure_analysis, schema, strategy
        )

        # Stage 3: Execute and validate with retry loop
        logger.info("\n--- Stage 3: Execution & Validation ---")

        for attempt in range(self.max_retries):
            logger.info(f"\nAttempt {attempt + 1}/{self.max_retries}")

            try:
                # Execute the code
                result_df = self._execute_code(transformation_code, encoded_data['dataframe'])
                logger.info(f"Code executed successfully. Result shape: {result_df.shape}")

                # Validate the result
                validation_result = self._validate_result(
                    result_df, schema, encoded_data, strategy
                )

                if validation_result['is_valid']:
                    logger.info("✓ Validation PASSED")
                    return {
                        'normalized_df': result_df,
                        'transformation_code': transformation_code,
                        'transformation_strategy': strategy,
                        'validation_result': validation_result,
                        'attempts': attempt + 1
                    }
                else:
                    logger.warning(f"✗ Validation FAILED: {validation_result['errors']}")

                    if attempt < self.max_retries - 1:
                        # Generate feedback and regenerate code
                        feedback = self._generate_structured_feedback(
                            validation_result, result_df, encoded_data['dataframe'], strategy
                        )
                        transformation_code = self._regenerate_code_with_feedback(
                            encoded_data, structure_analysis, schema, strategy,
                            transformation_code, feedback
                        )
                    else:
                        logger.error("Max retries reached, returning best effort result")
                        return {
                            'normalized_df': result_df,
                            'transformation_code': transformation_code,
                            'transformation_strategy': strategy,
                            'validation_result': validation_result,
                            'attempts': attempt + 1
                        }

            except Exception as e:
                error_msg = str(e)
                error_trace = traceback.format_exc()
                logger.error(f"Execution error: {error_msg}")

                if attempt < self.max_retries - 1:
                    feedback = {
                        'error_type': 'EXECUTION_ERROR',
                        'error_message': error_msg,
                        'error_trace': error_trace,
                        'suggestion': 'Fix the code to handle this error'
                    }
                    transformation_code = self._regenerate_code_with_feedback(
                        encoded_data, structure_analysis, schema, strategy,
                        transformation_code, feedback
                    )
                else:
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("Transformation failed after all retries")

    # =========================================================================
    # Stage 1: Strategy Generation
    # =========================================================================

    def _generate_strategy(self,
                           encoded_data: Dict[str, Any],
                           structure_analysis: Dict[str, Any],
                           schema: Dict[str, Any]) -> Dict[str, Any]:
        """Generate free-form transformation strategy using LLM."""

        prompt = self._create_strategy_prompt(encoded_data, structure_analysis, schema)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_strategy_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                # temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )

            result_text = response.choices[0].message.content
            strategy = self._parse_json_response(result_text)

            logger.debug(f"Generated strategy: {json.dumps(strategy, indent=2, ensure_ascii=False)[:1000]}...")
            return strategy

        except Exception as e:
            logger.error(f"Error generating strategy: {e}")
            return self._get_default_strategy(schema)

    def _get_strategy_system_prompt(self) -> str:
        """System prompt for strategy generation."""
        return """You are an expert data engineer creating transformation strategies for messy spreadsheets.

## YOUR TASK
Create a step-by-step transformation strategy to convert a messy spreadsheet into tidy format.

## PRINCIPLES
1. Each step should have a clear goal and description
2. Describe transformations in plain language - no predefined operations
3. Be specific about row/column indices and data mappings
4. Think about edge cases and validation points

## STRATEGY SHOULD COVER
1. Data region extraction (skip headers/footers)
2. Handling special patterns (bilingual, section markers, etc.)
3. Reshaping (unpivot wide columns, merge rows, etc.)
4. Data cleaning and type conversion
5. Final column selection and naming

Output valid JSON only."""

    def _create_strategy_prompt(self,
                                encoded_data: Dict[str, Any],
                                structure_analysis: Dict[str, Any],
                                schema: Dict[str, Any]) -> str:
        """Create prompt for strategy generation."""
        df = encoded_data['dataframe']

        # Create sample data view
        sample_data = self._create_sample_data_view(df, structure_analysis)

        # Summarize key info
        structure_summary = self._summarize_structure(structure_analysis)
        schema_summary = self._summarize_schema(schema)

        prompt = f"""Create a transformation strategy for this spreadsheet.

## SOURCE DATA
Shape: {df.shape[0]} rows × {df.shape[1]} columns

{sample_data}

## STRUCTURE ANALYSIS
{structure_summary}

## TARGET SCHEMA
{schema_summary}

## YOUR TASK
Create a detailed transformation strategy. Think through:

1. How to extract the data region (which rows to include/exclude)
2. How to handle any special patterns (bilingual rows, section markers)
3. How to reshape the data (unpivot, merge, etc.)
4. How to create each target column
5. Validation checkpoints

## OUTPUT FORMAT (JSON)

{{
  "strategy_summary": "Brief description of the overall transformation approach",
  
  "transformation_steps": [
    {{
      "step_number": 1,
      "goal": "What this step achieves",
      "description": "Detailed free-form description of what to do. Be specific about indices and logic.",
      "input_state": "Description of data state before this step",
      "output_state": "Description of data state after this step",
      "key_logic": "The core transformation logic in plain language",
      "pandas_hints": "Optional hints about pandas operations that might help"
    }}
  ],
  
  "column_creation_details": [
    {{
      "target_column": "column_name",
      "source": "Where this data comes from",
      "transformation": "How to create this column"
    }}
  ],
  
  "validation_checkpoints": [
    {{
      "after_step": 1,
      "check_type": "row_count | column_exists | value_spot_check",
      "description": "What to verify",
      "expected_value": "What the expected result is"
    }}
  ],
  
  "edge_cases": [
    {{
      "case": "Description of edge case",
      "handling": "How to handle it"
    }}
  ]
}}

Be thorough and specific. Output only the JSON."""

        return prompt

    def _create_sample_data_view(self, df: pd.DataFrame, structure_analysis: Dict[str, Any]) -> str:
        """Create sample data view for strategy prompt."""
        lines = []

        # Show all header rows
        header_rows = structure_analysis.get('header_structure', {}).get('header_rows', [0])
        lines.append("HEADER ROWS:")
        for row_idx in header_rows:
            if row_idx < len(df):
                row_vals = [f"[{j}]={repr(str(df.iloc[row_idx, j])[:30])}"
                            for j in range(len(df.columns))
                            if pd.notna(df.iloc[row_idx, j]) and str(df.iloc[row_idx, j]).strip()]
                lines.append(f"  Row {row_idx}: {', '.join(row_vals)}")

        # Show data region sample
        data_region = structure_analysis.get('data_region', {})
        start_row = data_region.get('start_row', 1)
        end_row = min(data_region.get('end_row', len(df)-1), start_row + 15)

        lines.append(f"\nDATA ROWS ({start_row} to {end_row}):")
        for i in range(start_row, end_row + 1):
            if i < len(df):
                row_vals = [f"[{j}]={repr(str(df.iloc[i, j])[:25])}"
                            for j in range(len(df.columns))
                            if pd.notna(df.iloc[i, j]) and str(df.iloc[i, j]).strip()]
                lines.append(f"  Row {i}: {', '.join(row_vals)}")

        return "\n".join(lines)
    def _check_if_filtering_needed(self,rows_to_exclude: str, details: dict) -> bool:
        """
        Determine if row filtering is actually needed.

        Returns True only if there are actual aggregation rows (Total, Sum, etc.)
        that need to be excluded.

        Returns False if:
        - rows_to_exclude is empty, None, or indicates "none"
        - The "summary values" are actually just different category types (not aggregations)
        """
        if not rows_to_exclude:
            return False

        # Normalize the guidance text
        rows_to_exclude_lower = rows_to_exclude.lower().strip()

        # Phrases that indicate NO filtering is needed
        no_filter_phrases = [
            'none',
            'none specifically',
            'no rows',
            'keep all',
            'preserve all',
            'no filtering',
            'not applicable',
            'n/a',
            'all rows are valid',
            'no exclusion'
        ]

        for phrase in no_filter_phrases:
            if phrase in rows_to_exclude_lower:
                return False

        # Check if summary_values contain actual aggregation keywords
        aggregation_keywords = [
            'total', 'sum', 'subtotal', 'aggregate', 'all',
            '合計', '小計', '總計', '总计', '小计', '合计'
        ]

        summary_values = details.get('summary_values', [])
        has_real_aggregation = False

        for value in summary_values:
            value_lower = str(value).lower()
            for keyword in aggregation_keywords:
                if keyword in value_lower:
                    has_real_aggregation = True
                    break
            if has_real_aggregation:
                break

        # Only filter if we found actual aggregation keywords
        # If guidance says to exclude but no aggregation keywords found,
        # be conservative and preserve all data
        return has_real_aggregation
    def _summarize_structure(self, structure_analysis: dict) -> str:
        """
        Summarize structure analysis for prompt.

        FIXED: Now properly handles implicit aggregation by:
        1. Checking if rows_to_exclude actually requires filtering
        2. Using conditional split logic instead of mandatory filtering
        3. Preserving all valid observation rows
        """
        lines = []

        # Semantic understanding
        sem = structure_analysis.get('semantic_understanding', {})
        lines.append(f"Data represents: {sem.get('data_description', 'Unknown')}")

        # Header info
        header = structure_analysis.get('header_structure', {})
        lines.append(f"Header rows: {header.get('header_rows', [])}")
        lines.append(f"Header levels: {header.get('num_levels', 1)}")

        # Data region
        data_region = structure_analysis.get('data_region', {})
        lines.append(f"Data region: rows {data_region.get('start_row')} to {data_region.get('end_row')}")

        # Row patterns
        row_patterns = structure_analysis.get('row_patterns', {})
        if row_patterns.get('has_bilingual_rows'):
            bilingual = row_patterns.get('bilingual_details', {})
            lines.append(f"Bilingual: {bilingual.get('pattern', 'yes')}, relationship: {bilingual.get('data_relationship', 'unknown')}")
            if bilingual.get('if_different_types'):
                diff = bilingual['if_different_types']
                lines.append(f"  CN rows: {diff.get('cn_row_contains')}, EN rows: {diff.get('en_row_contains')}")

        if row_patterns.get('has_section_markers'):
            markers = row_patterns.get('section_marker_details', {})
            lines.append(f"Section markers: {markers.get('marker_values', [])} at column {markers.get('marker_column')}")

        # =========================================================================
        # FIXED: Improved implicit aggregation handling
        # =========================================================================
        implicit_agg = structure_analysis.get('implicit_aggregation', {})
        if implicit_agg.get('has_implicit_aggregation'):
            lines.append(f"\n{'='*60}")
            lines.append(f"IMPLICIT AGGREGATION DETECTED")
            lines.append(f"{'='*60}")

            details = implicit_agg.get('detection_details', {})
            guidance = implicit_agg.get('transformation_guidance', {})

            # Get the rows_to_exclude guidance from structure analysis
            rows_to_exclude = guidance.get('rows_to_exclude', '')

            # Determine if actual row filtering is needed
            needs_row_filtering = self._check_if_filtering_needed(rows_to_exclude, details)

            # Document the category column and detected patterns
            lines.append(f"\nCategory column: '{details.get('category_column', 'unknown')}'")
            lines.append(f"Delimiter detected: '{details.get('delimiter', 'N/A')}'")
            lines.append(f"Additional dimension: '{details.get('additional_dimension', 'N/A')}'")

            # CRITICAL: Conditional row filtering based on actual guidance
            if needs_row_filtering:
                # Only add filtering instructions if truly needed
                lines.append(f"\n--- ROW FILTERING REQUIRED ---")
                lines.append(f"Rows to exclude: {rows_to_exclude}")
                lines.append(f"Values to REMOVE (true aggregation rows): {details.get('summary_values', [])}")
                lines.append(f"Values to KEEP: {details.get('detail_values', [])}")
            else:
                # No filtering needed - all rows are valid observations
                lines.append(f"\n--- NO ROW FILTERING NEEDED ---")
                lines.append(f"IMPORTANT: ALL rows are valid observation records - DO NOT filter any rows!")
                lines.append(f"")
                lines.append(f"The data contains multiple category types:")
                lines.append(f"  - Some rows have values WITH delimiter (contain additional dimension like gender)")
                lines.append(f"  - Some rows have values WITHOUT delimiter (different category, no sub-dimension)")
                lines.append(f"  - BOTH types are valid observations and MUST be preserved in the output")
                lines.append(f"")
                lines.append(f"DO NOT confuse 'no delimiter' with 'aggregation row'!")
                lines.append(f"Rows without delimiter are simply a DIFFERENT CATEGORY, not summaries/totals.")

            # FIXED: Conditional split operation - apply to ALL rows
            if details.get('delimiter'):
                lines.append(f"\n--- CONDITIONAL COLUMN SPLIT (apply to ALL rows) ---")
                lines.append(f"Column to process: '{guidance.get('column_to_split', 'identified by LLM')}'")
                lines.append(f"Delimiter: '{details.get('delimiter')}'")
                lines.append(f"Additional dimension to extract: '{details.get('additional_dimension')}'")
                lines.append(f"Expected new columns: {guidance.get('expected_new_columns', [])}")
                lines.append(f"")
                lines.append(f"SPLIT LOGIC (must handle BOTH cases):")
                lines.append(f"  CASE 1 - Row contains delimiter '{details.get('delimiter')}':")
                lines.append(f"    - Split the value into [base_type, additional_dimension]")
                lines.append(f"    - Example: 'Physical abuse - Male' -> base='Physical abuse', gender='Male'")
                lines.append(f"  CASE 2 - Row does NOT contain delimiter:")
                lines.append(f"    - Keep original value as base_type (do not modify)")
                lines.append(f"    - Set additional_dimension to None/NULL")
                lines.append(f"    - Example: 'Physical abuse' -> base='Physical abuse', gender=None")
                lines.append(f"")
                lines.append(f"CRITICAL: The split logic must work for ALL rows, not just filtered rows!")

            # Sample data for reference
            samples = implicit_agg.get('sample_detail_rows', [])
            if samples:
                lines.append(f"\nSAMPLE DATA (rows WITH delimiter, for verification):")
                for sample in samples[:2]:
                    lines.append(f"  {sample.get('all_columns', sample)}")

        # Column patterns
        col_patterns = structure_analysis.get('column_patterns', {})
        if col_patterns.get('value_columns'):
            lines.append(f"\nValue columns to unpivot: {col_patterns.get('value_columns')}")
        if col_patterns.get('aggregate_columns'):
            lines.append(f"Aggregate columns to exclude: {col_patterns.get('aggregate_columns')}")

        return "\n".join(lines)

    def _summarize_schema(self, schema: Dict[str, Any]) -> str:
        """Summarize target schema for prompt."""
        lines = []

        obs = schema.get('observation_unit', {})
        lines.append(f"Observation unit: {obs.get('description', 'Unknown')}")

        lines.append("\nTarget columns:")
        for col in schema.get('target_columns', []):
            dim_marker = "[DIM]" if col.get('is_dimension') else "[VAL]"
            lines.append(f"  {dim_marker} {col.get('name')} ({col.get('data_type')}): {col.get('description', '')[:50]}")

        expected = schema.get('expected_output', {})
        lines.append(f"\nExpected: {expected.get('row_count_estimate', '?')} rows ({expected.get('row_count_formula', 'unknown')})")

        # Validation samples
        samples = schema.get('validation_samples', [])
        if samples:
            lines.append("\nValidation samples:")
            for sample in samples[:2]:
                lines.append(f"  {sample.get('description', '')}: {sample.get('expected_row', {})}")

        return "\n".join(lines)

    def _get_default_strategy(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        """Return default strategy if generation fails."""
        return {
            'strategy_summary': 'Default pass-through transformation',
            'transformation_steps': [
                {
                    'step_number': 1,
                    'goal': 'Copy data as-is',
                    'description': 'Return the dataframe with minimal changes',
                    'input_state': 'Raw dataframe',
                    'output_state': 'Same dataframe',
                    'key_logic': 'df.copy()',
                    'pandas_hints': 'df.copy()'
                }
            ],
            'column_creation_details': [],
            'validation_checkpoints': [],
            'edge_cases': []
        }

    # =========================================================================
    # Stage 2: Code Generation
    # =========================================================================

    def _generate_code_from_strategy(self,
                                     encoded_data: Dict[str, Any],
                                     structure_analysis: Dict[str, Any],
                                     schema: Dict[str, Any],
                                     strategy: Dict[str, Any]) -> str:
        """Generate Python code from the transformation strategy."""

        prompt = self._create_code_generation_prompt(
            encoded_data, structure_analysis, schema, strategy
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_code_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                # temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )

            code = response.choices[0].message.content
            code = self._extract_code(code)

            logger.debug(f"Generated code:\n{code[:500]}...")
            return code

        except Exception as e:
            logger.error(f"Error generating code: {e}")
            raise

    def _get_code_system_prompt(self) -> str:
        """System prompt for code generation."""
        return """You are an expert Python programmer specializing in pandas data transformations.

## YOUR TASK
Implement a transformation strategy as executable Python code.

## REQUIREMENTS
1. Define a `transform(df)` function that takes a DataFrame and returns the transformed DataFrame
2. Follow the strategy steps exactly
3. Add comments referencing strategy step numbers
4. Include print statements for debugging (showing row counts, sample data at each step)
5. Handle edge cases robustly (missing values, type conversion errors)
6. Use only pandas, numpy, and standard library

## CODE STRUCTURE
```python
import pandas as pd
import numpy as np

def transform(df):
    \"\"\"
    Transform messy spreadsheet to tidy format.
    \"\"\"
    print(f"Input shape: {df.shape}")
    
    # === Step 1: [Goal from strategy] ===
    # [implementation]
    print(f"After step 1: {result.shape}")
    
    # === Step 2: [Goal from strategy] ===
    # [implementation]
    
    # ... more steps ...
    
    return result
```

Output ONLY the Python code, no markdown formatting."""

    def _create_code_generation_prompt(self,
                                       encoded_data: Dict[str, Any],
                                       structure_analysis: Dict[str, Any],
                                       schema: Dict[str, Any],
                                       strategy: Dict[str, Any]) -> str:
        """Create prompt for code generation."""
        df = encoded_data['dataframe']

        # Format strategy steps
        steps_text = self._format_strategy_steps(strategy)

        # Get sample data for verification
        sample_data = self._get_sample_for_code_gen(df, structure_analysis)

        # Target columns
        target_cols = [col['name'] for col in schema.get('target_columns', [])]

        prompt = f"""Implement this transformation strategy as Python code.

## TRANSFORMATION STRATEGY
{steps_text}

## SOURCE DATA INFO
- Shape: {df.shape}
- Columns: {list(df.columns)}

{sample_data}

## TARGET OUTPUT
- Columns: {target_cols}
- Expected rows: {schema.get('expected_output', {}).get('row_count_estimate', 'unknown')}

## COLUMN CREATION DETAILS
{json.dumps(strategy.get('column_creation_details', []), indent=2, ensure_ascii=False)}

## VALIDATION SAMPLES (use for spot-checking)
{json.dumps(schema.get('validation_samples', []), indent=2, ensure_ascii=False)}

## REQUIREMENTS
1. Function signature: `def transform(df):`
2. Return the transformed DataFrame
3. Add print statements showing progress
4. Handle errors gracefully
5. Match the target column names exactly

Generate the Python code now. Output ONLY code, no explanations or markdown."""

        return prompt

    def _format_strategy_steps(self, strategy: Dict[str, Any]) -> str:
        """Format strategy steps for code generation prompt."""
        lines = [f"Summary: {strategy.get('strategy_summary', '')}"]
        lines.append("")

        for step in strategy.get('transformation_steps', []):
            lines.append(f"### Step {step.get('step_number', '?')}: {step.get('goal', '')}")
            lines.append(f"Description: {step.get('description', '')}")
            lines.append(f"Key logic: {step.get('key_logic', '')}")
            if step.get('pandas_hints'):
                lines.append(f"Pandas hints: {step.get('pandas_hints')}")
            lines.append(f"Input: {step.get('input_state', '')}")
            lines.append(f"Output: {step.get('output_state', '')}")
            lines.append("")

        return "\n".join(lines)

    def _get_sample_for_code_gen(self, df: pd.DataFrame, structure_analysis: Dict[str, Any]) -> str:
        """Get sample data for code generation."""
        lines = ["SAMPLE DATA:"]

        data_region = structure_analysis.get('data_region', {})
        start_row = data_region.get('start_row', 0)

        # Show a few data rows
        for i in range(start_row, min(start_row + 6, len(df))):
            row_vals = []
            for j in range(min(12, len(df.columns))):
                val = df.iloc[i, j]
                if pd.notna(val):
                    row_vals.append(f"[{j}]={repr(val)[:20]}")
            lines.append(f"Row {i}: {', '.join(row_vals)}")

        return "\n".join(lines)

    def _extract_code(self, text: str) -> str:
        """Extract Python code from response."""
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

    # =========================================================================
    # Stage 3: Execution and Validation
    # =========================================================================

    def _execute_code(self, code: str, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the transformation code safely."""
        exec_globals = {
            'pd': pd,
            'np': np,
            'df': df.copy(),
            'print': print
        }

        exec(code, exec_globals)

        if 'transform' not in exec_globals:
            raise ValueError("Generated code must define a 'transform' function")

        transform_func = exec_globals['transform']
        result_df = transform_func(df.copy())

        return result_df

    def _validate_result(self,
                         result_df: pd.DataFrame,
                         schema: Dict[str, Any],
                         encoded_data: Dict[str, Any],
                         strategy: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive validation including semantic sampling.
        """
        errors = []
        warnings = []
        checks_performed = []

        # Check 1: Not empty
        if result_df.empty:
            errors.append("Result DataFrame is empty")
        checks_performed.append({'check': 'not_empty', 'passed': not result_df.empty})

        # Check 2: Expected columns exist
        expected_cols = [col['name'] for col in schema.get('target_columns', [])]
        actual_cols = list(result_df.columns)
        missing_cols = set(expected_cols) - set(actual_cols)
        extra_cols = set(actual_cols) - set(expected_cols)

        if missing_cols:
            errors.append(f"Missing columns: {missing_cols}")
        if extra_cols:
            warnings.append(f"Extra columns: {extra_cols}")

        checks_performed.append({
            'check': 'columns_match',
            'passed': len(missing_cols) == 0,
            'missing': list(missing_cols),
            'extra': list(extra_cols)
        })

        # Check 3: Row count reasonability
        expected_count = schema.get('expected_output', {}).get('row_count_estimate', 0)
        actual_count = len(result_df)

        if expected_count > 0:
            ratio = actual_count / expected_count
            if ratio < 0.5:
                errors.append(f"Row count too low: {actual_count} vs expected {expected_count}")
            elif ratio > 2.0:
                warnings.append(f"Row count higher than expected: {actual_count} vs expected {expected_count}")

        checks_performed.append({
            'check': 'row_count',
            'passed': 0.5 <= (actual_count / expected_count if expected_count > 0 else 1) <= 2.0,
            'actual': actual_count,
            'expected': expected_count
        })

        # Check 4: Semantic sampling validation
        sample_checks = self._validate_samples(result_df, schema, encoded_data)
        checks_performed.extend(sample_checks)

        for check in sample_checks:
            if not check.get('passed', True):
                errors.append(f"Sample validation failed: {check.get('description', 'unknown')}")

        # Check 5: Null percentage check
        for col in result_df.columns:
            null_pct = result_df[col].isnull().sum() / len(result_df) * 100 if len(result_df) > 0 else 0
            if null_pct > 50:
                warnings.append(f"Column '{col}' has {null_pct:.1f}% null values")

        return {
            'is_valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'checks_performed': checks_performed,
            'actual_shape': result_df.shape,
            'expected_shape': (expected_count, len(expected_cols))
        }

    def _validate_samples(self,
                          result_df: pd.DataFrame,
                          schema: Dict[str, Any],
                          encoded_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate using semantic samples from schema."""
        sample_checks = []

        validation_samples = schema.get('validation_samples', [])

        for sample in validation_samples:
            expected_row = sample.get('expected_row', {})
            if not expected_row:
                continue

            check = {
                'check': 'semantic_sample',
                'description': sample.get('description', 'Sample validation'),
                'expected': expected_row,
                'passed': False,
                'actual': None,
                'error': None
            }

            try:
                # Build query conditions
                conditions = pd.Series([True] * len(result_df))

                for col, expected_val in expected_row.items():
                    if col in result_df.columns:
                        if pd.isna(expected_val):
                            conditions &= result_df[col].isna()
                        else:
                            # Handle numeric comparison with tolerance
                            if isinstance(expected_val, (int, float)):
                                conditions &= (
                                        (result_df[col] == expected_val) |
                                        (abs(result_df[col] - expected_val) < 0.01)
                                )
                            else:
                                conditions &= (result_df[col] == expected_val)

                matching_rows = result_df[conditions]

                if len(matching_rows) > 0:
                    check['passed'] = True
                    check['actual'] = matching_rows.iloc[0].to_dict()
                else:
                    check['error'] = "No matching row found"
                    # Try to find partial matches for debugging
                    for col, val in expected_row.items():
                        if col in result_df.columns:
                            matches = result_df[result_df[col] == val] if not pd.isna(val) else result_df[result_df[col].isna()]
                            if len(matches) == 0:
                                check['error'] += f"; Column '{col}'={val} not found"
                                break

            except Exception as e:
                check['error'] = str(e)

            sample_checks.append(check)

        return sample_checks

    def _generate_structured_feedback(self,
                                      validation_result: Dict[str, Any],
                                      result_df: pd.DataFrame,
                                      source_df: pd.DataFrame,
                                      strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Generate structured feedback for code regeneration."""
        feedback = {
            'status': 'VALIDATION_FAILED',
            'issues': [],
            'debugging_info': {}
        }

        # Analyze each failed check
        for check in validation_result.get('checks_performed', []):
            if not check.get('passed', True):
                issue = {
                    'check_type': check.get('check'),
                    'details': check,
                    'diagnosis': None,
                    'suggested_fix': None
                }

                if check['check'] == 'row_count':
                    expected = check.get('expected', 0)
                    actual = check.get('actual', 0)
                    if actual < expected:
                        issue['diagnosis'] = f"Lost {expected - actual} rows during transformation"
                        issue['suggested_fix'] = "Check: 1) data_start_row is correct, 2) row filtering isn't too aggressive, 3) merge/join operations preserve rows"
                    else:
                        issue['diagnosis'] = f"Created {actual - expected} extra rows"
                        issue['suggested_fix'] = "Check: 1) unpivot created duplicates, 2) merge created cartesian product"

                elif check['check'] == 'columns_match':
                    missing = check.get('missing', [])
                    if missing:
                        issue['diagnosis'] = f"Missing columns: {missing}"
                        issue['suggested_fix'] = f"Ensure these columns are created: {missing}"

                elif check['check'] == 'semantic_sample':
                    issue['diagnosis'] = f"Value mismatch: {check.get('error', 'unknown')}"
                    issue['suggested_fix'] = "Check column mapping and data extraction logic"

                feedback['issues'].append(issue)

        # Add debugging info
        feedback['debugging_info'] = {
            'result_shape': result_df.shape,
            'result_columns': list(result_df.columns),
            'result_sample': result_df.head(5).to_dict() if len(result_df) > 0 else {},
            'source_shape': source_df.shape
        }

        return feedback

    def _regenerate_code_with_feedback(self,
                                       encoded_data: Dict[str, Any],
                                       structure_analysis: Dict[str, Any],
                                       schema: Dict[str, Any],
                                       strategy: Dict[str, Any],
                                       previous_code: str,
                                       feedback: Dict[str, Any]) -> str:
        """Regenerate code using structured feedback."""
        logger.info("Regenerating code with feedback...")

        prompt = f"""The previous transformation code failed validation. Fix it based on the feedback.

## PREVIOUS CODE
```python
{previous_code}
```

## VALIDATION FEEDBACK
{json.dumps(feedback, indent=2, ensure_ascii=False, default=str)}

## ISSUES TO FIX
"""
        for issue in feedback.get('issues', []):
            prompt += f"""
### Issue: {issue.get('check_type')}
- Diagnosis: {issue.get('diagnosis')}
- Suggested Fix: {issue.get('suggested_fix')}
"""

        prompt += f"""

## ORIGINAL STRATEGY (for reference)
{json.dumps(strategy.get('transformation_steps', []), indent=2, ensure_ascii=False)}

## TARGET COLUMNS
{[col['name'] for col in schema.get('target_columns', [])]}

## DEBUGGING INFO
- Result had shape: {feedback.get('debugging_info', {}).get('result_shape')}
- Result columns: {feedback.get('debugging_info', {}).get('result_columns')}

Fix the code to address all issues. Output ONLY the corrected Python code."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_code_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                # temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens
            )

            code = response.choices[0].message.content
            code = self._extract_code(code)
            return code

        except Exception as e:
            logger.error(f"Error regenerating code: {e}")
            return previous_code

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response."""
        response_text = response_text.strip()

        if response_text.startswith('```'):
            lines = response_text.split('\n')
            response_text = '\n'.join(lines[1:-1])
            if response_text.startswith('json'):
                response_text = response_text[4:].strip()

        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            import re
            match = re.search(r'\{[\s\S]*\}', response_text)
            if match:
                try:
                    return json.loads(match.group())
                except:
                    pass
            raise ValueError("Could not extract valid JSON from LLM response")