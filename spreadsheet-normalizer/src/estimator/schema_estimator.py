"""
Schema Estimator Module - Hybrid Approach (BEST PRACTICE)
Combines visual table context with precise metadata for optimal schema analysis
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
    Hybrid schema estimator:
    - Uses table samples for VISUAL understanding and pattern recognition
    - Uses metadata for PRECISE values, statistics, and edge cases
    - Balances intuition with accuracy
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the schema estimator with configuration."""
        self.config = config
        self.standardize_names = config.get('standardize_names', True)
        self.detect_types = config.get('detect_types', True)
        self.merge_similar_columns = config.get('merge_similar_columns', False)

        # 🔥 支持多种 LLM 提供商
        self.llm_provider = config.get('llm_provider', 'openai')
        
        if self.llm_provider == 'qwen':
            api_key = os.getenv('DASHSCOPE_API_KEY')
            if not api_key:
                raise ValueError("DASHSCOPE_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            self.model = config.get('model', 'qwen-max')
            
        elif self.llm_provider == 'gemini':
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv('GEMINI_BASE_URL', 'https://oa.api2d.net/v1')
            )
            self.model = config.get('model', 'gemini-2.0-flash-exp')
            
        elif self.llm_provider == 'claude':
            api_key = os.getenv('API2D_API_KEY')
            if not api_key:
                raise ValueError("API2D_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv('API2D_BASE_URL', 'https://oa.api2d.net/v1')
            )
            self.model = config.get('model', 'claude-3-5-sonnet-20241022')
            
        elif self.llm_provider == 'deepseek':
            api_key = os.getenv('DEEPSEEK_API_KEY')
            if not api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            
            self.client = OpenAI(
                api_key=api_key,
                base_url=os.getenv('DEEPSEEK_BASE_URL', 'https://api.deepseek.com/v1')
            )
            self.model = config.get('model', 'deepseek-chat')
            
        else:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")

            self.client = OpenAI(api_key=api_key)
            self.model = config.get('model', os.getenv('OPENAI_MODEL', 'gpt-4o-mini'))

        # LLM settings
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4000)
        
        logger.info(f"Initialized SchemaEstimator (Hybrid) with {self.llm_provider} ({self.model})")

    def estimate_schema(self,
                        encoded_data: Dict[str, Any],
                        structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate schema using hybrid approach: visual + metadata."""
        logger.info("Estimating normalized schema with HYBRID approach...")

        # Verify we have both dataframe and metadata
        if 'dataframe' not in encoded_data:
            raise ValueError("encoded_data must contain 'dataframe'")
        
        metadata = encoded_data.get('metadata', {})
        if 'columns' not in metadata:
            logger.warning("Missing 'columns' in metadata. Will use basic metadata only.")

        # Create hybrid prompt
        prompt = self._create_hybrid_prompt(encoded_data, structure_analysis)

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
            return self._get_default_schema(encoded_data, structure_analysis)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for hybrid schema estimation."""
        return """You are an expert data engineer specializing in database schema design and normalization.

Your approach:
1. Look at the TABLE STRUCTURE to understand patterns, relationships, and hierarchies
2. Use METADATA for precise values, frequencies, and edge cases
3. Design practical, normalized schemas that balance clarity and completeness

Be observant, precise, and practical. Output valid JSON only."""

    def _create_hybrid_prompt(self,
                              encoded_data: Dict[str, Any],
                              structure_analysis: Dict[str, Any]) -> str:
        """
        Create hybrid prompt combining visual table context with precise metadata.
        
        STRATEGY:
        1. Show actual table data (first 30 rows) for pattern recognition
        2. Provide detailed metadata for precision and edge cases
        3. Highlight structural issues from analysis
        4. Give clear, step-by-step instructions
        """
        metadata = encoded_data['metadata']
        df = encoded_data['dataframe']
        
        # === PART 1: VISUAL CONTEXT (Pattern Recognition) ===
        sample_rows = min(30, len(df))
        sample_data = df.head(sample_rows).to_string(max_colwidth=50, max_rows=sample_rows)
        
        # Also show last few rows to catch aggregates
        if len(df) > sample_rows:
            last_rows = df.tail(5).to_string(max_colwidth=50, max_rows=5)
        else:
            last_rows = ""
        
        # === PART 2: METADATA (Precision) ===
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
            "pattern_type": "composite_column | implicit_aggregation | bilingual | mixed_types | etc",
            "affected_columns": ["column1", "column2"],
            "description": "What you observed in the table",
            "evidence": "Reference to specific rows/values in the sample",
            "solution": "How you'll handle it"
        }}
    ],
    
    "column_schema": [
        {{
            "original_column": "exact column name from input",
            "normalized_name": "clean_snake_case_name",
            "data_type": "boolean | boolean_with_na | categorical | integer | numeric | date | string | identifier",
            "operation": "keep | split | merge | drop | convert",
            "description": "Brief description",
            "reasoning": "Why this transformation"
        }}
    ],
    
    "split_operations": [
        {{
            "source_column": "exact column name",
            "split_pattern": "exact delimiter (e.g., ' - ', '/', etc)",
            "target_columns": ["new_col1", "new_col2"],
            "logic": "How to split (e.g., 'split on delimiter, take first/second part')",
            "evidence": "Cite metadata: 'Delimiter appears in X% of values'"
        }}
    ],
    
    "type_conversions": [
        {{
            "column": "column name",
            "source_type": "current pandas dtype",
            "target_type": "boolean_with_na | categorical | etc",
            "value_mapping": {{
                "actual_value_from_metadata_1": mapped_value_1,
                "actual_value_from_metadata_2": mapped_value_2
            }},
            "handle_nulls": "strategy for null/NA values",
            "edge_cases": ["list any special cases from metadata"]
        }}
    ],
    
    "aggregation_handling": {{
        "has_implicit_aggregation": true/false,
        "strategy": "split_tables | filter_aggregates | none",
        "affected_rows": "row indices or description",
        "reasoning": "Why this approach"
    }},
    
    "normalization_plan": [
        "Step 1: Remove/flag aggregate rows if detected",
        "Step 2: Split composite columns based on evidence",
        "Step 3: Convert types using exact value mappings",
        "Step 4: Rename columns to snake_case",
        "Step 5: Validate output schema"
    ],
    
    "expected_output_columns": ["complete", "list", "of", "final", "column", "names"],
    
    "reasoning": "Overall explanation of your design decisions, referencing both the table structure you saw and the metadata precision"
}}

═══════════════════════════════════════════════════════════════
QUALITY CHECKLIST
═══════════════════════════════════════════════════════════════

Before outputting, verify:
✓ Did you look at the actual table rows to understand patterns?
✓ Did you use EXACT unique values from metadata (not guessed)?
✓ For splits, did you cite the delimiter frequency from metadata?
✓ For type conversions, did you create value_mapping using actual values?
✓ Did you address implicit aggregation if detected?
✓ Is your output column count correct? (original ± splits ± drops)

Output ONLY the JSON object, no additional commentary.
"""

        return prompt

    def _build_focused_metadata(self, columns_metadata: Dict[str, Any], df) -> str:
        """
        Build focused metadata view emphasizing actionable information.
        Less verbose than full metadata dump.
        """
        lines = []
        
        for col_name, col_meta in columns_metadata.items():
            lines.append(f"\n### `{col_name}`")
            
            # Core info
            inferred_type = col_meta.get('inferred_type', 'unknown')
            unique_count = col_meta.get('unique_count', 0)
            null_pct = col_meta.get('null_percentage', 0)
            
            lines.append(f"- **Inferred type**: {inferred_type}")
            lines.append(f"- **Cardinality**: {unique_count} unique values ({null_pct:.1f}% null)")
            
            # Unique values (most important for type conversion)
            unique_values = col_meta.get('unique_values', [])
            if unique_values:
                if len(unique_values) <= 10:
                    lines.append(f"- **All unique values**: {unique_values}")
                else:
                    lines.append(f"- **Top unique values**: {unique_values[:10]}")
                    lines.append(f"  (showing 10 of {len(unique_values)})")
            
            # Value frequency (for categorical/boolean)
            value_counts = col_meta.get('value_counts', {})
            if value_counts and len(value_counts) <= 20:
                lines.append(f"- **Value distribution**:")
                for val, count in list(value_counts.items())[:10]:
                    pct = (count / len(df)) * 100
                    lines.append(f"  - `{val}`: {count} ({pct:.1f}%)")
                if len(value_counts) > 10:
                    lines.append(f"  - ... and {len(value_counts)-10} more")
            
            # Delimiters (critical for splitting)
            delimiters = col_meta.get('potential_delimiters', [])
            if delimiters:
                lines.append(f"- **🔥 Potential delimiters detected:**")
                for delim_info in delimiters[:3]:
                    delim = delim_info['delimiter']
                    pct = delim_info['percentage']
                    consistent = delim_info.get('is_consistent', False)
                    lines.append(f"  - `{repr(delim)}`: appears in {pct:.0f}% of values")
                    if consistent:
                        lines.append(f"    ✓ Split pattern is consistent")
                    if delim_info.get('sample_split'):
                        lines.append(f"    Example: {delim_info['sample_split']}")
            
            # Bilingual flag
            if col_meta.get('has_bilingual_content'):
                lines.append(f"- **🌐 Contains bilingual content** (mixed scripts)")
            
            # Statistics for numeric
            stats = col_meta.get('statistics', {})
            if stats and inferred_type in ['integer', 'numeric']:
                lines.append(
                    f"- **Range**: {stats.get('min')} to {stats.get('max')} "
                    f"(mean: {stats.get('mean', 0):.2f})"
                )
        
        return '\n'.join(lines)

    def _build_basic_metadata_from_df(self, df) -> str:
        """Fallback: Build basic metadata directly from dataframe if no enhanced metadata."""
        lines = ["\n⚠️  Using basic metadata (enhanced metadata not available)\n"]
        
        for col in df.columns:
            lines.append(f"\n### `{col}`")
            lines.append(f"- Type: {df[col].dtype}")
            lines.append(f"- Unique: {df[col].nunique()}")
            lines.append(f"- Nulls: {df[col].isnull().sum()} ({df[col].isnull().mean()*100:.1f}%)")
            
            # Show unique values if few
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) <= 10:
                lines.append(f"- Unique values: {list(unique_vals)}")
            else:
                lines.append(f"- Sample values: {list(unique_vals[:5])}")
        
        return '\n'.join(lines)

    def _summarize_structure_analysis(self, structure_analysis: Dict[str, Any]) -> str:
        """Summarize key findings from structure analysis."""
        lines = []
        
        structure_type = structure_analysis.get('structure_type', 'unknown')
        lines.append(f"**Structure Type**: {structure_type}")
        
        header_rows = structure_analysis.get('header_rows', [])
        if header_rows:
            lines.append(f"**Header Rows**: {header_rows}")
        
        data_rows = structure_analysis.get('data_rows', [])
        if isinstance(data_rows, list) and data_rows:
            if len(data_rows) > 10:
                lines.append(f"**Data Rows**: {len(data_rows)} rows (indices {min(data_rows)}-{max(data_rows)})")
            else:
                lines.append(f"**Data Rows**: {data_rows}")
        elif isinstance(data_rows, str):
            lines.append(f"**Data Rows**: {data_rows}")
        
        metadata_rows = structure_analysis.get('metadata_rows', [])
        if metadata_rows:
            lines.append(f"**Metadata Rows**: {metadata_rows} (titles, notes, etc.)")
        
        aggregate_rows = structure_analysis.get('aggregate_rows', [])
        if aggregate_rows:
            lines.append(f"**⚠️  Explicit Aggregate Rows**: {aggregate_rows} (totals, subtotals)")
        
        return '\n'.join(lines)

    def _format_implicit_aggregation(self, structure_analysis: Dict[str, Any]) -> str:
        """Format implicit aggregation information if detected."""
        implicit_agg = structure_analysis.get('implicit_aggregation', {})
        
        if not implicit_agg.get('has_implicit_aggregation'):
            return ""
        
        lines = ["\n🔥 **IMPLICIT AGGREGATION DETECTED** 🔥\n"]
        lines.append("This table mixes summary rows and detail rows!\n")
        
        hierarchies = implicit_agg.get('aggregation_hierarchies', [])
        if hierarchies:
            lines.append("**Detected Hierarchies:**")
            for h in hierarchies:
                summary_cat = h.get('summary_category', '')
                detail_cat = h.get('detail_category', '')
                extra_dim = h.get('additional_dimension', '')
                lines.append(f"- `{summary_cat}` (summary)")
                lines.append(f"  └→ `{detail_cat}` (adds dimension: {extra_dim})")
        
        summary_rows = implicit_agg.get('summary_rows', [])
        detail_rows = implicit_agg.get('detail_rows', [])
        
        if summary_rows:
            lines.append(f"\n**Summary rows**: {len(summary_rows)} rows")
            lines.append(f"  Indices: {summary_rows[:10]}{'...' if len(summary_rows) > 10 else ''}")
        
        if detail_rows:
            lines.append(f"**Detail rows**: {len(detail_rows)} rows")
            lines.append(f"  Indices: {detail_rows[:10]}{'...' if len(detail_rows) > 10 else ''}")
        
        lines.append("\n**⚠️  You MUST address this in your normalization plan:**")
        lines.append("  Option 1: Split into separate tables (summary table + detail table)")
        lines.append("  Option 2: Add a 'level' column and filter out summary rows")
        lines.append("  Option 3: Keep only detail rows if they're complete")
        
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
        """Validate and enrich the schema result."""
        
        # Ensure all required fields exist
        required_fields = {
            'identified_patterns': [],
            'column_schema': [],
            'split_operations': [],
            'type_conversions': [],
            'aggregation_handling': {},
            'normalization_plan': ['Apply default normalization'],
            'expected_output_columns': [],
            'reasoning': ''
        }
        
        for field, default in required_fields.items():
            if field not in schema:
                schema[field] = default
        
        # Build expected output columns if missing
        if not schema['expected_output_columns']:
            output_cols = []
            for col_def in schema.get('column_schema', []):
                if col_def.get('operation') != 'drop':
                    if 'normalized_name' in col_def:
                        output_cols.append(col_def['normalized_name'])
            
            # Add split columns
            for split_op in schema.get('split_operations', []):
                output_cols.extend(split_op.get('target_columns', []))
            
            schema['expected_output_columns'] = output_cols
        
        # Validate column names (snake_case)
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
        
        # Add reference to source metadata
        schema['source_metadata'] = encoded_data.get('metadata', {})
        
        return schema

    def _get_default_schema(self,
                            encoded_data: Dict[str, Any],
                            structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Return a default schema if LLM fails."""
        metadata = encoded_data.get('metadata', {})
        df = encoded_data.get('dataframe')
        
        if df is not None:
            original_columns = df.columns.tolist()
        else:
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
                'reasoning': 'Default: keep as-is'
            })

        return {
            'identified_patterns': [],
            'column_schema': column_schema,
            'split_operations': [],
            'type_conversions': [],
            'aggregation_handling': {'has_implicit_aggregation': False, 'strategy': 'none'},
            'normalization_plan': [
                'Keep all columns as-is',
                'Standardize column names to snake_case',
                'Remove special characters'
            ],
            'expected_output_columns': [col['normalized_name'] for col in column_schema],
            'reasoning': 'Default schema due to LLM error - manual review recommended',
            'source_metadata': metadata
        }