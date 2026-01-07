"""
Structure Analyzer Module (Enhanced)
Analyzes table structure using LLM semantic reasoning to identify:
- Header structure (single/multi-level)
- Data regions
- Row patterns (bilingual, grouped by year, etc.)
- Column patterns (dimensions vs values)
- Implicit aggregation hierarchies
"""

import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import os
import pandas as pd
import re
import numpy as np
logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """
    Analyzes spreadsheet structure using LLM semantic reasoning.

    Key principle: Let LLM understand the SEMANTIC structure, not pattern match.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the structure analyzer with configuration."""
        self.config = config
        self.detect_implicit_aggregates = config.get('detect_implicit_aggregates', True)
        self.use_color_detection = config.get('use_color_detection', True)
        self.color_threshold = config.get('color_threshold', 0.1)  # 降低阈值
        self.min_colored_cells = config.get('min_colored_cells', 1)  # 至少 1 个

        logger.info(
            f"Color detection: {'enabled' if self.use_color_detection else 'disabled'}"
        )
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
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable not set")
            
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
        self.max_tokens = config.get('max_tokens', 3000)
        self.max_tokens = config.get('max_tokens', 4000)
        
        logger.info(f"Initialized StructureAnalyzer with {self.llm_provider} ({self.model})")

    def analyze(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure of the encoded spreadsheet using LLM semantic reasoning.

        Returns comprehensive structure analysis including:
        - semantic_understanding: What the data represents
        - header_structure: How headers are organized
        - data_region: Where data starts/ends
        - row_patterns: Patterns in row organization
        - column_patterns: Patterns in column organization
        - implicit_aggregation: Detected aggregation hierarchies (preserved feature)
        """
        logger.info("Analyzing table structure with LLM semantic reasoning...")

        df = encoded_data['dataframe']

        # Step 1: LLM-based semantic structure analysis
        llm_analysis = self._analyze_structure_with_llm(encoded_data)

        # Step 2: Detect implicit aggregation (preserved original feature)
        # This is a valuable deterministic check that complements LLM analysis
        if self.detect_implicit_aggregates:
            implicit_agg = self._detect_implicit_aggregation(df, llm_analysis)
            llm_analysis['implicit_aggregation'] = implicit_agg

            # If implicit aggregation detected, add to special patterns
            if implicit_agg.get('has_implicit_aggregation'):
                if 'special_patterns' not in llm_analysis:
                    llm_analysis['special_patterns'] = []
                llm_analysis['special_patterns'].append({
                    'pattern_type': 'implicit_aggregation',
                    'description': f"Detected {len(implicit_agg.get('aggregation_hierarchies', []))} aggregation hierarchies where summary rows coexist with detail rows",
                    'hierarchies': implicit_agg.get('aggregation_hierarchies', [])
                })

        logger.info("Structure analysis complete")
        return llm_analysis
            """分析表格结构 - 包含颜色检测"""
            logger.info("Analyzing table structure with LLM...")
            
            df = encoded_data['dataframe']
            
            # 🔥 颜色检测
            color_based_aggregation = []
            highlighted_rows = []
            color_info = {}
            
            if self.use_color_detection:
                try:
                    logger.info("Starting color-based detection...")
                    
                    # 提取颜色信息
                    color_info = self._extract_color_info(encoded_data)
                    
                    if color_info:
                        # 检测高亮行
                        highlighted_rows = self._detect_highlighted_rows(df, color_info)
                        
                        # 基于颜色检测聚合行
                        if highlighted_rows:
                            color_based_aggregation = self._detect_aggregation_by_color(
                                df, highlighted_rows
                            )
                            logger.info(
                                f"Color detection found {len(color_based_aggregation)} "
                                f"potential aggregation rows"
                            )
                    else:
                        logger.info("No color highlighting found in data")
                        
                except Exception as e:
                    logger.warning(f"Color detection failed: {e}")
                    import traceback
                    logger.debug(traceback.format_exc())
            
            # Step 1: LLM 分析
            llm_analysis = self._analyze_with_llm(encoded_data)
            
            # Step 2: 合并颜色检测的结果
            if color_based_aggregation:
                # 将颜色检测到的聚合行添加到 LLM 结果中
                existing_agg = set(llm_analysis.get('aggregate_rows', []))
                combined_agg = list(existing_agg | set(color_based_aggregation))
                llm_analysis['aggregate_rows'] = sorted(combined_agg)
                
                logger.info(
                    f"Combined aggregation rows: {len(existing_agg)} (LLM) + "
                    f"{len(color_based_aggregation)} (color) = {len(combined_agg)} (total)"
                )
                
                # 添加颜色分析信息到结果
                llm_analysis['color_analysis'] = {
                    'enabled': True,
                    'highlighted_rows': highlighted_rows,
                    'color_detected_aggregation': color_based_aggregation,
                    'total_highlighted': len(highlighted_rows),
                    'total_aggregation': len(color_based_aggregation)
                }
            else:
                llm_analysis['color_analysis'] = {
                    'enabled': self.use_color_detection,
                    'highlighted_rows': [],
                    'color_detected_aggregation': [],
                    'message': 'No color-based aggregation detected'
                }
            
            # Step 3: 隐式聚合检测
            if self.detect_implicit_aggregates:
                implicit_agg = self._detect_implicit_aggregation(df, llm_analysis)
                llm_analysis['implicit_aggregation'] = implicit_agg
            
            logger.info("Structure analysis complete")
            return llm_analysis

    def _analyze_structure_with_llm(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to perform semantic structure analysis."""
        prompt = self._create_structure_analysis_prompt(encoded_data)

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

            # Parse JSON response
            analysis_result = self._parse_llm_response(result_text)

            # Validate and set defaults
            analysis_result = self._validate_analysis(analysis_result, encoded_data)

            return analysis_result

        except Exception as e:
            logger.error(f"Error in LLM structure analysis: {e}")
            return self._get_default_analysis(encoded_data)

    def _get_system_prompt(self) -> str:
        """System prompt for structure analysis."""
        return """You are an expert data analyst specializing in understanding spreadsheet structures.

Your task is to perform SEMANTIC analysis of spreadsheets - understanding what the data represents and how it's organized, not just mechanical pattern matching.

## TIDY DATA PRINCIPLES (Your Guide)
A tidy dataset has:
1. Each variable forms a column
2. Each observation forms a row
3. Each type of observational unit forms a table

Most spreadsheets are NOT tidy. Your job is to understand their structure so they can be transformed into tidy format.

## COMMON MESSY PATTERNS TO IDENTIFY
1. Column headers that are actually VALUES (e.g., years 2011, 2016, 2021 as column headers)
2. Multiple variables stored in one column
3. Variables stored in both rows AND columns
4. Multiple types of data interleaved (e.g., counts and percentages in alternating rows)
5. Multi-level headers spanning multiple rows
6. Bilingual content (Chinese/English pairs)
7. Section markers (e.g., year appearing once to mark a group of rows)
8. Implicit aggregation (totals mixed with breakdowns)

Be thorough and precise. Output valid JSON only."""

    def _create_structure_analysis_prompt(self, encoded_data: Dict[str, Any]) -> str:
        """Create prompt for structure analysis."""
        df = encoded_data['dataframe']
        encoded_text = encoded_data.get('encoded_text', '')[:3000]

        # Create detailed view of first 30 rows
        rows_preview = self._create_rows_preview(df, max_rows=30)

        # Get column statistics
        col_stats = self._get_column_statistics(df)

        prompt = f"""Analyze the semantic structure of this spreadsheet.

## SPREADSHEET OVERVIEW
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Column names from pandas: {df.columns.tolist()}

## DETAILED ROW CONTENTS (first 30 rows)
{rows_preview}

## COLUMN STATISTICS
{col_stats}

## ENCODED REPRESENTATION (SpreadsheetLLM format)
{encoded_text}

## YOUR ANALYSIS TASK

Analyze this spreadsheet and identify:

1. **Semantic Understanding**: What does this data represent? What is being measured?

2. **Header Structure**: 
   - Which rows contain headers?
   - Are there multi-level headers (headers spanning multiple rows)?
   - What does each header level represent?

3. **Data Region**:
   - Where does the actual data start (row index)?
   - Where does it end?
   - Are there any footer/note rows to exclude?

4. **Row Patterns**:
   - Are there bilingual rows (Chinese/English pairs)?
   - If bilingual, do both rows contain the SAME data (translation) or DIFFERENT data (e.g., counts vs percentages)?
   - Are there section markers (e.g., year appearing to mark a group)?
   - Are there total/subtotal rows mixed with detail rows?

5. **Column Patterns**:
   - Which columns are ID/dimension columns (should remain as columns in tidy format)?
   - Which columns contain values that should be unpivoted?
   - Are column headers actually data values (e.g., years, age groups)?

6. **Special Patterns**:
   - Any implicit aggregation (totals coexisting with breakdowns)?
   - Any merged cells or spanning headers?
   - Any other structural quirks?

## OUTPUT FORMAT (JSON)

{{
  "semantic_understanding": {{
    "data_description": "What this spreadsheet contains",
    "observation_unit": "What constitutes one observation in tidy format",
    "key_variables": ["List of variables/dimensions identified"]
  }},
  
  "header_structure": {{
    "header_rows": [list of row indices that are headers],
    "num_levels": 1,
    "level_details": [
      {{
        "level": 0,
        "row_index": 0,
        "semantic_meaning": "What this header level represents",
        "values_found": ["sample values"]
      }}
    ],
    "column_header_mapping": {{
      "description": "How column positions map to header values",
      "id_columns": [{{\"col_index\": 0, \"semantic\": \"year\"}}],
      "value_columns": [{{\"col_indices\": [3,4,5], \"header_values\": [\"<15\", \"15-24\", \"25-34\"], \"semantic\": \"age_group\"}}]
    }}
  }},
  
  "data_region": {{
    "start_row": 7,
    "end_row": 49,
    "notes": "Any notes about the data region"
  }},
  
  "row_patterns": {{
    "has_bilingual_rows": true,
    "bilingual_details": {{
      "pattern": "alternating",
      "cn_rows": "even indices starting from data_start",
      "en_rows": "odd indices",
      "data_relationship": "SAME_DATA_TRANSLATION | DIFFERENT_DATA_TYPES",
      "if_different_types": {{
        "cn_row_contains": "counts",
        "en_row_contains": "percentages"
      }}
    }},
    "has_section_markers": true,
    "section_marker_details": {{
      "marker_column": 0,
      "marker_rows": [7],
      "marker_values": [2011],
      "semantic": "year"
    }},
    "has_total_rows": false,
    "total_row_indices": []
  }},
  
  "column_patterns": {{
    "id_columns": [
      {{"col_index": 0, "name": "Year", "notes": "Section marker, needs forward fill"}},
      {{"col_index": 1, "name": "Ethnicity", "notes": "Bilingual"}}
    ],
    "value_columns": [
      {{"col_indices": [3,4,5,6,7,8,9], "represents": "age_group", "header_row": 4}}
    ],
    "aggregate_columns": [
      {{"col_index": 10, "type": "total", "should_exclude": true}}
    ],
    "metadata_columns": [
      {{"col_index": 11, "name": "median_age", "notes": "Only in CN rows"}}
    ]
  }},
  
  "special_patterns": [
    {{
      "pattern_type": "name_of_pattern",
      "description": "Detailed description",
      "affected_rows_or_cols": "specifics"
    }}
  ],
  
  "transformation_complexity": "low | medium | high",
  "transformation_notes": "Key challenges for transforming this to tidy format"
}}

Output ONLY the JSON object, no other text."""

        return prompt

    def _create_rows_preview(self, df: pd.DataFrame, max_rows: int = 30) -> str:
        """Create detailed preview of rows for LLM analysis."""
        lines = []
        for i in range(min(max_rows, len(df))):
            row_content = []
            for j in range(len(df.columns)):
                val = df.iloc[i, j]
                if pd.notna(val) and str(val).strip():
                    val_str = str(val)[:50]  # Truncate long values
                    row_content.append(f"[{j}]={repr(val_str)}")

            if row_content:
                lines.append(f"Row {i}: {', '.join(row_content)}")
            else:
                lines.append(f"Row {i}: (empty)")

        if len(df) > max_rows:
            lines.append(f"... ({len(df) - max_rows} more rows)")

        return "\n".join(lines)

    def _get_column_statistics(self, df: pd.DataFrame) -> str:
        """Get statistics for each column."""
        lines = []
        for j, col in enumerate(df.columns):
            non_null = df.iloc[:, j].dropna()
            if len(non_null) == 0:
                lines.append(f"Col {j}: All empty")
                continue

            # Check data types
            sample_vals = non_null.head(5).tolist()
            numeric_count = sum(1 for v in non_null if isinstance(v, (int, float)))
            string_count = len(non_null) - numeric_count

            # Check for Chinese characters
            has_chinese = any(
                any('\u4e00' <= c <= '\u9fff' for c in str(v))
                for v in non_null.head(10)
            )

            lines.append(
                f"Col {j}: {len(non_null)} non-null, "
                f"numeric={numeric_count}, string={string_count}, "
                f"has_chinese={has_chinese}, "
                f"samples={sample_vals[:3]}"
            )

        return "\n".join(lines)

    def _detect_implicit_aggregation(self,
                                     df: pd.DataFrame,
                                     llm_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect implicit aggregation by checking category hierarchy and numerical verification.

        PRESERVED ORIGINAL FEATURE: This deterministic check complements LLM analysis.
        """
        logger.info("Detecting implicit aggregation patterns...")

        result = {
            'has_implicit_aggregation': False,
            'aggregation_hierarchies': [],
            'summary_rows': [],
            'detail_rows': []
        }

        if len(df) < 2:
            return result

        # Try to identify category column
        category_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if any(keyword in col_str for keyword in ['category', '類別', 'type', '項目']):
                category_col = col
                break

        # Also check from LLM analysis
        if not category_col and 'column_patterns' in llm_analysis:
            id_cols = llm_analysis.get('column_patterns', {}).get('id_columns', [])
            for id_col in id_cols:
                if 'category' in str(id_col.get('name', '')).lower():
                    col_idx = id_col.get('col_index')
                    if col_idx is not None and col_idx < len(df.columns):
                        category_col = df.columns[col_idx]
                        break

        if not category_col:
            logger.debug("No category column found for implicit aggregation detection")
            return result

        # Get unique categories
        categories = df[category_col].dropna().astype(str).unique()
        logger.debug(f"Found {len(categories)} unique categories in column '{category_col}'")

        if len(categories) < 2:
            return result

        # Find category pairs where one contains the other
        hierarchies = []

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                if i >= j:
                    continue

                cat1_str = str(cat1).strip()
                cat2_str = str(cat2).strip()

                # Check if one is substring of other (hierarchical relationship)
                if cat1_str in cat2_str and len(cat2_str) > len(cat1_str):
                    summary_cat = cat1_str
                    detail_cat = cat2_str
                elif cat2_str in cat1_str and len(cat1_str) > len(cat2_str):
                    summary_cat = cat2_str
                    detail_cat = cat1_str
                else:
                    continue

                # Get row indices for each category
                summary_indices = df[df[category_col].astype(str) == summary_cat].index.tolist()
                detail_indices = df[df[category_col].astype(str) == detail_cat].index.tolist()

                if len(summary_indices) > 0 and len(detail_indices) > 0:
                    if len(detail_indices) > len(summary_indices):
                        extra_dimension = detail_cat.replace(summary_cat, '').strip().lstrip('及').strip()

                        hierarchies.append({
                            'summary_category': summary_cat,
                            'detail_category': detail_cat,
                            'additional_dimension': extra_dimension,
                            'summary_row_count': len(summary_indices),
                            'detail_row_count': len(detail_indices)
                        })

                        result['summary_rows'].extend(summary_indices)
                        result['detail_rows'].extend(detail_indices)

                        logger.info(f"Found hierarchy: '{summary_cat}' → '{detail_cat}'")

        if hierarchies:
            result['summary_rows'] = sorted(list(set(result['summary_rows'])))
            result['detail_rows'] = sorted(list(set(result['detail_rows'])))
            result['has_implicit_aggregation'] = True
            result['aggregation_hierarchies'] = hierarchies

            logger.info(f"DETECTED: {len(hierarchies)} implicit aggregation hierarchies")
        else:
            logger.info("No implicit aggregation detected")

        return result

    def _verify_aggregation_relationship(self,
                                         df: pd.DataFrame,
                                         category_col: str,
                                         value_col: str,
                                         item_col: Optional[str],
                                         summary_category: str,
                                         detail_category: str) -> Optional[Dict[str, Any]]:
        """
        Verify if summary_category rows are aggregates of detail_category rows.
        Returns hierarchy info if verified, None otherwise.
        """
        summary_df = df[df[category_col] == summary_category].copy()
        detail_df = df[df[category_col] == detail_category].copy()

        if len(summary_df) == 0 or len(detail_df) == 0:
            return None

        # If we have item columns, verify item by item
        if item_col:
            verified_items = []

            for item in summary_df[item_col].unique():
                summary_rows = summary_df[summary_df[item_col] == item]
                if len(summary_rows) == 0:
                    continue

                summary_value = pd.to_numeric(summary_rows[value_col], errors='coerce').sum()

                # Find matching detail rows
                # The detail item might have additional suffixes (e.g., "身體虐待 - 男性")
                item_base = str(item).split(' - ')[0]
                detail_matches = detail_df[
                    detail_df[item_col].astype(str).str.contains(item_base, na=False, regex=False)
                ]

                if len(detail_matches) == 0:
                    continue

                detail_value = pd.to_numeric(detail_matches[value_col], errors='coerce').sum()

                # Check if values match (allow small rounding error)
                if abs(summary_value - detail_value) < 1:
                    verified_items.append({
                        'item': str(item),
                        'summary_value': float(summary_value),
                        'detail_value': float(detail_value),
                        'detail_count': len(detail_matches)
                    })

            # If we verified at least half of the items, consider it a valid hierarchy
            if len(verified_items) >= len(summary_df) * 0.5:
                # Extract the additional dimension
                extra_dimension = detail_category.replace(summary_category, '').strip()
                extra_dimension = extra_dimension.lstrip('及').strip()

                return {
                    'summary_category': summary_category,
                    'detail_category': detail_category,
                    'additional_dimension': extra_dimension,
                    'verified_items': verified_items[:5],  # Only keep first 5 for logging
                    'total_verified': len(verified_items),
                    'verification_rate': len(verified_items) / len(summary_df)
                }

        return None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for structure analysis."""
        return """You are an expert data analyst specializing in spreadsheet structure analysis.

Your task is to analyze the physical structure of tables and identify different types of rows:
- Header rows (column names)
- Data rows (actual data)
- Metadata rows (titles, dates, notes, descriptions)
- Aggregate rows (totals, subtotals, sums)
- Column groupings and hierarchies

Be precise and analytical. Output valid JSON only."""

    def _create_analysis_prompt(self, encoded_data: Dict[str, Any]) -> str:
        """Create the analysis prompt for the LLM."""
        metadata = encoded_data.get('metadata', {})
        encoded_text = encoded_data.get('encoded_text', '')[:1500]
        df = encoded_data['dataframe']

        # Get row and column counts from dataframe directly
        row_count = len(df)
        column_count = len(df.columns)
        column_names = df.columns.tolist()

        # Get sample of first and last few rows
        first_rows = df.head(10).to_string() if len(df) > 0 else "No data"

        prompt = f"""Analyze the structure of this spreadsheet table.

## Table Metadata:
- Total rows: {row_count}
- Total columns: {column_count}
- Column names: {column_names}

## Sample Data (first 10 rows):
{first_rows}

## Encoded Representation:
{encoded_text}

## Your Analysis Task:

1. **Identify Header Rows**: Which row(s) contain column headers?
   - Look for descriptive text, mixed languages, or field names
   - May be multiple rows for multi-level headers

2. **Identify Metadata Rows**: Which rows are NOT data (titles, dates, notes)?
   - Usually at the top
   - Contain descriptive text, dates, or documentation

3. **Identify Data Rows**: Which rows contain actual data?
   - Regular pattern of values
   - Numeric or categorical data

4. **Identify Aggregate Rows**: Which rows are totals/subtotals?
   - Look for keywords: "Total", "Sum", "小計", "總計", "合計"
   - Usually have special formatting or position

5. **Detect Implicit Aggregation**: 
   - Are there category fields with different levels of detail?
   - If one category name contains another, it might indicate hierarchy
   - NOTE: You'll analyze semantically; numerical verification will be done separately

6. **Column Groupings**: Are columns organized into groups?
   - Merged headers
   - Parent-child relationships

7. **Structure Type**: What type of table is this?
   - "simple": Single header, straightforward rows
   - "multi_header": Multiple header rows
   - "pivot": Pivot table format
   - "hierarchical": Nested categories
   - "irregular": Unusual structure

## Output Format (JSON only):

{{
    "header_rows": [row_indices],
    "data_rows": ["start-end"],
    "metadata_rows": [row_indices],
    "aggregate_rows": [row_indices],
    "column_groups": [
        {{
            "parent_column": "column_name",
            "child_columns": ["col1", "col2", ...]
        }}
    ],
    "potential_category_hierarchy": [
        {{
            "broad_category": "category_name",
            "detailed_category": "category_name_with_more_dimensions",
            "reasoning": "why this might be hierarchical"
        }}
    ],
    "structure_type": "simple|multi_header|pivot|hierarchical|irregular",
    "confidence": 0.0-1.0,
    "recommendations": ["recommendation 1", "recommendation 2", ...]
}}

**CRITICAL**: 
- Row indices are 0-based (first row is 0)
- Use ranges like "9-1319" for continuous data rows
- Be conservative with aggregate_rows (only mark if clearly labeled)
- For potential_category_hierarchy, look for semantic relationships in category field names

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

    def _validate_analysis(self,
                           analysis: Dict[str, Any],
                           encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set defaults for analysis result."""
        df = encoded_data['dataframe']

        # Ensure required fields exist with defaults
        defaults = {
            'semantic_understanding': {
                'data_description': 'Unknown',
                'observation_unit': 'Unknown',
                'key_variables': []
            },
            'header_structure': {
                'header_rows': [0],
                'num_levels': 1,
                'level_details': [],
                'column_header_mapping': {}
            },
            'data_region': {
                'start_row': 1,
                'end_row': len(df) - 1,
                'notes': ''
            },
            'row_patterns': {
                'has_bilingual_rows': False,
                'has_section_markers': False,
                'has_total_rows': False
            },
            'column_patterns': {
                'id_columns': [],
                'value_columns': [],
                'aggregate_columns': [],
                'metadata_columns': []
            },
            'special_patterns': [],
            'transformation_complexity': 'medium',
            'transformation_notes': ''
        }

        for key, default_value in defaults.items():
            if key not in analysis:
                analysis[key] = default_value
            elif isinstance(default_value, dict):
                for sub_key, sub_default in default_value.items():
                    if sub_key not in analysis[key]:
                        analysis[key][sub_key] = sub_default

        # Validate row indices are within bounds
        data_region = analysis.get('data_region', {})
        if data_region.get('start_row', 0) < 0:
            data_region['start_row'] = 0
        if data_region.get('end_row', 0) >= len(df):
            data_region['end_row'] = len(df) - 1

        return analysis

    def _get_default_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return default analysis if LLM fails."""
        df = encoded_data['dataframe']

        return {
            'header_rows': [0] if row_count > 0 else [],
            'data_rows': ['1-' + str(row_count-1)] if row_count > 1 else [],
            'metadata_rows': [],
            'aggregate_rows': [],
            'column_groups': [],
            'potential_category_hierarchy': [],
            'structure_type': 'simple',
            'confidence': 0.3,
            'recommendations': ['Default analysis used due to LLM error']
        }
    
    def _excel_to_index(self, cell_ref: str) -> Optional[Tuple[int, int]]:
        """将 Excel 坐标转换为 (row, col) 索引"""
        match = re.match(r'([A-Z]+)(\d+)', cell_ref.upper())
        if not match:
            return None
        
        col_letters, row_num = match.groups()
        
        col = 0
        for char in col_letters:
            col = col * 26 + (ord(char) - ord('A') + 1)
        col -= 1
        
        row = int(row_num) - 1
        return (row, col)

    def _parse_cell_range(self, range_ref: str) -> List[Tuple[int, int]]:
        """解析单元格范围"""
        if ':' not in range_ref:
            cell = self._excel_to_index(range_ref)
            return [cell] if cell else []
        
        start_ref, end_ref = range_ref.split(':')
        start = self._excel_to_index(start_ref)
        end = self._excel_to_index(end_ref)
        
        if not start or not end:
            return []
        
        start_row, start_col = start
        end_row, end_col = end
        
        cells = []
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                cells.append((row, col))
        
        return cells
    
    def _extract_color_info(self, encoded_data: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
        """从 encoded_data 中提取颜色信息"""
        color_map = {}
        
        # 获取格式信息
        formats = encoded_data.get('formats', {})
        df = encoded_data['dataframe']
        
        # 🔥 详细调试
        logger.info(f"=== Color Detection Debug ===")
        logger.info(f"Encoded data keys: {list(encoded_data.keys())}")
        logger.info(f"Formats field exists: {'formats' in encoded_data}")
        logger.info(f"Formats type: {type(formats)}")
        logger.info(f"Formats count: {len(formats) if formats else 0}")
        
        if not formats:
            logger.info("⚠️ No format information available - formats field is empty or missing")
            return color_map
        
        logger.info(f"✅ Processing {len(formats)} format groups")
        
        # 显示前 2 个格式样例
        for i, (key, value) in enumerate(list(formats.items())[:2]):
            logger.info(f"Sample format {i+1}:")
            logger.info(f"  Key (first 200 chars): {key[:200]}")
            logger.info(f"  Cells: {value}")
        
        # 统计每行的高亮单元格
        row_highlight_count = {}
        row_format_info = {}
        
        highlight_count = 0  # 统计找到多少个 highlight 格式
        
        # 遍历每个格式组
        for format_json_str, cell_list in formats.items():
            try:
                # 解析格式 JSON 字符串
                format_info = json.loads(format_json_str)
                
                # 检查 _semantic 字段
                semantic = format_info.get('_semantic', 'plain')
                
                if semantic == 'highlight':
                    highlight_count += 1
                    logger.info(f"✅ Found highlight format #{highlight_count}: cells={cell_list}")
                    
                    # 获取颜色信息
                    fill = format_info.get('fill', {})
                    fg_color = fill.get('fg_color', 'none')
                    pattern_type = fill.get('pattern_type', 'none')
                    
                    font = format_info.get('font', {})
                    font_color = font.get('color', 'none')
                    bold = font.get('bold', False)
                    
                    # 🔥 添加详细调试
                    logger.info(f"  Processing {len(cell_list)} cell references...")
                    
                    # 遍历使用这种格式的单元格
                    for cell_ref in cell_list:
                        # 解析单元格范围
                        cells = self._parse_cell_range(cell_ref)
                        
                        # 🔥 添加这个日志
                        logger.info(f"  Cell ref '{cell_ref}' → {len(cells)} cells: {cells}")
                        
                        # 记录每个单元格
                        for row, col in cells:
                            # 🔥 添加这个日志
                            logger.debug(f"    Recording cell: row={row}, col={col}")
                            
                            if row not in row_highlight_count:
                                row_highlight_count[row] = 0
                                row_format_info[row] = {
                                    'colors': set(),
                                    'font_colors': set(),
                                    'bold': False
                                }
                            
                            row_highlight_count[row] += 1
                            row_format_info[row]['colors'].add(fg_color)
                            row_format_info[row]['font_colors'].add(font_color)
                            if bold:
                                row_format_info[row]['bold'] = True
                    
                    # 🔥 添加这个日志，显示当前的 row_highlight_count
                    logger.info(f"  Current row_highlight_count: {dict(row_highlight_count)}")
                else:
                    logger.debug(f"Skipping format with _semantic='{semantic}'")
            
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse format JSON: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing format group: {e}")
                import traceback
                logger.debug(traceback.format_exc())
                continue
        
        # 🔥 添加这个日志
        logger.info(f"Total row_highlight_count entries: {len(row_highlight_count)}")
        logger.info(f"row_highlight_count content: {dict(row_highlight_count)}")
        
        # 构建最终的 color_map
        total_cols = len(df.columns)
        
        for row_idx, count in row_highlight_count.items():
            # 🔥 添加这个日志
            logger.info(f"Checking row {row_idx}: count={count}, min_required={self.min_colored_cells}")
            
            if count >= self.min_colored_cells:
                format_details = row_format_info[row_idx]
                
                color_map[row_idx] = {
                    'has_background_color': True,
                    'colored_cells': count,
                    'total_cells': total_cols,
                    'color_ratio': count / total_cols,
                    'colors': list(format_details['colors']),
                    'font_colors': list(format_details['font_colors']),
                    'bold': format_details['bold']
                }
                
                logger.info(
                    f"✅ Added row {row_idx} to color_map: {count}/{total_cols} cells highlighted"
                )
            else:
                logger.info(f"❌ Skipped row {row_idx}: count ({count}) < min_required ({self.min_colored_cells})")
        
        logger.info(f"Total highlight formats found: {highlight_count}")
        logger.info(f"Found {len(color_map)} rows with color highlighting")
        logger.info(f"=== End Color Detection Debug ===")
        
        return color_map
    
    def _detect_highlighted_rows(self, 
                            df: pd.DataFrame,
                            color_info: Dict[int, Dict[str, Any]]) -> List[int]:
        """
        检测被高亮的行

        Args:
            df: DataFrame
            color_info: 颜色信息字典
            
        Returns:
            高亮行的索引列表
        """
        highlighted_rows = []

        for row_idx, info in color_info.items():
            # 🔥 检查颜色比例是否超过阈值
            if info['color_ratio'] >= self.color_threshold:
                highlighted_rows.append(row_idx)
                logger.debug(
                    f"Row {row_idx} is highlighted: "
                    f"{info['colored_cells']}/{info['total_cells']} cells "
                    f"({info['color_ratio']:.1%}), colors: {info['colors']}"
                )

        logger.info(f"Detected {len(highlighted_rows)} highlighted rows")
        return highlighted_rows
    
    def _detect_aggregation_by_color(self,
                                     df: pd.DataFrame,
                                     highlighted_rows: List[int]) -> List[int]:
        """
        通过颜色 + 数值特征检测聚合行
        
        检测策略：
        1. 高亮行 + 包含总计关键词 → 聚合行
        2. 高亮行 + 数值明显大于其他行 → 聚合行
        3. 高亮行 + 在表格底部 → 聚合行
        4. 高亮行 + 在表格顶部且有数值 → 聚合行
        """
        aggregation_rows = []
        
        # 获取所有数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        logger.debug(
            f"Checking {len(highlighted_rows)} highlighted rows for aggregation patterns"
        )
        
        for row_idx in highlighted_rows:
            is_aggregation = False
            reasons = []
            
            try:
                row = df.iloc[row_idx]
                
                # 检查 1: 是否包含总计关键词
                row_text = ' '.join([str(v) for v in row.values if pd.notna(v)])
                row_text_lower = row_text.lower()
                
                aggregation_keywords = [
                    'total', 'sum', 'subtotal', 'grand total', 'average', 'avg',
                    '总计', '小计', '合计', '總計', '汇总', '彙總', '总和',
                    '平均', '均值', 'suma', 'promedio'
                ]
                
                for keyword in aggregation_keywords:
                    if keyword in row_text_lower:
                        is_aggregation = True
                        reasons.append(f"contains keyword '{keyword}'")
                        break
                
                    # 检查 2: 数值是否明显较大
                # ============================================
                # 检查 2: 局部数值比较（🔥 改进）
                # ============================================
                if not is_aggregation and len(numeric_cols) > 0:
                    try:
                        row_sum = pd.to_numeric(row[numeric_cols], errors='coerce').sum()
                        
                        # 🔥 关键改进：只比较相邻的 N 行
                        window_size = 5  # 前后各5行
                        start_idx = max(0, row_idx - window_size)
                        end_idx = min(len(df), row_idx + window_size + 1)
                        
                        # 获取相邻行（排除当前行）
                        neighbor_indices = [i for i in range(start_idx, end_idx) if i != row_idx]
                        neighbor_rows = df.iloc[neighbor_indices]
                        
                        if len(neighbor_rows) > 0:
                            neighbor_sums = neighbor_rows[numeric_cols].apply(
                                lambda x: pd.to_numeric(x, errors='coerce')
                            ).sum(axis=1)
                            
                            avg_neighbor_sum = neighbor_sums.mean()
                            max_neighbor_sum = neighbor_sums.max()
                            
                            # 🔥 两个条件之一满足即可：
                            # 条件 A: 比相邻平均大很多（2倍）
                            # 条件 B: 比相邻最大值大（聚合了多行）
                            if pd.notna(row_sum) and pd.notna(avg_neighbor_sum):
                                if row_sum > avg_neighbor_sum * 1.5:
                                    is_aggregation = True
                                    reasons.append(
                                        f"local sum ({row_sum:.2f}) >> "
                                        f"neighbor avg ({avg_neighbor_sum:.2f})"
                                    )
                                elif pd.notna(max_neighbor_sum) and row_sum > max_neighbor_sum * 1.3:
                                    is_aggregation = True
                                    reasons.append(
                                        f"sum ({row_sum:.2f}) > "
                                        f"max neighbor ({max_neighbor_sum:.2f})"
                                    )
                    
                    except Exception as e:
                        logger.debug(f"Error in numerical check for row {row_idx}: {e}")

                if not is_aggregation:
                    # 聚合行通常在分类列为空或特殊值
                    # 例如：类别列是空的，只有数值列有值
                    non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
                    
                    if len(non_numeric_cols) > 0:
                        # 统计非数值列的空值比例
                        null_ratio = row[non_numeric_cols].isna().sum() / len(non_numeric_cols)
                        
                        # 如果非数值列大部分是空的，但数值列有值
                        if null_ratio > 0.5:  # 50% 以上的非数值列为空
                            if len(numeric_cols) > 0:
                                numeric_values = row[numeric_cols].notna().sum()
                                if numeric_values >= len(numeric_cols) * 0.3:  # 30% 数值列有值
                                    is_aggregation = True
                                    reasons.append(
                                        f"empty pattern: {null_ratio:.1%} non-numeric cols empty, "
                                        f"but has numeric values"
                                    )
                
                # 检查 3: 是否在表格底部（最后 10%）
                if not is_aggregation:
                    if row_idx >= len(df) * 0.9:
                        is_aggregation = True
                        reasons.append("near table bottom")
                
                # 检查 4: 是否在表格顶部（前 5 行）且包含数值
                if not is_aggregation and row_idx < 5:
                    if len(numeric_cols) > 0:
                        non_null_count = row[numeric_cols].notna().sum()
                        if non_null_count >= len(numeric_cols) * 0.5:
                            is_aggregation = True
                            reasons.append("near table top with numeric values")
                
                if is_aggregation:
                    aggregation_rows.append(row_idx)
                    logger.info(
                        f"Row {row_idx} identified as aggregation: {', '.join(reasons)}"
                    )
            
            except Exception as e:
                logger.warning(f"Error analyzing row {row_idx}: {e}")
        
        logger.info(
            f"Identified {len(aggregation_rows)} aggregation rows based on "
            f"color + patterns"
        )
        return aggregation_rows
            'semantic_understanding': {
                'data_description': 'Unable to analyze - using defaults',
                'observation_unit': 'Unknown',
                'key_variables': list(df.columns)
            },
            'header_structure': {
                'header_rows': [0],
                'num_levels': 1,
                'level_details': [],
                'column_header_mapping': {}
            },
            'data_region': {
                'start_row': 1,
                'end_row': len(df) - 1,
                'notes': 'Default analysis'
            },
            'row_patterns': {
                'has_bilingual_rows': False,
                'has_section_markers': False,
                'has_total_rows': False
            },
            'column_patterns': {
                'id_columns': [{'col_index': 0, 'name': str(df.columns[0])}] if len(df.columns) > 0 else [],
                'value_columns': [],
                'aggregate_columns': [],
                'metadata_columns': []
            },
            'special_patterns': [],
            'transformation_complexity': 'unknown',
            'transformation_notes': 'LLM analysis failed, using minimal defaults',
            'implicit_aggregation': {
                'has_implicit_aggregation': False,
                'aggregation_hierarchies': [],
                'summary_rows': [],
                'detail_rows': []
            }
        }
