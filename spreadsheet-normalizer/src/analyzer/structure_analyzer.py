"""
Structure Analyzer Module
Uses LLM to analyze the structure of spreadsheets and identify layout patterns
"""

import json
import logging
from typing import Dict, List, Any, Optional
from openai import OpenAI
import os

logger = logging.getLogger(__name__)


class StructureAnalyzer:
    """
    Analyzes the structure of encoded spreadsheets using LLM.
    Identifies headers, data rows, metadata, aggregates, and hierarchical structures.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the analyzer with configuration.

        Args:
            config: Configuration dictionary containing analyzer settings
        """
        self.config = config
        self.min_header_rows = config.get('min_header_rows', 1)
        self.max_header_rows = config.get('max_header_rows', 5)
        self.confidence_threshold = config.get('confidence_threshold', 0.7)

        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = OpenAI(api_key=api_key)
        self.model = os.getenv('OPENAI_MODEL', 'gpt-4o-mini')

        # LLM settings
        self.temperature = config.get('temperature', 0.1)
        self.max_tokens = config.get('max_tokens', 4000)

    def analyze(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze the structure of the encoded spreadsheet.

        Args:
            encoded_data: Output from SpreadsheetEncoder

        Returns:
            Dictionary containing structural analysis
        """
        logger.info("Analyzing table structure with LLM...")

        encoded_text = encoded_data['encoded_text']
        metadata = encoded_data['metadata']

        # Create analysis prompt
        prompt = self._create_analysis_prompt(encoded_text, metadata)

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
            analysis_result = self._parse_llm_response(result_text)

            # Validate and enrich the result
            analysis_result = self._validate_analysis(analysis_result, encoded_data)

            logger.info("Structure analysis complete")
            return analysis_result

        except Exception as e:
            logger.error(f"Error in structure analysis: {e}")
            # Return default analysis if LLM fails
            return self._get_default_analysis(encoded_data)

    def _get_system_prompt(self) -> str:
        """Get the system prompt for structure analysis."""
        return """You are an expert data analyst specializing in spreadsheet structure analysis.
Your task is to analyze the structure of messy, human-oriented spreadsheets and identify:
1. Header rows (column names, may be multi-level)
2. Data rows (actual observations/records)
3. Metadata rows (titles, descriptions, notes)
4. Aggregate rows (subtotals, totals, summaries embedded in data)
5. Hierarchical column groupings (if multi-level headers exist)

Focus on understanding the STRUCTURE and LAYOUT, not the content meaning.
Be precise and analytical. Output your analysis in valid JSON format."""

    def _create_analysis_prompt(self, encoded_text: str, metadata: Dict[str, Any]) -> str:
        """Create the analysis prompt for the LLM."""

        prompt = f"""Analyze the following spreadsheet structure:

                ## Table Preview:
                {encoded_text[:3000]}  
                
                ## Metadata:
                - Total rows: {metadata['num_rows']}
                - Total columns: {metadata['num_cols']}
                - Column names: {metadata['column_names']}
                - Compression ratio: {metadata.get('compression_ratio', 'N/A')}x
                - Detected issues: {metadata.get('potential_issues', [])}
                
                ## Your Task:
                Carefully analyze this table structure and identify:
                
                1. **Header Rows**: Which rows contain column headers?
                   - Are there multi-level headers?
                   - Do headers contain both native language and English names?
                
                2. **Data Rows**: Which rows contain actual data observations?
                   - Look for consistent data patterns
                
                3. **Metadata Rows**: Any rows containing titles, descriptions, or notes?
                
                4. **Aggregate Rows**: Any rows with subtotals, totals, or category summaries?
                
                5. **Column Hierarchy**: If there are multi-level headers, describe the hierarchy
                
                6. **Structure Type**: Classify the table structure:
                   - "simple": Single header row, flat structure
                   - "multi_header": Multiple header rows with hierarchy
                   - "mixed": Contains embedded aggregates or irregular structure
                   - "complex": Multiple issues requiring extensive restructuring
                
                7. **Normalization Recommendations**: What transformations are needed?
                
                ## Output Format:
                Provide your analysis as a valid JSON object with this structure:
                {{
                    "header_rows": [list of row indices, 0-based],
                    "data_rows": [list of row indices or range like "5-100"],
                    "metadata_rows": [list of row indices],
                    "aggregate_rows": [list of row indices],
                    "column_groups": [
                        {{
                            "parent_column": "column name or description",
                            "child_columns": ["list of columns under this parent"]
                        }}
                    ],
                    "structure_type": "simple|multi_header|mixed|complex",
                    "confidence": 0.0-1.0,
                    "recommendations": ["list of specific recommendations"],
                    "reasoning": "brief explanation of your analysis"
                }}
                
                Output ONLY the JSON object, no additional text."""

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
                for match in matches:
                    try:
                        return json.loads(match)
                    except:
                        continue
            raise ValueError("Could not extract valid JSON from LLM response")

    def _validate_analysis(self, analysis: Dict[str, Any], encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enrich the analysis result."""
        num_rows = encoded_data['metadata']['num_rows']

        # Ensure all required fields exist
        required_fields = ['header_rows', 'data_rows', 'structure_type', 'recommendations']
        for field in required_fields:
            if field not in analysis:
                logger.warning(f"Missing field in analysis: {field}")
                analysis[field] = [] if field != 'structure_type' else 'unknown'

        # Convert data_rows range notation to list if needed
        if isinstance(analysis.get('data_rows'), str):
            if '-' in analysis['data_rows']:
                start, end = map(int, analysis['data_rows'].split('-'))
                analysis['data_rows'] = list(range(start, min(end + 1, num_rows)))

        # Ensure indices are within bounds
        for field in ['header_rows', 'data_rows', 'metadata_rows', 'aggregate_rows']:
            if field in analysis and isinstance(analysis[field], list):
                analysis[field] = [idx for idx in analysis[field] if 0 <= idx < num_rows]

        # Add confidence score if missing
        if 'confidence' not in analysis:
            analysis['confidence'] = 0.5

        # Add default column groups if missing
        if 'column_groups' not in analysis:
            analysis['column_groups'] = []

        return analysis

    def _get_default_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Return a default analysis if LLM fails."""
        num_rows = encoded_data['metadata']['num_rows']

        return {
            'header_rows': [0],
            'data_rows': list(range(1, num_rows)),
            'metadata_rows': [],
            'aggregate_rows': [],
            'column_groups': [],
            'structure_type': 'simple',
            'confidence': 0.3,
            'recommendations': ['Manual review recommended - automatic analysis failed'],
            'reasoning': 'Default analysis due to LLM error'
        }