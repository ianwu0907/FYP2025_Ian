"""
Table Normalizer Pipeline (Enhanced)
Orchestrates the complete normalization process using semantic reasoning.

Pipeline stages:
1. Encoding: Compress spreadsheet to LLM-friendly representation
2. Structure Analysis: Semantic understanding of table structure
3. Schema Estimation: Derive ideal tidy schema
4. Transformation: Two-stage (Strategy → Code) generation and execution
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

from ..encoder import SpreadsheetEncoder
from ..analyzer import StructureAnalyzer
from ..estimator import SchemaEstimator
from ..generator import TransformationGenerator

logger = logging.getLogger(__name__)


class TableNormalizer:
    """
    Main pipeline that orchestrates the complete table normalization process.

    Uses LLM semantic reasoning throughout:
    - Structure analysis identifies patterns through understanding, not matching
    - Schema estimation derives ideal tidy format based on tidy data principles
    - Transformation uses free-form strategy generation
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the normalizer pipeline."""
        self.config = config

        # Initialize all components
        logger.info("Initializing Table Normalizer Pipeline (Enhanced)...")
        logger.info("  Using LLM semantic reasoning for generalized normalization")

        self.encoder = SpreadsheetEncoder(config.get('encoder', {}))
        self.analyzer = StructureAnalyzer(config.get('analyzer', {}) | config.get('llm', {}))
        self.estimator = SchemaEstimator(config.get('estimator', {}) | config.get('llm', {}))
        self.generator = TransformationGenerator(config.get('generator', {}) | config.get('llm', {}))

        # Output configuration
        self.output_config = config.get('output', {})
        self.save_intermediate = config.get('logging', {}).get('save_intermediate', True)
        self.output_dir = Path(config.get('logging', {}).get('output_dir', 'output'))
        self.output_dir.mkdir(exist_ok=True)

        logger.info("Pipeline initialized successfully")

    def normalize(self, input_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete normalization pipeline.

        Args:
            input_file: Path to input spreadsheet (.xlsx or .csv)
            output_file: Path to output file (default: output/normalized_output.csv)

        Returns:
            Dict containing:
            - normalized_df: The transformed DataFrame
            - output_path: Path to saved output
            - pipeline_log: Execution log
            - intermediate_results: All intermediate outputs
        """
        start_time = time.time()
        logger.info(f"Starting normalization pipeline for: {input_file}")

        pipeline_log = {
            'input_file': input_file,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stages': {}
        }

        try:
            # ================================================================
            # Stage 1: ENCODING
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 1: SPREADSHEET ENCODING")
            logger.info("=" * 80)

            encoded_data = self._run_encoding(input_file)
            pipeline_log['stages']['encoding'] = {
                'status': 'success',
                'shape': encoded_data['original_shape'],
                'compression_ratio': encoded_data['metadata'].get('compression_ratio', 0)
            }

            if self.save_intermediate:
                self._save_intermediate('01_encoded_data.json', {
                    'metadata': encoded_data['metadata'],
                    'encoded_text_preview': encoded_data['encoded_text'][:2000]
                })

            # ================================================================
            # Stage 2: STRUCTURE ANALYSIS (LLM Semantic Reasoning)
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: STRUCTURE ANALYSIS (Semantic Reasoning)")
            logger.info("=" * 80)

            structure_analysis = self._run_structure_analysis(encoded_data)
            pipeline_log['stages']['structure_analysis'] = {
                'status': 'success',
                'has_bilingual': structure_analysis.get('row_patterns', {}).get('has_bilingual_rows', False),
                'has_section_markers': structure_analysis.get('row_patterns', {}).get('has_section_markers', False),
                'has_implicit_aggregation': structure_analysis.get('implicit_aggregation', {}).get('has_implicit_aggregation', False),
                'transformation_complexity': structure_analysis.get('transformation_complexity', 'unknown')
            }

            if self.save_intermediate:
                self._save_intermediate('02_structure_analysis.json', structure_analysis)

            # ================================================================
            # Stage 3: SCHEMA ESTIMATION (Tidy Data Principles)
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3: SCHEMA ESTIMATION (Tidy Data Principles)")
            logger.info("=" * 80)

            schema = self._run_schema_estimation(encoded_data, structure_analysis)
            pipeline_log['stages']['schema_estimation'] = {
                'status': 'success',
                'observation_unit': schema.get('observation_unit', {}).get('description', 'unknown'),
                'num_target_columns': len(schema.get('target_columns', [])),
                'expected_rows': schema.get('expected_output', {}).get('row_count_estimate', 0)
            }

            if self.save_intermediate:
                self._save_intermediate('03_schema.json', schema)

            # ================================================================
            # Stage 4: TRANSFORMATION (Strategy → Code → Execute → Validate)
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 4: TRANSFORMATION (Two-Stage Generation)")
            logger.info("=" * 80)

            transformation_result = self._run_transformation(
                encoded_data, structure_analysis, schema
            )
            pipeline_log['stages']['transformation'] = {
                'status': 'success',
                'output_shape': transformation_result['normalized_df'].shape,
                'validation': transformation_result['validation_result'],
                'attempts': transformation_result.get('attempts', 1)
            }

            if self.save_intermediate:
                self._save_intermediate('04_transformation_strategy.json',
                                        transformation_result.get('transformation_strategy', {}))
                self._save_intermediate('05_transformation_code.py',
                                        transformation_result['transformation_code'],
                                        is_text=True)

            # ================================================================
            # Stage 5: SAVE OUTPUT
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 5: SAVING OUTPUT")
            logger.info("=" * 80)

            if output_file is None:
                output_file = self.output_dir / 'normalized_output.csv'

            output_path = self._save_output(transformation_result['normalized_df'], output_file)
            pipeline_log['output_file'] = str(output_path)

            # Calculate execution time
            elapsed_time = time.time() - start_time
            pipeline_log['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            pipeline_log['elapsed_seconds'] = round(elapsed_time, 2)
            pipeline_log['status'] = 'success'

            # Print summary
            self._print_summary(pipeline_log, transformation_result, encoded_data)

            # Save pipeline log
            if self.save_intermediate:
                self._save_intermediate('pipeline_log.json', pipeline_log)

            return {
                'normalized_df': transformation_result['normalized_df'],
                'output_path': output_path,
                'pipeline_log': pipeline_log,
                'intermediate_results': {
                    'encoded_data': encoded_data,
                    'structure_analysis': structure_analysis,
                    'schema': schema,
                    'transformation': transformation_result
                }
            }

        except Exception as e:
            logger.error(f"\n{'=' * 80}")
            logger.error(f"PIPELINE FAILED: {e}")
            logger.error(f"{'=' * 80}\n")

            pipeline_log['status'] = 'failed'
            pipeline_log['error'] = str(e)
            pipeline_log['elapsed_seconds'] = round(time.time() - start_time, 2)

            if self.save_intermediate:
                self._save_intermediate('pipeline_log.json', pipeline_log)

            raise

    def _run_encoding(self, input_file: str) -> Dict[str, Any]:
        """Run the encoding stage."""
        df = self.encoder.load_file(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")

        encoded_data = self.encoder.encode(df)
        logger.info(f"Encoding complete. Compression: {encoded_data['metadata'].get('compression_ratio', 0):.2f}x")

        return encoded_data

    def _run_structure_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the structure analysis stage with LLM semantic reasoning."""
        analysis = self.analyzer.analyze(encoded_data)

        # Log key findings
        sem = analysis.get('semantic_understanding', {})
        logger.info(f"Data description: {sem.get('data_description', 'Unknown')[:100]}")

        row_patterns = analysis.get('row_patterns', {})
        if row_patterns.get('has_bilingual_rows'):
            bilingual = row_patterns.get('bilingual_details', {})
            logger.info(f"Bilingual pattern: {bilingual.get('data_relationship', 'unknown')}")

        if row_patterns.get('has_section_markers'):
            markers = row_patterns.get('section_marker_details', {})
            logger.info(f"Section markers: {markers.get('marker_values', [])}")

        implicit_agg = analysis.get('implicit_aggregation', {})
        if implicit_agg.get('has_implicit_aggregation'):
            logger.info(f"Implicit aggregation detected: {len(implicit_agg.get('aggregation_hierarchies', []))} hierarchies")
            logger.info(f"  Summary rows to exclude: {len(implicit_agg.get('summary_rows', []))}")

        logger.info(f"Transformation complexity: {analysis.get('transformation_complexity', 'unknown')}")

        return analysis

    def _run_schema_estimation(self,
                               encoded_data: Dict[str, Any],
                               structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run the schema estimation stage based on tidy data principles."""
        schema = self.estimator.estimate_schema(encoded_data, structure_analysis)

        # Log target schema
        obs_unit = schema.get('observation_unit', {})
        logger.info(f"Observation unit: {obs_unit.get('description', 'Unknown')}")

        target_cols = schema.get('target_columns', [])
        logger.info(f"Target columns ({len(target_cols)}):")
        for col in target_cols:
            dim_marker = "[DIM]" if col.get('is_dimension') else "[VAL]"
            logger.info(f"  {dim_marker} {col.get('name')} ({col.get('data_type')})")

        expected = schema.get('expected_output', {})
        logger.info(f"Expected output: {expected.get('row_count_estimate', '?')} rows")

        return schema

    def _run_transformation(self,
                            encoded_data: Dict[str, Any],
                            structure_analysis: Dict[str, Any],
                            schema: Dict[str, Any]) -> Dict[str, Any]:
        """Run the two-stage transformation generation and execution."""
        result = self.generator.generate_and_execute(
            encoded_data, structure_analysis, schema
        )

        # Log results
        logger.info(f"Transformation completed in {result.get('attempts', 1)} attempt(s)")
        logger.info(f"Output shape: {result['normalized_df'].shape}")

        validation = result['validation_result']
        if validation['is_valid']:
            logger.info("✓ Validation PASSED")
        else:
            logger.warning(f"✗ Validation issues: {validation.get('errors', [])}")

        if validation.get('warnings'):
            for warning in validation['warnings']:
                logger.warning(f"  Warning: {warning}")

        return result

    def _save_output(self, df: pd.DataFrame, output_file: str) -> Path:
        """Save the normalized DataFrame to file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_format = self.output_config.get('format', 'csv')
        encoding = self.output_config.get('encoding', 'utf-8')
        include_index = self.output_config.get('index', False)

        if output_format == 'csv' or output_path.suffix == '.csv':
            df.to_csv(output_path, index=include_index, encoding=encoding)
        elif output_format == 'xlsx' or output_path.suffix == '.xlsx':
            df.to_excel(output_path, index=include_index, engine='openpyxl')
        else:
            df.to_csv(output_path, index=include_index, encoding=encoding)

        logger.info(f"Saved output to: {output_path}")
        return output_path

    def _save_intermediate(self, filename: str, data: Any, is_text: bool = False):
        """Save intermediate results to the output directory."""
        try:
            filepath = self.output_dir / filename

            if is_text:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(data)
            else:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False, default=str)

            logger.debug(f"Saved intermediate result: {filename}")
        except Exception as e:
            logger.warning(f"Failed to save intermediate result {filename}: {e}")

    def _print_summary(self,
                       pipeline_log: Dict[str, Any],
                       transformation_result: Dict[str, Any],
                       encoded_data: Dict[str, Any]):
        """Print final pipeline summary."""
        logger.info("\n" + "=" * 80)
        logger.info("NORMALIZATION SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Status: {pipeline_log.get('status', 'unknown').upper()}")
        logger.info(f"Input shape: {encoded_data['original_shape']}")
        logger.info(f"Output shape: {transformation_result['normalized_df'].shape}")
        logger.info(f"Compression: {encoded_data['metadata'].get('compression_ratio', 0):.2f}x")
        logger.info(f"Output file: {pipeline_log.get('output_file', 'N/A')}")
        logger.info(f"Execution time: {pipeline_log.get('elapsed_seconds', 0)} seconds")
        logger.info(f"Transformation attempts: {transformation_result.get('attempts', 1)}")

        validation = transformation_result['validation_result']
        logger.info(f"Validation: {'PASSED' if validation['is_valid'] else 'FAILED'}")

        if validation.get('warnings'):
            logger.info(f"Warnings: {len(validation['warnings'])}")
            for w in validation['warnings'][:3]:
                logger.info(f"  - {w}")

        logger.info("=" * 80 + "\n")
