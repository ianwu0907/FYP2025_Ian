"""
Table Normalizer Pipeline (Redesigned)
Orchestrates the complete normalization process.

Pipeline stages:
1. Encoding:                Compress spreadsheet to LLM-friendly representation
2. Irregularity Detection:  Detect structural irregularities (physical + LLM)
3. Schema Estimation:       Design target tidy schema (LLM + guidance)
4. Transformation:          Generate code, execute, validate, retry
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import json
import time

from ..encoder import SpreadsheetEncoder
from ..detector import IrregularityDetector
from ..estimator import SchemaEstimator
from ..generator import TransformationGenerator
from ..metrics import tidiness_metrics

logger = logging.getLogger(__name__)


class TableNormalizer:
    """
    Main pipeline that orchestrates the complete table normalization process.

    Flow:
      Encoder → IrregularityDetector → SchemaEstimator → TransformationGenerator
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the normalizer pipeline."""
        self.config = config

        logger.info("Initializing Table Normalizer Pipeline...")

        self.encoder = SpreadsheetEncoder(config.get('encoder', {}))
        self.detector = IrregularityDetector(config.get('detector', {}) | config.get('llm', {}))
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

            # --- Compute BEFORE tidiness metrics on raw input ---
            logger.info("\n" + "-" * 60)
            logger.info("TIDINESS METRICS [BEFORE]")
            logger.info("-" * 60)
            raw_df = encoded_data['dataframe']
            metrics_before = tidiness_metrics.compute_all_metrics(
                raw_df, label="BEFORE"
            )
            pipeline_log['metrics_before'] = metrics_before
            if self.save_intermediate:
                self._save_intermediate('metrics_before.json', metrics_before)

            # ================================================================
            # Stage 2: IRREGULARITY DETECTION
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: IRREGULARITY DETECTION")
            logger.info("=" * 80)

            detection_result = self._run_detection(encoded_data)
            pipeline_log['stages']['detection'] = {
                'status': 'success',
                'irregularities': detection_result['labels'],
                'num_irregularities': len(detection_result['labels']),
            }

            if self.save_intermediate:
                self._save_intermediate('02_detection_result.json', {
                    'physical': detection_result['physical'],
                    'irregularities': detection_result['irregularities'],
                    'labels': detection_result['labels'],
                })

            # ================================================================
            # Stage 3: SCHEMA ESTIMATION
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3: SCHEMA ESTIMATION")
            logger.info("=" * 80)

            schema = self._run_schema_estimation(encoded_data, detection_result)
            pipeline_log['stages']['schema_estimation'] = {
                'status': 'success',
                'observation_unit': schema.get('observation_unit', {}).get('description', 'unknown'),
                'num_target_columns': len(schema.get('target_columns', [])),
                'expected_rows': schema.get('expected_output', {}).get('row_count_estimate', 0)
            }

            if self.save_intermediate:
                # Remove non-serializable fields before saving
                schema_to_save = {k: v for k, v in schema.items()
                                  if k not in ('detection_result',)}
                self._save_intermediate('03_schema.json', schema_to_save)

            # ================================================================
            # Stage 4: TRANSFORMATION
            # ================================================================
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 4: TRANSFORMATION")
            logger.info("=" * 80)

            transformation_result = self._run_transformation(
                encoded_data, detection_result, schema
            )
            pipeline_log['stages']['transformation'] = {
                'status': 'success',
                'output_shape': transformation_result['normalized_df'].shape,
                'validation': transformation_result['validation_result'],
                'attempts': transformation_result.get('attempts', 1)
            }

            if self.save_intermediate:
                self._save_intermediate('04_transformation_code.py',
                                        transformation_result['transformation_code'],
                                        is_text=True)

            # --- Compute AFTER tidiness metrics on transformed output ---
            logger.info("\n" + "-" * 60)
            logger.info("TIDINESS METRICS [AFTER]")
            logger.info("-" * 60)
            result_df = transformation_result['normalized_df']

            # Use schema target columns as dimension columns for NMI
            dim_cols_for_nmi = [
                c["name"] for c in schema.get("target_columns", [])
                if c.get("role") == "dimension"
                   or c.get("is_dimension", False)
                   or c.get("data_type") == "string"
            ]
            # Filter to columns that actually exist in result
            dim_cols_for_nmi = [
                c for c in dim_cols_for_nmi if c in result_df.columns
            ]

            metrics_after = tidiness_metrics.compute_all_metrics(
                result_df,
                dim_cols=dim_cols_for_nmi if dim_cols_for_nmi else None,
                label="AFTER"
            )
            pipeline_log['metrics_after'] = metrics_after

            # --- Compare BEFORE vs AFTER ---
            logger.info("\n" + "=" * 65)
            metrics_comparison = tidiness_metrics.compare_metrics(
                metrics_before, metrics_after
            )
            tidiness_metrics.log_comparison(metrics_comparison)
            logger.info("=" * 65)

            pipeline_log['metrics_comparison'] = metrics_comparison
            if self.save_intermediate:
                self._save_intermediate('metrics_after.json', metrics_after)
                self._save_intermediate('metrics_comparison.json',
                                        metrics_comparison)

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
            self._print_summary(pipeline_log, transformation_result, encoded_data,
                                detection_result)

            # Save pipeline log
            if self.save_intermediate:
                self._save_intermediate('pipeline_log.json', pipeline_log)

            return {
                'normalized_df': transformation_result['normalized_df'],
                'output_path': output_path,
                'pipeline_log': pipeline_log,
                'metrics': {
                    'before': metrics_before,
                    'after': metrics_after,
                    'comparison': metrics_comparison,
                },
                'intermediate_results': {
                    'encoded_data': encoded_data,
                    'detection_result': detection_result,
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

    # ==================================================================
    # Stage runners
    # ==================================================================

    def _run_encoding(self, input_file: str) -> Dict[str, Any]:
        """Run the encoding stage."""
        df = self.encoder.load_file(input_file)
        logger.info(f"Loaded data with shape: {df.shape}")

        encoded_data = self.encoder.encode(df)
        logger.info(f"Encoding complete. Compression: "
                    f"{encoded_data['metadata'].get('compression_ratio', 0):.2f}x")

        return encoded_data

    def _run_detection(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run irregularity detection."""
        detection_result = self.detector.detect(encoded_data)

        labels = detection_result['labels']
        logger.info(f"Detected {len(labels)} irregularities: {labels}")

        for ir in detection_result['irregularities']:
            logger.info(f"  {ir['label']}: {ir.get('evidence', '')[:80]}")

        return detection_result

    def _run_schema_estimation(self,
                               encoded_data: Dict[str, Any],
                               detection_result: Dict[str, Any]) -> Dict[str, Any]:
        """Run schema estimation."""
        schema = self.estimator.estimate_schema(encoded_data, detection_result)

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
                            detection_result: Dict[str, Any],
                            schema: Dict[str, Any]) -> Dict[str, Any]:
        """Run transformation generation and execution."""
        result = self.generator.generate_and_execute(
            encoded_data, detection_result, schema
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

    # ==================================================================
    # Output
    # ==================================================================

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
                       encoded_data: Dict[str, Any],
                       detection_result: Dict[str, Any]):
        """Print final pipeline summary."""
        logger.info("\n" + "=" * 80)
        logger.info("NORMALIZATION SUMMARY")
        logger.info("=" * 80)

        logger.info(f"Status: {pipeline_log.get('status', 'unknown').upper()}")
        logger.info(f"Input shape: {encoded_data['original_shape']}")
        logger.info(f"Output shape: {transformation_result['normalized_df'].shape}")
        logger.info(f"Compression: {encoded_data['metadata'].get('compression_ratio', 0):.2f}x")
        logger.info(f"Irregularities: {detection_result['labels']}")
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