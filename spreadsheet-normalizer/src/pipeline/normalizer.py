"""
Table Normalizer Pipeline
Orchestrates the complete normalization process
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
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the normalizer pipeline."""
        self.config = config

        # Initialize all components
        logger.info("Initializing Table Normalizer Pipeline...")
        logger.info("Using SpreadsheetLLM Encoder")

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
        """Execute the complete normalization pipeline."""
        start_time = time.time()
        logger.info(f"Starting normalization pipeline for: {input_file}")

        pipeline_log = {
            'input_file': input_file,
            'start_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'stages': {}
        }

        try:
            # Stage 1: Encoding
            logger.info("\n" + "="*80)
            logger.info("STAGE 1: SPREADSHEET ENCODING")
            logger.info("="*80)

            encoded_data = self._run_encoding(input_file)
            pipeline_log['stages']['encoding'] = {
                'status': 'success',
                'shape': encoded_data['original_shape'],
                'compression_ratio': encoded_data['metadata'].get('compression_ratio', 0)
            }

            if self.save_intermediate:
                self._save_intermediate('encoded_data.json', {
                    'metadata': encoded_data['metadata'],
                    'encoded_text': encoded_data['encoded_text'][:1000]
                })

            # Stage 2: Structure Analysis
            logger.info("\n" + "="*80)
            logger.info("STAGE 2: STRUCTURE ANALYSIS")
            logger.info("="*80)

            structure_analysis = self._run_analysis(encoded_data)
            pipeline_log['stages']['analysis'] = {
                'status': 'success',
                'structure_type': structure_analysis.get('structure_type'),
                'confidence': structure_analysis.get('confidence')
            }

            if self.save_intermediate:
                self._save_intermediate('structure_analysis.json', structure_analysis)

            # Stage 3: Schema Estimation
            logger.info("\n" + "="*80)
            logger.info("STAGE 3: SCHEMA ESTIMATION")
            logger.info("="*80)

            schema = self._run_schema_estimation(encoded_data, structure_analysis)
            pipeline_log['stages']['schema'] = {
                'status': 'success',
                'num_columns': len(schema.get('expected_output_columns', []))
            }

            if self.save_intermediate:
                self._save_intermediate('schema.json', schema)

            # Stage 4: Transformation Generation and Execution
            logger.info("\n" + "="*80)
            logger.info("STAGE 4: TRANSFORMATION GENERATION & EXECUTION")
            logger.info("="*80)

            transformation_result = self._run_transformation(
                encoded_data, structure_analysis, schema
            )
            pipeline_log['stages']['transformation'] = {
                'status': 'success',
                'output_shape': transformation_result['normalized_df'].shape,
                'validation': transformation_result['validation_result']
            }

            if self.save_intermediate:
                self._save_intermediate('transformation_code.py',
                                        transformation_result['transformation_code'],
                                        is_text=True)

            # Stage 5: Save Output
            logger.info("\n" + "="*80)
            logger.info("STAGE 5: SAVING OUTPUT")
            logger.info("="*80)

            if output_file is None:
                output_file = self.output_dir / 'normalized_output.csv'

            output_path = self._save_output(transformation_result['normalized_df'], output_file)
            pipeline_log['output_file'] = str(output_path)

            # Calculate execution time
            elapsed_time = time.time() - start_time
            pipeline_log['end_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
            pipeline_log['elapsed_seconds'] = round(elapsed_time, 2)

            logger.info(f"\n{'='*80}")
            logger.info(f"PIPELINE COMPLETED SUCCESSFULLY in {elapsed_time:.2f} seconds")
            logger.info(f"Output saved to: {output_path}")
            logger.info(f"{'='*80}\n")

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
            logger.error(f"\n{'='*80}")
            logger.error(f"PIPELINE FAILED: {e}")
            logger.error(f"{'='*80}\n")

            pipeline_log['status'] = 'failed'
            pipeline_log['error'] = str(e)

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

    def _run_analysis(self, encoded_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run the structure analysis stage."""
        analysis = self.analyzer.analyze(encoded_data)

        logger.info(f"Structure type: {analysis.get('structure_type')}")
        logger.info(f"Confidence: {analysis.get('confidence')}")
        logger.info(f"Header rows: {analysis.get('header_rows')}")

        return analysis

    def _run_schema_estimation(self,
                               encoded_data: Dict[str, Any],
                               structure_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Run the schema estimation stage."""
        schema = self.estimator.estimate_schema(encoded_data, structure_analysis)

        logger.info(f"Proposed columns: {schema.get('expected_output_columns')}")

        return schema

    def _run_transformation(self,
                            encoded_data: Dict[str, Any],
                            structure_analysis: Dict[str, Any],
                            schema: Dict[str, Any]) -> Dict[str, Any]:
        """Run the transformation generation and execution stage."""
        result = self.generator.generate_and_execute(
            encoded_data, structure_analysis, schema
        )

        logger.info(f"Transformation completed")
        logger.info(f"Output shape: {result['normalized_df'].shape}")
        logger.info(f"Validation: {result['validation_result']['is_valid']}")

        if result['validation_result']['warnings']:
            logger.warning(f"Warnings: {result['validation_result']['warnings']}")

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