#!/usr/bin/env python3
"""
Spreadsheet Normalizer - Main Entry Point (Enhanced)

Uses LLM semantic reasoning for generalized table normalization.
Transforms messy spreadsheets into tidy format based on tidy data principles.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv
from typing import Any, Dict
from src.pipeline import TableNormalizer


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    # Reduce noise from libraries
    logging.getLogger('openai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('httpx').setLevel(logging.WARNING)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_file = Path(config_path)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def display_single_table_summary(result: dict, logger):
    """Display summary for single table result."""
    logger.info("\n" + "="*80)
    logger.info("NORMALIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Status: SUCCESS")
    logger.info(f"Input shape: {result['intermediate_results']['encoded_data']['original_shape']}")
    logger.info(f"Output shape: {result['normalized_df'].shape}")
    logger.info(f"Compression: {result['intermediate_results']['encoded_data']['metadata']['compression_ratio']:.2f}x")
    logger.info(f"Output file: {result['output_path']}")
    logger.info(f"Execution time: {result['pipeline_log']['elapsed_seconds']} seconds")
    logger.info("="*80 + "\n")

    # Display warnings
    validation = result['intermediate_results']['transformation']['validation_result']
    if validation['warnings']:
        logger.warning("Warnings:")
        for warning in validation['warnings']:
            logger.warning(f"  - {warning}")


def display_multi_table_summary(result: Dict[str, Any], logger):
    """Display summary for multi-table normalization results."""
    logger.info("\n" + "="*80)
    logger.info("MULTI-TABLE NORMALIZATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Status: SUCCESS")
    logger.info(f"Number of tables detected: {result['num_tables']}")
    logger.info(f"Processing mode: {result['pipeline_log']['processing_mode']}")
    logger.info(f"Execution time: {result['pipeline_log']['elapsed_seconds']} seconds")
    logger.info("="*80)
    
    logger.info("\nTables processed:")
    for table in result['all_tables']:
        logger.info(f"  {table['table_id']}. {table['table_name']}")
        
        if 'normalized_df' in table:
            logger.info(f"     Shape: {table['normalized_df'].shape}")
            
            # ✅ Check if region exists before accessing
            if table.get('region'):
                logger.info(
                    f"     Region: rows [{table['region']['start_row']}:"
                    f"{table['region']['end_row']}], "
                    f"cols [{table['region']['start_col']}:"
                    f"{table['region']['end_col']}]"
                )
            
            # ✅ Add source info
            if table.get('source'):
                logger.info(f"     Source: {table['source']}")
            
            # ✅ Add hyperlink info if available
            if table.get('hyperlink_info'):
                info = table['hyperlink_info']
                logger.info(f"     Linked from: {info['from_cell']}")
            
            logger.info(f"     Status: ✓ Success")
        else:
            error_msg = table.get('error', 'Unknown error')
            logger.error(f"     Status: ✗ Failed - {error_msg}")
        
        logger.info("")
    
    logger.info("="*80)
def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Spreadsheet Normalizer - LLM-based table schema standardization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data.xlsx --output clean_data.csv
  python main.py --input messy_table.xlsx --output normalized.csv --verbose
  
Features:
  - Semantic structure analysis (not pattern matching)
  - Tidy data principles-based schema derivation
  - Two-stage transformation (Strategy → Code)
  - Automatic validation with semantic sampling
  - Preserved: Implicit aggregation detection
        """
    )

    parser.add_argument('--input', '-i', required=True,
                        help='Path to input spreadsheet file (.csv or .xlsx)')
    parser.add_argument('--output', '-o', default=None,
                        help='Path to output file (default: output/normalized_output.csv)')
    parser.add_argument('--config', '-c', default='config/config.yaml',
                        help='Path to configuration file (default: config/config.yaml)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    parser.add_argument('--no-split', action='store_true',
                        help='Disable table splitting even if configured')

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    try:
        # Load environment variables
        load_dotenv()
        logger.info("Environment variables loaded")

        # Load configuration
        config = load_config(args.config)
        logger.info(f"Configuration loaded from: {args.config}")

        # Override table splitting if --no-split is provided
        if args.no_split:
            if 'table_splitting' not in config:
                config['table_splitting'] = {}
            config['table_splitting']['enabled'] = False
            logger.info("Table splitting disabled via command line flag")

        # Validate input file
        input_file = Path(args.input)
        if not input_file.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)

        if input_file.suffix.lower() not in ['.csv', '.xlsx']:
            logger.error(f"Unsupported file format: {input_file.suffix}")
            logger.error("Supported formats: .csv, .xlsx")
            sys.exit(1)

        # Initialize pipeline
        logger.info("\n" + "=" * 80)
        logger.info("SPREADSHEET NORMALIZER (Enhanced)")
        logger.info("Using LLM Semantic Reasoning for Generalized Normalization")
        logger.info("=" * 80)
        logger.info(f"Input file: {args.input}")
        logger.info(f"Output file: {args.output or 'output/normalized_output.csv'}")
        
        # Display table splitting status
        splitting_enabled = config.get('table_splitting', {}).get('enabled', True)
        if splitting_enabled:
            processing_mode = config.get('table_splitting', {}).get('processing_mode', 'separate')
            logger.info(f"Table splitting: ENABLED (mode: {processing_mode})")
        else:
            logger.info(f"Table splitting: DISABLED")
        
        logger.info("="*80 + "\n")
        logger.info("=" * 80 + "\n")

        normalizer = TableNormalizer(config)

        # Run normalization
        result = normalizer.normalize(
            input_file=args.input,
            output_file=args.output
        )

        # Final status
        logger.info("\n✓ Normalization completed successfully!")
        logger.info(f"✓ Results saved to: {result['output_path']}")

        if config.get('logging', {}).get('save_intermediate', True):
            output_dir = Path(config.get('logging', {}).get('output_dir', 'output'))
            logger.info(f"✓ Intermediate results saved to: {output_dir}")

        # Print validation warnings if any
        validation = result['intermediate_results']['transformation']['validation_result']
        if validation.get('warnings'):
            logger.warning("\nWarnings:")
            for warning in validation['warnings']:
                logger.warning(f"  - {warning}")

    except KeyboardInterrupt:
        logger.info("\n\nProcess interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n\n✗ Error: {e}", exc_info=args.verbose)
        sys.exit(1)


if __name__ == '__main__':
    main()
