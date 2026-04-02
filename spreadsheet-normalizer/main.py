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
