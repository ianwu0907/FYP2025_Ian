#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: How to use the optimized encoder for single file encoding.
"""

import sys
import time
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add path to the encoder
sys.path.insert(0, str(Path(__file__).parent / "Spreadsheet_LLM_Encoder"))

# Import the optimized encoder
from Spreadsheet_LLM_Encoder_optimized import spreadsheet_llm_encode

def encode_file_example():
    """Example of encoding a single XLSX file."""

    # Input file (must be .xlsx format)
    input_file = "spreadsheet-normalizer/testdata.xlsx"

    # Output file (optional)
    output_file = "result/testdata_encoded_optimized.json"

    # Create output directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("OPTIMIZED ENCODER EXAMPLE")
    print("="*70)
    print(f"Input: {input_file}")
    print(f"Output: {output_file}")
    print("-"*70)

    # Start timing
    start_time = time.time()

    # Encode the file
    result = spreadsheet_llm_encode(
        excel_path=input_file,
        output_path=output_file,
        k=2  # Neighborhood distance for structural anchors
    )

    # Calculate elapsed time
    elapsed = time.time() - start_time

    # Print results
    if result:
        print(f"\n[SUCCESS] Encoding completed in {elapsed:.2f} seconds")

        if 'compression_metrics' in result:
            overall = result['compression_metrics'].get('overall', {})
            ratio = overall.get('overall_compression_ratio', 0)
            print(f"Compression ratio: {ratio:.2f}x")

            print(f"\nOutput saved to: {output_file}")
    else:
        print("\n[ERROR] Encoding failed")

    print("="*70)
    return result


if __name__ == "__main__":
    # Run the example
    encode_file_example()

    print("\nTIP: To encode your own files, modify the 'input_file' variable")
    print("     The encoder only supports .xlsx files (not .csv)")
