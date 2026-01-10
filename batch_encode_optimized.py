#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch encoding script using the OPTIMIZED encoder.
Processes all files in original_spreadsheets and saves to encoded_result folder.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
import pandas as pd

# Set stdout encoding to UTF-8 for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add spreadsheet encoder to path
sys.path.insert(0, str(Path(__file__).parent / "Spreadsheet_LLM_Encoder"))

import logging
logging.basicConfig(level=logging.WARNING)  # Reduce noise from encoder

from Spreadsheet_LLM_Encoder_optimized import spreadsheet_llm_encode

# Directories
INPUT_DIR = Path("original_spreadsheets")
OUTPUT_DIR = Path("encoded_result")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = OUTPUT_DIR / f"batch_encoding_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message, print_to_console=True):
    """Log message to file and optionally print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"

    if print_to_console:
        print(log_line)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')


def csv_to_xlsx_temp(csv_path):
    """Convert CSV to temporary XLSX file."""
    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Create temporary XLSX file
        temp_file = tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False)
        temp_xlsx_path = temp_file.name
        temp_file.close()

        # Write to XLSX
        df.to_excel(temp_xlsx_path, index=False, engine='openpyxl')

        return temp_xlsx_path
    except Exception as e:
        log_message(f"  [ERROR] CSV conversion failed: {str(e)}")
        return None


def encode_file(input_file):
    """Encode a single file using the optimized encoder."""
    file_name = input_file.name
    output_file = OUTPUT_DIR / f"{input_file.stem}_encoded.json"

    log_message(f"Processing: {file_name}")

    temp_xlsx = None
    try:
        # Handle CSV files - convert to temporary XLSX
        if input_file.suffix.lower() == '.csv':
            log_message(f"  Converting CSV to XLSX...")
            temp_xlsx = csv_to_xlsx_temp(input_file)
            if not temp_xlsx:
                return False
            file_to_encode = temp_xlsx
        else:
            file_to_encode = str(input_file)

        # Encode using optimized encoder
        import time
        start_time = time.time()

        result = spreadsheet_llm_encode(
            excel_path=file_to_encode,
            output_path=None,  # We'll save manually
            k=2
        )

        elapsed = time.time() - start_time

        if not result:
            log_message(f"  [FAIL] Encoding returned None")
            return False

        # Save result to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        log_message(f"  [OK] SUCCESS: {file_name} -> {output_file.name}")
        log_message(f"      Time: {elapsed:.2f}s")

        # Log compression stats
        if 'compression_metrics' in result:
            overall = result['compression_metrics'].get('overall', {})
            ratio = overall.get('overall_compression_ratio', 0)
            if ratio > 0:
                log_message(f"      Compression: {ratio:.2f}x")

        return True

    except Exception as e:
        log_message(f"  [ERROR] FAILED: {file_name}")
        log_message(f"      Error: {str(e)}")
        return False

    finally:
        # Clean up temporary XLSX file
        if temp_xlsx and os.path.exists(temp_xlsx):
            try:
                os.remove(temp_xlsx)
            except Exception:
                pass


def main():
    """Main batch encoding function."""
    log_message("=" * 80)
    log_message("BATCH ENCODING STARTED (OPTIMIZED ENCODER)")
    log_message("=" * 80)

    # Get all spreadsheet files
    spreadsheet_files = sorted(INPUT_DIR.glob("*.csv")) + \
                       sorted(INPUT_DIR.glob("*.xlsx")) + \
                       sorted(INPUT_DIR.glob("*.xls"))

    if not spreadsheet_files:
        log_message(f"No spreadsheet files found in {INPUT_DIR}")
        return

    log_message(f"Found {len(spreadsheet_files)} files to encode")
    log_message("")

    # Process each file
    successful = 0
    failed = 0
    total_time = 0

    import time
    batch_start = time.time()

    for i, file_path in enumerate(spreadsheet_files, 1):
        log_message(f"[{i}/{len(spreadsheet_files)}] Starting: {file_path.name}")

        file_start = time.time()
        if encode_file(file_path):
            successful += 1
        else:
            failed += 1
        file_time = time.time() - file_start
        total_time += file_time

        log_message("")

    batch_elapsed = time.time() - batch_start

    # Summary
    log_message("=" * 80)
    log_message("BATCH ENCODING COMPLETED")
    log_message("=" * 80)
    log_message(f"Total files: {len(spreadsheet_files)}")
    log_message(f"Successful: {successful}")
    log_message(f"Failed: {failed}")
    log_message(f"Success rate: {successful/len(spreadsheet_files)*100:.1f}%")
    log_message(f"")
    log_message(f"Total time: {batch_elapsed:.2f}s")
    log_message(f"Average time per file: {batch_elapsed/len(spreadsheet_files):.2f}s")
    log_message(f"")
    log_message(f"Encoded data saved to: {OUTPUT_DIR.absolute()}")
    log_message(f"Log file: {LOG_FILE.absolute()}")
    log_message("=" * 80)


if __name__ == "__main__":
    main()
