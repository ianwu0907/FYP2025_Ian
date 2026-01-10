#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing script for normalizing all spreadsheets in original_spreadsheets folder.
Processes all CSV/Excel files and saves results to the 'result' folder.
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

# Set stdout encoding to UTF-8 for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Directories
INPUT_DIR = Path("original_spreadsheets")
OUTPUT_DIR = Path("result")
NORMALIZER_DIR = Path("spreadsheet-normalizer")
NORMALIZER_SCRIPT = NORMALIZER_DIR / "main.py"

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True)

# Log file
LOG_FILE = OUTPUT_DIR / f"batch_processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message, print_to_console=True):
    """Log message to file and optionally print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"

    if print_to_console:
        print(log_line)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

def process_file(input_file):
    """Process a single spreadsheet file using the normalizer."""
    file_name = input_file.name
    output_file = OUTPUT_DIR / f"{input_file.stem}_normalized.csv"

    log_message(f"Processing: {file_name}")

    try:
        # Build command with config file path
        config_file = NORMALIZER_DIR / "config" / "config.yaml"
        cmd = [
            sys.executable,
            str(NORMALIZER_SCRIPT),
            "--input", str(input_file),
            "--output", str(output_file),
            "--config", str(config_file)
        ]

        # Run normalizer
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per file
        )

        if result.returncode == 0:
            log_message(f"[OK] SUCCESS: {file_name} -> {output_file.name}")
            return True
        else:
            log_message(f"[FAIL] FAILED: {file_name}")
            log_message(f"  Error: {result.stderr}", print_to_console=False)
            return False

    except subprocess.TimeoutExpired:
        log_message(f"[TIMEOUT] TIMEOUT: {file_name} (exceeded 5 minutes)")
        return False
    except Exception as e:
        log_message(f"[ERROR] ERROR: {file_name} - {str(e)}")
        return False

def main():
    """Main batch processing function."""
    log_message("=" * 80)
    log_message("BATCH NORMALIZATION STARTED")
    log_message("=" * 80)

    # Check if normalizer script exists
    if not NORMALIZER_SCRIPT.exists():
        log_message(f"ERROR: Normalizer script not found at {NORMALIZER_SCRIPT}")
        return

    # Get all spreadsheet files
    spreadsheet_files = sorted(INPUT_DIR.glob("*.csv")) + \
                       sorted(INPUT_DIR.glob("*.xlsx")) + \
                       sorted(INPUT_DIR.glob("*.xls"))

    if not spreadsheet_files:
        log_message(f"No spreadsheet files found in {INPUT_DIR}")
        return

    log_message(f"Found {len(spreadsheet_files)} files to process")
    log_message("")

    # Process each file
    successful = 0
    failed = 0

    for i, file_path in enumerate(spreadsheet_files, 1):
        log_message(f"[{i}/{len(spreadsheet_files)}] Starting: {file_path.name}")

        if process_file(file_path):
            successful += 1
        else:
            failed += 1

        log_message("")

    # Summary
    log_message("=" * 80)
    log_message("BATCH NORMALIZATION COMPLETED")
    log_message("=" * 80)
    log_message(f"Total files: {len(spreadsheet_files)}")
    log_message(f"Successful: {successful}")
    log_message(f"Failed: {failed}")
    log_message(f"Success rate: {successful/len(spreadsheet_files)*100:.1f}%")
    log_message(f"\nResults saved to: {OUTPUT_DIR.absolute()}")
    log_message(f"Log file: {LOG_FILE.absolute()}")

if __name__ == "__main__":
    main()
