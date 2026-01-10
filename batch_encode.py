#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch encoding script for spreadsheets.
Encodes all CSV files and saves only the encoded data (no normalization).
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Set stdout encoding to UTF-8 for Windows compatibility
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add spreadsheet-normalizer to path
sys.path.insert(0, str(Path(__file__).parent / "spreadsheet-normalizer"))

import yaml
from src.encoder.spreadsheet_encoder import SpreadsheetEncoder

# Directories
INPUT_DIR = Path("original_spreadsheets")
OUTPUT_DIR = Path("result/encoded")
CONFIG_FILE = Path("spreadsheet-normalizer/config/config.yaml")

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Log file
LOG_FILE = OUTPUT_DIR.parent / f"batch_encoding_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

def log_message(message, print_to_console=True):
    """Log message to file and optionally print to console."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_line = f"[{timestamp}] {message}"

    if print_to_console:
        print(log_line)

    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(log_line + '\n')

def load_config():
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        log_message(f"[ERROR] Config file not found: {CONFIG_FILE}")
        return None

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config

def make_json_serializable(obj):
    """Convert non-serializable objects to serializable format."""
    import pandas as pd
    import numpy as np

    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif pd.isna(obj):
        return None
    else:
        return obj

def encode_file(input_file, encoder):
    """Encode a single spreadsheet file."""
    file_name = input_file.name
    output_file = OUTPUT_DIR / f"{input_file.stem}_encoded.json"

    log_message(f"Processing: {file_name}")

    try:
        # Load file
        df = encoder.load_file(str(input_file))

        # Encode
        encoded_data = encoder.encode(df)

        # Convert to JSON-serializable format
        serializable_data = make_json_serializable(encoded_data)

        # Save encoded data as JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, indent=2, ensure_ascii=False)

        log_message(f"[OK] SUCCESS: {file_name} -> {output_file.name}")

        # Log encoding stats
        if 'metadata' in encoded_data:
            metadata = encoded_data['metadata']
            log_message(f"  Original shape: {metadata.get('original_shape', 'N/A')}")
            compression = metadata.get('compression_ratio', None)
            if compression is not None:
                log_message(f"  Compression ratio: {compression:.2f}x")

        return True

    except Exception as e:
        log_message(f"[ERROR] FAILED: {file_name}")
        log_message(f"  Error: {str(e)}")
        return False

def main():
    """Main batch encoding function."""
    log_message("=" * 80)
    log_message("BATCH ENCODING STARTED")
    log_message("=" * 80)

    # Load config
    config = load_config()
    if not config:
        return

    # Initialize encoder
    encoder_config = config.get('encoder', {})
    encoder = SpreadsheetEncoder(encoder_config)
    log_message("Encoder initialized")

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

    for i, file_path in enumerate(spreadsheet_files, 1):
        log_message(f"[{i}/{len(spreadsheet_files)}] Starting: {file_path.name}")

        if encode_file(file_path, encoder):
            successful += 1
        else:
            failed += 1

        log_message("")

    # Summary
    log_message("=" * 80)
    log_message("BATCH ENCODING COMPLETED")
    log_message("=" * 80)
    log_message(f"Total files: {len(spreadsheet_files)}")
    log_message(f"Successful: {successful}")
    log_message(f"Failed: {failed}")
    log_message(f"Success rate: {successful/len(spreadsheet_files)*100:.1f}%")
    log_message(f"\nEncoded data saved to: {OUTPUT_DIR.absolute()}")
    log_message(f"Log file: {LOG_FILE.absolute()}")

if __name__ == "__main__":
    main()
