#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick test to verify optimized encoder works correctly.
"""

import time
import sys
from pathlib import Path

# Set stdout encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add path
sys.path.insert(0, str(Path(__file__).parent / "Spreadsheet_LLM_Encoder"))

def test_encoder(file_path, use_optimized=False):
    """Test encoder on a single file."""
    print(f"\nTesting {'OPTIMIZED' if use_optimized else 'ORIGINAL'} encoder")
    print(f"File: {file_path}")

    try:
        if use_optimized:
            from Spreadsheet_LLM_Encoder_optimized import spreadsheet_llm_encode
        else:
            from Spreadsheet_LLM_Encoder import spreadsheet_llm_encode

        start = time.time()
        result = spreadsheet_llm_encode(str(file_path), k=2)
        elapsed = time.time() - start

        print(f"[OK] Completed in {elapsed:.2f}s")

        if result and 'compression_metrics' in result:
            overall = result['compression_metrics'].get('overall', {})
            ratio = overall.get('overall_compression_ratio', 0)
            print(f"     Compression: {ratio:.2f}x")

        return elapsed

    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Test on one XLSX file
    test_file = Path("spreadsheet-normalizer/testdata.xlsx")

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        sys.exit(1)

    print("="*60)
    print("ENCODER QUICK TEST")
    print("="*60)

    # Test original
    time_orig = test_encoder(test_file, use_optimized=False)

    # Test optimized
    time_opt = test_encoder(test_file, use_optimized=True)

    if time_orig and time_opt:
        speedup = time_orig / time_opt
        print(f"\n>>> Speedup: {speedup:.2f}x")
        print("="*60)
