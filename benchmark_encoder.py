#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmark script to compare original vs optimized encoder performance.
"""

import time
import sys
from pathlib import Path

# Set stdout encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "Spreadsheet_LLM_Encoder"))

def benchmark_single_file(file_path, use_optimized=False):
    """Benchmark encoding a single file."""
    print(f"\n{'='*80}")
    print(f"Benchmarking: {file_path.name}")
    print(f"Mode: {'OPTIMIZED' if use_optimized else 'ORIGINAL'}")
    print(f"{'='*80}")

    try:
        if use_optimized:
            from Spreadsheet_LLM_Encoder.Spreadsheet_LLM_Encoder_optimized import spreadsheet_llm_encode
        else:
            from Spreadsheet_LLM_Encoder.Spreadsheet_LLM_Encoder import spreadsheet_llm_encode

        start_time = time.time()
        result = spreadsheet_llm_encode(str(file_path), k=2)
        elapsed = time.time() - start_time

        print(f"[OK] Completed in {elapsed:.2f} seconds")

        # Extract metrics
        if result and 'compression_metrics' in result:
            overall = result['compression_metrics'].get('overall', {})
            ratio = overall.get('overall_compression_ratio', 0)
            print(f"     Compression ratio: {ratio:.2f}x")

        return {
            'success': True,
            'elapsed': elapsed,
            'result': result
        }

    except Exception as e:
        print(f"[ERROR] Failed: {str(e)}")
        return {
            'success': False,
            'elapsed': None,
            'error': str(e)
        }


def main():
    """Run benchmark comparison."""
    print("="*80)
    print("SPREADSHEET ENCODER PERFORMANCE BENCHMARK")
    print("="*80)

    # Test files
    test_files = [
        Path("original_spreadsheets/spreadsheet_001.csv"),
        Path("original_spreadsheets/spreadsheet_008.csv"),
        Path("original_spreadsheets/spreadsheet_014.csv"),
    ]

    results = []

    for test_file in test_files:
        if not test_file.exists():
            print(f"[SKIP] File not found: {test_file}")
            continue

        # Test original
        result_orig = benchmark_single_file(test_file, use_optimized=False)

        # Test optimized
        result_opt = benchmark_single_file(test_file, use_optimized=True)

        # Calculate speedup
        if result_orig['success'] and result_opt['success']:
            speedup = result_orig['elapsed'] / result_opt['elapsed']
            print(f"\n>>> SPEEDUP: {speedup:.2f}x faster with optimization")

            results.append({
                'file': test_file.name,
                'original': result_orig['elapsed'],
                'optimized': result_opt['elapsed'],
                'speedup': speedup
            })

    # Summary
    if results:
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        print(f"{'File':<30} {'Original':<12} {'Optimized':<12} {'Speedup':<10}")
        print("-"*80)

        for r in results:
            print(f"{r['file']:<30} {r['original']:>10.2f}s {r['optimized']:>10.2f}s {r['speedup']:>8.2f}x")

        avg_speedup = sum(r['speedup'] for r in results) / len(results)
        print("-"*80)
        print(f"{'AVERAGE SPEEDUP':<30} {'':<12} {'':<12} {avg_speedup:>8.2f}x")
        print("="*80)


if __name__ == "__main__":
    main()
