#!/usr/bin/env python3
"""
从 batch log 里找出失败的文件，建立 retry_input/ 目录（复制文件）。
.xls 失败的自动改用 .xlsx 版本。
用法: python prepare_retry.py [batch_log.txt]
"""

import sys
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_DIR = BASE_DIR / "tables_for_dataset"
NORMALIZED_DIR = BASE_DIR / "tables_normalized"
RETRY_DIR = BASE_DIR / "retry_input"

def find_latest_log() -> Path:
    logs = sorted(NORMALIZED_DIR.glob("batch_log_*.txt"))
    if not logs:
        print("No batch_log_*.txt found in tables_normalized/")
        sys.exit(1)
    return logs[-1]

def parse_failed_files(log_path: Path) -> list[str]:
    failed = []
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("[FAILED]"):
                # [FAILED] filename.xlsx  (...)  ERROR: ...
                filename = line.split("]")[1].strip().split("(")[0].strip()
                failed.append(filename)
    return failed

def main():
    log_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_log()
    print(f"Reading log: {log_path}\n")

    failed = parse_failed_files(log_path)
    print(f"Found {len(failed)} failed file(s)\n")

    RETRY_DIR.mkdir(exist_ok=True)
    # Clear existing retry_input
    for f in RETRY_DIR.iterdir():
        f.unlink()

    copied, skipped, not_found = 0, 0, 0

    for fname in failed:
        src = INPUT_DIR / fname
        stem = Path(fname).stem

        # .xls 失败的 → 改用 .xlsx 版本
        if fname.endswith(".xls") and not fname.endswith(".xlsx"):
            xlsx_src = INPUT_DIR / (stem + ".xlsx")
            if xlsx_src.exists():
                # 检查是否已经 normalized
                if (NORMALIZED_DIR / f"{stem}_normalized.csv").exists():
                    print(f"  SKIP (already normalized): {xlsx_src.name}")
                    skipped += 1
                    continue
                shutil.copy2(xlsx_src, RETRY_DIR / xlsx_src.name)
                print(f"  ADD (xls→xlsx): {xlsx_src.name}")
                copied += 1
            else:
                print(f"  MISSING xlsx: {stem}.xlsx not found, skipping")
                not_found += 1
            continue

        # 其他失败文件 → 检查是否已经 normalized（可能上次成功过）
        if (NORMALIZED_DIR / f"{stem}_normalized.csv").exists():
            print(f"  SKIP (already normalized): {fname}")
            skipped += 1
            continue

        if src.exists():
            shutil.copy2(src, RETRY_DIR / fname)
            print(f"  ADD: {fname}")
            copied += 1
        else:
            print(f"  MISSING: {fname}")
            not_found += 1

    print(f"\nRetry input ready: {RETRY_DIR}")
    print(f"  Files to retry: {copied}")
    print(f"  Already done (skipped): {skipped}")
    print(f"  Not found: {not_found}")
    print(f"\nNext step:")
    print(f"  cd ../spreadsheet-normalizer")
    print(f"  python batch_normalize.py -i \"../wtq_evaluation/retry_input/\" -o \"../wtq_evaluation/tables_normalized/\"")

if __name__ == "__main__":
    main()
