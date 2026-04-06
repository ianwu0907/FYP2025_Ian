"""
Split tables into correct/ and incorrect/ folders based on evaluation results.
Can split either normalized tables or original (pre-normalization) tables.

Usage:
  # Split normalized tables
  python split_by_correctness.py --results results/results_xxx.json --source ./tables_normalized_v3

  # Split original tables
  python split_by_correctness.py --results results/results_xxx.json --source ./tables_for_dataset
"""

import argparse
import shutil
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to evaluation results JSON")
    parser.add_argument("--source", required=True, help="Source directory to copy files from")
    args = parser.parse_args()

    results_path = Path(args.results)
    source_dir = Path(args.source)

    correct_dir = source_dir.parent / (source_dir.name + "_correct")
    incorrect_dir = source_dir.parent / (source_dir.name + "_incorrect")
    correct_dir.mkdir(exist_ok=True)
    incorrect_dir.mkdir(exist_ok=True)

    with open(results_path, encoding="utf-8") as f:
        data = json.load(f)

    # Aggregate by table_file: correct if ALL questions on that table are correct
    table_results: dict[str, list[bool]] = {}
    for r in data["results"]:
        tf = r["table_file"]
        if tf not in table_results:
            table_results[tf] = []
        table_results[tf].append(r["normalized_correct"])

    correct_tables = [tf for tf, bools in table_results.items() if all(bools)]
    incorrect_tables = [tf for tf, bools in table_results.items() if not all(bools)]

    def find_file(stem: str) -> Path | None:
        # Try exact filename first, then various extensions
        candidates = [
            source_dir / (stem + ".xlsx"),
            source_dir / (stem + ".csv"),
            source_dir / (stem + ".xls"),
            source_dir / (stem + "_normalized.csv"),
            source_dir / (stem + "_normalized.xlsx"),
        ]
        for p in candidates:
            if p.exists():
                return p
        # Fallback: search by stem
        for f in source_dir.iterdir():
            if f.stem == stem:
                return f
        return None

    copied_correct = copied_incorrect = missing = 0

    for tf in correct_tables:
        stem = Path(tf).stem
        src = find_file(stem)
        if src:
            shutil.copy2(src, correct_dir / src.name)
            copied_correct += 1
        else:
            print(f"  [MISSING] {tf}")
            missing += 1

    for tf in incorrect_tables:
        stem = Path(tf).stem
        src = find_file(stem)
        if src:
            shutil.copy2(src, incorrect_dir / src.name)
            copied_incorrect += 1
        else:
            missing += 1

    print(f"Correct tables:   {copied_correct} -> {correct_dir.name}/")
    print(f"Incorrect tables: {copied_incorrect} -> {incorrect_dir.name}/")
    if missing:
        print(f"Missing files:    {missing}")


if __name__ == "__main__":
    main()
