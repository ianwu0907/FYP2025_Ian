"""
Rename cleaned_data.xlsx files and remove folders.
Usage: python rename_and_cleanup.py --input_dir ./cleanmyexcel --output_dir ./tables_normalized
"""

import argparse
import shutil
from pathlib import Path


def main(input_dir: str, output_dir: str):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    folders = sorted([f for f in input_path.iterdir() if f.is_dir()])
    if not folders:
        print(f"No folders found in {input_dir}")
        return

    print(f"Found {len(folders)} folders\n")
    success, failed = 0, 0

    for folder in folders:
        cleaned_file = folder / "cleaned_data.xlsx"

        if not cleaned_file.exists():
            # Try to find any xlsx inside
            xlsx_files = list(folder.glob("*.xlsx"))
            if xlsx_files:
                cleaned_file = xlsx_files[0]
                print(f"[{folder.name}] No cleaned_data.xlsx, using {cleaned_file.name}")
            else:
                print(f"[{folder.name}] ERROR: No .xlsx found, skipping")
                failed += 1
                continue

        new_name = f"{folder.name}_normalized.xlsx"
        dest = output_path / new_name

        shutil.copy2(cleaned_file, dest)
        shutil.rmtree(folder)
        print(f"[{folder.name}] → {new_name}")
        success += 1

    print(f"\nDone: {success} renamed, {failed} failed")
    print(f"Output: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./cleanmyexcel")
    parser.add_argument("--output_dir", default="./tables_normalized")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir)
