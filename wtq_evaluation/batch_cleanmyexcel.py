"""
Batch processor for cleanmyexcel.io
Usage: python batch_cleanmyexcel.py --input_dir ./tables_original/structural --output_dir ./cleanmyexcel --start 8 --end 34
"""

import base64
import time
import argparse
import requests
import zipfile
from pathlib import Path

SUBMIT_URL = "https://e3us4kgm49.execute-api.eu-west-3.amazonaws.com/default/cleanmyexcel-submit-lambda"

HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://cleanmyexcel.io",
    "referer": "https://cleanmyexcel.io/",
    "user-agent": "Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
}

POLL_SCHEDULE = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120]


def process_file(filepath: Path, output_dir: Path) -> float | None:
    size_kb = filepath.stat().st_size / 1024
    if size_kb > 100:
        print(f"  WARNING: {size_kb:.1f}KB — exceeds 100KB limit, skipping")
        return None

    with open(filepath, "rb") as f:
        file_content = base64.b64encode(f.read()).decode("utf-8")

    t0 = time.time()
    resp = requests.post(SUBMIT_URL, headers=HEADERS,
                         json={"file_content": file_content, "file_name": filepath.name},
                         timeout=30)
    if resp.status_code != 200:
        print(f"  ERROR: Submit failed ({resp.status_code}): {resp.text[:200]}")
        return None

    download_url = resp.json()
    if not isinstance(download_url, str) or not download_url.startswith("http"):
        print(f"  ERROR: Unexpected response: {str(download_url)[:200]}")
        return None

    print(f"  Submitted in {time.time()-t0:.1f}s, polling for result...")

    elapsed = None
    for wait in POLL_SCHEDULE:
        if wait > 0:
            print(f"  Waiting {wait}s...")
            time.sleep(wait)
        t_now = time.time() - t0
        dl = requests.get(download_url, timeout=10)
        print(f"  t={t_now:.0f}s — status {dl.status_code}")
        if dl.status_code == 200:
            elapsed = t_now
            zip_path = output_dir / (filepath.stem + "_cleaned.zip")
            with open(zip_path, "wb") as f:
                f.write(dl.content)
            extract_dir = output_dir / filepath.stem
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
                extracted = z.namelist()
            zip_path.unlink()
            print(f"  Done in {elapsed:.1f}s — {len(extracted)} file(s) → {extract_dir.name}/")
            for name in extracted:
                print(f"    - {name}")
            break
    else:
        print(f"  FAILED: no response after {POLL_SCHEDULE[-1]}s")

    return elapsed


def main(input_dir: str, output_dir: str, delay: float, start: int, end: int):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_files = sorted(input_path.glob("*.xlsx")) + sorted(input_path.glob("*.csv"))

    # Filter by structural number (e.g. structural_8.xlsx → 8)
    def get_num(p: Path):
        stem = p.stem  # e.g. "structural_8"
        parts = stem.rsplit("_", 1)
        try:
            return int(parts[-1])
        except ValueError:
            return -1

    files = [f for f in all_files if start <= get_num(f) <= end]
    files = sorted(files, key=get_num)

    if not files:
        print(f"No files found matching structural_{start} to structural_{end}")
        return

    print(f"Processing structural_{start} to structural_{end}: {len(files)} files")
    for f in files:
        print(f"  {f.name}")
    print()

    times = []
    failed = []

    for i, filepath in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {filepath.name} ({filepath.stat().st_size/1024:.1f}KB)")
        elapsed = process_file(filepath, output_path)

        if elapsed is not None:
            times.append(elapsed)
        else:
            failed.append(filepath.name)

        if i < len(files):
            print(f"  Waiting {delay}s before next file...\n")
            time.sleep(delay)

    print(f"\n{'='*50}")
    print(f"Finished: {len(times)}/{len(files)} succeeded")
    if times:
        print(f"Min time:     {min(times):.1f}s")
        print(f"Max time:     {max(times):.1f}s")
        print(f"Average time: {sum(times)/len(times):.1f}s")
        print(f"Total time:   {sum(times)/60:.1f} minutes")
    if failed:
        print(f"\nFailed files:")
        for name in failed:
            print(f"  - {name}")
    print(f"\nOutput: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="./tables_original/structural")
    parser.add_argument("--output_dir", default="./cleanmyexcel")
    parser.add_argument("--delay", type=float, default=3.0)
    parser.add_argument("--start", type=int, default=1, help="Start file number (e.g. 8)")
    parser.add_argument("--end", type=int, default=999, help="End file number (e.g. 34)")
    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.delay, args.start, args.end)