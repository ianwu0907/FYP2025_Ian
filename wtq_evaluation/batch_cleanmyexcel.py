"""
Batch processor for cleanmyexcel.io — supports any filename + concurrent processing.

Usage:
  python batch_cleanmyexcel.py                            # default: tables_for_dataset → cleanmyexcel
  python batch_cleanmyexcel.py --workers 5                # 5 concurrent uploads
  python batch_cleanmyexcel.py --input_dir ./my_tables    # custom input
  python batch_cleanmyexcel.py --dry-run                  # preview only, no API calls
"""

import base64
import time
import argparse
import zipfile
import logging
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

SUBMIT_URL = "https://e3us4kgm49.execute-api.eu-west-3.amazonaws.com/default/cleanmyexcel-submit-lambda"
HEADERS = {
    "accept": "*/*",
    "content-type": "application/json",
    "origin": "https://cleanmyexcel.io",
    "referer": "https://cleanmyexcel.io/",
    "user-agent": "Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
}
POLL_SCHEDULE = [0, 5, 10, 15, 20, 30, 45, 60, 90, 120, 150, 180, 240, 300, 360, 420, 480, 540, 600]


def already_done(filepath: Path, output_dir: Path) -> bool:
    """Check if this file has already been processed."""
    return (output_dir / f"{filepath.stem}_normalized.xlsx").exists()


def process_file(filepath: Path, output_dir: Path) -> dict:
    """Submit one file to cleanmyexcel.io, poll for result, extract zip."""
    result = {"file": filepath.name, "status": "failed", "elapsed": 0.0, "error": None}

    with open(filepath, "rb") as f:
        file_content = base64.b64encode(f.read()).decode("utf-8")

    t0 = time.time()
    try:
        resp = requests.post(
            SUBMIT_URL, headers=HEADERS,
            json={"file_content": file_content, "file_name": filepath.name},
            timeout=30,
        )
    except requests.RequestException as e:
        result["error"] = f"Submit error: {e}"
        return result

    if resp.status_code != 200:
        result["error"] = f"Submit failed ({resp.status_code}): {resp.text[:200]}"
        return result

    download_url = resp.json()
    if not isinstance(download_url, str) or not download_url.startswith("http"):
        result["error"] = f"Unexpected response: {str(download_url)[:200]}"
        return result

    # Poll for result
    for wait in POLL_SCHEDULE:
        if wait > 0:
            time.sleep(wait)
        try:
            dl = requests.get(download_url, timeout=15)
        except requests.RequestException:
            continue

        if dl.status_code == 200:
            elapsed = time.time() - t0
            zip_path = output_dir / (filepath.stem + "_cleaned.zip")
            with open(zip_path, "wb") as f:
                f.write(dl.content)

            # Extract zip → output_dir/{stem}/
            extract_dir = output_dir / filepath.stem
            extract_dir.mkdir(exist_ok=True)
            with zipfile.ZipFile(zip_path, "r") as z:
                z.extractall(extract_dir)
                extracted = z.namelist()
            zip_path.unlink()

            # Rename to {stem}_normalized.xlsx
            cleaned = extract_dir / "cleaned_data.xlsx"
            if not cleaned.exists():
                xlsx_files = list(extract_dir.glob("*.xlsx"))
                if xlsx_files:
                    cleaned = xlsx_files[0]
            if cleaned.exists():
                dest = output_dir / f"{filepath.stem}_normalized.xlsx"
                import shutil
                shutil.copy2(cleaned, dest)
                shutil.rmtree(extract_dir)
                result["status"] = "success"
                result["elapsed"] = round(elapsed, 1)
            else:
                result["error"] = f"No xlsx found in zip: {extracted}"
            return result

    result["error"] = f"No response after {POLL_SCHEDULE[-1]}s polling"
    return result


def main():
    parser = argparse.ArgumentParser(description="Batch cleanmyexcel.io processor")
    parser.add_argument("--input_dir", default="./tables_for_dataset")
    parser.add_argument("--output_dir", default="./cleanmyexcel")
    parser.add_argument("--workers", type=int, default=3,
                        help="Number of concurrent uploads (default: 3)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview which files would be processed, no API calls")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Collect all xlsx/csv files, skip .xls (old format)
    all_files = sorted(input_path.glob("*.xlsx")) + sorted(input_path.glob("*.csv"))

    # Split: already done vs pending
    done = [f for f in all_files if already_done(f, output_path)]
    pending = [f for f in all_files if not already_done(f, output_path)]

    logger.info(f"Input:   {input_path.resolve()}")
    logger.info(f"Output:  {output_path.resolve()}")
    logger.info(f"Total:   {len(all_files)} files")
    logger.info(f"Already done (skip): {len(done)}")
    logger.info(f"To process: {len(pending)}")

    if args.dry_run:
        logger.info("\n-- DRY RUN: files that would be submitted --")
        for f in pending:
            logger.info(f"  {f.name} ({f.stat().st_size/1024:.1f}KB)")
        logger.info(f"\nTotal to submit: {len(pending)}")
        return

    if not pending:
        logger.info("Nothing to do.")
        return

    logger.info(f"\nStarting with {args.workers} concurrent worker(s)...\n")

    success_times = []
    failed = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_file, f, output_path): f for f in pending}
        completed = 0
        for future in as_completed(futures):
            completed += 1
            r = future.result()
            if r["status"] == "success":
                success_times.append(r["elapsed"])
                logger.info(f"[{completed}/{len(pending)}] OK  {r['file']}  ({r['elapsed']}s)")
            else:
                failed.append(r)
                logger.error(f"[{completed}/{len(pending)}] FAIL  {r['file']}  {r['error']}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info(f"Done: {len(success_times)} succeeded  |  {len(failed)} failed")
    if success_times:
        logger.info(f"Avg time: {sum(success_times)/len(success_times):.1f}s  |  "
                    f"Total: {sum(success_times)/60:.1f} min")
    if failed:
        logger.info("\nFailed files:")
        for r in failed:
            logger.info(f"  - {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
