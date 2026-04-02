#!/usr/bin/env python3
"""
Batch Normalizer - 批量转换表格
用法:
  python batch_normalize.py                          # 处理 input/ 目录，输出到 batch_output/
  python batch_normalize.py -i my_tables/ -o results/
  python batch_normalize.py -i my_tables/ --skip-existing
"""

import argparse
import logging
import sys
import time
import yaml
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


def load_config(config_path: Path) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def collect_files(input_dir: Path) -> list[Path]:
    files = []
    for ext in ("*.xlsx", "*.xls", "*.csv"):
        files.extend(sorted(input_dir.glob(ext)))
    return files


def process_file(normalizer, input_file: Path, output_dir: Path) -> dict:
    output_file = output_dir / f"{input_file.stem}_normalized.csv"
    start = time.time()
    try:
        result = normalizer.normalize(str(input_file), str(output_file))
        elapsed = round(time.time() - start, 1)
        status = result.get("pipeline_log", {}).get("status", "unknown")
        return {"status": status, "output": output_file, "elapsed": elapsed}
    except Exception as e:
        elapsed = round(time.time() - start, 1)
        return {"status": "failed", "error": str(e), "elapsed": elapsed}


def main():
    parser = argparse.ArgumentParser(description="Batch normalize spreadsheets")
    parser.add_argument("-i", "--input", default="input",
                        help="Input directory (default: input/)")
    parser.add_argument("-o", "--output", default="batch_output",
                        help="Output directory (default: batch_output/)")
    parser.add_argument("-c", "--config", default="config/config.yaml",
                        help="Config file (default: config/config.yaml)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip files that already have a normalized output")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    config_path = Path(args.config)

    # Validation
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir.absolute()}")
        sys.exit(1)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect files
    files = collect_files(input_dir)
    if not files:
        logger.error(f"No xlsx/xls/csv files found in {input_dir.absolute()}")
        sys.exit(1)

    # Skip existing
    if args.skip_existing:
        before = len(files)
        files = [f for f in files
                 if not (output_dir / f"{f.stem}_normalized.csv").exists()]
        skipped = before - len(files)
        if skipped:
            logger.info(f"Skipping {skipped} already-normalized file(s)")

    if not files:
        logger.info("All files already processed.")
        return

    logger.info(f"Found {len(files)} file(s) to process")
    logger.info(f"Input:  {input_dir.absolute()}")
    logger.info(f"Output: {output_dir.absolute()}")
    logger.info("=" * 60)

    # Load config and init normalizer ONCE (reuse across all files)
    config = load_config(config_path)
    # Redirect intermediate outputs to batch_output/intermediate/
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    config["logging"]["output_dir"] = str(intermediate_dir)

    from src.pipeline import TableNormalizer
    normalizer = TableNormalizer(config)

    # Process
    results = []
    for i, f in enumerate(files, 1):
        logger.info(f"[{i}/{len(files)}] {f.name}")
        r = process_file(normalizer, f, output_dir)
        r["file"] = f.name
        results.append(r)

        if r["status"] == "success":
            logger.info(f"  OK  -> {r['output'].name}  ({r['elapsed']}s)")
        else:
            logger.error(f"  FAIL -> {r.get('error', 'unknown error')}  ({r['elapsed']}s)")

    # Summary
    succeeded = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]
    total_time = sum(r["elapsed"] for r in results)

    logger.info("=" * 60)
    logger.info(f"Done: {len(succeeded)}/{len(results)} succeeded  |  "
                f"Failed: {len(failed)}  |  Total time: {total_time:.1f}s")

    if failed:
        logger.info("\nFailed files:")
        for r in failed:
            logger.info(f"  - {r['file']}: {r.get('error', '')}")

    # Save log
    log_path = output_dir / f"batch_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(log_path, "w", encoding="utf-8") as lf:
        for r in results:
            line = f"[{r['status'].upper()}] {r['file']}  ({r['elapsed']}s)"
            if r["status"] != "success":
                line += f"  ERROR: {r.get('error', '')}"
            lf.write(line + "\n")
    logger.info(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
