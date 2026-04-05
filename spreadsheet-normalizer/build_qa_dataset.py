"""
QA Dataset Builder

Scans a raw/ folder and a tidy/ folder, pairs files by stem name,
generates QA pairs from each golden (tidy) CSV, and writes the
complete dataset to a single JSON file.

Folder convention:
    raw/
        population.xlsx        ← original messy spreadsheet
        poverty_2017.xlsx
        ...
    tidy/
        population.csv         ← human-curated tidy version
        poverty_2017.csv
        ...

Raw files can be .xlsx or .csv.
Tidy (golden) files must be .csv.

Pairing is done by file stem (case-insensitive).
Files present in raw/ but missing in tidy/ are skipped with a warning,
and vice versa.

Output JSON schema:
    [
        {
            "table_id":     "population",
            "raw_file":     "raw/population.xlsx",
            "tidy_file":    "tidy/population.csv",
            "qa_pairs": [
                {
                    "qtype":        "lookup",
                    "question":     "Find the Population value ...",
                    "answer":       12345.0,
                    "_ref_dim":     "Year",
                    "_ref_x":       "2020",
                    "_ref_measure": "Population"
                },
                ...
            ]
        },
        ...
    ]
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from code_retrievability import generate_qa_pairs

logger = logging.getLogger(__name__)


# ======================================================================
# File loading
# ======================================================================

def _load_raw(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a raw file (.xlsx or .csv) into a DataFrame.
    For Excel, loads the first sheet with no header parsing
    (raw structure preserved).
    """
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, sheet_name=0, header=0)
        elif path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=0)
        else:
            logger.warning(f"Unsupported raw file format: {path}")
            return None
    except Exception as e:
        logger.warning(f"Failed to load raw file {path}: {type(e).__name__}: {e}")
        return None


def _load_tidy(path: Path) -> Optional[pd.DataFrame]:
    """
    Load a golden tidy CSV into a DataFrame.
    First row is treated as the header (tidy files always have clean headers).
    """
    try:
        return pd.read_csv(path, header=0)
    except Exception as e:
        logger.warning(f"Failed to load tidy file {path}: {type(e).__name__}: {e}")
        return None


# ======================================================================
# Folder pairing
# ======================================================================

def _pair_files(
        raw_dir: Path,
        tidy_dir: Path,
) -> List[Dict[str, Path]]:
    """
    Pair raw and tidy files by stem name (case-insensitive).

    Tidy files follow the naming convention <stem>_normalized.csv,
    e.g. population_normalized.csv pairs with population.xlsx.
    The "_normalized" suffix is stripped before matching.

    Returns:
        List of dicts: { "table_id", "raw_path", "tidy_path" }
    """
    raw_files = {
        p.stem.lower(): p
        for p in raw_dir.iterdir()
        if p.suffix.lower() in {".xlsx", ".xls", ".csv"}
    }

    # Strip "_normalized" suffix from tidy stems before pairing.
    # e.g. "population_normalized" -> key "population"
    tidy_files = {}
    for p in tidy_dir.iterdir():
        if p.suffix.lower() != ".csv":
            continue
        stem = p.stem.lower()
        if stem.endswith("_normalized"):
            stem = stem[: -len("_normalized")]
        tidy_files[stem] = p

    raw_stems  = set(raw_files.keys())
    tidy_stems = set(tidy_files.keys())

    # Warn about unpaired files
    for stem in raw_stems - tidy_stems:
        logger.warning(
            f"raw/{raw_files[stem].name} has no matching tidy file -- skipped."
        )
    for stem in tidy_stems - raw_stems:
        logger.warning(
            f"tidy/{tidy_files[stem].name} has no matching raw file -- skipped."
        )

    pairs = []
    for stem in sorted(raw_stems & tidy_stems):
        pairs.append({
            "table_id":  stem,
            "raw_path":  raw_files[stem],
            "tidy_path": tidy_files[stem],
        })

    logger.info(f"Paired {len(pairs)} tables from {raw_dir} and {tidy_dir}.")
    return pairs


# ======================================================================
# Dataset builder
# ======================================================================

def build_qa_dataset(
        raw_dir: str,
        tidy_dir: str,
        output_path: str,
        n_per_type: int = 5,
        random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Build a QA dataset from paired raw/ and tidy/ folders.

    For each paired table:
      1. Load the golden tidy CSV as reference_tidy_df.
      2. Generate QA pairs (lookup + aggregation) from the tidy df.
      3. Record the raw file path for later use in evaluation.

    Writes the full dataset to output_path as JSON.

    Args:
        raw_dir:      Path to folder containing raw spreadsheets.
        tidy_dir:     Path to folder containing golden tidy CSVs.
        output_path:  Path to write the output JSON dataset.
        n_per_type:   Max QA pairs per query type per table.
        random_state: Seed for reproducibility.

    Returns:
        The dataset as a list of table entry dicts.
    """
    raw_dir  = Path(raw_dir)
    tidy_dir = Path(tidy_dir)

    if not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    if not tidy_dir.exists():
        raise FileNotFoundError(f"tidy_dir not found: {tidy_dir}")

    pairs   = _pair_files(raw_dir, tidy_dir)
    dataset = []

    for entry in pairs:
        table_id  = entry["table_id"]
        raw_path  = entry["raw_path"]
        tidy_path = entry["tidy_path"]

        logger.info(f"Processing table: {table_id}")

        # Load tidy df (used as reference for QA generation)
        tidy_df = _load_tidy(tidy_path)
        if tidy_df is None:
            logger.warning(f"  Skipping {table_id}: could not load tidy file.")
            continue

        # Generate QA pairs from golden tidy df
        qa_pairs = generate_qa_pairs(
            reference_tidy_df=tidy_df,
            n_per_type=n_per_type,
            random_state=random_state,
        )

        if not qa_pairs:
            logger.warning(
                f"  Skipping {table_id}: no QA pairs generated "
                f"(could not detect dim/measure columns)."
            )
            continue

        logger.info(
            f"  Generated {len(qa_pairs)} pairs "
            f"({sum(1 for p in qa_pairs if p['qtype']=='lookup')} lookup, "
            f"{sum(1 for p in qa_pairs if p['qtype']=='aggregation')} aggregation)"
        )

        dataset.append({
            "table_id":  table_id,
            "raw_file":  str(raw_path),
            "tidy_file": str(tidy_path),
            "qa_pairs":  qa_pairs,
        })

    # Write dataset to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    total_pairs = sum(len(e["qa_pairs"]) for e in dataset)
    logger.info(
        f"Dataset written to {output_path}: "
        f"{len(dataset)} tables, {total_pairs} QA pairs total."
    )

    return dataset


# ======================================================================
# Dataset loading (for evaluation)
# ======================================================================

def load_qa_dataset(path: str) -> List[Dict[str, Any]]:
    """
    Load a previously built QA dataset from JSON.

    Args:
        path: Path to the JSON dataset file.

    Returns:
        List of table entry dicts as written by build_qa_dataset.
    """
    with open(path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(
        f"Loaded QA dataset from {path}: "
        f"{len(dataset)} tables, "
        f"{sum(len(e['qa_pairs']) for e in dataset)} pairs."
    )
    return dataset


# ======================================================================
# Dataset summary
# ======================================================================

def summarise_dataset(dataset: List[Dict[str, Any]]):
    """Log a summary of the dataset contents."""
    total_pairs  = sum(len(e["qa_pairs"]) for e in dataset)
    n_lookup     = sum(
        sum(1 for p in e["qa_pairs"] if p["qtype"] == "lookup")
        for e in dataset
    )
    n_aggregation = total_pairs - n_lookup

    logger.info("QA DATASET SUMMARY")
    logger.info("-" * 50)
    logger.info(f"  Tables:            {len(dataset)}")
    logger.info(f"  Total QA pairs:    {total_pairs}")
    logger.info(f"  Lookup pairs:      {n_lookup}")
    logger.info(f"  Aggregation pairs: {n_aggregation}")
    logger.info("-" * 50)

    for entry in dataset:
        n  = len(entry["qa_pairs"])
        nl = sum(1 for p in entry["qa_pairs"] if p["qtype"] == "lookup")
        na = n - nl
        logger.info(
            f"  {entry['table_id']:<30} "
            f"{n:>3} pairs  ({nl} lookup, {na} agg)"
        )


# ======================================================================
# CLI entry point
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    import argparse

    parser = argparse.ArgumentParser(
        description="Build a QA dataset from paired raw/ and tidy/ folders."
    )
    parser.add_argument("--raw",    required=True, help="Path to raw/ folder")
    parser.add_argument("--tidy",   required=True, help="Path to tidy/ folder")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--n",      type=int, default=5,
                        help="Max QA pairs per query type per table (default: 5)")
    parser.add_argument("--seed",   type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    dataset = build_qa_dataset(
        raw_dir=args.raw,
        tidy_dir=args.tidy,
        output_path=args.output,
        n_per_type=args.n,
        random_state=args.seed,
    )
    summarise_dataset(dataset)