"""
Evaluation Runner

For each table in the QA dataset:
  1. Load raw_df from the raw file path recorded in the dataset.
  2. Run raw_df through the normalization pipeline → pipeline_df.
  3. Load reference_tidy_df from the golden tidy CSV.
  4. Run QA evaluation on raw_df and pipeline_df using the pre-generated
     QA pairs (slot-filling + template execution).
  5. Record per-table and aggregate results.
  6. Write a full evaluation report to JSON.

Usage:
    python run_evaluation.py \
        --dataset  dataset/qa_pairs.json \
        --config   config/config.yaml \
        --output   eval/results.json \
        --model    gpt-4o-mini

The evaluation report JSON schema:
    {
        "summary": {
            "n_tables":            int,
            "n_tables_failed":     int,
            "n_qa_pairs":          int,
            "raw":      { "execution_rate", "accuracy",
                          "by_type": { "lookup": {...}, "aggregation": {...} } },
            "pipeline": { ... }
        },
        "tables": [
            {
                "table_id":        str,
                "pipeline_status": "success" | "failed",
                "pipeline_error":  str | null,
                "n_qa_pairs":      int,
                "scores": {
                    "raw":      { "execution_rate", "accuracy", "by_type" },
                    "pipeline": { ... }
                },
                "qa_pairs": [...]   # full per-question detail
            },
            ...
        ]
    }
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from dotenv import load_dotenv
from openai import OpenAI

from src.pipeline import TableNormalizer
from build_qa_dataset import load_qa_dataset
from code_retrievability import (
    evaluate_code_retrievability,
    compute_scores,
    log_results,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Raw file loader
# ======================================================================

def _load_raw_df(raw_file: str) -> Optional[pd.DataFrame]:
    """Load raw file (.xlsx or .csv) as-is, preserving messy structure."""
    path = Path(raw_file)
    try:
        if path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(path, header=0)
        elif path.suffix.lower() == ".csv":
            return pd.read_csv(path, header=0)
        else:
            logger.warning(f"Unsupported format: {path}")
            return None
    except Exception as e:
        logger.warning(f"Failed to load {path}: {type(e).__name__}: {e}")
        return None


def _load_tidy_df(tidy_file: str) -> Optional[pd.DataFrame]:
    """Load golden tidy CSV as reference."""
    try:
        return pd.read_csv(tidy_file, header=0)
    except Exception as e:
        logger.warning(f"Failed to load {tidy_file}: {type(e).__name__}: {e}")
        return None


# ======================================================================
# Per-table evaluation
# ======================================================================

def _run_table(
        entry: Dict[str, Any],
        normalizer: TableNormalizer,
        client: OpenAI,
        model: str,
        preview_rows: int,
        max_distinct: int,
        pipeline_output_dir: Path,
) -> Dict[str, Any]:
    """
    Run the full evaluation pipeline for a single table entry.

    Returns a table result dict regardless of pipeline success/failure —
    if the pipeline fails, raw QA is still recorded and pipeline scores
    are set to None.
    """
    table_id  = entry["table_id"]
    raw_file  = entry["raw_file"]
    tidy_file = entry["tidy_file"]
    qa_pairs  = entry["qa_pairs"]

    logger.info(f"\n{'='*72}")
    logger.info(f"TABLE: {table_id}")
    logger.info(f"{'='*72}")

    result = {
        "table_id":        table_id,
        "pipeline_status": None,
        "pipeline_error":  None,
        "n_qa_pairs":      len(qa_pairs),
        "scores":          {},
        "qa_pairs":        [],
    }

    # ── Load raw df ───────────────────────────────────────────────────
    raw_df = _load_raw_df(raw_file)
    if raw_df is None:
        result["pipeline_status"] = "failed"
        result["pipeline_error"]  = f"Could not load raw file: {raw_file}"
        logger.error(f"  Could not load raw file, skipping table.")
        return result

    # ── Load reference tidy df ────────────────────────────────────────
    reference_tidy_df = _load_tidy_df(tidy_file)
    if reference_tidy_df is None:
        result["pipeline_status"] = "failed"
        result["pipeline_error"]  = f"Could not load tidy file: {tidy_file}"
        logger.error(f"  Could not load tidy file, skipping table.")
        return result

    # ── Run normalisation pipeline ────────────────────────────────────
    pipeline_df    = None
    pipeline_error = None

    output_file = pipeline_output_dir / f"{table_id}_pipeline_output.csv"
    try:
        logger.info(f"  Running normalisation pipeline...")
        t0 = time.time()
        pipeline_result = normalizer.normalize(
            input_file=raw_file,
            output_file=str(output_file),
        )
        pipeline_df = pipeline_result["normalized_df"]
        elapsed = round(time.time() - t0, 1)
        logger.info(
            f"  Pipeline succeeded in {elapsed}s. "
            f"Output shape: {pipeline_df.shape}"
        )
        result["pipeline_status"] = "success"

    except Exception as e:
        pipeline_error = f"{type(e).__name__}: {e}"
        result["pipeline_status"] = "failed"
        result["pipeline_error"]  = pipeline_error
        logger.error(f"  Pipeline FAILED: {pipeline_error}")

    # ── Normalise column names (integer col names / float-int years) ─────
    # Fix: stringify all column names so JSON-returned slot names always match.
    raw_df.columns            = raw_df.columns.astype(str)
    reference_tidy_df.columns = reference_tidy_df.columns.astype(str)
    if pipeline_df is not None:
        pipeline_df.columns   = pipeline_df.columns.astype(str)

    # ── Run QA evaluation ─────────────────────────────────────────────
    # Even if pipeline failed we still evaluate on raw_df alone,
    # using reference_tidy_df as both pipeline_df and reference if needed.
    effective_pipeline_df = pipeline_df if pipeline_df is not None \
        else reference_tidy_df

    logger.info(f"  Running QA evaluation ({len(qa_pairs)} pairs)...")
    evaluation = evaluate_code_retrievability(
        raw_df=raw_df,
        pipeline_df=effective_pipeline_df,
        reference_tidy_df=reference_tidy_df,
        client=client,
        model=model,
        # QA pairs already generated — pass n_per_type=0 to skip generation
        # and inject pre-built pairs directly.
        _prebuilt_pairs=qa_pairs,
        preview_rows=preview_rows,
        max_distinct=max_distinct,
    )

    log_results(evaluation)

    result["scores"]   = evaluation["scores"]
    result["qa_pairs"] = evaluation["qa_pairs"]

    # If pipeline failed, mark pipeline scores explicitly as unavailable
    if pipeline_df is None:
        result["scores"]["pipeline"] = {
            "n": 0,
            "execution_rate": None,
            "accuracy": None,
            "note": "pipeline_failed",
        }

    return result


# ======================================================================
# Aggregate scoring
# ======================================================================

def _aggregate_scores(table_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute dataset-level aggregate scores.

    Fix 9: strict paired filtering — only tables where pipeline succeeded
    are included in BOTH raw and pipeline aggregates. This ensures the
    delta (pipeline − raw) is computed on identical table sets.
    """
    # Only tables where pipeline succeeded contribute to the comparison.
    valid_tables = [t for t in table_results if t["pipeline_status"] == "success"]
    n_failed     = len(table_results) - len(valid_tables)

    def _mean_from(dicts: list, key: str) -> Optional[float]:
        vals = [d[key] for d in dicts if d.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _agg(label: str) -> Dict[str, Any]:
        rows = [t["scores"][label] for t in valid_tables
                if t["scores"].get(label) and
                t["scores"][label].get("execution_rate") is not None]
        if not rows:
            return {}

        def _mean(key):
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        by_type = {}
        for qtype in ["lookup", "aggregation"]:
            sub = [r.get("by_type", {}).get(qtype, {}) for r in rows]
            by_type[qtype] = {
                "execution_rate": _mean_from(sub, "execution_rate"),
                "accuracy":       _mean_from(sub, "accuracy"),
            }

        return {
            "n_tables":       len(rows),
            "execution_rate": _mean("execution_rate"),
            "accuracy":       _mean("accuracy"),
            "by_type":        by_type,
        }

    return {
        "n_tables":           len(table_results),
        "n_tables_evaluated": len(valid_tables),
        "n_tables_failed":    n_failed,
        "n_qa_pairs":         sum(t["n_qa_pairs"] for t in table_results),
        # Both raw and pipeline are aggregated over the same valid_tables set.
        "raw":      _agg("raw"),
        "pipeline": _agg("pipeline"),
    }




# ======================================================================
# Report logging
# ======================================================================

def log_summary(summary: Dict[str, Any]):
    """Pretty-print the aggregate evaluation summary."""
    logger.info("\n" + "=" * 76)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 76)
    logger.info(f"  Tables total     : {summary['n_tables']}")
    logger.info(f"  Pipeline success : {summary.get('n_tables_evaluated', 'n/a')} (paired for delta)")
    logger.info(f"  Pipeline failures: {summary['n_tables_failed']}")
    logger.info(f"  Total QA pairs   : {summary['n_qa_pairs']}")
    logger.info("")
    logger.info(
        f"  {'Metric':<38} {'Raw':>8} {'Pipeline':>9} {'Delta':>8}"
    )
    logger.info("-" * 76)

    raw_s = summary.get("raw",      {})
    pip_s = summary.get("pipeline", {})

    def _row(label, raw_val, pip_val):
        if raw_val is None or pip_val is None:
            logger.info(f"  {label:<38}  {'n/a':>8}  {'n/a':>9}")
            return
        delta = round(pip_val - raw_val, 4)
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
        logger.info(
            f"  {label:<38} {raw_val:>8.4f} {pip_val:>9.4f} "
            f"{delta:>+8.4f}  {arrow}"
        )

    _row("execution_rate (all)",
         raw_s.get("execution_rate"), pip_s.get("execution_rate"))
    _row("accuracy (all)",
         raw_s.get("accuracy"), pip_s.get("accuracy"))

    for qtype in ["lookup", "aggregation"]:
        raw_bt = raw_s.get("by_type", {}).get(qtype, {})
        pip_bt = pip_s.get("by_type", {}).get(qtype, {})
        _row(f"  execution_rate [{qtype}]",
             raw_bt.get("execution_rate"), pip_bt.get("execution_rate"))
        _row(f"  accuracy [{qtype}]",
             raw_bt.get("accuracy"), pip_bt.get("accuracy"))

    logger.info("=" * 76)


# ======================================================================
# Main runner
# ======================================================================

def run_evaluation(
        dataset_path: str,
        config_path: str,
        output_path: str,
        model: str = "gpt-4o-mini",
        preview_rows: int = 15,
        max_distinct: int = 20,
        pipeline_output_dir: str = "eval/pipeline_outputs",
) -> Dict[str, Any]:
    """
    Run the full evaluation over the QA dataset.

    Args:
        dataset_path:       Path to qa_pairs.json built by build_qa_dataset.py.
        config_path:        Path to the normaliser config YAML.
        output_path:        Path to write the evaluation results JSON.
        model:              LLM model for slot filling.
        preview_rows:       Rows shown to LLM per table.
        max_distinct:       Max distinct values shown per column.
        pipeline_output_dir: Directory to save pipeline CSV outputs.

    Returns:
        Full evaluation report dict.
    """
    load_dotenv()

    # Setup
    dataset   = load_qa_dataset(dataset_path)
    config    = yaml.safe_load(open(config_path, encoding="utf-8"))
    client    = OpenAI()
    normalizer = TableNormalizer(config)

    pipeline_output_dir = Path(pipeline_output_dir)
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-table evaluation
    table_results = []
    for entry in dataset:
        table_result = _run_table(
            entry=entry,
            normalizer=normalizer,
            client=client,
            model=model,
            preview_rows=preview_rows,
            max_distinct=max_distinct,
            pipeline_output_dir=pipeline_output_dir,
        )
        table_results.append(table_result)

        # Write partial results after each table so progress is not lost
        _write_report(table_results, output_path, partial=True)

    # Aggregate
    summary = _aggregate_scores(table_results)
    log_summary(summary)

    report = {
        "summary": summary,
        "tables":  table_results,
    }
    _write_report(table_results, output_path, summary=summary)
    logger.info(f"\nEvaluation report written to: {output_path}")
    return report


def _write_report(
        table_results: List[Dict[str, Any]],
        output_path: Path,
        partial: bool = False,
        summary: Optional[Dict] = None,
):
    """Write current results to JSON."""
    report = {
        "partial": partial,
        "summary": summary or {},
        "tables":  table_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


# ======================================================================
# Patch evaluate_code_retrievability to accept prebuilt pairs
# ======================================================================
# evaluate_code_retrievability in code_retrievability.py always calls
# generate_qa_pairs internally. We need a variant that accepts
# pre-generated pairs from the dataset JSON instead.
#
# Rather than modifying the original file, we monkey-patch here by
# importing the internals we need.

from code_retrievability import (
    fill_slots_via_llm,
    execute_template,
    is_correct,
)


def evaluate_code_retrievability(
        raw_df: pd.DataFrame,
        pipeline_df: pd.DataFrame,
        reference_tidy_df: pd.DataFrame,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        preview_rows: int = 15,
        max_distinct: int = 20,
        _prebuilt_pairs: Optional[List[Dict]] = None,
        **kwargs,
) -> Dict[str, Any]:
    """
    Extended evaluate_code_retrievability that accepts pre-generated QA pairs.

    When _prebuilt_pairs is provided, QA generation is skipped and the
    supplied pairs are used directly. This is the expected path when
    running batch evaluation from a pre-built dataset JSON.
    """
    from code_retrievability import generate_qa_pairs

    qa_pairs = _prebuilt_pairs if _prebuilt_pairs is not None \
        else generate_qa_pairs(reference_tidy_df)

    if not qa_pairs:
        logger.warning("No QA pairs available — skipping table.")
        return {"qa_pairs": [], "scores": {}}

    # Work on a deep copy of pairs so we don't mutate the dataset JSON.
    # Results are accumulated in evaluated_pairs which is what we score.
    evaluated_pairs = []
    for pair in qa_pairs:
        pair = dict(pair)               # shallow copy of the pair dict
        pair["results"] = {}            # fresh results dict

        for label, df in [("raw", raw_df), ("pipeline", pipeline_df)]:
            logger.info(
                f"    [{label}] {pair['qtype']}: "
                f"{pair['question'][:65]}..."
            )

            slots = fill_slots_via_llm(
                question=pair["question"],
                qtype=pair["qtype"],
                df=df,
                client=client,
                model=model,
                preview_rows=preview_rows,
                max_distinct=max_distinct,
            )

            if slots is None:
                pair["results"][label] = {
                    "slots":    None,
                    "executed": False,
                    "result":   None,
                    "error":    "SlotFillingError: LLM did not return valid slots",
                    "correct":  False,
                }
                continue

            result, error = execute_template(pair["qtype"], slots, df)
            pair["results"][label] = {
                "slots":    slots,
                "executed": error is None,
                "result":   result,
                "error":    error,
                "correct":  is_correct(result, pair["answer"]),
            }

        evaluated_pairs.append(pair)

    return {
        "qa_pairs": evaluated_pairs,
        "scores":   compute_scores(evaluated_pairs),
    }


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress noisy pipeline logs during batch evaluation
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate normalisation pipeline on pre-built QA dataset."
    )
    parser.add_argument("--dataset",  required=True,
                        help="Path to qa_pairs.json from build_qa_dataset.py")
    parser.add_argument("--config",   required=True,
                        help="Path to normaliser config YAML")
    parser.add_argument("--output",   required=True,
                        help="Path to write evaluation results JSON")
    parser.add_argument("--model",    default="gpt-4o-mini",
                        help="LLM model for slot filling (default: gpt-4o-mini)")
    parser.add_argument("--preview",  type=int, default=15,
                        help="Data rows shown to LLM per table (default: 15)")
    parser.add_argument("--distinct", type=int, default=20,
                        help="Max distinct values shown per column (default: 20)")
    parser.add_argument("--pipeline-out", default="eval/pipeline_outputs",
                        help="Directory for pipeline CSV outputs")
    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        config_path=args.config,
        output_path=args.output,
        model=args.model,
        preview_rows=args.preview,
        max_distinct=args.distinct,
        pipeline_output_dir=args.pipeline_out,
    )