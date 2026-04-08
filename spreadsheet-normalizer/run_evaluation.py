"""
Evaluation Runner

Runs the full code-retrievability evaluation over a pre-built QA dataset.
All parameters are read from the project config YAML (config/config.yaml)
under the `evaluation:` key.  The two most common overrides (dataset path
and output path) can also be supplied on the command line.

Usage (default — everything from config):
    python run_evaluation.py --config config/config.yaml

Usage (override dataset / output):
    python run_evaluation.py --config config/config.yaml \\
        --dataset dataset/qa_pairs_v2.json \\
        --output  eval/results_v2.json

Config block (add to config/config.yaml):
    evaluation:
      dataset:              dataset/qa_pairs.json
      output:               eval/results.json
      model:                gpt-4o-mini
      preview_rows:         15
      max_distinct:         20
      pipeline_output_dir:  eval/pipeline_outputs

Evaluation report JSON schema:
    {
        "summary": {
            "n_tables":           int,   # total tables in dataset
            "n_tables_evaluated": int,   # tables where pipeline succeeded (paired)
            "n_tables_failed":    int,
            "n_qa_pairs":         int,
            "raw":      { "n_tables", "execution_rate", "accuracy",
                          "by_type": { "lookup": {...}, "aggregation": {...} } },
            "pipeline": { ... }   # same structure; raw & pipeline on same tables
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
                "qa_pairs": [...]
            },
            ...
        ]
    }
"""

import json
import logging
import os
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
    fill_slots_via_llm,
    execute_template,
    is_correct,
    compute_scores,
    log_results,
    generate_qa_pairs,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Config helpers
# ======================================================================

# Defaults used when the evaluation block is absent from config.
_EVAL_DEFAULTS: Dict[str, Any] = {
    "dataset":              "dataset/qa_pairs.json",
    "output":               "eval/results.json",
    "model":                "gpt-4o-mini",
    "preview_rows":         15,
    "max_distinct":         20,
    "pipeline_output_dir":  "eval/pipeline_outputs",
}

def _deduplicate_columns(df: pd.DataFrame) -> pd.DataFrame:

    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}_{i}" if i != 0 else str(dup) for i in range(sum(cols == dup))
        ]
    df.columns = cols
    return df
def _load_eval_config(config_path: str,
                      dataset_override: Optional[str] = None,
                      output_override:  Optional[str] = None) -> Dict[str, Any]:
    """
    Load the project config YAML and extract the evaluation sub-config.
    Command-line overrides take precedence over config values, which take
    precedence over _EVAL_DEFAULTS.

    Returns a flat dict of all evaluation parameters.
    """
    with open(config_path, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)

    eval_cfg: Dict[str, Any] = {**_EVAL_DEFAULTS,
                                **full_config.get("evaluation", {})}

    # CLI overrides (None means "not supplied", so don't overwrite)
    if dataset_override is not None:
        eval_cfg["dataset"] = dataset_override
    if output_override is not None:
        eval_cfg["output"] = output_override

    return eval_cfg, full_config


# ======================================================================
# File loaders
# ======================================================================

def _load_raw_df(raw_file: str) -> Optional[pd.DataFrame]:
    """Load raw spreadsheet (.xlsx or .csv) preserving its messy structure."""
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
    """Load the golden tidy CSV as the QA ground-truth reference."""
    try:
        return pd.read_csv(tidy_file, header=0)
    except Exception as e:
        logger.warning(f"Failed to load {tidy_file}: {type(e).__name__}: {e}")
        return None


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stringify all column names so LLM-returned slot names (always strings)
    match regardless of whether pandas read them as integers (e.g. years).
    """
    df = df.copy()
    df.columns = df.columns.astype(str)
    return df


# ======================================================================
# QA evaluation (prebuilt-pairs variant)
# ======================================================================

def _evaluate_table(
        raw_df: pd.DataFrame,
        pipeline_df: pd.DataFrame,
        reference_tidy_df: pd.DataFrame,
        qa_pairs: List[Dict[str, Any]],
        client: OpenAI,
        model: str,
        preview_rows: int,
        max_distinct: int,
) -> Dict[str, Any]:
    """
    Run slot-filling QA evaluation on raw_df and pipeline_df using
    pre-generated qa_pairs (ground truth from reference_tidy_df).

    Works on shallow copies of pairs so the dataset JSON is never mutated.
    """
    if not qa_pairs:
        logger.warning("No QA pairs — skipping table.")
        return {"qa_pairs": [], "scores": {}}

    evaluated_pairs = []
    for pair in qa_pairs:
        pair = dict(pair)
        pair["results"] = {}
        expected_values = []
        if "_ref_x" in pair: expected_values.append(pair["_ref_x"])
        if "_filters" in pair: expected_values.extend(pair["_filters"].values())

        for label, df in [("raw", raw_df), ("pipeline", pipeline_df)]:
            if df is None:
                pair["results"][label] = { "executed": False, "error": "Pipeline failed..." }
                continue
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
                target_values=expected_values,
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
# Per-table runner
# ======================================================================

def _run_table(
        entry: Dict[str, Any],
        normalizer: TableNormalizer,
        client: OpenAI,
        model: str,
        preview_rows: int,
        max_distinct: int,
        pipeline_output_dir: Path,
        full_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    """
    Full pipeline + QA evaluation for a single dataset entry.

    Always returns a result dict. If the pipeline fails, raw QA is still
    recorded; pipeline scores are set to None and excluded from aggregate.
    """
    table_start_time = time.time()
    table_id  = entry["table_id"]
    raw_file  = entry["raw_file"]
    tidy_file = entry["tidy_file"]
    qa_pairs  = entry["qa_pairs"]

    logger.info(f"\n{'='*72}")
    logger.info(f"TABLE: {table_id}")
    logger.info(f"{'='*72}")

    result: Dict[str, Any] = {
        "table_id":        table_id,
        "pipeline_status": None,
        "pipeline_error":  None,
        "pipeline_sec": 0.0,
        "n_qa_pairs":      len(qa_pairs),
        "scores":          {},
        "qa_pairs":        [],
    }

    # ── Load inputs ───────────────────────────────────────────────────
    raw_df = _load_raw_df(raw_file)

    if raw_df is None:
        result["pipeline_status"] = "failed"
        result["pipeline_error"]  = f"Could not load raw file: {raw_file}"
        logger.error(f"  Could not load raw file, skipping table.")
        return result

    reference_tidy_df = _load_tidy_df(tidy_file)
    if reference_tidy_df is None:
        result["pipeline_status"] = "failed"
        result["pipeline_error"]  = f"Could not load tidy file: {tidy_file}"
        logger.error("  Could not load tidy file — skipping table.")
        return result

    # ── Run normalisation pipeline ────────────────────────────────────
    pipeline_df: Optional[pd.DataFrame] = None
    output_file = pipeline_output_dir / f"{table_id}_pipeline_output.csv"
    pipeline_start_time = time.time()
    _pipeline_max_retries = 3
    for _attempt in range(_pipeline_max_retries):
        try:
            logger.info(f"  Running normalisation pipeline (attempt {_attempt+1}/{_pipeline_max_retries})...")
            t0 = time.time()
            # Fresh normalizer on retry to reset HTTP connections
            if _attempt > 0:
                normalizer = TableNormalizer(full_config)
            pipeline_result = normalizer.normalize(
                input_file=raw_file,
                output_file=str(output_file),
            )
            pipeline_df = pipeline_result["normalized_df"]
            result["pipeline_sec"] = round(time.time() - pipeline_start_time, 2)
            logger.info(
                f"  Pipeline succeeded in {round(time.time()-t0, 1)}s. "
                f"Output shape: {pipeline_df.shape}"
            )
            result["pipeline_status"] = "success"
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            if _attempt < _pipeline_max_retries - 1:
                wait = 15 * (_attempt + 1)
                logger.warning(f"  Pipeline FAILED (attempt {_attempt+1}): {err} — retrying in {wait}s...")
                time.sleep(wait)
            else:
                result["pipeline_sec"] = round(time.time() - pipeline_start_time, 2)
                result["pipeline_status"] = "failed"
                result["pipeline_error"]  = err
                logger.error(f"  Pipeline FAILED after {_pipeline_max_retries} attempts: {err}")

    # ── Normalise column names (fixes integer/float col names) ────────
    raw_df = _deduplicate_columns(raw_df)
    raw_df = _normalise_columns(raw_df)
    reference_tidy_df = _normalise_columns(reference_tidy_df)
    if pipeline_df is not None:
        pipeline_df = _deduplicate_columns(pipeline_df)
        pipeline_df = _normalise_columns(pipeline_df)

    # ── QA evaluation ─────────────────────────────────────────────────
    # If pipeline failed, evaluate pipeline slot against reference_tidy_df
    # so that the failure is recorded but we still get a raw score.
    effective_pipeline_df = pipeline_df if pipeline_df is not None \
        else reference_tidy_df

    logger.info(f"  Running QA evaluation ({len(qa_pairs)} pairs)...")
    evaluation = _evaluate_table(
        raw_df=raw_df,
        pipeline_df=effective_pipeline_df,
        reference_tidy_df=reference_tidy_df,
        qa_pairs=qa_pairs,
        client=client,
        model=model,
        preview_rows=preview_rows,
        max_distinct=max_distinct,
    )

    log_results(evaluation)
    result["scores"]   = evaluation["scores"]
    result["qa_pairs"] = evaluation["qa_pairs"]

    # Mark pipeline scores unavailable when pipeline failed
    if pipeline_df is None:
        result["scores"]["pipeline"] = {
            "n":              0,
            "execution_rate": None,
            "accuracy":       None,
            "note":           "pipeline_failed",
        }

    return result


# ======================================================================
# Aggregate scoring
# ======================================================================

def _aggregate_scores(table_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate evaluation scores across tables.

    Strict paired filtering: only tables where the pipeline succeeded are
    included in BOTH raw and pipeline aggregates. This ensures the delta
    (pipeline − raw) is computed on identical table sets and is not
    confounded by tables that the pipeline could not process.
    """
    valid_tables = [t for t in table_results if t["pipeline_status"] == "success"]
    n_failed     = len(table_results) - len(valid_tables)
    avg_pipeline_sec = 0.0
    if valid_tables:
        avg_pipeline_sec = sum(t.get("pipeline_sec", 0) for t in valid_tables) / len(valid_tables)
    def _mean_from(dicts: List[Dict], key: str) -> Optional[float]:
        vals = [d[key] for d in dicts if d.get(key) is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    def _agg(label: str) -> Dict[str, Any]:
        rows = [
            t["scores"][label] for t in valid_tables
            if t["scores"].get(label)
               and t["scores"][label].get("execution_rate") is not None
        ]
        if not rows:
            return {}

        def _mean(key: str) -> Optional[float]:
            vals = [r[key] for r in rows if r.get(key) is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        by_type: Dict[str, Any] = {}
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
        # raw and pipeline computed on the same valid_tables set (paired).
        "avg_pipeline_sec":   round(avg_pipeline_sec, 2),
        "raw":      _agg("raw"),
        "pipeline": _agg("pipeline"),
    }


# ======================================================================
# Summary logging
# ======================================================================

def log_summary(summary: Dict[str, Any]) -> None:
    """Pretty-print the aggregate evaluation summary."""
    logger.info("\n" + "=" * 76)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 76)
    logger.info(f"  Tables total     : {summary['n_tables']}")
    logger.info(
        f"  Paired (success) : {summary.get('n_tables_evaluated', 'n/a')} "
        f"(delta computed on this set)"
    )
    logger.info(f"  Pipeline failed  : {summary['n_tables_failed']}")
    logger.info(f"  Total QA pairs   : {summary['n_qa_pairs']}")
    logger.info(f"  Avg Pipeline Time: {summary.get('avg_pipeline_sec', 0.0)}s per table")
    logger.info("")
    logger.info(
        f"  {'Metric':<38} {'Raw':>8} {'Pipeline':>9} {'Delta':>8}"
    )
    logger.info("-" * 76)

    raw_s = summary.get("raw",      {})
    pip_s = summary.get("pipeline", {})

    def _row(label: str, raw_val: Optional[float],
             pip_val: Optional[float]) -> None:
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
         raw_s.get("accuracy"),       pip_s.get("accuracy"))

    for qtype in ["lookup", "aggregation"]:
        raw_bt = raw_s.get("by_type", {}).get(qtype, {})
        pip_bt = pip_s.get("by_type", {}).get(qtype, {})
        _row(f"  execution_rate [{qtype}]",
             raw_bt.get("execution_rate"), pip_bt.get("execution_rate"))
        _row(f"  accuracy [{qtype}]",
             raw_bt.get("accuracy"),       pip_bt.get("accuracy"))

    logger.info("=" * 76)


# ======================================================================
# Report writer
# ======================================================================

def _write_report(
        table_results: List[Dict[str, Any]],
        output_path: Path,
        partial: bool = False,
        summary: Optional[Dict] = None,
) -> None:
    """Serialise current results to JSON (partial or final)."""
    report = {
        "partial": partial,
        "summary": summary or {},
        "tables":  table_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)


# ======================================================================
# Provider env-var configuration
# ======================================================================

def _configure_env_for_model(model: str) -> None:
    """
    Override OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL in os.environ
    so that all pipeline modules (which read these vars) hit the correct
    provider, regardless of what .env contains.
    """
    m = model.lower()

    if m.startswith("deepseek"):
        api_key  = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        os.environ["OPENAI_API_KEY"]  = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_MODEL"]    = model
        logger.info(f"[env] Using DeepSeek provider: base_url={base_url}, model={model}")

    elif m.startswith("qwen") or m.startswith("qwq"):
        api_key  = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
        os.environ["OPENAI_API_KEY"]  = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_MODEL"]    = model
        logger.info(f"[env] Using Qwen/DashScope provider: base_url={base_url}, model={model}")

    elif m.startswith("gemini"):
        api_key  = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY", "")
        base_url = os.getenv("GEMINI_BASE_URL", "https://oa.api2d.net/v1")
        os.environ["OPENAI_API_KEY"]  = api_key
        os.environ["OPENAI_BASE_URL"] = base_url
        os.environ["OPENAI_MODEL"]    = model
        logger.info(f"[env] Using Gemini provider: base_url={base_url}, model={model}")

    elif m.startswith("minimax") or "minimax" in m:
        # MiniMax uses OPENAI_API_KEY / OPENAI_BASE_URL as-is from .env
        logger.info(f"[env] Using MiniMax provider (from .env): model={model}")

    else:
        # Generic OpenAI-compatible: leave env vars as set in terminal / .env
        logger.info(f"[env] Using default OpenAI provider: model={model}")


# ======================================================================
# Main runner
# ======================================================================

def run_evaluation(
        config_path: str,
        dataset_override: Optional[str] = None,
        output_override:  Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the full evaluation over the QA dataset.

    All parameters (model, preview_rows, etc.) are read from the
    `evaluation:` block in config_path.  dataset_override and
    output_override allow CLI callers to redirect without editing config.

    Returns the full evaluation report dict.
    """
    load_dotenv()

    eval_cfg, full_config = _load_eval_config(
        config_path, dataset_override, output_override
    )

    logger.info("Evaluation configuration:")
    for k, v in eval_cfg.items():
        logger.info(f"  {k}: {v}")

    # Resolve paths
    dataset_path        = eval_cfg["dataset"]
    output_path         = Path(eval_cfg["output"])
    model               = eval_cfg["model"]
    preview_rows        = int(eval_cfg["preview_rows"])
    max_distinct        = int(eval_cfg["max_distinct"])
    pipeline_output_dir = Path(eval_cfg["pipeline_output_dir"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pipeline_output_dir.mkdir(parents=True, exist_ok=True)

    # Override OPENAI_* env vars based on the evaluation model so that
    # pipeline modules (which read OPENAI_API_KEY / OPENAI_BASE_URL) always
    # hit the correct provider, regardless of what .env contains.
    _configure_env_for_model(model)

    # Initialise shared resources
    dataset    = load_qa_dataset(dataset_path)
    client     = OpenAI()

    # Resume from existing checkpoint if present
    # Only skip tables where pipeline SUCCEEDED — failed ones will be retried.
    table_results: List[Dict[str, Any]] = []
    success_ids: set = set()
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            prev_tables = existing.get("tables", [])
            # Keep only successful results; failed ones are dropped so they re-run
            table_results = [t for t in prev_tables if t.get("pipeline_status") == "success"]
            success_ids = {t["table_id"] for t in table_results}
            n_failed = len(prev_tables) - len(table_results)
            if success_ids:
                logger.info(f"Resuming: {len(success_ids)} succeeded (skipping), {n_failed} failed (retrying).")
        except Exception:
            pass

    # Per-table evaluation
    for entry in dataset:
        if entry["table_id"] in success_ids:
            logger.info(f"Skipping (already succeeded): {entry['table_id']}")
            continue
        # Create a fresh normalizer per table to avoid stale HTTP connections
        normalizer = TableNormalizer(full_config)
        table_result = _run_table(
            entry=entry,
            normalizer=normalizer,
            client=client,
            model=model,
            preview_rows=preview_rows,
            max_distinct=max_distinct,
            pipeline_output_dir=pipeline_output_dir,
            full_config=full_config,
        )
        table_results.append(table_result)
        # Checkpoint after each table so progress is not lost on crash
        _write_report(table_results, output_path, partial=True)

    # Final aggregate and report
    summary = _aggregate_scores(table_results)
    log_summary(summary)

    report = {"summary": summary, "tables": table_results}
    _write_report(table_results, output_path, partial=False, summary=summary)
    logger.info(f"\nEvaluation report written to: {output_path}")
    return report


# ======================================================================
# CLI
# ======================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the normalisation pipeline on a pre-built QA dataset. "
            "All parameters are read from config/config.yaml under the "
            "'evaluation:' key. --dataset and --output override config values."
        )
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to project config YAML (e.g. config/config.yaml)",
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Override config evaluation.dataset path",
    )
    parser.add_argument(
        "--output", default=None,
        help="Override config evaluation.output path",
    )
    args = parser.parse_args()

    run_evaluation(
        config_path=args.config,
        dataset_override=args.dataset,
        output_override=args.output,
    )