"""
WTQ-style Evaluation Script
Evaluates spreadsheet normalization quality by comparing LLM QA accuracy
on original vs normalized tables.

Original table answers are cached in results/original_answers_{provider}_{model}.json
so subsequent evaluations (e.g. cleanmyexcel, your own tool) skip re-querying raw tables.

Usage:
    python evaluate.py --provider openai --model gpt-4o-mini
    python evaluate.py --provider gemini --model gemini-2.0-flash
    python evaluate.py --provider openai --model gpt-4o-mini --questions qa-001,qa-005,qa-010
    python evaluate.py --provider gemini --no-cache   # force re-query originals
"""

import os
import sys
import json
import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ── metrics imports ────────────────────────────────────────────────────────
NORMALIZER_DIR = Path(__file__).parent.parent / "spreadsheet-normalizer"
sys.path.insert(0, str(NORMALIZER_DIR))
try:
    from src.metrics.tidiness_metrics import compute_all_metrics, compare_metrics
    TIDINESS_METRICS_AVAILABLE = True
except ImportError:
    TIDINESS_METRICS_AVAILABLE = False

try:
    from structure_metrics import TableQualityMetrics
    STRUCTURE_METRICS_AVAILABLE = True
    _tqm = TableQualityMetrics()
except ImportError:
    STRUCTURE_METRICS_AVAILABLE = False

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
QA_DATASET = BASE_DIR / "custom_qa_dataset_updated.json"
ORIGINAL_DIR = BASE_DIR / "tables_for_dataset"
NORMALIZED_DIR = BASE_DIR / "tables_normalized"
RESULTS_DIR = BASE_DIR / "results"

ORIGINAL_SUFFIX = ".xlsx.csv"
NORMALIZED_SUFFIX = ".xlsx_normalized.csv"


# ── answer cache ──────────────────────────────────────────────────────────
def cache_path(provider: str, model: str, side: str) -> Path:
    """side: 'original' or a normalized-dir tag (e.g. 'cleanmyexcel')"""
    safe_model = model.replace("/", "_").replace(".", "_").replace("-", "_")
    return RESULTS_DIR / f"answers_{side}_{provider}_{safe_model}.json"


def load_cache(provider: str, model: str, side: str) -> dict:
    path = cache_path(provider, model, side)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        answers = {
            k: {"answer": v["answer"]} if isinstance(v, dict) else {"answer": v}
            for k, v in data["answers"].items()
        }
        print(f"  [cache] Loaded {len(answers)} {side} answers from {path.name}")
        return answers
    return {}


def save_cache(provider: str, model: str, side: str, answers: dict) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = cache_path(provider, model, side)
    data = {
        "provider": provider,
        "model": model,
        "side": side,
        "updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total": len(answers),
        "answers": answers,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [cache] Saved {len(answers)} {side} answers → {path.name}")


# ── LLM client factory ─────────────────────────────────────────────────────
def make_client(provider: str) -> tuple[OpenAI, str]:
    """Return (client, default_model) for the given provider."""
    load_dotenv()
    if provider == "openai":
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        default_model = "gpt-4o-mini"
    elif provider == "gemini":
        client = OpenAI(
            api_key=os.environ.get("GEMINI_API_KEY", os.environ.get("API2D_API_KEY")),
            base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"),
        )
        default_model = "gemini-2.0-flash"
    elif provider == "deepseek":
        client = OpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        )
        default_model = "deepseek-chat"
    elif provider == "qwen":
        client = OpenAI(
            api_key=os.environ["DASHSCOPE_API_KEY"],
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        default_model = "qwen2.5-7b-instruct-1m"
    elif provider == "api2d":
        client = OpenAI(
            api_key=os.environ.get("API2D_API_KEY"),
            base_url=os.environ.get("API2D_BASE_URL", "https://oa.api2d.net/v1"),
        )
        default_model = "o3-mini"
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: openai, gemini, deepseek, qwen, api2d")
    return client, default_model


# ── table loading ──────────────────────────────────────────────────────────
def compute_table_metrics(orig_path: Path, norm_path: Path) -> dict:
    """Compute tidiness and structural metrics for original and normalized tables."""
    result = {}
    try:
        orig_df = pd.read_excel(orig_path, dtype=str) if orig_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(orig_path, dtype=str)
        norm_df = pd.read_excel(norm_path, dtype=str) if norm_path.suffix.lower() in (".xlsx", ".xls") else pd.read_csv(norm_path, dtype=str)
    except Exception as e:
        return {"error": str(e)}

    if TIDINESS_METRICS_AVAILABLE:
        try:
            before = compute_all_metrics(orig_df, label="BEFORE")
            after = compute_all_metrics(norm_df, label="AFTER")
            comparison = compare_metrics(before, after)
            result["tidiness"] = {
                "before": {k: v for k, v in before.items() if k != "shape"},
                "after": {k: v for k, v in after.items() if k != "shape"},
                "comparison": comparison,
            }
        except Exception as e:
            result["tidiness_error"] = str(e)

    if STRUCTURE_METRICS_AVAILABLE:
        try:
            before_s = _tqm.compute_structural(orig_df)
            after_s = _tqm.compute_structural(norm_df)
            result["structural"] = {
                "before": {k: v for k, v in before_s.items() if k != "column_type_purity_detail"},
                "after": {k: v for k, v in after_s.items() if k != "column_type_purity_detail"},
            }
        except Exception as e:
            result["structural_error"] = str(e)

    return result


def load_table_as_text(path: Path) -> str:
    """Load a CSV or Excel file and return as CSV text for the LLM prompt."""
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str)
        else:
            df = pd.read_csv(path, dtype=str)
    except Exception as e:
        return f"[Error loading table: {e}]"

    df = df.fillna("")
    return df.to_csv(index=False)


def resolve_path(candidates: list[Path]) -> Path | None:
    """Return the first existing path from candidates, or None."""
    for p in candidates:
        if p.exists():
            return p
    return None


def get_table_paths(table_file: str, normalized_dir: Path) -> tuple[Path | None, Path | None]:
    """
    Resolve original and normalized table paths, trying multiple naming conventions:
      Original:   {base}.xlsx  →  {base}.csv
      Normalized: {base}_normalized.xlsx  →  {base}_normalized.csv
    """
    base = table_file.replace(".xlsx", "").replace(".csv", "")

    original = resolve_path([
        ORIGINAL_DIR / f"{base}.xlsx",
        ORIGINAL_DIR / f"{base}.csv",
        ORIGINAL_DIR / table_file,
    ])

    normalized = resolve_path([
        normalized_dir / f"{base}_normalized.xlsx",
        normalized_dir / f"{base}_normalized.csv",
        normalized_dir / f"{base}.xlsx_normalized.csv",
    ])

    return original, normalized


# ── LLM QA ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are a precise data analyst. You will be given a CSV table and a question.
Answer the question using ONLY the data in the table.
Rules:
- Reply with ONLY the answer value (number, name, or short phrase). No explanation.
- If the answer requires calculation (sum, difference, percentage), compute it and give the numeric result.
- If the answer cannot be determined from the table, reply with exactly: UNKNOWN
- For text answers, use the English name if available.
"""

USER_TEMPLATE = """\
Table:
{table}

Question: {question}
"""


def ask_llm(client: OpenAI, model: str, table_text: str, question: str) -> str:
    """Send table + question to LLM and return the raw answer string."""
    # o1/o3 models use max_completion_tokens and don't support temperature
    is_reasoning_model = any(model.startswith(p) for p in ("o1", "o3"))
    kwargs = {"max_completion_tokens": 512} if is_reasoning_model else {"temperature": 0, "max_tokens": 256}
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    table=table_text,
                    question=question,
                )},
            ],
            **kwargs,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ── answer matching ─────────────────────────────────────────────────────────
def normalize_number(s: str) -> str | None:
    """Try to parse s as a number and return canonical string, or None."""
    s = s.strip().replace(",", "").replace("%", "")
    try:
        val = float(s)
        # Return as int string if it's a whole number
        return str(int(val)) if val == int(val) else str(round(val, 4))
    except ValueError:
        return None


def is_correct(answer: str, ground_truths: list[str]) -> bool:
    """
    Check if the model answer matches any ground truth.

    Matching rules (in order):
    1. UNKNOWN / ERROR → always False
    2. Exact case-insensitive match
    3. Numeric equivalence (handles trailing .0, commas, etc.)
    4. Any ground truth token appears in the answer (substring, case-insensitive)
    """
    if not answer or answer.upper() == "UNKNOWN" or answer.startswith("ERROR"):
        return False

    ans_lower = answer.lower().strip()

    for gt in ground_truths:
        gt_lower = gt.lower().strip()

        # Rule 2: exact match
        if ans_lower == gt_lower:
            return True

        # Rule 3: numeric equivalence
        ans_num = normalize_number(answer)
        gt_num = normalize_number(gt)
        if ans_num is not None and gt_num is not None and ans_num == gt_num:
            return True

        # Rule 4: ground truth appears as whole word/phrase in answer
        if re.search(r'\b' + re.escape(gt_lower) + r'\b', ans_lower):
            return True

    return False


# ── main evaluation loop ───────────────────────────────────────────────────
def run_evaluation(
    provider: str,
    model: str,
    normalized_dir: Path = None,
    question_ids: list[str] | None = None,
    verbose: bool = False,
    no_cache: bool = False,
) -> dict:
    client, _ = make_client(provider)
    RESULTS_DIR.mkdir(exist_ok=True)

    if normalized_dir is None:
        normalized_dir = NORMALIZED_DIR
    norm_tag = normalized_dir.name  # e.g. "cleanmyexcel", "tables_normalized"

    with open(QA_DATASET, encoding="utf-8") as f:
        dataset = json.load(f)

    qa_pairs = dataset["qa_pairs"]
    if question_ids:
        qa_pairs = [q for q in qa_pairs if q["id"] in question_ids]
        if not qa_pairs:
            raise ValueError(f"No matching question IDs found: {question_ids}")

    # Load both caches
    orig_cache = {} if no_cache else load_cache(provider, model, "original")
    norm_cache = {} if no_cache else load_cache(provider, model, norm_tag)
    orig_hits = orig_new = norm_hits = norm_new = 0

    orig_table_texts: dict[str, str] = {}
    norm_table_texts: dict[str, str] = {}
    metrics_cache: dict[str, dict] = {}

    results = []
    orig_correct = 0
    norm_correct = 0

    for i, qa in enumerate(qa_pairs, 1):
        table_file = qa["table_file"]
        question = qa["question"]
        ground_truths = qa["answers"]
        qa_id = qa["id"]

        if table_file not in orig_table_texts:
            orig_path, norm_path = get_table_paths(table_file, normalized_dir)
            orig_table_texts[table_file] = load_table_as_text(orig_path) if orig_path else None
            norm_table_texts[table_file] = load_table_as_text(norm_path) if norm_path else None
            # Compute metrics once per table pair
            if orig_path and norm_path:
                metrics_cache[table_file] = compute_table_metrics(orig_path, norm_path)
            else:
                metrics_cache[table_file] = {}
            if orig_path is None:
                print(f"  [WARN] original table not found: {table_file}")
            if norm_path is None:
                print(f"  [WARN] normalized table not found: {table_file}")

        orig_text = orig_table_texts.get(table_file)
        norm_text = norm_table_texts.get(table_file)

        # Original side
        cached_orig = orig_cache.get(qa_id)
        if cached_orig and not cached_orig["answer"].startswith("ERROR"):
            orig_answer = cached_orig["answer"]
            orig_hits += 1
        elif orig_text:
            orig_answer = ask_llm(client, model, orig_text, question)
            orig_cache[qa_id] = {"answer": orig_answer}
            orig_new += 1
        else:
            orig_answer = "[Table not found]"
        orig_ok = is_correct(orig_answer, ground_truths)

        # Normalized side
        cached_norm = norm_cache.get(qa_id)
        if cached_norm and not cached_norm["answer"].startswith("ERROR"):
            norm_answer = cached_norm["answer"]
            norm_hits += 1
        elif norm_text:
            norm_answer = ask_llm(client, model, norm_text, question)
            norm_cache[qa_id] = {"answer": norm_answer}
            norm_new += 1
        else:
            norm_answer = "[Normalized not found]"
        norm_ok = is_correct(norm_answer, ground_truths)

        if orig_ok:
            orig_correct += 1
        if norm_ok:
            norm_correct += 1

        entry = {
            "id": qa_id,
            "question": question,
            "ground_truth": ground_truths,
            "original_answer": orig_answer,
            "original_correct": orig_ok,
            "normalized_answer": norm_answer,
            "normalized_correct": norm_ok,
            "table_file": table_file,
        }
        if metrics_cache.get(table_file):
            entry["metrics"] = metrics_cache[table_file]
        results.append(entry)

        status = f"[{i:02d}/{len(qa_pairs)}] {qa_id}"
        orig_mark = "✓" if orig_ok else "✗"
        norm_mark = "✓" if norm_ok else "✗"
        orig_tag = " (cached)" if cached_orig and not no_cache else ""
        norm_tag_str = " (cached)" if cached_norm and not no_cache else ""
        if verbose:
            print(f"{status}  orig={orig_mark}{orig_tag} ({orig_answer[:40]!r})  norm={norm_mark}{norm_tag_str} ({norm_answer[:40]!r})")
        else:
            print(f"{status}  orig={orig_mark}{orig_tag}  norm={norm_mark}{norm_tag_str}")

    # Save updated caches
    if not no_cache:
        if orig_new > 0:
            save_cache(provider, model, "original", orig_cache)
        if norm_new > 0:
            save_cache(provider, model, norm_tag, norm_cache)
    print(f"  [cache] original: {orig_hits} hits, {orig_new} new  |  {norm_tag}: {norm_hits} hits, {norm_new} new")

    total = len(results)

    # Aggregate tidiness metrics across all tables
    tidiness_summary = {}
    if TIDINESS_METRICS_AVAILABLE:
        metric_keys = ["cell_coverage", "row_completeness_uniformity", "column_type_homogeneity",
                       "data_row_ratio", "column_completeness_min", "header_uniqueness",
                       "type_consistency", "substring_containment", "inter_column_nmi"]
        for k in metric_keys:
            before_vals = [r["metrics"]["tidiness"]["before"][k] for r in results
                           if r.get("metrics", {}).get("tidiness", {}).get("before", {}).get(k) is not None]
            after_vals = [r["metrics"]["tidiness"]["after"][k] for r in results
                          if r.get("metrics", {}).get("tidiness", {}).get("after", {}).get(k) is not None]
            if before_vals and after_vals:
                tidiness_summary[k] = {
                    "avg_before": round(sum(before_vals) / len(before_vals), 4),
                    "avg_after": round(sum(after_vals) / len(after_vals), 4),
                    "avg_delta": round(sum(after_vals) / len(after_vals) - sum(before_vals) / len(before_vals), 4),
                }

    summary = {
        "total": total,
        "normalized_dir": str(normalized_dir),
        "original_correct": orig_correct,
        "normalized_correct": norm_correct,
        "original_accuracy": round(orig_correct / total * 100, 1) if total else 0,
        "normalized_accuracy": round(norm_correct / total * 100, 1) if total else 0,
        "tidiness_metrics_avg": tidiness_summary,
    }

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "qa_model": model,
        "qa_provider": provider,
        "normalized_dir": str(normalized_dir),
        "summary": summary,
        "results": results,
    }

    ts = output["timestamp"]
    out_path = RESULTS_DIR / f"results_{norm_tag}_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Results saved → {out_path.name}")
    print(f"Normalized dir: {normalized_dir}")
    print(f"Original  accuracy: {summary['original_accuracy']}%  ({orig_correct}/{total})")
    print(f"Normalized accuracy: {summary['normalized_accuracy']}%  ({norm_correct}/{total})")
    delta = summary["normalized_accuracy"] - summary["original_accuracy"]
    sign = "+" if delta >= 0 else ""
    print(f"Improvement: {sign}{delta:.1f}%")
    if tidiness_summary:
        print(f"\n── Tidiness Metrics (avg across {len(metrics_cache)} tables) ──")
        print(f"  {'Metric':<35} {'Before':>7} {'After':>7} {'Delta':>7}")
        print(f"  {'-'*58}")
        for k, v in tidiness_summary.items():
            sign_d = "+" if v['avg_delta'] >= 0 else ""
            print(f"  {k:<35} {v['avg_before']:>7.4f} {v['avg_after']:>7.4f} {sign_d}{v['avg_delta']:>6.4f}")
    print(f"{'='*50}")

    return output


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate normalization quality via LLM QA")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "gemini", "deepseek", "qwen", "api2d"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--normalized_dir", default=None,
                        help="Directory with normalized tables (default: tables_normalized/)")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question IDs, e.g. qa-001,qa-002")
    parser.add_argument("--verbose", action="store_true",
                        help="Print answer details for each question")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable cache, re-query everything")
    args = parser.parse_args()

    _, default_model = make_client(args.provider)
    model = args.model or default_model
    normalized_dir = Path(args.normalized_dir) if args.normalized_dir else None

    question_ids = None
    if args.questions:
        question_ids = [q.strip() for q in args.questions.split(",")]

    print(f"Provider: {args.provider}  Model: {model}")
    print(f"Normalized dir: {normalized_dir or NORMALIZED_DIR}")
    print(f"Questions: {len(question_ids) if question_ids else 'all'}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print()

    run_evaluation(
        provider=args.provider,
        model=model,
        normalized_dir=normalized_dir,
        question_ids=question_ids,
        verbose=args.verbose,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
