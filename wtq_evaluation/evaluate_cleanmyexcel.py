"""
CleanMyExcel Evaluation Script
Evaluates cleanmyexcel.io normalization quality by comparing LLM QA accuracy
on original vs CleanMyExcel-processed tables.

Original table answers are cached in results/original_answers_{provider}_{model}.json
so subsequent evaluations (e.g. your own tool) skip re-querying the raw tables.

Usage:
    python evaluate_cleanmyexcel.py --provider openai --model gpt-4o-mini
    python evaluate_cleanmyexcel.py --provider gemini --model gemini-2.0-flash
    python evaluate_cleanmyexcel.py --provider gemini --questions qa-001,qa-005 --verbose
    python evaluate_cleanmyexcel.py --provider gemini --no-cache   # force re-query originals
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
QA_DATASET = BASE_DIR / "custom_qa_dataset_updated.json"
ORIGINAL_DIR = BASE_DIR / "tables_original"
CLEANMYEXCEL_DIR = BASE_DIR / "cleanmyexcel"
RESULTS_DIR = BASE_DIR / "results"


# ── LLM client factory ─────────────────────────────────────────────────────
def make_client(provider: str) -> tuple[OpenAI, str]:
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


# ── original answer cache ──────────────────────────────────────────────────
def cache_path(provider: str, model: str) -> Path:
    """Return path to the original-answers cache file for this provider+model."""
    safe_model = model.replace("/", "_").replace(".", "_").replace("-", "_")
    return RESULTS_DIR / f"original_answers_{provider}_{safe_model}.json"


def load_cache(provider: str, model: str) -> dict:
    """Load existing cache, or return empty dict."""
    path = cache_path(provider, model)
    if path.exists():
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        print(f"  [cache] Loaded {len(data['answers'])} original answers from {path.name}")
        return data["answers"]
    return {}


def save_cache(provider: str, model: str, answers: dict) -> None:
    """Persist original answers cache to disk."""
    RESULTS_DIR.mkdir(exist_ok=True)
    path = cache_path(provider, model)
    data = {
        "provider": provider,
        "model": model,
        "updated": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "total": len(answers),
        "answers": answers,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  [cache] Saved {len(answers)} original answers → {path.name}")


# ── table loading ──────────────────────────────────────────────────────────
def load_table_as_text(path: Path, max_rows: int = 200) -> str:
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str)
        else:
            df = pd.read_csv(path, dtype=str)
    except Exception as e:
        return f"[Error loading table: {e}]"

    df = df.fillna("")
    if len(df) > max_rows:
        df = df.head(max_rows)

    return df.to_csv(index=False)


def resolve_path(candidates: list[Path]) -> Path | None:
    for p in candidates:
        if p.exists():
            return p
    return None


def get_original_path(table_file: str) -> Path | None:
    base = table_file.replace(".xlsx", "")
    return resolve_path([
        ORIGINAL_DIR / f"{base}.xlsx.csv",
        ORIGINAL_DIR / f"{base}.csv",
        ORIGINAL_DIR / f"{base}.xlsx",
        ORIGINAL_DIR / table_file,
        *ORIGINAL_DIR.rglob(f"{base}.xlsx.csv"),
        *ORIGINAL_DIR.rglob(f"{base}.csv"),
        *ORIGINAL_DIR.rglob(f"{base}.xlsx"),
    ])


def get_cleanmyexcel_path(table_file: str) -> Path | None:
    stem = Path(table_file).stem   # "long1"
    name = Path(table_file).name   # "long1.xlsx"
    base = table_file.replace(".xlsx", "")
    return resolve_path([
        CLEANMYEXCEL_DIR / f"{stem}_normalized.xlsx",
        CLEANMYEXCEL_DIR / f"{name}_normalized.xlsx",
        CLEANMYEXCEL_DIR / f"{base}_normalized.xlsx",
        CLEANMYEXCEL_DIR / f"{stem}.xlsx",
    ])


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
    s = s.strip().replace(",", "").replace("%", "")
    try:
        val = float(s)
        return str(int(val)) if val == int(val) else str(round(val, 4))
    except ValueError:
        return None


def is_correct(answer: str, ground_truths: list[str]) -> bool:
    if not answer or answer.upper() == "UNKNOWN" or answer.startswith("ERROR"):
        return False

    ans_lower = answer.lower().strip()

    for gt in ground_truths:
        gt_lower = gt.lower().strip()

        if ans_lower == gt_lower:
            return True

        ans_num = normalize_number(answer)
        gt_num = normalize_number(gt)
        if ans_num is not None and gt_num is not None and ans_num == gt_num:
            return True

        if gt_lower in ans_lower:
            return True

    return False


# ── main evaluation loop ───────────────────────────────────────────────────
def run_evaluation(
    provider: str,
    model: str,
    question_ids: list[str] | None = None,
    verbose: bool = False,
    use_cache: bool = True,
) -> dict:
    client, _ = make_client(provider)
    RESULTS_DIR.mkdir(exist_ok=True)

    with open(QA_DATASET, encoding="utf-8") as f:
        dataset = json.load(f)

    qa_pairs = dataset["qa_pairs"]
    if question_ids:
        qa_pairs = [q for q in qa_pairs if q["id"] in question_ids]
        if not qa_pairs:
            raise ValueError(f"No matching question IDs found: {question_ids}")

    # Load original answer cache
    orig_cache: dict = load_cache(provider, model) if use_cache else {}
    cache_hits = 0
    cache_new = 0

    orig_table_texts: dict[str, str] = {}   # table_file → CSV text (for new cache entries)
    cme_table_texts: dict[str, str] = {}

    results = []
    orig_correct = 0
    cme_correct = 0
    skipped = 0

    for i, qa in enumerate(qa_pairs, 1):
        table_file = qa["table_file"]
        question = qa["question"]
        ground_truths = qa["answers"]
        qa_id = qa["id"]

        # Resolve table files (only load text once per table)
        if table_file not in orig_table_texts:
            orig_path = get_original_path(table_file)
            if orig_path is None:
                print(f"  [WARN] original table not found for {table_file}, skipping")
                orig_table_texts[table_file] = None
            else:
                orig_table_texts[table_file] = load_table_as_text(orig_path)

        if table_file not in cme_table_texts:
            cme_path = get_cleanmyexcel_path(table_file)
            if cme_path is None:
                print(f"  [WARN] cleanmyexcel table not found for {table_file}, skipping")
                cme_table_texts[table_file] = None
            else:
                cme_table_texts[table_file] = load_table_as_text(cme_path)

        if orig_table_texts.get(table_file) is None or cme_table_texts.get(table_file) is None:
            skipped += 1
            continue

        # Original answer: use cache or query LLM
        # Treat ERROR entries as cache misses so they get retried
        cached = orig_cache.get(qa_id)
        if cached and not cached["answer"].startswith("ERROR"):
            orig_answer = cached["answer"]
            orig_ok = cached["correct"]
            cache_hits += 1
        else:
            orig_answer = ask_llm(client, model, orig_table_texts[table_file], question)
            orig_ok = is_correct(orig_answer, ground_truths)
            orig_cache[qa_id] = {"answer": orig_answer, "correct": orig_ok}
            cache_new += 1

        # CleanMyExcel answer: always query
        cme_answer = ask_llm(client, model, cme_table_texts[table_file], question)
        cme_ok = is_correct(cme_answer, ground_truths)

        if orig_ok:
            orig_correct += 1
        if cme_ok:
            cme_correct += 1

        result = {
            "id": qa_id,
            "question": question,
            "ground_truth": ground_truths,
            "original_answer": orig_answer,
            "original_correct": orig_ok,
            "cleanmyexcel_answer": cme_answer,
            "cleanmyexcel_correct": cme_ok,
            "table_file": table_file,
        }
        results.append(result)

        status = f"[{i:02d}/{len(qa_pairs)}] {qa_id}"
        orig_mark = "✓" if orig_ok else "✗"
        cme_mark = "✓" if cme_ok else "✗"
        cached_tag = " (cached)" if qa_id in orig_cache and cache_hits > 0 else ""
        if verbose:
            print(f"{status}  orig={orig_mark}{cached_tag} ({orig_answer[:40]!r})  cme={cme_mark} ({cme_answer[:40]!r})")
        else:
            print(f"{status}  orig={orig_mark}{cached_tag}  cme={cme_mark}")

    # Save updated cache
    if cache_new > 0:
        save_cache(provider, model, orig_cache)
    elif use_cache:
        print(f"  [cache] All {cache_hits} original answers served from cache, no update needed")

    total = len(results)
    summary = {
        "total_attempted": total,
        "skipped": skipped,
        "original_correct": orig_correct,
        "cleanmyexcel_correct": cme_correct,
        "original_accuracy": round(orig_correct / total * 100, 1) if total else 0,
        "cleanmyexcel_accuracy": round(cme_correct / total * 100, 1) if total else 0,
        "cache_hits": cache_hits,
        "cache_new_entries": cache_new,
    }

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "qa_model": model,
        "qa_provider": provider,
        "tool": "cleanmyexcel",
        "summary": summary,
        "results": results,
    }

    ts = output["timestamp"]
    out_path = RESULTS_DIR / f"results_cleanmyexcel_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Results saved → {out_path.name}")
    if skipped:
        print(f"Skipped (table not found): {skipped}")
    print(f"Cache: {cache_hits} hits, {cache_new} new entries")
    print(f"Original      accuracy: {summary['original_accuracy']}%  ({orig_correct}/{total})")
    print(f"CleanMyExcel  accuracy: {summary['cleanmyexcel_accuracy']}%  ({cme_correct}/{total})")
    delta = summary["cleanmyexcel_accuracy"] - summary["original_accuracy"]
    sign = "+" if delta >= 0 else ""
    print(f"Improvement: {sign}{delta:.1f}%")
    print(f"{'='*50}")

    return output


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate CleanMyExcel quality via LLM QA")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "gemini", "deepseek", "qwen", "api2d"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question IDs to run, e.g. qa-001,qa-002")
    parser.add_argument("--verbose", action="store_true",
                        help="Print answer details for each question")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-query original tables, ignore existing cache")
    args = parser.parse_args()

    _, default_model = make_client(args.provider)
    model = args.model or default_model

    question_ids = None
    if args.questions:
        question_ids = [q.strip() for q in args.questions.split(",")]

    print(f"Provider: {args.provider}  Model: {model}")
    print(f"Tool: CleanMyExcel  Dir: {CLEANMYEXCEL_DIR}")
    print(f"Questions: {len(question_ids) if question_ids else 'all'}")
    print(f"Cache: {'disabled' if args.no_cache else 'enabled'}")
    print()

    run_evaluation(
        provider=args.provider,
        model=model,
        question_ids=question_ids,
        verbose=args.verbose,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
