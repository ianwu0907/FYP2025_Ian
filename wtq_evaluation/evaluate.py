"""
WTQ-style Evaluation Script
Evaluates spreadsheet normalization quality by comparing LLM QA accuracy
on original vs normalized tables.

Usage:
    python evaluate.py --provider openai --model gpt-4o-mini
    python evaluate.py --provider gemini --model gemini-2.0-flash
    python evaluate.py --provider openai --model gpt-4o-mini --questions qa-001,qa-005,qa-010
"""

import os
import json
import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ── paths ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
QA_DATASET = BASE_DIR / "custom_qa_dataset.json"
ORIGINAL_DIR = BASE_DIR / "tables_original"
NORMALIZED_DIR = BASE_DIR / "tables_normalized"
RESULTS_DIR = BASE_DIR / "results"

ORIGINAL_SUFFIX = ".xlsx.csv"
NORMALIZED_SUFFIX = ".xlsx_normalized.csv"


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
    else:
        raise ValueError(f"Unsupported provider: {provider}. Choose from: openai, gemini, deepseek")
    return client, default_model


# ── table loading ──────────────────────────────────────────────────────────
def load_table_as_text(path: Path, max_rows: int = 200) -> str:
    """Load a CSV or Excel file and return as CSV text for the LLM prompt."""
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
    """Return the first existing path from candidates, or None."""
    for p in candidates:
        if p.exists():
            return p
    return None


def get_table_paths(table_file: str) -> tuple[Path | None, Path | None]:
    """
    Resolve original and normalized table paths, trying multiple naming conventions:
      Original:   {base}.xlsx.csv  →  {base}.csv  →  {base}.xlsx
      Normalized: {base}.xlsx_normalized.csv  →  {base}_normalized.csv  →  {base}_normalized.xlsx
    """
    base = table_file.replace(".xlsx", "")

    original = resolve_path([
        ORIGINAL_DIR / f"{base}.xlsx.csv",
        ORIGINAL_DIR / f"{base}.csv",
        ORIGINAL_DIR / f"{base}.xlsx",
        ORIGINAL_DIR / table_file,        # exact filename as-is
    ])

    normalized = resolve_path([
        NORMALIZED_DIR / f"{base}.xlsx_normalized.csv",
        NORMALIZED_DIR / f"{base}_normalized.csv",
        NORMALIZED_DIR / f"{base}_normalized.xlsx",
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
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=256,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_TEMPLATE.format(
                    table=table_text,
                    question=question,
                )},
            ],
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

        # Rule 4: ground truth appears as substring in answer
        if gt_lower in ans_lower:
            return True

    return False


# ── main evaluation loop ───────────────────────────────────────────────────
def run_evaluation(
    provider: str,
    model: str,
    question_ids: list[str] | None = None,
    verbose: bool = False,
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

    # Pre-load all tables (cache by filename to avoid repeated reads)
    table_cache: dict[str, tuple[str, str]] = {}

    results = []
    orig_correct = 0
    norm_correct = 0

    for i, qa in enumerate(qa_pairs, 1):
        table_file = qa["table_file"]
        question = qa["question"]
        ground_truths = qa["answers"]

        if table_file not in table_cache:
            orig_path, norm_path = get_table_paths(table_file)
            if orig_path is None:
                print(f"  [WARN] original table not found for {table_file}, skipping")
                table_cache[table_file] = ("[Table file not found]", "[Table file not found]")
            elif norm_path is None:
                print(f"  [WARN] normalized table not found for {table_file}, skipping normalized side")
                table_cache[table_file] = (
                    load_table_as_text(orig_path),
                    "[Normalized table file not found]",
                )
            else:
                table_cache[table_file] = (
                    load_table_as_text(orig_path),
                    load_table_as_text(norm_path),
                )
        orig_text, norm_text = table_cache[table_file]

        orig_answer = ask_llm(client, model, orig_text, question)
        norm_answer = ask_llm(client, model, norm_text, question)

        orig_ok = is_correct(orig_answer, ground_truths)
        norm_ok = is_correct(norm_answer, ground_truths)

        if orig_ok:
            orig_correct += 1
        if norm_ok:
            norm_correct += 1

        result = {
            "id": qa["id"],
            "question": question,
            "ground_truth": ground_truths,
            "original_answer": orig_answer,
            "original_correct": orig_ok,
            "normalized_answer": norm_answer,
            "normalized_correct": norm_ok,
            "table_file": table_file,
        }
        results.append(result)

        status = f"[{i:02d}/{len(qa_pairs)}] {qa['id']}"
        orig_mark = "✓" if orig_ok else "✗"
        norm_mark = "✓" if norm_ok else "✗"
        if verbose:
            print(f"{status}  orig={orig_mark} ({orig_answer[:40]!r})  norm={norm_mark} ({norm_answer[:40]!r})")
        else:
            print(f"{status}  orig={orig_mark}  norm={norm_mark}")

    total = len(results)
    summary = {
        "total": total,
        "original_correct": orig_correct,
        "normalized_correct": norm_correct,
        "original_accuracy": round(orig_correct / total * 100, 1) if total else 0,
        "normalized_accuracy": round(norm_correct / total * 100, 1) if total else 0,
    }

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "qa_model": model,
        "qa_provider": provider,
        "summary": summary,
        "results": results,
    }

    ts = output["timestamp"]
    out_path = RESULTS_DIR / f"results_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Results saved → {out_path.name}")
    print(f"Original  accuracy: {summary['original_accuracy']}%  ({orig_correct}/{total})")
    print(f"Normalized accuracy: {summary['normalized_accuracy']}%  ({norm_correct}/{total})")
    delta = summary["normalized_accuracy"] - summary["original_accuracy"]
    sign = "+" if delta >= 0 else ""
    print(f"Improvement: {sign}{delta:.1f}%")
    print(f"{'='*50}")

    return output


# ── CLI ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Evaluate normalization quality via LLM QA")
    parser.add_argument("--provider", default="openai",
                        choices=["openai", "gemini", "deepseek"],
                        help="LLM provider (default: openai)")
    parser.add_argument("--model", default=None,
                        help="Model name (default: provider's default)")
    parser.add_argument("--questions", default=None,
                        help="Comma-separated question IDs to run, e.g. qa-001,qa-002")
    parser.add_argument("--verbose", action="store_true",
                        help="Print answer details for each question")
    args = parser.parse_args()

    # Resolve default model per provider
    _, default_model = make_client(args.provider)
    model = args.model or default_model

    question_ids = None
    if args.questions:
        question_ids = [q.strip() for q in args.questions.split(",")]

    print(f"Provider: {args.provider}  Model: {model}")
    print(f"Questions: {len(question_ids) if question_ids else 'all (40)'}")
    print()

    run_evaluation(
        provider=args.provider,
        model=model,
        question_ids=question_ids,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
