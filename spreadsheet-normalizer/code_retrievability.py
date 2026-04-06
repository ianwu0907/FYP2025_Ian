"""
Code Retrievability Evaluator

Measures how well an LLM can fill schema slots in fixed query templates
when given raw vs. pipeline-normalised tables. Isolates table structure
quality from LLM coding ability — the code logic is fixed; only the
schema slot values (column names, filter values) are filled by the LLM.

Evaluation flow:
  1. Generate QA pairs with known numeric answers from reference_tidy_df.
     Questions are phrased as semantic intent (not exposing column names),
     so they remain valid even when the pipeline output uses different
     column names or value representations.
  2. For each QA pair, present the same question + full column profile
     (column names + distinct values) to the LLM for both raw_df and
     pipeline_df. The LLM infers the correct slot values by inspecting
     both column names and data content.
  3. Slot 'x' must be chosen from the column's actual distinct values,
     not invented — this ensures template execution never fails due to
     a value that doesn't exist in the table.
  4. Execute the filled template against the actual DataFrame.
  5. Compare numeric result to ground-truth answer (1% relative tolerance).
  6. Report execution_rate and accuracy, broken down by query type.

Design rationale:
  - Fixed templates          → LLM coding ability is not a confound.
  - Semantic questions       → schema mismatch between reference and
                               pipeline output does not unfairly penalise
                               the pipeline.
  - Column profiles          → LLM can infer semantics from data content
                               even when column names are Unnamed/opaque.
  - x from distinct values   → execution never fails due to a fabricated
                               filter value; difficulty is in semantic
                               mapping, not string guessing.
  - reference_tidy_df ≠ pipeline_df → ground truth generation and
                               evaluation target are fully decoupled.
"""

import json
import logging
import math
import textwrap
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from openai import OpenAI

from src.metrics.tidiness_metrics import _detect_dim_cols, _detect_measure_cols, _is_ratio_col

logger = logging.getLogger(__name__)


# ======================================================================
# Query Templates
# ======================================================================

def _norm_str(s: str) -> str:
    """Normalise whole-number float strings to int strings.
    "2020.0" -> "2020";  "3.14" -> "3.14" (unchanged).
    Fix 5: ensures LLM-filled slots always match column profile values.
    """
    try:
        import math as _m
        f = float(s)
        if not _m.isnan(f) and not _m.isinf(f) and f == int(f):
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


TEMPLATES = {
    "lookup": {
        # Uses injected local vars (filters_dict, B_col) — no string
        # placeholders — so single quotes in values never cause SyntaxError.
        # Fix 1.
        "code": (
            "import pandas as pd\n"
            "mask = pd.Series([True] * len(df), index=df.index)\n"
            "for _col, _val in filters_dict.items():\n"
            "    _col_vals = df[_col].astype(str).str.strip().apply(_norm_str)\n"
            "    mask &= _col_vals == _norm_str(str(_val))\n"
            "result = pd.to_numeric(df[mask][B_col], errors='coerce').dropna().iloc[0]"
        ),
        "slots": ["filters", "B"],
    },
    "aggregation": {
        "code": (
            "import pandas as pd\n"
            "mask = pd.Series([True] * len(df), index=df.index)\n"
            "for _col, _val in filters_dict.items():\n"
            "    _col_vals = df[_col].astype(str).str.strip().apply(_norm_str)\n"
            "    mask &= _col_vals == _norm_str(str(_val))\n"
            "if mask.sum() == 0:\n"
            "    result = None\n"
            "else:\n"
            "    result = pd.to_numeric(df[mask][B_col], errors='coerce').sum()"
        ),
        "slots": ["filters", "B"],
    },
}


# ======================================================================
# QA Pair Generation  (from reference_tidy_df only)
# ======================================================================

def _make_semantic_question(qtype: str, dim: str, x: str,
                            measure: str,
                            filters: dict = None) -> str:
    """
    Build a natural-language question that describes semantic intent
    without exposing the exact column name or value string from
    reference_tidy_df.

    For lookup with composite key, filters contains all dim→val pairs.
    """
    if qtype == "lookup":
        if filters and len(filters) > 1:
            conditions = " and ".join(
                f"{col} is '{val}'" for col, val in filters.items()
            )
            return f"Find the {measure} value where {conditions}."
        return (
            f"Find the {measure} value for the group "
            f"where {dim} is '{x}'."
        )
    else:  # aggregation
        return (
            f"What is the total {measure} across all records "
            f"where {dim} is '{x}'?"
        )


def generate_qa_pairs(
        reference_tidy_df: pd.DataFrame,
        n_per_type: int = 5,
        random_state: int = 42,
) -> List[Dict[str, Any]]:
    """
    Generate QA pairs with exact numeric ground-truth answers from
    reference_tidy_df.

    Public fields in each pair dict:
        qtype    — "lookup" | "aggregation"
        question — semantic natural-language question (no raw column names)
        answer   — exact numeric ground truth (float)

    Internal fields (used for logging / debugging, NOT passed to LLM):
        _ref_dim     — dimension column name in reference_tidy_df
        _ref_x       — filter value as it appears in reference_tidy_df
        _ref_measure — measure column name in reference_tidy_df

    Args:
        reference_tidy_df: Normalised, tidy DataFrame used as ground truth.
        n_per_type:        Max QA pairs generated per query type.
        random_state:      Seed for reproducibility.

    Returns:
        List of QA pair dicts.
    """
    rng      = np.random.default_rng(random_state)

    # Lookup can use all numeric columns including percent/rate.
    lookup_measure_cols = _detect_measure_cols(reference_tidy_df,
                                               exclude_ratio=False)
    # Aggregation excludes ratio columns — summing percent is meaningless.
    agg_measure_cols    = _detect_measure_cols(reference_tidy_df,
                                               exclude_ratio=True)

    # For QA generation, use all string-dominant columns as dim candidates —
    # more permissive than _detect_dim_cols, which excludes high-cardinality
    # columns. The lookup uniqueness check (sum() == 1) handles ambiguity
    # separately, so there is no need to pre-filter by cardinality here.
    def _qa_dim_cols(df: pd.DataFrame) -> List:
        cols = []
        for col in df.columns:
            non_empty = df[col].dropna()
            if len(non_empty) == 0:
                continue
            n_str = sum(1 for v in non_empty if isinstance(v, str))
            if n_str / len(non_empty) > 0.5:
                cols.append(col)
        return cols

    dim_cols = _qa_dim_cols(reference_tidy_df)

    if not dim_cols or not lookup_measure_cols:
        # Detailed diagnostic to help understand why detection failed
        logger.warning(
            "generate_qa_pairs: could not detect dim/measure columns. "
            "Returning empty list."
        )
        logger.warning(f"  DataFrame shape : {reference_tidy_df.shape}")
        logger.warning(f"  Columns         : {list(reference_tidy_df.columns)}")
        logger.warning(f"  _qa_dim_cols    : {dim_cols}")
        logger.warning(f"  lookup_measures : {lookup_measure_cols}")
        for col in reference_tidy_df.columns:
            non_empty = reference_tidy_df[col].dropna()
            if len(non_empty) == 0:
                logger.warning(f"    {str(col):<30} ALL NULL")
                continue
            n_str = sum(1 for v in non_empty if isinstance(v, str))
            cleaned = non_empty.astype(str).str.replace(",", "", regex=False)
            parsed  = pd.to_numeric(cleaned, errors="coerce")
            num_rate = parsed.notna().sum() / max(len(reference_tidy_df), 1)
            str_frac = n_str / len(non_empty)
            logger.warning(
                f"    {str(col):<30} str={str_frac:.2f}  "
                f"num={num_rate:.2f}  unique={reference_tidy_df[col].nunique()}/{len(reference_tidy_df)}"
            )
        return []

    pairs = []

    # ── Lookup pairs ──────────────────────────────────────────────────
    # Composite key: use ALL columns except the target measure as filters.
    # This correctly includes temporal/numeric dimensions like `year` that
    # are not detected as string dim_cols but are essential for row identity.
    # Every row in a proper tidy table should be uniquely identified by
    # (all other columns) → the composite key is always the full row minus
    # the measure being queried.
    candidates = []
    all_cols = list(reference_tidy_df.columns)

    for measure in lookup_measure_cols:
        measure_idx = all_cols.index(measure)
        # All columns except the measure become the composite key
        key_col_indices = [i for i, c in enumerate(all_cols) if c != measure]
        key_col_names   = [all_cols[i] for i in key_col_indices]

        if not key_col_names:
            continue

        # Build sub-table: key cols + measure col
        sub = reference_tidy_df.iloc[:, key_col_indices + [measure_idx]].copy()
        sub.columns = [f"_k{i}" for i in range(len(key_col_names))] + ["_measure"]
        sub["_measure"] = pd.to_numeric(
            sub["_measure"].astype(str).str.replace(",", "", regex=False),
            errors="coerce"
        )
        sub = sub.dropna(subset=["_measure"])
        if sub.empty:
            continue

        for _, row in sub.iterrows():
            filters = {
                key_col_names[i]: str(row[f"_k{i}"]).strip()
                for i in range(len(key_col_names))
            }
            # Verify composite key uniquely identifies exactly one row
            mask = pd.Series([True] * len(sub), index=sub.index)
            for i in range(len(key_col_names)):
                mask &= sub[f"_k{i}"].astype(str).str.strip() == filters[key_col_names[i]]
            if mask.sum() == 1:
                # Use the first string-type key col as the "primary dim"
                # for question generation (most semantically meaningful)
                primary_dim = next(
                    (c for c in key_col_names
                     if c in dim_cols and not _is_ratio_col(c)),
                    key_col_names[0]
                )
                candidates.append({
                    "qtype":        "lookup",
                    "_ref_dim":     primary_dim,
                    "_ref_x":       filters[primary_dim],
                    "_ref_measure": measure,
                    "_filters":     filters,
                    "answer":       float(row["_measure"]),
                })

    logger.info(
        f"generate_qa_pairs [{reference_tidy_df.shape}]: "
        f"{len(candidates)} lookup candidates from "
        f"{len(dim_cols)} dim x {len(lookup_measure_cols)} measure cols"
    )
    if not candidates:
        # Help diagnose why — log per (dim, measure) pair
        for dim in dim_cols[:3]:
            for measure in lookup_measure_cols[:3]:
                if dim == measure:
                    continue
                try:
                    dim_idx     = list(reference_tidy_df.columns).index(dim)
                    measure_idx = list(reference_tidy_df.columns).index(measure)
                    sub = reference_tidy_df.iloc[:, [dim_idx, measure_idx]].copy()
                    sub.columns = ["_dim", "_measure"]
                    sub["_measure"] = pd.to_numeric(
                        sub["_measure"].astype(str).str.replace(",", "", regex=False),
                        errors="coerce"
                    )
                    sub_clean = sub.dropna()
                    vc = sub_clean["_dim"].astype(str).str.strip().value_counts()
                    multi = (vc > 1).sum()
                    logger.warning(
                        f"  dim={dim!r} x measure={measure!r}: "
                        f"{len(sub_clean)} non-null rows, "
                        f"{multi}/{len(vc)} dim values appear >1 time "
                        f"(all non-unique → 0 lookup candidates)"
                    )
                except Exception as e:
                    logger.warning(f"  dim={dim!r} x measure={measure!r}: ERROR {e}")

    if candidates:
        idxs = rng.choice(len(candidates),
                          size=min(n_per_type, len(candidates)),
                          replace=False)
        for i in idxs:
            c = candidates[i]
            c["question"] = _make_semantic_question(
                "lookup", c["_ref_dim"], c["_ref_x"], c["_ref_measure"],
                filters=c.get("_filters")
            )
            pairs.append(c)

    # ── Aggregation pairs ─────────────────────────────────────────────
    # Only keep groups where >1 row contributes, so the sum is non-trivial.
    # Ratio columns (percent, rate, etc.) are excluded — their sums are
    # semantically invalid.
    agg_candidates = []
    for dim in dim_cols:
        for measure in agg_measure_cols:
            # Use positional selection to avoid duplicate-column-name ambiguity.
            dim_idx     = list(reference_tidy_df.columns).index(dim)
            measure_idx = list(reference_tidy_df.columns).index(measure)
            sub = reference_tidy_df.iloc[:, [dim_idx, measure_idx]].copy()
            sub.columns = ["_dim", "_measure"]
            sub["_measure"] = pd.to_numeric(sub["_measure"], errors="coerce")
            sub = sub.dropna()
            if sub.empty:
                continue
            counts  = sub.groupby("_dim")["_measure"].count()
            grouped = sub.groupby("_dim")["_measure"].sum().reset_index()
            grouped = grouped[grouped["_dim"].map(counts) > 1]
            if grouped.empty:
                continue
            for _, row in grouped.iterrows():
                agg_candidates.append({
                    "qtype":        "aggregation",
                    "_ref_dim":     dim,
                    "_ref_x":       str(row["_dim"]).strip(),
                    "_ref_measure": measure,
                    "answer":       float(row["_measure"]),
                })

    logger.info(
        f"generate_qa_pairs: "
        f"{len(agg_candidates)} aggregation candidates "
        f"(excluded ratio cols: {[c for c in _detect_measure_cols(reference_tidy_df, exclude_ratio=False) if _is_ratio_col(c)]})"
    )

    if agg_candidates:
        idxs = rng.choice(len(agg_candidates),
                          size=min(n_per_type, len(agg_candidates)),
                          replace=False)
        for i in idxs:
            c = agg_candidates[i]
            c["question"] = _make_semantic_question(
                "aggregation", c["_ref_dim"], c["_ref_x"], c["_ref_measure"]
            )
            pairs.append(c)

    logger.info(
        f"generate_qa_pairs: {len(pairs)} pairs generated "
        f"({sum(1 for p in pairs if p['qtype']=='lookup')} lookup, "
        f"{sum(1 for p in pairs if p['qtype']=='aggregation')} aggregation)"
    )
    return pairs


# ======================================================================
# Column Profile Builder
# ======================================================================

def _build_column_profiles(df: pd.DataFrame,
                           max_distinct: int = 20,
                           must_include: List[Any] = None) -> Dict[str, List[str]]:
    """
    Ensures that 'must_include' values (Ground Truth) are prioritized in the
    profile to avoid prompt truncation blind spots.
    """
    targets = set([_norm_str(str(v).strip()) for v in (must_include or [])])
    profiles = {}
    for col in df.columns:
        # 获取所有规范化后的唯一值
        all_vals = [_norm_str(v) for v in df[col].dropna().astype(str).str.strip().unique()]

        if len(all_vals) > max_distinct:
            guaranteed = [v for v in all_vals if v in targets]
            others = [v for v in all_vals if v not in targets]

            final_vals = (guaranteed + others)[:max_distinct]
        else:
            final_vals = all_vals

        profiles[str(col)] = final_vals
    return profiles


# ======================================================================
# LLM Slot Filling
# ======================================================================

_SLOT_SYSTEM_PROMPT = textwrap.dedent("""
You are a data analyst. You will be given:
  1. A question describing a data retrieval intent using concept labels
     that may not match the actual column names in the table.
  2. The table's column names, a data preview, and per-column distinct values.
  3. A query template to fill. Both LOOKUP and AGGREGATION templates now use
     the exact same slot structure: "filters" (a dict) and "B" (measure column).

Slot filling rules:
  - "filters": a JSON object mapping each dimension column name to its filter
    value. Include ALL dimension columns needed to isolate the target data
    (for LOOKUP, this uniquely identifies a single row; for AGGREGATION, this
    defines the group to sum). Every key must be EXACTLY a column name; every
    value must be EXACTLY a string from that column's distinct values list.
  - "B": EXACTLY one column name — the measure to retrieve or sum.

General rules:
  - Use BOTH column names AND distinct values to infer semantic meaning.
  - All column names and values must be copied exactly from what is shown.
  - Output ONLY a valid JSON object. No explanation, no markdown fences.

Example output (for BOTH Lookup and Aggregation):
{"filters": {"country_of_birth": "cameroon", "gender": "female"}, "B": "count"}
""").strip()


def _build_slot_prompt(
        question: str,
        qtype: str,
        df: pd.DataFrame,
        preview_rows: int = 15,
        max_distinct: int = 20,
        target_values: list = None,
) -> str:
    """
    Build the user prompt for slot filling, including:
      - The semantic question
      - Column names
      - Data preview rows
      - Per-column distinct value profiles (allows LLM to infer semantics
        from data content even when column names are opaque)
      - The template code to fill
    """
    template_code = TEMPLATES[qtype]["code"]
    col_list      = [str(c) for c in df.columns]
    preview       = df.head(preview_rows).to_string(index=False)
    profiles = _build_column_profiles(df, max_distinct=max_distinct, must_include=target_values)

    profiles_str = "\n".join(
        f'  "{col}": {vals}' for col, vals in profiles.items()
    )

    return textwrap.dedent(f"""
    Question (semantic intent):
    {question}

    Table column names:
    {col_list}

    Table preview (first {preview_rows} rows):
    {preview}

    Column profiles (column name → distinct values present in this table):
    {profiles_str}

    Template to fill (slots are {{A}}, {{x}}, {{B}}):
    {template_code}

    Remember:
    - Infer column semantics from BOTH column names AND distinct values.
    - All column names and values must match exactly what is shown.
    - For LOOKUP: output {{"filters": {{...}}, "B": "..."}}.
    - For AGGREGATION: output {{"A": "...", "x": "...", "B": "..."}}.
    - Output ONLY the JSON object. No explanation, no markdown.
    """).strip()


def fill_slots_via_llm(
        question: str,
        qtype: str,
        df: pd.DataFrame,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        preview_rows: int = 15,
        max_distinct: int = 20,
        target_values: list = None
) -> Optional[Dict[str, str]]:
    """
    Ask the LLM to fill template slots for the given question and table.

    Post-fill validation:
      - A must be a real column name.
      - x must exist in column A's actual distinct values.
      - B must be a real column name.

    Returns:
        Dict with keys A, x, B (all strings), or None on failure.
    """
    user_prompt = _build_slot_prompt(
        question=question,
        qtype=qtype,
        df=df,
        preview_rows=preview_rows,
        max_distinct=max_distinct,
        target_values=target_values
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _SLOT_SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=12800,
        )
        raw = response.choices[0].message.content.strip()

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        slots    = json.loads(raw)
        required = TEMPLATES[qtype]["slots"]

        if not all(k in slots for k in required):
            logger.warning(f"fill_slots_via_llm: missing required slots in: {raw}")
            return None

        col_strs = [str(c) for c in df.columns]

        filters = slots.get("filters")
        if not isinstance(filters, dict) or not filters:
            logger.warning(f"fill_slots_via_llm: filters must be a non-empty dict, got: {filters}")
            return None

        for col, val in list(filters.items()):
            if col not in col_strs:
                logger.warning(f"fill_slots_via_llm: filters key '{col}' is not a column.")
                return None
            # Fix 4: strip val before comparing (LLM may include whitespace).
            # Fix 5: normalise float-int ("2020.0" == "2020").
            actual_vals = [_norm_str(v) for v in
                           df[col].dropna().astype(str).str.strip().unique()]
            val_norm = _norm_str(str(val).strip())
            if val_norm not in actual_vals:
                logger.warning(
                    f"fill_slots_via_llm: filters['{col}']='{val}' not in column. "
                    f"Available: {actual_vals[:10]}"
                )
                return None

        b = slots.get("B")
        if b not in col_strs:
            logger.warning(f"fill_slots_via_llm: B='{b}' is not a column.")
            return None

        return {"filters": {k: str(v) for k, v in filters.items()}, "B": str(b)}

    except (json.JSONDecodeError, KeyError, Exception) as e:
        logger.warning(f"fill_slots_via_llm failed: {type(e).__name__}: {e}")
        return None


# ======================================================================
# Template Execution
# ======================================================================

def execute_template(
        qtype: str,
        slots: Dict[str, str],
        df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[str]]:
    """
    Execute fixed template via local_vars injection — no string replace().
    Fix 1: single quotes in values never cause SyntaxError.
    Fix 8: aggregation returns error when filter matches 0 rows, distinguishing
    genuine sum=0 from "no rows found" (both produce result=0.0 without this).

    Returns:
        (result_float, error_str) — exactly one of them is None.
    """
    code = TEMPLATES[qtype]["code"]
    local_vars: Dict[str, Any] = {"df": df.copy(), "pd": pd, "_norm_str": _norm_str}

    local_vars["filters_dict"] = slots.get("filters", {})
    local_vars["B_col"]        = slots["B"]

    try:
        exec(code, {}, local_vars)  # noqa: S102
        result = local_vars.get("result")

        if isinstance(result, pd.Series):
            if len(result) == 1:
                result = float(result.iloc[0])
            else:
                return None, f"ResultError: expected scalar, got Series of len {len(result)}"
        elif result is None:
            return None, "ResultError: result is None"
        else:
            result = float(result)

        if math.isnan(result):
            return None, "ResultError: result is NaN"

        return result, None

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ======================================================================
# Answer Comparison
# ======================================================================

def is_correct(
        result: Optional[float],
        answer: float,
        rtol: float = 0.01,
) -> bool:
    """
    True if result matches answer within 1% relative tolerance.
    """
    if result is None:
        return False
    if answer == 0.0:
        return abs(result) < 1e-6
    return abs(result - answer) / abs(answer) <= rtol


# ======================================================================
# Main Evaluation Pipeline
# ======================================================================

def evaluate_code_retrievability(
        raw_df: pd.DataFrame,
        pipeline_df: pd.DataFrame,
        reference_tidy_df: pd.DataFrame,
        client: OpenAI,
        model: str = "gpt-4o-mini",
        n_per_type: int = 5,
        preview_rows: int = 15,
        max_distinct: int = 20,
        random_state: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate code retrievability of raw_df vs. pipeline_df.

    QA pairs (with numeric ground-truth answers) are generated from
    reference_tidy_df, which may differ from pipeline_df in column names
    and value representations. Questions are phrased semantically so they
    remain valid regardless of the target table's schema.

    Args:
        raw_df:            Original, un-normalised DataFrame.
        pipeline_df:       DataFrame produced by the normalisation pipeline.
        reference_tidy_df: Canonical tidy DataFrame used only for ground-
                           truth generation. Can be human-curated or the
                           pipeline output itself if no separate reference
                           is available.
        client:            OpenAI client instance.
        model:             LLM model identifier.
        n_per_type:        Max QA pairs per query type.
        preview_rows:      Data rows shown to LLM in table preview.
        max_distinct:      Max distinct values shown per column in profiles.
        random_state:      Seed for QA pair sampling reproducibility.

    Returns:
        Dict:
            "qa_pairs" — list of QA pair dicts with results filled in.
            "scores"   — summary scores for raw and pipeline.
    """
    qa_pairs = generate_qa_pairs(
        reference_tidy_df,
        n_per_type=n_per_type,
        random_state=random_state,
    )
    if not qa_pairs:
        logger.error("No QA pairs generated — cannot evaluate.")
        return {"qa_pairs": [], "scores": {}}

    for pair in qa_pairs:
        pair["results"] = {}

        for label, df in [("raw", raw_df), ("pipeline", pipeline_df)]:
            logger.info(
                f"  [{label}] {pair['qtype']}: "
                f"{pair['question'][:70]}..."
            )
            target_vals = []
            if "_ref_x" in pair: target_vals.append(pair["_ref_x"])
            if "_filters" in pair: target_vals.extend(pair["_filters"].values())

            slots = fill_slots_via_llm(
                question=pair["question"],
                qtype=pair["qtype"],
                df=df,
                client=client,
                model=model,
                preview_rows=preview_rows,
                max_distinct=max_distinct,
                target_values=target_vals,
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

    return {
        "qa_pairs": qa_pairs,
        "scores":   compute_scores(qa_pairs),
    }


# ======================================================================
# Scoring
# ======================================================================

def compute_scores(qa_pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Summarise evaluation results.

    Returns:
        {
            "raw":      { "n", "execution_rate", "accuracy",
                          "by_type": { "lookup": {...},
                                       "aggregation": {...} } },
            "pipeline": { ... },
        }
    """
    scores = {}

    for label in ["raw", "pipeline"]:
        all_pairs = [p for p in qa_pairs if label in p.get("results", {})]
        if not all_pairs:
            scores[label] = {}
            continue

        def _rate(pairs, key):
            if not pairs:
                return 0.0
            return round(
                sum(p["results"][label][key] for p in pairs) / len(pairs), 4
            )

        by_type = {}
        for qtype in ["lookup", "aggregation"]:
            sub = [p for p in all_pairs if p["qtype"] == qtype]
            by_type[qtype] = {
                "n":              len(sub),
                "execution_rate": _rate(sub, "executed"),
                "accuracy":       _rate(sub, "correct"),
            }

        scores[label] = {
            "n":              len(all_pairs),
            "execution_rate": _rate(all_pairs, "executed"),
            "accuracy":       _rate(all_pairs, "correct"),
            "by_type":        by_type,
        }

    return scores


# ======================================================================
# Logging / Reporting
# ======================================================================

def log_results(evaluation: Dict[str, Any]):
    """Pretty-print full evaluation results to the logger."""
    qa_pairs = evaluation.get("qa_pairs", [])
    scores   = evaluation.get("scores",   {})

    # ── Per-question detail ───────────────────────────────────────────
    logger.info("CODE RETRIEVABILITY — PER-QUESTION DETAIL")
    logger.info("=" * 76)

    for i, pair in enumerate(qa_pairs, 1):
        logger.info(f"  Q{i} [{pair['qtype']}] {pair['question']}")
        logger.info(
            f"       answer={pair['answer']}  "
            f"(ref: {pair['_ref_dim']}='{pair['_ref_x']}' "
            f"→ {pair['_ref_measure']})"
        )
        for label in ["raw", "pipeline"]:
            r = pair.get("results", {}).get(label)
            if r is None:
                continue
            mark       = "✓" if r["correct"] else "✗"
            slots_str  = json.dumps(r["slots"] or {})
            result_str = (
                f"{r['result']:.4f}" if r["result"] is not None else "—"
            )
            err_str = f"  [{r['error']}]" if r["error"] else ""
            logger.info(
                f"       {label:<8}  {mark}  slots={slots_str}  "
                f"result={result_str}{err_str}"
            )
        logger.info("")

    # ── Score summary ─────────────────────────────────────────────────
    logger.info("CODE RETRIEVABILITY — SCORE SUMMARY")
    logger.info("-" * 76)
    logger.info(
        f"  {'Metric':<38} {'Raw':>8} {'Pipeline':>9} {'Delta':>8}"
    )
    logger.info("-" * 76)

    raw_s = scores.get("raw",      {})
    pip_s = scores.get("pipeline", {})

    def _row(label, raw_val, pip_val):
        delta = round(pip_val - raw_val, 4)
        arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "=")
        logger.info(
            f"  {label:<38} {raw_val:>8.4f} {pip_val:>9.4f} "
            f"{delta:>+8.4f}  {arrow}"
        )

    _row("execution_rate (all)",
         raw_s.get("execution_rate", 0.0),
         pip_s.get("execution_rate", 0.0))
    _row("accuracy (all)",
         raw_s.get("accuracy", 0.0),
         pip_s.get("accuracy", 0.0))

    for qtype in ["lookup", "aggregation"]:
        raw_bt = raw_s.get("by_type", {}).get(qtype, {})
        pip_bt = pip_s.get("by_type", {}).get(qtype, {})
        _row(f"  execution_rate [{qtype}]",
             raw_bt.get("execution_rate", 0.0),
             pip_bt.get("execution_rate", 0.0))
        _row(f"  accuracy [{qtype}]",
             raw_bt.get("accuracy", 0.0),
             pip_bt.get("accuracy", 0.0))

    logger.info("-" * 76)
    logger.info(
        f"  n(raw)={raw_s.get('n', 0)}  "
        f"n(pipeline)={pip_s.get('n', 0)}"
    )