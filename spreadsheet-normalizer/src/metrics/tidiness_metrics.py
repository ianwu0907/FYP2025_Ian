"""
Tidiness Metrics — Quantitative assessment of table structural quality.

Computes a suite of metrics on any DataFrame, designed to be called
before (raw input) and after (tidy output) transformation to measure
improvement.

References:
- Cell Coverage, Row Completeness: adapted from Rosin (1999), Adelfio & Samet (2013)
- Column Type Homogeneity: Proactive Wrangler, Guo et al. (UIST 2011)
- Type Consistency: Abedjan, Golab & Naumann (VLDB Journal 2015)
- Column Completeness: Pipino, Lee & Wang (CACM 2002)
- Inter-Column NMI: standard information-theoretic measure
- Substring Containment: novel metric for granularity mixing detection
- Groupby Queryability: novel metric for analytical operability
"""

import logging
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter

logger = logging.getLogger(__name__)


# ======================================================================
# Shared helpers
# ======================================================================

def _detect_dim_cols(df: pd.DataFrame) -> List:
    """
    Detect dimension columns: string-dominant and not a unique-key column.

    Cardinality rule: unique values < n_rows (at least one value repeats,
    OR the column has fewer distinct values than rows). The old 50% threshold
    was too strict for small tables — a 7-row table with 6 distinct category
    labels is a valid dimension column.
    Used by inter_column_nmi and groupby_queryability.
    """
    n_rows = len(df)
    dim_cols = []
    for col in df.columns:
        non_empty = df[col].dropna()
        if len(non_empty) == 0:
            continue
        n_str = sum(1 for v in non_empty if isinstance(v, str))
        if n_str / len(non_empty) > 0.5:
            # Exclude only pure ID columns where every value is unique
            if df[col].nunique() < n_rows:
                dim_cols.append(col)
    return dim_cols


# Column name substrings that indicate ratio/proportion columns.
# Summing such columns across rows produces meaningless answers,
# so they are excluded from aggregation but valid for lookup.
_RATIO_COL_PATTERNS = {"percent", "pct", "rate", "ratio", "proportion", "share", "%"}


def _is_ratio_col(col_name: str) -> bool:
    name = str(col_name).lower()
    return any(pat in name for pat in _RATIO_COL_PATTERNS)


def _detect_measure_cols(df: pd.DataFrame,
                         exclude_ratio: bool = False) -> List:
    """
    Detect measure columns: >50% of non-null values parse as numeric.
    Handles comma-formatted numbers (e.g. "15,765") by stripping commas
    before parsing.

    Args:
        exclude_ratio: If True, columns whose names suggest percentage/ratio
                       (percent, rate, ratio, etc.) are excluded.
                       Use True for aggregation (sum), False for lookup.
    """
    measure_cols = []
    for col in df.columns:
        if exclude_ratio and _is_ratio_col(col):
            continue
        series = df[col].dropna().astype(str).str.replace(",", "", regex=False)
        parsed = pd.to_numeric(series, errors="coerce")
        if parsed.notna().sum() / max(len(df), 1) >= 0.5:
            measure_cols.append(col)
    return measure_cols


# ======================================================================
# Individual metrics
# ======================================================================

def cell_coverage(df: pd.DataFrame) -> float:
    """
    Ratio of non-empty cells to total cells in the bounding rectangle.
    Measures rectangularity / data density.
    Perfect rectangle with no gaps → 1.0.
    """
    total = df.shape[0] * df.shape[1]
    if total == 0:
        return 0.0
    non_empty = df.notna().sum().sum()
    return round(non_empty / total, 4)


def row_completeness_uniformity(df: pd.DataFrame) -> float:
    """
    1 - std(per-row fill rate).
    Measures whether all rows have similar completeness.
    Uniform rows → 1.0.  Mixed metadata/data rows → lower.
    """
    if df.empty:
        return 0.0
    row_fill = df.notna().mean(axis=1)
    return round(1.0 - row_fill.std(), 4)


def column_type_homogeneity(df: pd.DataFrame) -> float:
    """
    Herfindahl-Hirschman Index over {numeric, text, empty} per column.
    Average across all columns.
    HHI = sum(fraction_i^2).  Perfectly homogeneous column → 1.0.
    Source: Proactive Wrangler (Guo et al., 2011).
    """
    if df.empty or df.shape[1] == 0:
        return 0.0

    scores = []
    for col in df.columns:
        total = len(df)
        if total == 0:
            continue

        series = df[col]
        n_empty = series.isna().sum()
        non_empty = series.dropna()

        # Classify non-empty values
        numeric_mask = pd.to_numeric(non_empty, errors='coerce').notna()
        n_numeric = numeric_mask.sum()
        n_text = len(non_empty) - n_numeric

        fracs = [n_numeric / total, n_text / total, n_empty / total]
        hhi = sum(f ** 2 for f in fracs)
        scores.append(hhi)

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def data_row_ratio(df: pd.DataFrame) -> float:
    """
    Fraction of rows that contain at least one numeric value.
    Measures physical data density (excludes headers, metadata, blanks).
    Source: adapted from Adelfio & Samet (2013).
    """
    if df.empty:
        return 0.0

    def _has_numeric(row):
        for val in row:
            if pd.isna(val):
                continue
            if isinstance(val, (int, float, np.integer, np.floating)):
                return True
            if isinstance(val, str):
                s = val.strip()
                try:
                    float(s.replace(',', ''))
                    return True
                except ValueError:
                    pass
        return False

    n_data = sum(1 for _, row in df.iterrows() if _has_numeric(row))
    return round(n_data / len(df), 4)


def column_completeness_min(df: pd.DataFrame) -> float:
    """
    Minimum per-column non-null rate (bottleneck column).
    Source: Pipino, Lee & Wang (CACM 2002).
    Tidy table dimension columns should have no nulls → 1.0.
    """
    if df.empty or df.shape[1] == 0:
        return 0.0
    col_completeness = df.notna().mean()
    return round(col_completeness.min(), 4)


def header_uniqueness(df: pd.DataFrame) -> float:
    """
    Fraction of column names that are unique.
    Auto-generated or duplicate headers → low.  Tidy headers → 1.0.
    """
    if df.shape[1] == 0:
        return 0.0
    n_unique = len(set(str(c) for c in df.columns))
    return round(n_unique / df.shape[1], 4)


def type_consistency(df: pd.DataFrame) -> float:
    """
    Per column: fraction of non-empty values matching the dominant type.
    Average across columns.
    Source: Abedjan, Golab & Naumann (VLDB Journal 2015).
    """
    if df.empty or df.shape[1] == 0:
        return 0.0

    scores = []
    for col in df.columns:
        non_empty = df[col].dropna()
        if len(non_empty) == 0:
            scores.append(0.0)
            continue
        n_numeric = pd.to_numeric(non_empty, errors='coerce').notna().sum()
        n_text = len(non_empty) - n_numeric
        scores.append(max(n_numeric, n_text) / len(non_empty))

    return round(sum(scores) / len(scores), 4) if scores else 0.0


def substring_containment_rate(df: pd.DataFrame,
                               text_cols: Optional[List] = None) -> float:
    """
    For text columns, measure how often one distinct value is a substring
    of another.  High rate → mixed granularity (e.g., 'Physical abuse'
    co-existing with 'Physical abuse - Male').

    Checks only short ⊂ long direction.  O(n^2 * L) per column.
    Average across checked columns.
    """
    if df.empty:
        return 0.0

    if text_cols is None:
        # Auto-detect text columns (>50% string values)
        text_cols = []
        for col in df.columns:
            non_empty = df[col].dropna()
            if len(non_empty) == 0:
                continue
            n_str = sum(1 for v in non_empty if isinstance(v, str))
            if n_str / len(non_empty) > 0.5:
                text_cols.append(col)

    if not text_cols:
        return 0.0

    col_scores = []
    for col in text_cols:
        values = sorted(
            df[col].dropna().astype(str).str.strip().unique(),
            key=len
        )
        n = len(values)
        if n < 2:
            col_scores.append(0.0)
            continue

        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                if len(values[i]) < len(values[j]) and values[i] in values[j]:
                    count += 1

        total_pairs = n * (n - 1)
        col_scores.append(count / total_pairs if total_pairs > 0 else 0.0)

    return round(sum(col_scores) / len(col_scores), 4) if col_scores else 0.0


def inter_column_nmi(df: pd.DataFrame,
                     dim_cols: Optional[List] = None) -> float:
    """
    Average pairwise Normalized Mutual Information across dimension columns.
    High NMI → columns encode overlapping information (poor separation).
    Low NMI → columns are semantically independent (good).

    If dim_cols not provided, uses _detect_dim_cols().
    """
    if df.empty:
        return 0.0

    if dim_cols is None:
        dim_cols = _detect_dim_cols(df)

    if len(dim_cols) < 2:
        return 0.0

    def _nmi(col_a, col_b):
        a_vals = df[col_a].astype(str)
        b_vals = df[col_b].astype(str)
        pairs = list(zip(a_vals, b_vals))
        n = len(pairs)
        if n == 0:
            return 0.0

        joint = Counter(pairs)
        margin_a = Counter(a_vals)
        margin_b = Counter(b_vals)

        mi = 0.0
        for (a, b), c_ab in joint.items():
            p_ab = c_ab / n
            p_a = margin_a[a] / n
            p_b = margin_b[b] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += p_ab * math.log(p_ab / (p_a * p_b))

        h_a = -sum((c / n) * math.log(c / n)
                   for c in margin_a.values() if c > 0)
        h_b = -sum((c / n) * math.log(c / n)
                   for c in margin_b.values() if c > 0)

        if h_a == 0 or h_b == 0:
            return 0.0
        return mi / math.sqrt(h_a * h_b)

    total = 0.0
    count = 0
    for i in range(len(dim_cols)):
        for j in range(i + 1, len(dim_cols)):
            total += _nmi(dim_cols[i], dim_cols[j])
            count += 1

    return round(total / count, 4) if count > 0 else 0.0


def groupby_queryability(df: pd.DataFrame) -> Tuple[float, str]:
    """
    Measures how many (dim, measure) groupby operations produce a
    meaningful aggregation — i.e. the result has fewer rows than the
    input (real grouping occurred) and at least 2 groups.

    Three cases:
      - No dim cols detected          → 0.0, mode="no_dims"
        (table has no groupable structure at all)
      - Dim cols but no measure cols  → size-only probe, mode="size_only"
        (purely categorical table; groupby().size() is still meaningful)
      - Both dim and measure cols     → size + sum probes, mode="full"

    Returns:
        (score, mode) where score ∈ [0.0, 1.0] and mode is a string
        describing which probe set was used.

    Higher is better.
    """
    n_rows = len(df)
    if n_rows < 2:
        return 0.0, "no_dims"

    dim_cols = _detect_dim_cols(df)
    if not dim_cols:
        return 0.0, "no_dims"

    measure_cols = _detect_measure_cols(df)

    def _meaningful(result) -> bool:
        """True if groupby produced real aggregation."""
        return len(result) < n_rows and len(result) >= 2

    if not measure_cols:
        # Size-only mode: groupby().size() across all dim cols
        successes = 0
        for dim in dim_cols:
            try:
                result = df.groupby(dim, dropna=False).size()
                if _meaningful(result):
                    successes += 1
            except Exception:
                pass
        total = len(dim_cols)
        score = round(successes / total, 4) if total > 0 else 0.0
        return score, "size_only"

    # Full mode: both size and sum probes across all (dim, measure) pairs
    successes = 0
    total = 0
    for dim in dim_cols:
        # size probe
        total += 1
        try:
            result = df.groupby(dim, dropna=False).size()
            if _meaningful(result):
                successes += 1
        except Exception:
            pass

        # sum probe per measure col
        for measure in measure_cols:
            total += 1
            try:
                parsed = pd.to_numeric(
                    df[measure].astype(str).str.replace(",", "", regex=False),
                    errors="coerce",
                )
                temp = df.copy()
                temp[measure] = parsed
                result = temp.groupby(dim, dropna=False)[measure].sum()
                if _meaningful(result):
                    successes += 1
            except Exception:
                pass

    score = round(successes / total, 4) if total > 0 else 0.0
    return score, "full"


# ======================================================================
# Composite scorer
# ======================================================================

def compute_all_metrics(df: pd.DataFrame,
                        dim_cols: Optional[List] = None,
                        text_cols: Optional[List] = None,
                        label: str = "") -> Dict[str, Any]:
    """
    Compute all tidiness metrics on a DataFrame.

    Args:
        df:        The DataFrame to evaluate.
        dim_cols:  Dimension columns for NMI (auto-detected if None).
        text_cols: Text columns for substring containment (auto-detected if None).
        label:     Label for logging (e.g., "BEFORE", "AFTER").

    Returns:
        Dict with metric names → values, plus shape info.
        groupby_queryability_mode is stored separately for logging
        but not included in compare_metrics scoring.
    """
    if label:
        logger.info(f"Computing tidiness metrics [{label}]...")

    gq_score, gq_mode = groupby_queryability(df)

    metrics = {
        "shape": {"rows": df.shape[0], "cols": df.shape[1]},
        "cell_coverage": cell_coverage(df),
        "row_completeness_uniformity": row_completeness_uniformity(df),
        "column_type_homogeneity": column_type_homogeneity(df),
        "data_row_ratio": data_row_ratio(df),
        "column_completeness_min": column_completeness_min(df),
        "header_uniqueness": header_uniqueness(df),
        "type_consistency": type_consistency(df),
        "substring_containment": substring_containment_rate(df, text_cols),
        "inter_column_nmi": inter_column_nmi(df, dim_cols),
        "groupby_queryability": gq_score,
        "groupby_queryability_mode": gq_mode,
    }

    if label:
        logger.info(f"  Shape: {metrics['shape']}")
        for k, v in metrics.items():
            if k == "shape":
                continue
            if k == "groupby_queryability":
                logger.info(f"  {k}: {v}  (mode={metrics['groupby_queryability_mode']})")
            elif k == "groupby_queryability_mode":
                continue
            else:
                logger.info(f"  {k}: {v}")

    return metrics


def compare_metrics(before: Dict[str, Any],
                    after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare before/after metrics and compute deltas.

    Returns:
        Dict with each metric's before, after, delta, and direction.

    Note: groupby_queryability_mode is metadata, not scored.
    """
    comparison = {}

    # Metrics where higher = better
    higher_is_better = {
        "cell_coverage", "row_completeness_uniformity",
        "column_type_homogeneity", "data_row_ratio",
        "column_completeness_min", "header_uniqueness",
        "type_consistency", "groupby_queryability",
    }
    # Metrics where lower = better
    lower_is_better = {
        "substring_containment", "inter_column_nmi",
    }

    all_metrics = higher_is_better | lower_is_better

    for metric in all_metrics:
        b = before.get(metric, 0)
        a = after.get(metric, 0)
        delta = round(a - b, 4)

        if metric in higher_is_better:
            improved = delta > 0.01
            degraded = delta < -0.01
        else:
            improved = delta < -0.01
            degraded = delta > 0.01

        if improved:
            direction = "✓ improved"
        elif degraded:
            direction = "✗ degraded"
        else:
            direction = "= unchanged"

        comparison[metric] = {
            "before": b,
            "after": a,
            "delta": delta,
            "direction": direction,
        }

    # Attach mode info as metadata (not scored)
    comparison["_groupby_queryability_mode"] = {
        "before": before.get("groupby_queryability_mode", "n/a"),
        "after": after.get("groupby_queryability_mode", "n/a"),
    }

    return comparison


def log_comparison(comparison: Dict[str, Any]):
    """Pretty-print the before/after comparison."""
    logger.info("TIDINESS METRICS COMPARISON")
    logger.info("-" * 65)
    logger.info(f"  {'Metric':<32} {'Before':>7} {'After':>7} {'Delta':>7}  Dir")
    logger.info("-" * 65)

    for metric, data in comparison.items():
        # Skip metadata entries
        if metric.startswith("_"):
            continue
        logger.info(
            f"  {metric:<32} {data['before']:>7.4f} {data['after']:>7.4f} "
            f"{data['delta']:>+7.4f}  {data['direction']}"
        )

    # Log groupby mode metadata
    mode_info = comparison.get("_groupby_queryability_mode", {})
    if mode_info:
        logger.info(
            f"  groupby_queryability mode:       "
            f"before={mode_info['before']}, after={mode_info['after']}"
        )

    logger.info("-" * 65)

    # Count improvements / degradations
    scored = {k: v for k, v in comparison.items() if not k.startswith("_")}
    n_improved = sum(1 for d in scored.values()
                     if d['direction'].startswith("✓"))
    n_degraded = sum(1 for d in scored.values()
                     if d['direction'].startswith("✗"))
    n_unchanged = len(scored) - n_improved - n_degraded

    logger.info(
        f"  Summary: {n_improved} improved, {n_degraded} degraded, "
        f"{n_unchanged} unchanged"
    )