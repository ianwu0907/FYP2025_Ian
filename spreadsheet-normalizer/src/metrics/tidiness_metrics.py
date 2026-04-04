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
"""

import logging
import math
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from collections import Counter

logger = logging.getLogger(__name__)


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

    If dim_cols not provided, uses all text columns.
    """
    if df.empty:
        return 0.0

    if dim_cols is None:
        dim_cols = []
        for col in df.columns:
            non_empty = df[col].dropna()
            if len(non_empty) == 0:
                continue
            n_str = sum(1 for v in non_empty if isinstance(v, str))
            if n_str / len(non_empty) > 0.5:
                dim_cols.append(col)

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
    """
    if label:
        logger.info(f"Computing tidiness metrics [{label}]...")

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
    }

    if label:
        logger.info(f"  Shape: {metrics['shape']}")
        for k, v in metrics.items():
            if k != "shape":
                logger.info(f"  {k}: {v}")

    return metrics


def compare_metrics(before: Dict[str, Any],
                    after: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compare before/after metrics and compute deltas.

    Returns:
        Dict with each metric's before, after, delta, and direction.
    """
    comparison = {}

    # Metrics where higher = better
    higher_is_better = {
        "cell_coverage", "row_completeness_uniformity",
        "column_type_homogeneity", "data_row_ratio",
        "column_completeness_min", "header_uniqueness",
        "type_consistency",
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

    return comparison


def log_comparison(comparison: Dict[str, Any]):
    """Pretty-print the before/after comparison."""
    logger.info("TIDINESS METRICS COMPARISON")
    logger.info("-" * 65)
    logger.info(f"  {'Metric':<32} {'Before':>7} {'After':>7} {'Delta':>7}  Dir")
    logger.info("-" * 65)

    for metric, data in comparison.items():
        logger.info(
            f"  {metric:<32} {data['before']:>7.4f} {data['after']:>7.4f} "
            f"{data['delta']:>+7.4f}  {data['direction']}"
        )

    logger.info("-" * 65)

    # Count improvements / degradations
    n_improved = sum(1 for d in comparison.values()
                     if d['direction'].startswith("✓"))
    n_degraded = sum(1 for d in comparison.values()
                     if d['direction'].startswith("✗"))
    n_unchanged = len(comparison) - n_improved - n_degraded

    logger.info(
        f"  Summary: {n_improved} improved, {n_degraded} degraded, "
        f"{n_unchanged} unchanged"
    )