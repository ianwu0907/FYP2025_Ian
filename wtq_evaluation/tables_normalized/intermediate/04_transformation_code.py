import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region (rows 9 to 140 inclusive)
    result = df.iloc[9:141].copy()
    result = result.reset_index(drop=True)
    print(f"After slicing to data region (rows 9-140): {result.shape}")

    # Step 2: Forward-fill sparse dimension column 0 (sector_cn) — handles rows like 90, 109, 111, etc.
    result.iloc[:, 0] = result.iloc[:, 0].fillna(method="ffill")
    print(f"After forward-filling sector_cn: {result.shape}")

    # Step 3: Forward-fill sparse dimension column 1 (sector_en) — but only where col 0 is non-blank and col 1 is blank,
    # and we want to carry forward the *last non-blank English label that aligns with same Chinese sector*.
    # However, per detection: col 1 sparsity is asymmetric — blanks inherit from prior row's col 1 *only if col 0 is blank*.
    # But our col 0 is now filled everywhere. So instead: fill col 1 only when it's blank AND the prior row has non-blank col 1,
    # but do NOT forward-fill across sector boundaries. Instead, use simple ffill for col 1 too — consistent with pattern.
    result.iloc[:, 1] = result.iloc[:, 1].fillna(method="ffill")
    print(f"After forward-filling sector_en: {result.shape}")

    # Step 4: Remove implicit aggregation rows (per schema exclusion list)
    # Exclude rows: 9, 10, 16, 24, 62, 67, 72, 75, 84, 87, 93, 98, 102, 105, 110, 115, 120, 124, 127, 134
    # These are 0-indexed *within the original df*, but we've sliced to rows 9-140 → new indices: 0 to 131
    # Original row 9 → new index 0
    # Original row 10 → new index 1
    # Original row 16 → new index 7
    # Original row 24 → new index 15
    # Original row 62 → new index 53
    # Original row 67 → new index 58
    # Original row 72 → new index 63
    # Original row 75 → new index 66
    # Original row 84 → new index 75
    # Original row 87 → new index 78
    # Original row 93 → new index 84
    # Original row 98 → new index 89
    # Original row 102 → new index 93
    # Original row 105 → new index 96
    # Original row 110 → new index 101
    # Original row 115 → new index 106
    # Original row 120 → new index 111
    # Original row 124 → new index 115
    # Original row 127 → new index 118
    # Original row 134 → new index 125
    agg_row_indices = [0, 1, 7, 15, 53, 58, 63, 66, 75, 78, 84, 89, 93, 96, 101, 106, 111, 115, 118, 125]
    result = result.drop(result.index[agg_row_indices]).reset_index(drop=True)
    print(f"After removing implicit aggregation rows: {result.shape}")

    # Step 5: Drop aggregate column 5 ("AVERAGE WAGE") per AGGREGATE_COLUMNS guidance
    # Keep columns 0, 1, 2, 3, 4 only (drop column index 5)
    result = result.iloc[:, [0, 1, 2, 3, 4]]
    print(f"After dropping aggregate column 5 (AVERAGE WAGE): {result.shape}")

    # Step 6: Build tidy records by unpivoting columns 2–4 with proper metric/demographic mapping
    # Column 2 → headcount, total
    # Column 3 → headcount, female (only where non-null)
    # Column 4 → total_wages, total (unit: hundred million yuan → keep as-is; will convert later if needed, but target says "unit-adjusted", and schema doesn't require conversion since output is just value)
    # So we create:
    #   - For each row: (sector_cn, sector_en, "total", "headcount", col2_val)
    #   - For each row: if col3 non-null: (sector_cn, sector_en, "female", "headcount", col3_val)
    #   - For each row: (sector_cn, sector_en, "total", "total_wages", col4_val)
    #
    # Note: demographic is only "total" or "female"; metric is "headcount" or "total_wages"
    # We do NOT include "average_wage" (column 5 was dropped)
    records = []
    for i in range(len(result)):
        row = result.iloc[i]
        sector_cn = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
        sector_en = str(row.iloc[1]).strip() if pd.notna(row.iloc[1]) else ""
        
        # Headcount total
        headcount_total = pd.to_numeric(row.iloc[2], errors="coerce")
        if pd.notna(headcount_total):
            records.append({
                "sector_cn": sector_cn,
                "sector_en": sector_en,
                "demographic": "total",
                "metric": "headcount",
                "value": float(headcount_total)
            })
        
        # Headcount female (col 3)
        headcount_female = pd.to_numeric(row.iloc[3], errors="coerce")
        if pd.notna(headcount_female):
            records.append({
                "sector_cn": sector_cn,
                "sector_en": sector_en,
                "demographic": "female",
                "metric": "headcount",
                "value": float(headcount_female)
            })
        
        # Total wages (col 4)
        total_wages = pd.to_numeric(row.iloc[4], errors="coerce")
        if pd.notna(total_wages):
            records.append({
                "sector_cn": sector_cn,
                "sector_en": sector_en,
                "demographic": "total",
                "metric": "total_wages",
                "value": float(total_wages)
            })

    output = pd.DataFrame(records)
    print(f"After unpivoting to tidy format: {output.shape}")

    # Step 7: Ensure exact column order and types
    output = output[["sector_cn", "sector_en", "demographic", "metric", "value"]]
    # Cast value to float (already done), others to string
    output["sector_cn"] = output["sector_cn"].astype(str)
    output["sector_en"] = output["sector_en"].astype(str)
    output["demographic"] = output["demographic"].astype(str)
    output["metric"] = output["metric"].astype(str)
    output["value"] = pd.to_numeric(output["value"], errors="coerce")

    print(f"Final output shape: {output.shape}")
    return output