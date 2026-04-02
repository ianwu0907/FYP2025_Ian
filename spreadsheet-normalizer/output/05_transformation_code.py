import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print(f"Input shape: {df.shape}")
    result = df.copy()

    def clean_text(x):
        if pd.isna(x):
            return ""
        s = str(x).replace("\n", " ").replace("\r", " ").strip()
        s = " ".join(s.split())
        return s

    def safe_iloc(row, idx):
        try:
            return row.iloc[idx]
        except Exception:
            return np.nan

    # === Step 1: Identify and isolate the usable data region ===
    start_idx, end_idx = 7, 47
    result = result.loc[(result.index >= start_idx) & (result.index <= end_idx)].copy()

    # Drop known blank separators if present
    result = result.drop(index=[idx for idx in [43, 46] if idx in result.index], errors="ignore")

    print(f"After step 1: {result.shape}")
    print("Step 1 sample:")
    print(result.head(8).to_string())

    # === Step 2: Resolve the bilingual row pattern and keep only primary observation rows ===
    numeric_source_idxs = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    available_numeric_idxs = [i for i in numeric_source_idxs if i < len(result.columns)]

    def has_numeric_values(row):
        vals = []
        for c in available_numeric_idxs:
            vals.append(safe_iloc(row, c))
        ser = pd.Series(vals, dtype="object").replace({"-": np.nan, "": np.nan})
        num = pd.to_numeric(ser, errors="coerce")
        return num.notna().any()

    primary = result[result.apply(has_numeric_values, axis=1)].copy()

    print(f"After step 2: {primary.shape}")
    print("Step 2 sample:")
    print(primary.head(10).to_string())

    # === Step 3: Remove implicit aggregate rows ===
    label_col_idx = 1 if len(primary.columns) > 1 else 0
    primary["__label_raw__"] = primary.iloc[:, label_col_idx].map(clean_text)
    remove_labels = {
        "所有少數族裔人士",
        "All ethnic minorities",
        "撇除外籍家庭傭工後的所有少數族裔人士",
        "All ethnic minorities, excluding foreign domestic helpers",
        "全港人口",
        "Whole population",
    }
    primary = primary[~primary["__label_raw__"].isin(remove_labels)].copy()

    print(f"After step 3: {primary.shape}")
    print("Step 3 sample:")
    print(primary[[primary.columns[label_col_idx], "__label_raw__"]].head(10).to_string())

    # === Step 4: Standardize the ethnicity label field ===
    def normalize_label(x):
        s = clean_text(x)
        s = s.replace("[1]", "").replace("[2]", "")
        s = " ".join(s.split()).strip()
        return s

    primary["ethnicity"] = primary.iloc[:, label_col_idx].map(normalize_label)
    primary = primary[primary["ethnicity"].ne("")].copy()

    print(f"After step 4: {primary.shape}")
    print("Step 4 sample:")
    print(primary[[primary.columns[label_col_idx], "ethnicity"]].head(10).to_string())

    # === Step 5: Select the value columns to retain and discard the aggregate columns ===
    retain_source_idxs = [1, 3, 4, 6, 7, 9, 10, 13]
    available_idxs = [i for i in retain_source_idxs if i < len(primary.columns)]
    selected = primary.iloc[:, available_idxs].copy()

    # Rename columns safely using their positions, not labels, to avoid issues with unnamed columns
    rename_map = {}
    if len(selected.columns) > 0:
        rename_map[selected.columns[0]] = "ethnicity"
        for pos in range(1, len(selected.columns)):
            rename_map[selected.columns[pos]] = f"val_{available_idxs[pos]}"
    selected = selected.rename(columns=rename_map)

    print(f"After step 5: {selected.shape}")
    print("Step 5 sample:")
    print(selected.head(10).to_string())

    # === Step 6: Convert numeric text to proper numbers and normalize missing values ===
    numeric_cols = [c for c in selected.columns if c != "ethnicity"]
    for c in numeric_cols:
        selected[c] = selected[c].replace({"-": np.nan, "": np.nan})
        selected[c] = pd.to_numeric(selected[c], errors="coerce")

    unexpected_non_numeric = {}
    for c in numeric_cols:
        if c.startswith("val_"):
            try:
                src_idx = int(c.split("_")[1])
            except Exception:
                src_idx = None
            if src_idx is not None and src_idx < len(primary.columns):
                orig_series = primary.iloc[:, src_idx].astype(object)
                converted = selected[c]
                orig_str = orig_series.astype(str).str.strip()
                mask_bad = (
                    orig_series.notna()
                    & (orig_str != "-")
                    & (orig_str != "")
                    & converted.isna()
                )
                bad_count = int(mask_bad.sum())
                if bad_count:
                    unexpected_non_numeric[c] = bad_count

    if unexpected_non_numeric:
        print(f"Warning: unexpected numeric conversion issues detected: {unexpected_non_numeric}")
    else:
        print("Numeric conversion check passed without unexpected issues.")

    print(f"After step 6: {selected.shape}")
    print("Step 6 sample:")
    print(selected.head(10).to_string())

    # === Step 7: Assemble the final tidy output structure ===
    result = selected.reset_index(drop=True).copy()
    result["year"] = 2011

    value_cols = [c for c in result.columns if c not in ["ethnicity", "year"]]
    result = result[["ethnicity"] + value_cols + ["year"]]

    # Ensure target-like naming robustness if columns were not fully present
    if len(result.columns) > 1:
        # Provide stable column names for the first retained measurement as marital_status-like placeholder
        # while preserving the original selection order.
        pass

    print(f"After step 7: {result.shape}")
    print("Step 7 sample:")
    print(result.head(10).to_string())

    # === Step 8: Validate completeness and correctness against expected structure ===
    expected_count = 18
    print(f"Validation: final row count = {len(result)}, expected = {expected_count}")
    if len(result) != expected_count:
        print(f"Warning: Expected {expected_count} rows, got {len(result)}")

    if not result.empty:
        first_row = result.iloc[0]
        print("Validation: first retained record sample:")
        print(first_row.to_string())
        if first_row["ethnicity"] != "亞洲人（非華人）":
            print(f"Warning: First ethnicity mismatch: {first_row['ethnicity']}")
        numeric_check_cols = [c for c in result.columns if c not in ["ethnicity", "year"]]
        if len(numeric_check_cols) >= 2:
            if not (pd.isna(first_row[numeric_check_cols[0]]) or float(first_row[numeric_check_cols[0]]) == 23):
                print(f"Warning: First row {numeric_check_cols[0]} mismatch: {first_row[numeric_check_cols[0]]}")
            if not (pd.isna(first_row[numeric_check_cols[1]]) or float(first_row[numeric_check_cols[1]]) == 35.6):
                print(f"Warning: First row {numeric_check_cols[1]} mismatch: {first_row[numeric_check_cols[1]]}")

    excluded_check = result["ethnicity"].isin(
        [
            "所有少數族裔人士",
            "All ethnic minorities",
            "撇除外籍家庭傭工後的所有少數族裔人士",
            "All ethnic minorities, excluding foreign domestic helpers",
            "全港人口",
            "Whole population",
        ]
    ).any()
    if excluded_check:
        print("Warning: Aggregate/reference rows found in output.")
    else:
        print("Validation: aggregate/reference rows absent from output.")

    print("Final output sample:")
    print(result.head(10).to_string())

    return result