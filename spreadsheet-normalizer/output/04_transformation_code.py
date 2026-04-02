import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    result = df.iloc[5:48].copy()
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {result.shape}")
    print(result.head())

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {result.shape}")
    print(result.head())

    # Step 3: Identify bilingual row pairs — keep only Chinese rows (have data)
    rows_to_keep = []
    for i in range(len(result)):
        # Check if this row has numeric data in value columns
        has_data = False
        for j in [3, 4, 5, 6, 7, 8]:
            val = result.iloc[i, j]
            if pd.notna(val) and str(val).strip():
                has_data = True
                break
        if has_data:
            rows_to_keep.append(i)
    result_cn = result.iloc[rows_to_keep].copy().reset_index(drop=True)

    # Get English labels from the row after each Chinese row
    en_labels = []
    for i in rows_to_keep:
        if i + 1 < len(result):
            en_labels.append(str(result.iloc[i + 1, 0]).strip())
        else:
            en_labels.append("")
    result_cn["ethnicity_en"] = en_labels
    print(f"After bilingual merge: {result_cn.shape}")
    print(result_cn.head())

    # Step 4: Remove aggregation rows
    agg_kw = ["所有少數族裔人士", "撇除外籍家庭傭工後的所有少數族裔人士", "全港人口"]
    mask = result_cn.iloc[:, 0].apply(
        lambda x: not any(k in str(x).strip() for k in agg_kw) if pd.notna(x) else True
    )
    result_cn = result_cn[mask].reset_index(drop=True)
    print(f"After removing aggregation rows: {result_cn.shape}")
    print(result_cn.head())

    # Step 5: Forward-fill the year column
    result_cn.iloc[:, 0] = result_cn.iloc[:, 0].fillna(method="ffill")
    print(f"After forward-filling year: {result_cn.shape}")
    print(result_cn.head())

    # Step 6: Unpivot year groups
    # Column mapping: {3: ("2011", "Number"), 4: ("2011", "%"),
    #                  5: ("2016", "Number"), 6: ("2016", "%"),
    #                  7: ("2021", "Number"), 8: ("2021", "%")}
    records = []
    for i in range(len(result_cn)):
        row = result_cn.iloc[i]
        base = {"ethnicity_cn": str(row.iloc[0]).split("\n")[0].strip(),
                "ethnicity_en": row["ethnicity_en"]}
        for year, number_col, pct_col in [("2011", 3, 4), ("2016", 5, 6), ("2021", 7, 8)]:
            record = {**base, "year": int(year)}
            record["value_type"] = "Number"
            record["value"] = pd.to_numeric(row.iloc[number_col], errors="coerce")
            records.append(record)
            record = {**base, "year": int(year)}
            record["value_type"] = "%"
            record["value"] = pd.to_numeric(row.iloc[pct_col], errors="coerce")
            records.append(record)

    output = pd.DataFrame(records)
    output = output[["ethnicity_cn", "ethnicity_en", "year", "value_type", "value"]]
    print(f"Final output: {output.shape}")
    print(output.head())
    return output