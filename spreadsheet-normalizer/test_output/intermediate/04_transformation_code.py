import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Forward-fill column 0 (region_of_birth)
    df.iloc[:, 0] = df.iloc[:, 0].fillna(method="ffill")
    print(f"After forward-filling region_of_birth: {df.shape}")
    print(df.head())

    # Step 2: Identify and handle section header rows
    df["section"] = None
    for i in range(len(df)):
        row = df.iloc[i]
        if pd.notna(row.iloc[0]) and all(pd.isna(row.iloc[j]) or str(row.iloc[j]).strip() == "" for j in [1, 2, 3, 4]):
            df.iloc[i, df.columns.get_loc("section")] = row.iloc[0]
    df["section"] = df["section"].fillna(method="ffill")
    df = df.dropna(subset=[df.columns[j] for j in [1, 2, 3, 4]], how="all").reset_index(drop=True)
    print(f"After handling section header rows: {df.shape}")
    print(df.head())

    # Step 3: Create the column mapping for education_level and value_type
    column_map = {}
    current_group = None
    for j in range(1, 5):
        val = df.iloc[0, j]
        if pd.notna(val) and str(val).strip():
            current_group = str(val).strip()
        if current_group:
            if j < 3:
                column_map[j] = (current_group, "percent")
            else:
                column_map[j] = (current_group, "percent")

    # Step 4: Unpivot the data using the column mapping
    records = []
    for _, row in df.iterrows():
        base = {"region_of_birth": row.iloc[0]}
        for col_idx, (education_level, value_type) in column_map.items():
            record = {**base, "education_level": education_level, "value_type": value_type}
            record["percentage"] = pd.to_numeric(row.iloc[col_idx], errors="coerce")
            records.append(record)

    result = pd.DataFrame(records)
    result = result[["region_of_birth", "education_level", "value_type", "percentage"]]
    print(f"Final output: {result.shape}")
    print(result.head())
    return result