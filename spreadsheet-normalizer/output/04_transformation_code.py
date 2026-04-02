import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    result = df.iloc[6:55].copy()  # Adjust slicing to include more rows
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Remove only true aggregation rows or metadata
    result = result[~result.iloc[:, 1].str.contains('總計|合計|所有少數族裔人士', na=False)]
    print(f"After removing aggregation rows: {result.shape}")

    # Step 3: Forward fill the Year column
    result.iloc[:, 0] = result.iloc[:, 0].ffill()
    print(f"After forward filling the Year column: {result.shape}")

    # Step 4: Build multi-level header semantics
    col_semantics = {}
    header_row_indices = [2, 3, 4] 
    for j in range(result.shape[1]):
        parts = []
        for hr in header_row_indices:
            val = df.iloc[hr, j]
            if pd.notna(val) and str(val).strip():
                parts.append(str(val).strip())
        col_semantics[j] = " | ".join(parts) if parts else f"col_{j}"

    # Step 5: Create a new DataFrame with only relevant columns
    numeric_cols = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    ethnicities = []
    data_rows = result.values.tolist()
    
    for i in range(len(data_rows)):
        if i % 2 == 0:  # Consider Chinese rows (even indices)
            cn_row = data_rows[i]
            ethnicity_cn = str(cn_row[1]).strip()
            year = str(cn_row[0]).strip()
            for age_group_idx, numeric_col in enumerate(numeric_cols):
                if numeric_col < len(cn_row):  # Ensure the index is within bounds
                    number = pd.to_numeric(cn_row[numeric_col], errors='coerce')
                    age_group = str(df.iloc[3, numeric_col]).strip()
                    ethnicities.append({
                        'year': int(year),
                        'ethnicity_cn': ethnicity_cn,
                        'ethnicity_en': str(data_rows[i + 1][1]).strip() if i + 1 < len(data_rows) else '',
                        'age_group': age_group,
                        'number': number
                    })
    
    final_df = pd.DataFrame(ethnicities)
    print(f"After merging bilingual rows and formatting: {final_df.shape}")

    # Step 6: Select final columns
    final_df = final_df[['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'number']]
    print(f"Final output shape: {final_df.shape}")

    return final_df.reset_index(drop=True)