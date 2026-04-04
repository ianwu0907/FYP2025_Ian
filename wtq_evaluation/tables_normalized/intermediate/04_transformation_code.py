import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Forward-fill sparse region column (column 0) BEFORE slicing
    # Replace empty strings with NaN, then forward-fill
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    print(f"After forward-filling region column: {df.shape}")

    # Step 2: Extract multi-level headers from rows 2 and 3 (0-indexed)
    # Row 2 = education_level top labels (starting at col 4), Row 3 = sex sub-labels
    # Build column mapping: {col_index: (education_level, sex)}
    column_map = {}
    
    # Extract education level labels from row 2 (index 2), starting at col 4, every 3 cols
    edu_labels = []
    for j in range(4, len(df.columns), 3):
        if j < len(df.columns) and pd.notna(df.iloc[2, j]):
            label = str(df.iloc[2, j]).strip()
            edu_labels.append(label)
    
    # For each education group (10 groups), assign the three columns: [Subtotal, Male, Female]
    # We only want Male and Female columns (skip Subtotal at index 0 of each group)
    for group_idx, edu_label in enumerate(edu_labels):
        base_col = 4 + group_idx * 3
        # Skip Subtotal (base_col), take Male (base_col+1) and Female (base_col+2)
        if base_col + 1 < len(df.columns) and pd.notna(df.iloc[3, base_col + 1]):
            if str(df.iloc[3, base_col + 1]).strip() == "Male":
                column_map[base_col + 1] = (edu_label, "Male")
        if base_col + 2 < len(df.columns) and pd.notna(df.iloc[3, base_col + 2]):
            if str(df.iloc[3, base_col + 2]).strip() == "Female":
                column_map[base_col + 2] = (edu_label, "Female")
    
    print(f"Built column_map with {len(column_map)} value columns")

    # Step 3: Collect records via loop over data rows (rows 5 to 42 inclusive)
    records = []
    # Skip metadata rows (0-4) and known blank rows (6, 12, 16, 24, 31, 37)
    blank_rows = {6, 12, 16, 24, 31, 37}
    
    for i in range(5, 43):
        if i in blank_rows:
            continue
            
        # Skip if row has no region (though we forward-filled, still check for safety)
        region_val = df.iloc[i, 0]
        if pd.isna(region_val) or str(region_val).strip() == "":
            continue
            
        region = str(region_val).strip()
        
        # Extract values for each (education_level, sex) pair in column_map
        for col_idx, (edu_level, sex) in column_map.items():
            val = df.iloc[i, col_idx]
            if pd.notna(val):
                try:
                    count_val = int(str(val).replace(',', '').strip())
                except (ValueError, TypeError):
                    count_val = np.nan
                records.append({
                    "region": region,
                    "education_level": edu_level,
                    "sex": sex,
                    "count": count_val
                })
    
    print(f"Collected {len(records)} records")

    # Step 4: Create DataFrame with exact target columns
    target_cols = ['region', 'education_level', 'sex', 'count']
    result = pd.DataFrame(records, columns=target_cols)
    
    # Drop rows with NaN count
    result = result.dropna(subset=['count'])
    result['count'] = result['count'].astype(int)
    
    print(f"Final output: {result.shape}")
    return result