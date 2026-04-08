import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Extract header rows for column mapping
    header_row_0 = df.iloc[0].fillna(method='ffill').astype(str).str.strip()
    header_row_1 = df.iloc[1].fillna(method='ffill').astype(str).str.strip()
    print(f"Header row 0: {list(header_row_0)}")
    print(f"Header row 1: {list(header_row_1)}")

    # Build column mapping for value columns (1-6)
    column_map = {}
    for j in range(1, 7):
        language = header_row_0.iloc[j]
        # Extract agricultural region from actual column name
        orig_col_name = str(df.columns[j])
        # Extract the number from "agricultural region X"
        region_num = orig_col_name.split()[-1]
        column_map[j] = (region_num, language)

    print(f"Column map: {column_map}")

    records = []
    # Data rows are 3-9 (0-indexed)
    for i in range(3, 10):
        category = str(df.iloc[i, 0]).strip()
        
        # Skip rows where category is "marital status" (section header)
        if category.lower() == "marital status":
            print(f"Skipping section header row {i}")
            continue
        
        # Determine category_type based on row index
        if 3 <= i <= 4:
            category_type = "sex"
        else:  # rows 6-9
            category_type = "marital_status"
        
        # Extract values from value columns
        for j in range(1, 7):
            value = df.iloc[i, j]
            if pd.isna(value):
                continue
                
            region_num, language = column_map[j]
            
            try:
                percent = float(value)
            except (ValueError, TypeError):
                continue
            
            records.append({
                "agricultural_region": region_num,
                "language": language,
                "category_type": category_type,
                "category": category,
                "percent": percent
            })
    
    print(f"Collected {len(records)} records")
    
    target_cols = ['agricultural_region', 'language', 'category_type', 'category', 'percent']
    result = pd.DataFrame(records, columns=target_cols)
    print(f"Final output: {result.shape}")
    return result