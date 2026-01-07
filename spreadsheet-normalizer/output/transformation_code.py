import pandas as pd
import numpy as np

def transform(df):
    # Step 1: Skip metadata rows
    df = df.iloc[6:]  # Skip the first 6 rows
    df.columns = df.iloc[0]  # Set the first row as header
    df = df[1:]  # Remove the header row from the data
    df.reset_index(drop=True, inplace=True)

    # Step 2: Define the year
    year = 2011  # As per the structure analysis

    # Step 3: Define the column mapping for marital status and sex
    marital_status_mapping = {
        "從未結婚": "never_married",
        "已婚": "married",
        "喪偶／離婚／分居": "widowed_divorced_separated"
    }
    
    sex_mapping = {
        "男": "male",
        "女": "female",
        "合計": "total"
    }

    # Step 4: Prepare to collect data
    data_rows = []

    # Step 5: Iterate through the DataFrame to extract relevant data
    for idx, row in df.iterrows():
        # Check if the row contains valid ethnicity data (Chinese)
        ethnicity_cn = row[1] if pd.notna(row[1]) and isinstance(row[1], str) and any('\u4e00' <= char <= '\u9fff' for char in row[1]) else None
        ethnicity_en = row[2] if pd.notna(row[2]) and isinstance(row[2], str) and any(char.isascii() for char in row[2]) else None
        
        # Skip rows without ethnicity data
        if ethnicity_cn is None:
            continue
        
        # Extract counts and percentages for each marital status and sex
        for col_idx in range(3, len(row)):
            count_value = row[col_idx]
            if pd.isna(count_value) or not isinstance(count_value, (int, float)):
                continue
            
            # Determine marital status and sex based on column index
            marital_status = None
            sex = None
            
            if col_idx in range(3, 6):
                marital_status = marital_status_mapping.get("從未結婚")
                sex = sex_mapping.get(df.columns[col_idx], None)
            elif col_idx in range(6, 9):
                marital_status = marital_status_mapping.get("已婚")
                sex = sex_mapping.get(df.columns[col_idx], None)
            elif col_idx in range(9, 12):
                marital_status = marital_status_mapping.get("喪偶／離婚／分居")
                sex = sex_mapping.get(df.columns[col_idx], None)
            else:
                continue  # Skip any columns outside the expected range
            
            # Append the data row
            data_rows.append({
                'year': year,
                'ethnicity_cn': ethnicity_cn,
                'ethnicity_en': ethnicity_en,
                'marital_status': marital_status,
                'sex': sex,
                'count': int(count_value) if count_value >= 0 else None,
                'percentage': float(count_value) if 0 <= count_value <= 100 else None
            })

    # Step 6: Create a DataFrame from the collected data
    tidy_df = pd.DataFrame(data_rows)

    # Step 7: Clean up the DataFrame by dropping rows with None values in critical columns
    tidy_df.dropna(subset=['year', 'ethnicity_cn', 'marital_status', 'sex', 'count', 'percentage'], inplace=True)

    # Step 8: Handle case where tidy_df might still be empty
    if tidy_df.empty:
        # Emit a warning if the resulting DataFrame is empty
        print("Warning: The resulting DataFrame is empty. Check the input data for validity.")

    return tidy_df