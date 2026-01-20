import pandas as pd
import numpy as np
import re

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print(f"Input shape: {df.shape}")
    
    # === Step 1: Extract the data region excluding non‐data header rows and footers ===
    # According to the strategy, row 6 (index 6) contains the reporting year and rows 7 to 48 contain paired ethnic data.
    try:
        data_region = df.iloc[6:49].copy()
    except Exception as e:
        print("Error extracting data region:", e)
        return pd.DataFrame()
    print(f"Data region shape (expected 43 rows): {data_region.shape}")
    
    # Extract the reporting year from row6 (first row in data_region), column 0
    year_value_raw = data_region.iloc[0, 0]
    try:
        reporting_year = int(pd.to_numeric(year_value_raw, errors='coerce'))
    except Exception as e:
        print("Error converting reporting year:", e)
        reporting_year = np.nan
    print(f"Reporting year extracted: {reporting_year}")
    
    # === Step 2: Identify and separate bilingual paired rows ===
    # Rows 7 to 48 (data_region rows excluding the first row) are paired:
    # Chinese rows (absolute counts) are at even positions and English rows (percentages) are the following odd positions.
    paired_data = data_region.iloc[1:].reset_index(drop=True)
    if len(paired_data) % 2 != 0:
        print("Warning: Paired data rows count is odd, dropping the last row for pairing.")
        paired_data = paired_data.iloc[:-1]
    # Split paired rows:
    df_cn = paired_data.iloc[0::2].reset_index(drop=True)
    df_en = paired_data.iloc[1::2].reset_index(drop=True)
    print(f"Number of Chinese rows (counts): {df_cn.shape[0]}")
    print(f"Number of English rows (percentages): {df_en.shape[0]}")
    
    # === Step 3: Map and assign header information for age groups ===
    # Extract header labels for age groups from header row 3 (columns 3 to 9)
    try:
        raw_headers = df.iloc[3, 3:10]
        header_mapping = {}
        for col, val in zip(range(3, 10), raw_headers):
            # Clean headers: remove newline characters and leading/trailing whitespace.
            cleaned = str(val).replace('\n', '').strip()
            # Remove extra spaces for age groups: remove space after '<' if present.
            cleaned = re.sub(r'^<\s*', '<', cleaned)
            # Remove spaces around '-' to standardize (e.g., "15 - 24" -> "15-24")
            cleaned = re.sub(r'\s*-\s*', '-', cleaned)
            header_mapping[col] = cleaned
    except Exception as e:
        print("Error extracting header mapping:", e)
        header_mapping = {}
    print("Header mapping for age groups:", header_mapping)
    
    # === Step 4: Reshape the data by unpivoting the age group columns and merging bilingual rows ===
    records = []
    
    def safe_int(x):
        try:
            val = pd.to_numeric(x, errors='coerce')
            return int(val) if pd.notnull(val) else np.nan
        except Exception:
            return np.nan

    def safe_float(x):
        try:
            val = pd.to_numeric(x, errors='coerce')
            return float(val) if pd.notnull(val) else np.nan
        except Exception:
            return np.nan

    # Iterate over each paired row to unpivot the age group columns (3 to 9) into tidy rows.
    for idx in range(df_cn.shape[0]):
        count_row = df_cn.iloc[idx]
        percent_row = df_en.iloc[idx]
        
        # Extract ethnicity names from column 1 and trim whitespace.
        ethnicity_cn = str(count_row.iloc[1]).strip() if pd.notnull(count_row.iloc[1]) else ""
        ethnicity_en = str(percent_row.iloc[1]).strip() if pd.notnull(percent_row.iloc[1]) else ""
        
        # Extract median age from count row (column 11)
        median_age_raw = count_row.iloc[11] if count_row.shape[0] > 11 else np.nan
        median_age = safe_float(median_age_raw)
        
        # Loop over age group columns (columns 3 to 9).
        for col in range(3, 10):
            age_group = header_mapping.get(col, f"col_{col}")
            count_val = safe_int(count_row.iloc[col])
            percentage_val = safe_float(percent_row.iloc[col])
            
            record = {
                "year": reporting_year,
                "ethnicity_cn": ethnicity_cn,
                "ethnicity_en": ethnicity_en,
                "age_group": age_group,
                "count": count_val,
                "percentage": percentage_val,
                "median_age": median_age
            }
            records.append(record)
    print(f"Total records created after unpivoting: {len(records)}")
    
    # Create final DataFrame from records
    result = pd.DataFrame(records)
    print(f"Final DataFrame shape: {result.shape}")
    print("Sample final data:")
    print(result.head())
    
    # === Step 5: Clean and convert data types appropriately ===
    try:
        result['year'] = pd.to_numeric(result['year'], errors='coerce').astype('Int64')
        result['count'] = pd.to_numeric(result['count'], errors='coerce').astype('Int64')
        result['percentage'] = pd.to_numeric(result['percentage'], errors='coerce')
        result['median_age'] = pd.to_numeric(result['median_age'], errors='coerce')
    except Exception as e:
        print("Error during type conversion:", e)
    
    print("Data types after conversion:")
    print(result.dtypes)
    
    return result