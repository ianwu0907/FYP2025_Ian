import pandas as pd
import numpy as np
import re

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Step 0: Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the main data region by skipping non-data header/footer rows ===
    try:
        # According to strategy, data rows are from row index 5 up to row index 102 (i.e. 98 rows expected)
        # If the dataframe has fewer rows than expected, take from row 5 to the end.
        data_start = 5
        data_end = 103  # not inclusive; i.e. rows 5 to 102
        if df.shape[0] >= data_end:
            df_region = df.iloc[data_start:data_end, :].copy()
        else:
            df_region = df.iloc[data_start:, :].copy()
        print("Step 1: Extracted data region shape:", df_region.shape)
        print("Step 1: Sample extracted rows (first 3):")
        print(df_region.head(3))
        
        # Ensure the dataframe is not empty
        if df_region.empty:
            print("Error in Step 1: Data region is empty after extraction.")
            return pd.DataFrame()
    except Exception as e:
        print("Error in Step 1:", e)
        return pd.DataFrame()
    
    # === Step 2: Pair bilingual rows and merge them into single records for each ethnic group ===
    try:
        # Reset index for pairing convenience
        df_region = df_region.reset_index(drop=True)
        
        # Ensure even number of rows; if odd, drop the last row and warn.
        if len(df_region) % 2 != 0:
            print("Warning in Step 2: Odd number of rows in data region. Dropping the last row for pairing.")
            df_region = df_region.iloc[:-1, :]
        print("Step 2: Data region shape after ensuring even number of rows:", df_region.shape)
        
        # Pair: every two consecutive rows form one record.
        # First row of each pair: Chinese-language info (and numeric data), second row: English-language label.
        df_cn = df_region.iloc[0::2].reset_index(drop=True)
        df_en = df_region.iloc[1::2].reset_index(drop=True)
        
        num_cols = df_cn.shape[1]
        # We expect numeric columns to exist at positions 3 to 8 (i.e. 6 columns total)
        required_numeric_indices = [3,4,5,6,7,8]
        if max(required_numeric_indices) >= num_cols:
            print("Error in Step 2: Not enough columns in the data region. Expected at least 9 columns, got", num_cols)
            return pd.DataFrame()
        
        # Create paired dataframe with desired columns
        paired = pd.DataFrame()
        # Assume column 0 contains ethnicity labels
        paired['ethnicity_cn'] = df_cn.iloc[:, 0]
        paired['ethnicity_en'] = df_en.iloc[:, 0]
        # Numeric data: for 2011, 2016, 2021 (assuming columns in positions 3-8 of Chinese row)
        paired['count_2011'] = df_cn.iloc[:, 3]
        paired['perc_2011']  = df_cn.iloc[:, 4]
        paired['count_2016'] = df_cn.iloc[:, 5]
        paired['perc_2016']  = df_cn.iloc[:, 6]
        paired['count_2021'] = df_cn.iloc[:, 7]
        paired['perc_2021']  = df_cn.iloc[:, 8]
        
        print("Step 2: Paired data shape:", paired.shape)
        print("Step 2: Sample paired data (first 3 rows):")
        print(paired.head(3))
        
        # Ensure we have some paired rows
        if paired.empty:
            print("Error in Step 2: No paired data created.")
            return pd.DataFrame()
    except Exception as e:
        print("Error in Step 2:", e)
        return pd.DataFrame()
    
    # === Step 3: Reshape wide-format year-based columns to long format (unpivot) ===
    try:
        records = []
        # For every paired row, create one record per year: 2011, 2016, 2021.
        for index, row in paired.iterrows():
            for year, count_col, perc_col in [(2011, 'count_2011', 'perc_2011'),
                                              (2016, 'count_2016', 'perc_2016'),
                                              (2021, 'count_2021', 'perc_2021')]:
                records.append({
                    'ethnicity_cn': row['ethnicity_cn'],
                    'ethnicity_en': row['ethnicity_en'],
                    'year': year,
                    'count': row[count_col],
                    'percentage': row[perc_col]
                })
        long_df = pd.DataFrame(records)
        print("Step 3: Long-format data shape:", long_df.shape)
        print("Step 3: Sample long-format data (first 6 rows):")
        print(long_df.head(6))
        
        if long_df.empty:
            print("Error in Step 3: Long-format data is empty after unpivoting.")
            return pd.DataFrame()
    except Exception as e:
        print("Error in Step 3:", e)
        return pd.DataFrame()
    
    # === Step 4: Clean and convert data types for numeric fields and extract footnotes ===
    try:
        # Function to clean a numeric string: remove non-digit, non-dot, non-minus characters.
        def clean_numeric(val):
            if pd.isnull(val):
                return np.nan
            val_str = str(val).strip()
            cleaned = re.sub(r'[^\d\.\-]', '', val_str)
            return cleaned
        
        # Clean count and percentage columns
        long_df['count'] = long_df['count'].apply(lambda x: clean_numeric(x))
        long_df['percentage'] = long_df['percentage'].apply(lambda x: clean_numeric(x))
        
        # Convert to numeric types: counts as Int64 (nullable integer), percentage as float.
        long_df['count'] = pd.to_numeric(long_df['count'], errors='coerce').astype('Int64')
        long_df['percentage'] = pd.to_numeric(long_df['percentage'], errors='coerce')
        
        # Function to extract footnotes from a string, e.g., "[1]"
        def extract_footnotes(text):
            if pd.isnull(text):
                return np.nan
            match = re.search(r'(\[[^\]]+\])', text)
            if match:
                return match.group(1)
            return np.nan
        
        # Extract footnotes from the Chinese ethnicity field and remove them from the text.
        long_df['footnotes'] = long_df['ethnicity_cn'].apply(extract_footnotes)
        long_df['ethnicity_cn'] = long_df['ethnicity_cn'].apply(lambda x: re.sub(r'\[[^\]]+\]', '', x).strip() if pd.notnull(x) else x)
        
        print("Step 4: Data types after cleaning:")
        print(long_df.dtypes)
        print("Step 4: Sample cleaned data (first 6 rows):")
        print(long_df.head(6))
    except Exception as e:
        print("Error in Step 4:", e)
        return pd.DataFrame()
    
    # === Step 5: Finalize and reorder target columns ===
    try:
        # According to target schema:
        target_cols = ['ethnicity_cn', 'ethnicity_en', 'footnotes', 'year', 'count', 'percentage']
        result = long_df[target_cols].copy()
        
        # Validate expected row count: Should equal number of ethnic groups * 3.
        num_ethnic_groups = paired.shape[0]
        expected_rows = num_ethnic_groups * 3
        actual_rows = result.shape[0]
        if actual_rows != expected_rows:
            print(f"Warning in Step 5: Expected {expected_rows} rows but got {actual_rows} rows.")
        print("Step 5: Final tidy dataframe shape:", result.shape)
        print("Step 5: Sample final tidy data (first 6 rows):")
        print(result.head(6))
        
        # Check that final dataframe has all required columns.
        missing_cols = [col for col in target_cols if col not in result.columns]
        if missing_cols:
            print("Error in Step 5: Missing columns in final DataFrame:", missing_cols)
            return pd.DataFrame()
    except Exception as e:
        print("Error in Step 5:", e)
        return pd.DataFrame()
    
    return result