import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print("Initial input shape:", df.shape)
    
    # === Step 1: Extract the data region by skipping header and footer rows ===
    # Retain rows 5 to 55 (i.e., using iloc[5:56])
    try:
        df_region = df.iloc[5:56].copy()  # Extract rows 5 to 55
    except Exception as e:
        print("Error during slicing rows:", e)
        return pd.DataFrame()
    print("After Step 1, data region shape:", df_region.shape)
    
    # Reset index to simplify pairing process
    df_region.reset_index(drop=True, inplace=True)
    print("Data region index reset. Head of data region:")
    print(df_region.head(4))
    
    # === Step 2: Pair bilingual rows and merge ethnicity names ===
    # Each ethnicity is split over 2 rows: first row for Chinese and second for English.
    merged_records = []
    n_rows = len(df_region)
    # If odd, warn and adjust
    if n_rows % 2 != 0:
        print("Warning: The number of rows in the data region is odd; the last row will be ignored.")
        n_rows = n_rows - 1

    for i in range(0, n_rows, 2):
        try:
            # Extract ethnicity names: first row gives Chinese version, second gives English.
            ethnicity_cn = str(df_region.iloc[i, 0]).strip() if not pd.isnull(df_region.iloc[i, 0]) else ""
            ethnicity_en = str(df_region.iloc[i+1, 0]).strip() if not pd.isnull(df_region.iloc[i+1, 0]) else ""
            
            # Extract measurement data from the first (Chinese) row
            # According to strategy, columns 3 & 4 correspond to 2011, 5 & 6 to 2016, 7 & 8 to 2021.
            m_2011_count = df_region.iloc[i, 3]
            m_2011_percentage = df_region.iloc[i, 4]
            m_2016_count = df_region.iloc[i, 5]
            m_2016_percentage = df_region.iloc[i, 6]
            m_2021_count = df_region.iloc[i, 7]
            m_2021_percentage = df_region.iloc[i, 8]
            
            record = {
                'ethnicity_cn': ethnicity_cn,
                'ethnicity_en': ethnicity_en,
                '2011_count': m_2011_count,
                '2011_percentage': m_2011_percentage,
                '2016_count': m_2016_count,
                '2016_percentage': m_2016_percentage,
                '2021_count': m_2021_count,
                '2021_percentage': m_2021_percentage
            }
            merged_records.append(record)
        except Exception as e:
            print(f"Error processing bilingual pair at index {i} and {i+1}: {e}")
    
    df_merged = pd.DataFrame(merged_records)
    print("After Step 2, merged data shape (one row per ethnicity):", df_merged.shape)
    print("Merged data sample:")
    print(df_merged.head())
    
    # === Step 3: Map header information to column positions for each year ===
    # Define mapping for the measurement columns for each year.
    year_mapping = {
        2011: ('2011_count', '2011_percentage'),
        2016: ('2016_count', '2016_percentage'),
        2021: ('2021_count', '2021_percentage')
    }
    print("Step 3 mapping defined:", year_mapping)
    
    # === Step 4: Unpivot measurement columns to create a long-form dataset ===
    long_records = []
    for idx, row in df_merged.iterrows():
        for year, (count_col, perc_col) in year_mapping.items():
            new_row = {
                'year': year,
                'ethnicity_cn': row['ethnicity_cn'],
                'ethnicity_en': row['ethnicity_en'],
                'count': row[count_col],
                'percentage': row[perc_col]
            }
            long_records.append(new_row)
    df_long = pd.DataFrame(long_records)
    print("After Step 4, long-form data shape:", df_long.shape)
    print("Long-form data sample:")
    print(df_long.head(9))
    
    # === Step 5: Convert data types and clean numerical data ===
    # Clean ethnicity strings.
    try:
        df_long['ethnicity_cn'] = df_long['ethnicity_cn'].astype(str).str.strip()
        df_long['ethnicity_en'] = df_long['ethnicity_en'].astype(str).str.strip()
    except Exception as e:
        print("Error cleaning ethnicity columns:", e)
    
    # Define safe conversion functions.
    def safe_int(val):
        try:
            # If already a numeric type (int/float), convert directly
            if isinstance(val, (int, np.integer)):
                return int(val)
            if isinstance(val, (float, np.floating)):
                return int(val)
            # Handle strings: remove commas, then convert
            val_str = str(val).replace(',', '').strip()
            # Convert through float to handle numbers like "365611.0"
            return int(float(val_str))
        except Exception:
            return np.nan

    def safe_float(val):
        try:
            if isinstance(val, (int, float, np.integer, np.floating)):
                return float(val)
            val_str = str(val).replace(',', '').strip()
            return float(val_str)
        except Exception:
            return np.nan

    df_long['count'] = df_long['count'].apply(safe_int)
    df_long['percentage'] = df_long['percentage'].apply(safe_float)
    
    print("After Step 5, cleaned data types. Sample data:")
    print(df_long.head(9))
    
    # === Step 6: Select and reorder final columns to match target schema ===
    # Final columns: ['year', 'ethnicity_cn', 'ethnicity_en', 'count', 'percentage']
    try:
        result = df_long[['year', 'ethnicity_cn', 'ethnicity_en', 'count', 'percentage']].copy()
    except Exception as e:
        print("Error reordering columns:", e)
        return pd.DataFrame()
    
    print("After Step 6, final transformed dataset shape:", result.shape)
    print("Final dataset sample:")
    print(result.head(10))
    
    return result