import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print(f"Input shape: {df.shape}")
    
    # === Step 1: Extract the core data region (rows 6 to 48) ===
    try:
        # rows 6 to 48 inclusive => use iloc[6:49]
        core_df = df.iloc[6:49, :].copy().reset_index(drop=True)
    except Exception as e:
        print("Error in Step 1 extracting core data region:", e)
        return None
    print(f"After Step 1: core_df shape = {core_df.shape}")
    
    # === Step 2: Propagate the section marker (year) to all ethnic group rows ===
    try:
        # The section marker is in the first row (row index 0) of core_df, column 0.
        year_value = core_df.iloc[0, 0]
        # Attempt to convert year_value to int if possible.
        year_value = int(str(year_value).strip()) if pd.notnull(year_value) else None
    except Exception as e:
        print("Error in Step 2 extracting year:", e)
        return None
    print(f"Extracted year: {year_value}")
    
    # The remaining rows (from index 1 onward) are bilingual paired ethnic data rows.
    try:
        data_df = core_df.iloc[1:, :].copy().reset_index(drop=True)
    except Exception as e:
        print("Error in obtaining ethnic data rows:", e)
        return None
    print(f"Data rows shape (after dropping section marker): {data_df.shape}")
    
    # === Step 3: Pair the bilingual rows and separate raw counts from percentages ===
    try:
        # Expect that the bilingual rows are in pairs:
        # First row of each pair: Chinese ethnicity name and raw counts (plus median age)
        # Second row of each pair: English ethnicity name and percentages
        if len(data_df) % 2 != 0:
            print("Warning: Ethnic data rows count is odd. Last row will be ignored.")
            data_df = data_df.iloc[:-1, :]  # drop last row if odd count
        # Split rows: even index of data_df are Chinese count rows, odd index are English percent rows.
        df_cn = data_df.iloc[0::2, :].copy().reset_index(drop=True)
        df_en = data_df.iloc[1::2, :].copy().reset_index(drop=True)
        
        # Add identifiers for merging later
        df_cn['pair_id'] = df_cn.index
        df_en['pair_id'] = df_en.index
        
        # Propagate year for both dataframes
        df_cn['year'] = year_value
        df_en['year'] = year_value
        
        # Extract ethnicity names, trimming whitespace; Chinese from df_cn and English from df_en.
        df_cn['ethnicity_cn'] = df_cn.iloc[:, 1].astype(str).str.strip()
        df_en['ethnicity_en'] = df_en.iloc[:, 1].astype(str).str.strip()
        
        # Extract median_age from df_cn from column index 11, and clean/convert it later.
        df_cn['median_age'] = df_cn.iloc[:, 11]
    except Exception as e:
        print("Error in Step 3 pairing bilingual rows:", e)
        return None
    print(f"Paired Chinese rows shape: {df_cn.shape}, English rows shape: {df_en.shape}")
    
    # === Step 4: Unpivot the age group specific columns and merge counts with percentages ===
    try:
        # Identify the age group columns: columns 3 to 9 (inclusive), exclude column 10 as aggregate total.
        # Use the column names from the original dataframe for positions 3 to 9.
        age_cols = list(df_cn.columns[3:10])
        # Define mapping from these column names to age group labels.
        # Based on the provided sample validation, we assume these labels:
        age_group_mapping = dict(zip(age_cols, ["<15", "15 - 24", "25 - 34", "35 - 44", "45 - 54", "55 - 64", "65+"]))
        
        # Melt the raw counts (from df_cn) for age groups
        melt_counts = pd.melt(
            df_cn,
            id_vars=["pair_id", "year", "ethnicity_cn", "median_age"],
            value_vars=age_cols,
            var_name="col",
            value_name="count"
        )
        # Map the melted column name to the age group label
        melt_counts['age_group'] = melt_counts['col'].map(age_group_mapping)
        melt_counts.drop(columns=["col"], inplace=True)
        
        # Melt the percentages (from df_en) for age groups
        melt_percent = pd.melt(
            df_en,
            id_vars=["pair_id", "year", "ethnicity_en"],
            value_vars=age_cols,
            var_name="col",
            value_name="percentage"
        )
        melt_percent['age_group'] = melt_percent['col'].map(age_group_mapping)
        melt_percent.drop(columns=["col"], inplace=True)
        
        # Merge the counts and percentages based on pair_id, year, and age_group
        merged = pd.merge(
            melt_counts,
            melt_percent,
            on=["pair_id", "year", "age_group"],
            how="inner"
        )
    except Exception as e:
        print("Error in Step 4 unpivoting age group columns:", e)
        return None
    print(f"After unpivot, merged data shape: {merged.shape}")
    
    # === Step 5: Data cleaning, type conversion and final column selection ===
    try:
        # Trim whitespace in string columns.
        merged['ethnicity_cn'] = merged['ethnicity_cn'].astype(str).str.strip()
        merged['ethnicity_en'] = merged['ethnicity_en'].astype(str).str.strip()
        merged['age_group'] = merged['age_group'].astype(str).str.strip()
        
        # Convert types: year -> int; count -> int; percentage and median_age -> float.
        merged['year'] = merged['year'].astype(int)
        
        # Convert count to numeric, errors coerced to NaN, then fill or leave NaN.
        merged['count'] = pd.to_numeric(merged['count'], errors='coerce')
        merged['percentage'] = pd.to_numeric(merged['percentage'], errors='coerce')
        merged['median_age'] = pd.to_numeric(merged['median_age'], errors='coerce')
        
        # Optionally drop rows with missing values in critical columns (if desired)
        merged = merged.dropna(subset=["count", "percentage", "median_age"])
        
        # Final selection and ordering of columns:
        result = merged[["year", "ethnicity_cn", "ethnicity_en", "age_group", "count", "percentage", "median_age"]].copy()
        
        # Reset index of final result
        result.reset_index(drop=True, inplace=True)
    except Exception as e:
        print("Error in Step 5 cleaning and type conversion:", e)
        return None
    print(f"Final data shape: {result.shape}")
    print("Final result sample:")
    print(result.head())
    
    return result

if __name__ == "__main__":
    # Example usage:
    # Create a sample dataframe to simulate the input.
    # Note: For real usage, the dataframe 'df' should be read from an Excel/CSV file.
    
    # Creating a dummy sample DataFrame with 49 rows x 13 columns based on the description.
    sample_data = {}
    col_names = ['表3.3            ', 'Unnamed: 1', '2011年、2016年及2021年按種族及年齡組別劃分的少數族裔人士數目', 
                 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5', 'Unnamed: 6', 'Unnamed: 7', 'Unnamed: 8', 'Unnamed: 9', 
                 'Unnamed: 10', 'Unnamed: 11', '索引 Index']
    # Create an empty DataFrame with correct columns.
    dummy_df = pd.DataFrame(np.nan, index=range(49), columns=col_names)
    
    # Set header rows (dummy values) for rows 0,2,3,4
    dummy_df.iloc[0] = [None]*13
    dummy_df.iloc[2] = [None]*13
    dummy_df.iloc[3] = [None]*13  # This row in real data gives age group labels but we use hard-coded mapping here.
    dummy_df.iloc[4] = [None]*13
    
    # Row 6: Section marker row with year '2011'
    dummy_df.iloc[6, 0] = "2011"
    
    # Add sample ethnic rows in bilingual pairs starting from row 7.
    # First pair: rows 7 and 8
    dummy_df.iloc[7, 1] = "亞洲人（非華人）"
    dummy_df.iloc[7, 3:10] = [23984, 25650, 152228, 103841, 41396, 13196, 5316]
    dummy_df.iloc[7, 10] = 365611
    dummy_df.iloc[7, 11] = 33.6
    
    dummy_df.iloc[8, 1] = "Asian (other than Chinese)"
    dummy_df.iloc[8, 3:10] = [6.6, 7, 41.6, 28.4, 11.3, 3.6, 1.5]
    dummy_df.iloc[8, 10] = 100
    
    # Second pair: rows 9 and 10
    dummy_df.iloc[9, 1] = "菲律賓人"
    dummy_df.iloc[9, 3:10] = [2918, 4016, 45840, 48555, 25243, 5768, 678]
    dummy_df.iloc[9, 10] = 133018
    dummy_df.iloc[9, 11] = 37.7
    
    dummy_df.iloc[10, 1] = "Filipino"
    dummy_df.iloc[10, 3:10] = [2.2, 3, 34.5, 36.5, 19, 4.3, 0.5]
    dummy_df.iloc[10, 10] = 100
    
    # Third pair: rows 11 and 12 (example additional pair)
    dummy_df.iloc[11, 1] = "印尼人"
    dummy_df.iloc[11, 3:10] = [302, 12405, 85764, 31846, 2255, 483, 322]
    dummy_df.iloc[11, 10] = 133377
    dummy_df.iloc[11, 11] = 30.7
    
    dummy_df.iloc[12, 1] = "Indonesian"
    dummy_df.iloc[12, 3:10] = [3.0, 12.5, 85.8, 31.8, 22.5, 4.8, 3.2]
    dummy_df.iloc[12, 10] = 100
    
    # For simplicity, fill the remaining rows with NaN (they won't be used).
    
    transformed = transform(dummy_df)
    print("Transformed DataFrame:")
    print(transformed)