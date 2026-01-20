import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    print("Initial DataFrame sample:\n", df.head(10))
    
    try:
        # === Step 1: Extract the relevant data region and remove extra headers/footers ===
        # Instead of hard-coding an end row, take all rows from row 6 to the end.
        # Also, only drop rows that are completely empty in all columns except the key ones.
        start_row = 6
        df_clean = df.iloc[start_row:].copy()  # use all rows from row 6 onward
        # Preserve the year row even if some columns are NaN.
        if df_clean.shape[0] == 0:
            print("No rows found after slicing starting at row 6.")
            return pd.DataFrame()
        print("After Step 1 - Raw data region shape (from row 6 on):", df_clean.shape)
        print("Data region sample:\n", df_clean.head(10))
        
        # Extract the year from the first row's first column
        year_val = df_clean.iloc[0, 0]
        try:
            year_val = int(str(year_val).strip())
        except Exception as e:
            print("Error converting year to int:", e)
            year_val = np.nan
        
        # Remove the year row and then drop rows that are completely empty in the identifying column (col 1)
        df_pairs = df_clean.iloc[1:].reset_index(drop=True)
        df_pairs = df_pairs.dropna(subset=[1]).reset_index(drop=True)
        print("After removing the year row and rows missing key field, df_pairs shape:", df_pairs.shape)
        print("df_pairs sample:\n", df_pairs.head(10))
        
        # === Step 2: Identify and separate bilingual data pairs ===
        # The data should alternate between count rows and percentage rows.
        n_rows = df_pairs.shape[0]
        if n_rows % 2 != 0:
            print("Warning: Number of rows in data pairs is odd. Dropping the last row for proper pairing.")
            df_pairs = df_pairs.iloc[:-1].reset_index(drop=True)
            n_rows = df_pairs.shape[0]
        print("Number of rows available for pairing:", n_rows)
        
        # Split into count rows and percentage rows based on alternating order.
        count_df = df_pairs.iloc[0::2].reset_index(drop=True)
        percent_df = df_pairs.iloc[1::2].reset_index(drop=True)
        
        # Add an identifier to each pair for later merging.
        count_df['pair_id'] = count_df.index
        percent_df['pair_id'] = percent_df.index
        
        # Create new columns with the correct mapping:
        # For count_df: column 1 is ethnicity in Chinese, column 11 is median_age.
        count_df['ethnicity_cn'] = count_df[1]
        count_df['median_age'] = count_df[11]
        count_df['year'] = year_val  # assign extracted year
        
        # For percent_df: column 1 is ethnicity in English.
        percent_df['ethnicity_en'] = percent_df[1]
        
        print("Count DataFrame shape:", count_df.shape)
        print("Percentage DataFrame shape:", percent_df.shape)
        print("Count DataFrame sample:\n", count_df.head(3))
        print("Percentage DataFrame sample:\n", percent_df.head(3))
        
        # === Step 3: Reshape data by unpivoting age group columns ===
        # Age group columns are defined for indices 3 to 9.
        age_group_mapping = {
            3: "<15",
            4: "15 - 24",
            5: "25 - 34",
            6: "35 - 44",
            7: "45 - 54",
            8: "55 - 64",
            9: "65+"
        }
        age_columns = list(age_group_mapping.keys())
        
        # Melt the count dataframe: id_vars include pair_id, ethnicity_cn, median_age, and year.
        count_melt = pd.melt(count_df, 
                             id_vars=['pair_id', 'ethnicity_cn', 'median_age', 'year'],
                             value_vars=age_columns,
                             var_name='age_col',
                             value_name='count')
        # Melt the percent dataframe: id_vars include pair_id and ethnicity_en.
        percent_melt = pd.melt(percent_df, 
                               id_vars=['pair_id', 'ethnicity_en'],
                               value_vars=age_columns,
                               var_name='age_col',
                               value_name='percentage')
        
        print("After melting - count_melt shape:", count_melt.shape)
        print("Count melt sample:\n", count_melt.head(6))
        print("After melting - percent_melt shape:", percent_melt.shape)
        print("Percent melt sample:\n", percent_melt.head(6))
        
        # === Step 4: Generate target columns and combine bilingual fields ===
        # Merge the melted dataframes on 'pair_id' and 'age_col'.
        merged = pd.merge(count_melt, percent_melt, on=['pair_id', 'age_col'], how='inner')
        print("After merging count and percentage melts, merged shape:", merged.shape)
        print("Merged sample:\n", merged.head(6))
        
        # Map the numerical age_col to age_group label using the defined mapping.
        merged['age_group'] = merged['age_col'].map(age_group_mapping)
        
        # Define safe conversion functions
        def safe_int(x):
            try:
                return int(float(x))
            except Exception as e:
                return np.nan

        def safe_float(x):
            try:
                return float(x)
            except Exception as e:
                return np.nan
        
        # Convert columns to their target data types
        merged['count'] = merged['count'].apply(safe_int)
        merged['percentage'] = merged['percentage'].apply(safe_float)
        merged['median_age'] = merged['median_age'].apply(safe_float)
        merged['year'] = merged['year'].apply(safe_int)
        
        # Select and reorder the target columns.
        result = merged[['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']].copy()
        
        # Ensure proper type assignment (using pandas nullable types when appropriate)
        result['year'] = result['year'].astype('Int64')
        result['count'] = result['count'].astype('Int64')
        result['percentage'] = result['percentage'].astype(float)
        result['median_age'] = result['median_age'].astype(float)
        
        print("After Step 4 - Final reshaped DataFrame shape:", result.shape)
        print("Final DataFrame sample:\n", result.head(10))
        
        # === Step 5: Validate and clean the final dataset ===
        missing_summary = result.isnull().sum()
        print("Missing values summary:\n", missing_summary)
        print("Final DataFrame datatypes:\n", result.dtypes)
        
        # Optional: Spot-check for the first ethnicity-age group pair for 2011.
        if not result.empty:
            spot_check = result[(result['year'] == year_val) &
                                (result['ethnicity_cn'] == "亞洲人（非華人）") &
                                (result['age_group'] == "<15")]
            if not spot_check.empty:
                print("Spot check for matching record:\n", spot_check.iloc[0].to_dict())
            else:
                print("Warning: No matching record found for the spot check criteria.")
        else:
            print("Final result is empty!")
        
        return result
        
    except Exception as e:
        print("An error occurred during transformation:", e)
        return pd.DataFrame()  # Return an empty DataFrame in case of error

if __name__ == "__main__":
    # Create a dummy dataframe that mimics the source structure for testing.
    # 49 rows and 13 columns as per instructions. Most rows are empty except for relevant ones.
    n_rows = 49
    n_cols = 13
    data = {i: [np.nan]*n_rows for i in range(n_cols)}
    
    # Populate sample rows:
    # Row 6: Year row -- only first column has the year
    row6 = [2011] + [np.nan]*(n_cols-1)
    # Row 7: Count row for '亞洲人（非華人）' with age group columns 3-9 and median age in col 11.
    row7 = [np.nan, "亞洲人（非華人）", np.nan, 23984, 25650, 152228, 103841, 41396, 13196, 5316, 365611, 33.6, np.nan]
    # Row 8: Percentage row for 'Asian (other than Chinese)' with age group percentages.
    row8 = [np.nan, "Asian (other than Chinese)", np.nan, 6.6, 7, 41.6, 28.4, 11.3, 3.6, 1.5, 100, np.nan, np.nan]
    # Row 9: Count row for '菲律賓人'
    row9 = [np.nan, "菲律賓人", np.nan, 2918, 4016, 45840, 48555, 25243, 5768, 678, 133018, 37.7, np.nan]
    # Row 10: Percentage row for 'Filipino'
    row10 = [np.nan, "Filipino", np.nan, 2.2, 3, 34.5, 36.5, 19, 4.3, 0.5, 100, np.nan, np.nan]
    
    # Insert the sample rows into the data dictionary at the specified indices:
    sample_rows = {6: row6, 7: row7, 8: row8, 9: row9, 10: row10}
    for idx, row in sample_rows.items():
        for col in range(n_cols):
            data[col][idx] = row[col]
    
    sample_df = pd.DataFrame(data)
    
    # Run the transform function on the sample data
    transformed_df = transform(sample_df)
    print("Transformed DataFrame:\n", transformed_df)