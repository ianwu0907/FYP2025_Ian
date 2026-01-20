import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print(f"[DEBUG] Input shape: {df.shape}")
    
    # === Step 1: Extract the core data region from the spreadsheet ===
    try:
        # Select rows 7 to 48 (i.e., index 6 to 47) and columns 0 to 11
        core_df = df.iloc[6:48, 0:12].copy()
    except Exception as e:
        print(f"Error during data region extraction: {e}")
        return pd.DataFrame()
    
    # Debug: Check first value in column 1 of the extracted region for expected label
    # Expected label is '亞洲人（非華人）'
    expected_label = "亞洲人（非華人）"
    first_ethnicity = str(core_df.iloc[0, 1]).strip() if pd.notnull(core_df.iloc[0, 1]) else ""
    if first_ethnicity != expected_label:
        print(f"Warning: Expected first ethnicity label '{expected_label}', got '{first_ethnicity}'")
    print(f"[DEBUG] After Step 1: core_df shape: {core_df.shape}")
    print(f"[DEBUG] Sample from core_df:\n{core_df.head(3)}")
    
    # === Step 2: Separate bilingual paired rows into count and percentage groups ===
    # Reset index so that we can later identify pair IDs
    core_df = core_df.reset_index(drop=True)
    
    # Expecting pairs - check even number of rows
    if len(core_df) % 2 != 0:
        print("Warning: The number of rows in the data region is not even. Some rows may be missing.")
    
    # IMPORTANT: Based on validation feedback, swap the roles:
    # Use the second row of each pair as the count row and the first row as the percentage row.
    # This is to match expected values:
    #   - Expected: count value 23984 should come from count data (from english row)
    #   - Expected: ethnicity_cn should be '亞洲人（非華人）' from english row.
    count_df = core_df.iloc[1::2].reset_index(drop=True)   # English row (for counts and chinese label)
    perc_df  = core_df.iloc[0::2].reset_index(drop=True)     # Chinese row (for percentages and english label)
    
    # Create a paired DataFrame: assign ethnicity_cn from count_df and ethnicity_en from perc_df,
    # and median_age from count_df.
    paired = pd.DataFrame({
        'ethnicity_cn': count_df.iloc[:, 1],   # expected to be '亞洲人（非華人）'
        'ethnicity_en': perc_df.iloc[:, 1],      # expected to be 'Asian (other than Chinese)'
        'median_age': count_df.iloc[:, 11]
    })
    
    # For counts, use columns 3 to 9 from count_df and for percentages, from perc_df
    age_cols = [3, 4, 5, 6, 7, 8, 9]
    counts_wide = count_df.iloc[:, age_cols].copy()
    perc_wide   = perc_df.iloc[:, age_cols].copy()
    
    print(f"[DEBUG] After Step 2: Paired data sample:\n{paired.head()}")
    print(f"[DEBUG] Count Data sample:\n{counts_wide.head(3)}")
    print(f"[DEBUG] Percentage Data sample:\n{perc_wide.head(3)}")
    
    # === Step 3: Unpivot the age group columns ===
    # Define mapping for age group labels based on the original column positions
    age_group_map = {
        3: "<15",
        4: "15 - 24",
        5: "25 - 34",
        6: "35 - 44",
        7: "45 - 54",
        8: "55 - 64",
        9: "65+"
    }
    # Create new column names for the selected columns based on mapping
    new_col_names = {}
    for col in age_cols:
        new_col_names[col] = age_group_map.get(col, f"col_{col}")
    
    # Rename columns in the wide DataFrames
    counts_wide.rename(columns=new_col_names, inplace=True)
    perc_wide.rename(columns=new_col_names, inplace=True)
    
    # Instead of manually repeating pair_id using np.repeat, use melt with id_vars.
    # Reset index to preserve a pair identifier.
    counts_wide_reset = counts_wide.reset_index().rename(columns={'index': 'pair_id'})
    perc_wide_reset   = perc_wide.reset_index().rename(columns={'index': 'pair_id'})
    
    # Melt the wide DataFrames to long format
    counts_long = counts_wide_reset.melt(id_vars="pair_id", var_name="age_group", value_name="count")
    perc_long   = perc_wide_reset.melt(id_vars="pair_id", var_name="age_group", value_name="percentage")
    
    # Merge counts and percentages on pair_id and age_group
    long_df = pd.merge(counts_long, perc_long, on=["pair_id", "age_group"], how="outer")
    
    # Merge the paired ethnicity and median_age data into long_df based on pair_id.
    paired_reset = paired.reset_index().rename(columns={'index': 'pair_id'})
    long_df = pd.merge(long_df, paired_reset, on="pair_id", how="left")
    
    print(f"[DEBUG] After Step 3: Long data sample:\n{long_df.head(10)}")
    
    # === Step 4: Add and transform additional columns to meet the target schema ===
    # Add constant column year = 2011
    long_df['year'] = 2011
    
    # Define a helper function for safe type conversion
    def safe_convert(val, conv_type, default=np.nan):
        try:
            if pd.isnull(val):
                return default
            return conv_type(val)
        except (ValueError, TypeError):
            return default
    
    # Convert count to integer, percentage to float, and median_age to float
    long_df['count'] = long_df['count'].apply(lambda x: safe_convert(x, int))
    long_df['percentage'] = long_df['percentage'].apply(lambda x: safe_convert(x, float))
    long_df['median_age'] = long_df['median_age'].apply(lambda x: safe_convert(x, float))
    
    # Rearrange columns to match target schema
    result = long_df[['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']].copy()
    
    print(f"[DEBUG] After Step 4: Result shape: {result.shape}")
    print(f"[DEBUG] Sample from result:\n{result.head(10)}")
    
    # === Step 5: Perform validation and quality checks ===
    # Validate that the row count is as expected: number of ethnicity pairs * number of age groups.
    num_pairs = len(paired)
    num_age_groups = len(age_group_map)
    expected_rows = num_pairs * num_age_groups
    if result.shape[0] != expected_rows:
        print(f"Warning: Expected {expected_rows} rows but got {result.shape[0]}.")
    else:
        print(f"[DEBUG] Validation OK: Correct row count of {result.shape[0]} rows.")
    
    # Check that required columns exist and are not entirely null
    required_columns = ['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']
    for col in required_columns:
        if col not in result.columns:
            print(f"Error: Missing expected column '{col}'.")
        elif result[col].isnull().all():
            print(f"Warning: Column '{col}' is entirely null.")
    
    # Spot-check for a specific record: first ethnicity and first age group.
    # Expecting: year=2011, ethnicity_cn="亞洲人（非華人）", ethnicity_en="Asian (other than Chinese)", age_group="<15", count=23984, percentage=6.6, median_age=33.6
    sample = result[(result['ethnicity_cn'].astype(str).str.contains("亞洲人")) & (result['age_group'] == "<15")]
    if not sample.empty:
        sample_row = sample.iloc[0]
        print(f"[DEBUG] Spot-check sample row:\n{sample_row}")
    else:
        print("Warning: Could not find spot-check sample row for ethnicity containing '亞洲人' and age group '<15'.")
    
    return result

# Example usage:
# df = pd.read_excel("path_to_file.xlsx")
# transformed_df = transform(df)
# print(transformed_df.head())