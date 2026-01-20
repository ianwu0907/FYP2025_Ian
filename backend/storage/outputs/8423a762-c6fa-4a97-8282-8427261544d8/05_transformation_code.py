import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Step 0: Starting transformation")
    print(f"Input shape: {df.shape}")
    
    # === Step 1: Extract the main data region and remove header and extraneous rows ===
    # We skip rows 0, 2, 3, 4; start from row index 6
    try:
        data = df.iloc[6:].reset_index(drop=True)
    except Exception as e:
        print("Error extracting rows from index 6 onward:", e)
        raise e
    print(f"After extracting data region: shape = {data.shape}")
    
    # === Step 2: Propagate the year across all observations ===
    # Identify section marker rows where column 0 is a year marker (numeric).
    # Create a 'year' column if cell in col 0 is digit, then forward fill it.
    try:
        # Convert column 0 to string, strip and check if it is numeric
        data['year'] = np.where(data.iloc[:, 0].astype(str).str.strip().str.isdigit(),
                                data.iloc[:, 0].astype(str).str.strip(),
                                np.nan)
        data['year'] = pd.Series(data['year']).ffill()
    except Exception as e:
        print("Error propagating year:", e)
        raise e
    print("After propagating year column, sample data:")
    print(data[['year']].head(10))
    
    # Remove the section marker rows (rows where column 0 is a pure digit)
    try:
        condition = ~data.iloc[:, 0].astype(str).str.strip().str.isdigit()
        data = data[condition].reset_index(drop=True)
    except Exception as e:
        print("Error removing marker rows:", e)
        raise e
    print(f"After removing section marker rows: shape = {data.shape}")
    
    # === Step 3: Separate bilingual rows into raw count and percentage pairs ===
    # The first row of each pair is raw count info (Chinese ethnicity, counts in cols 3-9, median age in col 11)
    # The following row is percentage info (English ethnicity, percentages in cols 3-9)
    if len(data) % 2 != 0:
        print("Warning: The number of rows after removing marker rows is odd; potential missing pair row.")
    try:
        raw_rows = data.iloc[::2].reset_index(drop=True)
        perc_rows = data.iloc[1::2].reset_index(drop=True)
        raw_rows['pair_id'] = raw_rows.index
        perc_rows['pair_id'] = perc_rows.index
    except Exception as e:
        print("Error separating bilingual pairs:", e)
        raise e
    print(f"Number of raw (count) rows: {raw_rows.shape[0]}")
    print(f"Number of percentage rows: {perc_rows.shape[0]}")
    
    # === Step 4: Unpivot the age group columns into a long format ===
    # Get age group labels from header row (row 3) for columns 3 to 9
    try:
        age_group_cols = df.columns[3:10]  # columns 3 through 9
        # Build mapping from original column name to cleaned header label from row 3
        age_group_mapping = {}
        for col in age_group_cols:
            label = str(df.loc[3, col]).strip()
            label = label.replace(" ", "")  # remove extra spaces within the label
            age_group_mapping[col] = label
        
        # Process raw_rows: Clean ethnicity_cn and median_age, then melt wide age-group count columns.
        raw_rows = raw_rows.copy()
        raw_rows['ethnicity_cn'] = raw_rows.iloc[:, 1].astype(str).str.strip()
        raw_rows['median_age'] = pd.to_numeric(raw_rows.iloc[:, 11].astype(str).str.strip(), errors='coerce')
        raw_melt = raw_rows.melt(id_vars=['year', 'ethnicity_cn', 'median_age', 'pair_id'],
                                  value_vars=list(age_group_cols),
                                  var_name='age_group_col',
                                  value_name='count')
        
        # Process percentage rows: Clean ethnicity_en and melt wide age-group percentage columns.
        perc_rows = perc_rows.copy()
        perc_rows['ethnicity_en'] = perc_rows.iloc[:, 1].astype(str).str.strip()
        perc_melt = perc_rows.melt(id_vars=['year', 'ethnicity_en', 'pair_id'],
                                    value_vars=list(age_group_cols),
                                    var_name='age_group_col',
                                    value_name='percentage')
        
        # Merge raw and percentage melts on 'year', 'pair_id', and 'age_group_col'
        merged = pd.merge(raw_melt, perc_melt, on=['year', 'pair_id', 'age_group_col'], how='inner')
        # Map the age_group_col to the actual age group label using header info
        merged['age_group'] = merged['age_group_col'].map(age_group_mapping)
    except Exception as e:
        print("Error during unpivoting:", e)
        raise e
    print(f"After unpivoting: shape = {merged.shape}")
    print("Sample unpivoted data:")
    print(merged.head(10))
    
    # === Step 5: Extract and convert data types for key values ===
    # Convert year to int, count to int, percentage to float, and median_age to float.
    # Use safe conversion to handle float-to-int conversion error by applying conversion via a lambda.
    def safe_int(x):
        try:
            # if number is not nan, convert to int; otherwise return pd.NA
            return int(x) if pd.notnull(x) else pd.NA
        except Exception:
            return pd.NA

    try:
        merged['year'] = pd.to_numeric(merged['year'], errors='coerce').apply(safe_int).astype('Int64')
        merged['count'] = pd.to_numeric(merged['count'].astype(str).str.replace(',', '').str.strip(),
                                        errors='coerce').apply(safe_int).astype('Int64')
        merged['percentage'] = pd.to_numeric(merged['percentage'].astype(str).str.strip(), errors='coerce')
        merged['median_age'] = pd.to_numeric(merged['median_age'], errors='coerce')
    except Exception as e:
        print("Error converting data types:", e)
        raise e
    print("After data type conversion:")
    print(merged[['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']].dtypes)
    
    # === Step 6: Select and rename final target columns ===
    # Final columns: ['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']
    try:
        result = merged[['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']].copy()
    except Exception as e:
        print("Error selecting final columns:", e)
        raise e
    print(f"After selecting final columns: shape = {result.shape}")
    print("Sample final transformed data:")
    print(result.head(10))
    
    # === Step 7: Validate transformation and perform edge-case checks ===
    # For example, check for the expected number of rows and sample record values.
    expected_row_count = 147
    actual_row_count = result.shape[0]
    if actual_row_count != expected_row_count:
        print(f"Warning: Expected {expected_row_count} rows but got {actual_row_count} rows.")
    else:
        print("Row count validation passed.")
    
    # Example spot-check: record for a specific ethnicity and age group.
    spot_check = result[(result['ethnicity_cn'] == "亞洲人（非華人）") & (result['age_group'] == "<15")]
    if not spot_check.empty:
        print("Spot check record for first ethnicity, first age group:")
        print(spot_check.iloc[0].to_dict())
    else:
        print("Warning: Spot check record not found.")
    
    print("Transformation completed successfully.")
    return result

if __name__ == "__main__":
    try:
        sample_df = pd.read_csv("sample_data.csv", header=None)
        transformed_df = transform(sample_df)
        print("Final transformed DataFrame head:")
        print(transformed_df.head())
    except Exception as e:
        print("Error during transformation testing:", e)