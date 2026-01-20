import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial input shape:", df.shape)
    
    # === Step 1: Extract the data region and remove header/footer rows ===
    # We use rows 6 to 48 (inclusive, i.e. 6:49 in iloc) as the working region.
    try:
        work_df = df.iloc[6:49].reset_index(drop=True)
    except Exception as e:
        print("Error slicing the data region:", e)
        return None
    print("After Step 1, shape:", work_df.shape)
    print("Sample data (first 5 rows):\n", work_df.head(5))
    
    # === Step 2: Forward-fill section markers (year) ===
    # Create a new column 'year' extracted from column 0.
    # We check if the value (converted to string) is a 4-digit number.
    def extract_year(x):
        try:
            # Convert to string, strip spaces, and check if it's all digits and length 4
            xs = str(x).strip()
            if xs.isdigit() and len(xs) == 4:
                return xs
            else:
                return np.nan
        except Exception as e:
            return np.nan
    
    work_df['year'] = work_df.iloc[:,0].apply(extract_year).ffill()
    print("After Step 2, sample 'year' column:\n", work_df['year'].head(10))
    
    # === Step 3: Identify and pair bilingual rows ===
    # We want to exclude the section marker rows which typically do not have an ethnicity name.
    # Assume that bilingual data rows have a non-null in column index 1.
    bilingual_df = work_df[work_df.iloc[:,1].notnull()].reset_index(drop=True)
    print("After filtering bilingual rows, shape:", bilingual_df.shape)
    
    # Ensure we have an even number of rows (each ethnicity represented by a pair)
    num_rows = bilingual_df.shape[0]
    if num_rows % 2 != 0:
        print("Warning: Number of bilingual rows is not even. Some data might be missing.")
    
    paired_records = []
    
    # Define mapping for age group columns from column indices 3 to 9
    age_group_map = {
        3: "<15",
        4: "15 - 24",
        5: "25 - 34",
        6: "35 - 44",
        7: "45 - 54",
        8: "55 - 64",
        9: "65+"
    }
    
    # Loop over the bilingual rows in steps of 2 (first = count/Chinese row; second = percentage/English row)
    for i in range(0, num_rows - 1, 2):
        try:
            count_row = bilingual_df.iloc[i]
            perc_row = bilingual_df.iloc[i+1]
            
            # Extract ethnicity names from column 1, trim whitespace
            ethnicity_cn = str(count_row.iloc[1]).strip() if pd.notnull(count_row.iloc[1]) else ""
            ethnicity_en = str(perc_row.iloc[1]).strip() if pd.notnull(perc_row.iloc[1]) else ""
            
            # Use the forward-filled 'year' from either row (they should be the same)
            year_val = count_row['year']
            
            # Get median_age from the count row at column 11, attempt conversion later
            median_age_val = count_row.iloc[11]
            
            # For each age group (columns 3 to 9), extract count and percentage values.
            for col_idx, age_group_label in age_group_map.items():
                # Extract and clean count (from count_row) and percentage (from perc_row)
                count_val = count_row.iloc[col_idx]
                perc_val = perc_row.iloc[col_idx]
                
                # Cleaning: Remove commas/spaces and convert to numeric if possible.
                try:
                    if isinstance(count_val, str):
                        count_clean = int(count_val.replace(",", "").strip())
                    elif pd.notnull(count_val):
                        count_clean = int(count_val)
                    else:
                        count_clean = np.nan
                except Exception as e:
                    print(f"Error converting count value at pair index {i}, col {col_idx}: {count_val} - {e}")
                    count_clean = np.nan
                    
                try:
                    if isinstance(perc_val, str):
                        perc_clean = float(perc_val.replace("%", "").strip())
                    elif pd.notnull(perc_val):
                        perc_clean = float(perc_val)
                    else:
                        perc_clean = np.nan
                except Exception as e:
                    print(f"Error converting percentage value at pair index {i}, col {col_idx}: {perc_val} - {e}")
                    perc_clean = np.nan
                
                # Append dictionary for this record
                paired_records.append({
                    "year": year_val,
                    "ethnicity_cn": ethnicity_cn,
                    "ethnicity_en": ethnicity_en,
                    "age_group": age_group_label,
                    "count": count_clean,
                    "percentage": perc_clean,
                    "median_age": median_age_val  # will convert later
                })
        except Exception as e:
            print("Error processing bilingual pair at index", i, ":", e)
    
    # Convert list of records to DataFrame
    result = pd.DataFrame(paired_records)
    print("After Step 3 and 4 (reshaping), shape:", result.shape)
    print("Sample of reshaped data:\n", result.head(10))
    
    # === Step 5: Clean data types and finalize column selection ===
    # Trim whitespace for string columns in ethnicity names and age_group
    result['ethnicity_cn'] = result['ethnicity_cn'].astype(str).str.strip()
    result['ethnicity_en'] = result['ethnicity_en'].astype(str).str.strip()
    result['age_group'] = result['age_group'].astype(str).str.strip()
    
    # Convert year to integer, count to integer, percentage and median_age to float.
    try:
        result['year'] = result['year'].astype(int)
    except Exception as e:
        print("Error converting year to int:", e)
    
    try:
        result['count'] = result['count'].astype(int)
    except Exception as e:
        print("Error converting count to int:", e)
    
    try:
        result['percentage'] = result['percentage'].astype(float)
    except Exception as e:
        print("Error converting percentage to float:", e)
    
    try:
        result['median_age'] = result['median_age'].apply(lambda x: str(x).strip() if pd.notnull(x) else x)
        result['median_age'] = result['median_age'].astype(float)
    except Exception as e:
        print("Error converting median_age to float:", e)
    
    # Select and order target columns
    target_columns = ['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']
    result = result[target_columns]
    
    print("Final output shape:", result.shape)
    print("Final sample data:\n", result.head(10))
    
    return result