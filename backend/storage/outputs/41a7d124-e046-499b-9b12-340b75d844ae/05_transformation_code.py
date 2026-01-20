import pandas as pd
import numpy as np
import re

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print(f"Input shape: {df.shape}")
    
    # === Step 1: Extract the core data region from the spreadsheet ===
    try:
        # Select rows from index 5 to 55 (upper bound excluded)
        df_data = df.iloc[5:56].copy()
        df_data.reset_index(drop=True, inplace=True)
    except Exception as e:
        print("Error during Step 1: Extracting core data region:", e)
        return pd.DataFrame()
    
    print(f"After Step 1, data region shape: {df_data.shape}")
    print("Sample data after Step 1:")
    print(df_data.head(3))
    
    # === Step 2: Pair bilingual rows and consolidate data for each ethnic group ===
    paired_records = []  # will store dictionaries for each paired group from Chinese (even row) and English (odd row)
    
    # iterate over the data region in steps of 2
    # note: if there is an odd number of rows, we'll skip the last unmatched row.
    for i in range(0, len(df_data) - 1, 2):
        try:
            # Get Chinese row (even index) and English row (odd index)
            row_cn = df_data.iloc[i]
            row_en = df_data.iloc[i+1]
            
            # Extract ethnicity names: column 0 (trim whitespaces)
            raw_eth_cn = str(row_cn.iloc[0]).strip() if pd.notnull(row_cn.iloc[0]) else ""
            raw_eth_en = str(row_en.iloc[0]).strip() if pd.notnull(row_en.iloc[0]) else ""
            
            # Look for supplementary notes in the Chinese name using regex e.g. text inside brackets []
            note_search = re.search(r'(\[.*?\])', raw_eth_cn)
            supplementary_notes = note_search.group(1) if note_search else None
            
            # Also remove the note from the ethnicity_cn text
            ethnicity_cn = re.sub(r'\[.*?\]', '', raw_eth_cn).strip()
            ethnicity_en = raw_eth_en  # No cleaning specific for notes in EN
            
            # Collect the measure columns for the Chinese row as numbers
            # They are at fixed column positions: 3,4 for 2011; 5,6 for 2016; 7,8 for 2021.
            # We'll retrieve the raw values from the Chinese row.
            # Some measure values might be missing, so we default to np.nan.
            try:
                count_2011 = row_cn.iloc[3]
                percentage_2011 = row_cn.iloc[4]
                count_2016 = row_cn.iloc[5]
                percentage_2016 = row_cn.iloc[6]
                count_2021 = row_cn.iloc[7]
                percentage_2021 = row_cn.iloc[8]
            except Exception as e:
                print(f"Error extracting measure columns for row index {i}: {e}")
                continue
            
            # Save the consolidated record with measure values as a dictionary; we'll unpivot in the next step.
            paired_records.append({
                "ethnicity_cn": ethnicity_cn,
                "ethnicity_en": ethnicity_en,
                "supplementary_notes": supplementary_notes,
                "2011_count": count_2011,
                "2011_percentage": percentage_2011,
                "2016_count": count_2016,
                "2016_percentage": percentage_2016,
                "2021_count": count_2021,
                "2021_percentage": percentage_2021
            })
        except Exception as e:
            print(f"Error processing row pair starting at index {i}: {e}")
    
    paired_df = pd.DataFrame(paired_records)
    print(f"After Step 2, paired records shape: {paired_df.shape}")
    print("Sample paired records:")
    print(paired_df.head(3))
    
    # === Step 3: Reshape the wide format of measures into a long format ===
    unpivoted_records = []
    # Define mapping for year groups
    year_groups = {
        2011: ('2011_count', '2011_percentage'),
        2016: ('2016_count', '2016_percentage'),
        2021: ('2021_count', '2021_percentage')
    }
    
    for idx, row in paired_df.iterrows():
        for year, (count_col, perc_col) in year_groups.items():
            record = {
                "ethnicity_cn": row["ethnicity_cn"],
                "ethnicity_en": row["ethnicity_en"],
                "supplementary_notes": row["supplementary_notes"],
                "year": year,
                "count": row[count_col],
                "percentage": row[perc_col]
            }
            unpivoted_records.append(record)
    
    long_df = pd.DataFrame(unpivoted_records)
    print(f"After Step 3, long format shape: {long_df.shape}")
    print("Sample of long format data:")
    print(long_df.head(6))
    
    # === Step 4: Clean and convert data types for numerical values ===
    def clean_numeric(val, dtype='int'):
        try:
            if pd.isnull(val):
                return None
            # Convert to string and remove commas and extra spaces
            val_str = str(val).replace(',', '').strip()
            # Remove stray symbols (keep digits, decimal point or minus sign)
            val_str = re.sub(r'[^\d\.-]', '', val_str)
            if dtype == 'int':
                return int(float(val_str))  # Convert via float to handle cases like "123.0"
            else:
                return float(val_str)
        except Exception as e:
            print(f"Numeric conversion error for value '{val}': {e}")
            return None
    
    # Clean count and percentage columns
    long_df['count'] = long_df['count'].apply(lambda x: clean_numeric(x, 'int'))
    long_df['percentage'] = long_df['percentage'].apply(lambda x: clean_numeric(x, 'float'))
    
    # Clean string columns: trim whitespaces
    long_df['ethnicity_cn'] = long_df['ethnicity_cn'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    long_df['ethnicity_en'] = long_df['ethnicity_en'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    print("After Step 4, cleaned numeric conversions:")
    print(long_df.head(6))
    
    # === Step 5: Select and rename the columns to match the target schema ===
    # Final target columns: ethnicity_cn, ethnicity_en, supplementary_notes, year, count, percentage
    result = long_df[['ethnicity_cn', 'ethnicity_en', 'supplementary_notes', 'year', 'count', 'percentage']].copy()
    
    print(f"After Step 5, final DataFrame shape: {result.shape}")
    print("Final sample data:")
    print(result.head(10))
    
    print("Transformation complete.")
    return result