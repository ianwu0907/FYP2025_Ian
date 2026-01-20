import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print(f"Initial DataFrame shape: {df.shape}")
    
    # === Step 1: Extract the data region from the spreadsheet ===
    # If the DataFrame already has exactly 56 rows (i.e. the data region), use it as is;
    # otherwise, assume header/footer rows are present and extract rows 5 to 55 (0-indexed rows 4 to 54).
    try:
        if df.shape[0] == 56:
            data_region = df.copy()
            print("Data region assumed to be the entire DataFrame.")
        else:
            data_region = df.iloc[4:55].copy()
            print("Data region extracted from rows 5 to 55 (1-indexed).")
        data_region.reset_index(drop=True, inplace=True)
        print(f"After Step 1 (extract data region) shape: {data_region.shape}")
        print("Data region sample:")
        print(data_region.head())
    except Exception as e:
        print("Error in Step 1:", e)
        return None

    # Define helper to check if a string contains Chinese characters
    def contains_chinese(text):
        if not isinstance(text, str):
            return False
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    # === Step 2: Identify and merge bilingual rows into single records ===
    # The first row of each pair contains the Chinese name and numeric data,
    # while the next row provides the English name.
    merged_records = []
    total_rows = data_region.shape[0]
    n_pairs = total_rows // 2
    for i in range(n_pairs):
        idx_first = i * 2
        idx_second = idx_first + 1
        try:
            # Get candidate texts from column 0 for bilingual names
            cand_cn = str(data_region.iloc[idx_first, 0]).strip() if pd.notnull(data_region.iloc[idx_first, 0]) else ''
            cand_en = str(data_region.iloc[idx_second, 0]).strip() if pd.notnull(data_region.iloc[idx_second, 0]) else ''
            
            # Heuristic: if one candidate contains Chinese and the other doesn't, assign accordingly.
            if contains_chinese(cand_cn) and not contains_chinese(cand_en):
                ethnicity_cn = cand_cn
                ethnicity_en = cand_en
            elif contains_chinese(cand_en) and not contains_chinese(cand_cn):
                ethnicity_cn = cand_en
                ethnicity_en = cand_cn
            else:
                # Default: assume first is Chinese and second is English.
                ethnicity_cn = cand_cn
                ethnicity_en = cand_en
            
            # Ensure that there are at least 10 columns to extract numeric data (columns 4-9)
            if data_region.shape[1] < 10:
                raise ValueError("Data region does not have enough columns; expected at least 10 columns.")
            
            # Extract numeric data from the first (Chinese) row.
            # According to the strategy, columns:
            #   col 4 and 5: 2011 count and percentage,
            #   col 6 and 7: 2016 count and percentage,
            #   col 8 and 9: 2021 count and percentage.
            count_2011 = data_region.iloc[idx_first, 4]
            percentage_2011 = data_region.iloc[idx_first, 5]
            count_2016 = data_region.iloc[idx_first, 6]
            percentage_2016 = data_region.iloc[idx_first, 7]
            count_2021 = data_region.iloc[idx_first, 8]
            percentage_2021 = data_region.iloc[idx_first, 9]
            
            merged_records.append({
                'ethnicity_cn': ethnicity_cn,
                'ethnicity_en': ethnicity_en,
                'count_2011': count_2011,
                'percentage_2011': percentage_2011,
                'count_2016': count_2016,
                'percentage_2016': percentage_2016,
                'count_2021': count_2021,
                'percentage_2021': percentage_2021
            })
        except Exception as e:
            print(f"Error merging pair at index {i}: {e}")
    
    # If there's an odd row remaining, process it assuming no English name available.
    if total_rows % 2 == 1:
        try:
            idx_last = total_rows - 1
            cand_cn = str(data_region.iloc[idx_last, 0]).strip() if pd.notnull(data_region.iloc[idx_last, 0]) else ''
            ethnicity_cn = cand_cn
            ethnicity_en = ''
            if data_region.shape[1] < 10:
                raise ValueError("Data region does not have enough columns.")
            count_2011 = data_region.iloc[idx_last, 4]
            percentage_2011 = data_region.iloc[idx_last, 5]
            count_2016 = data_region.iloc[idx_last, 6]
            percentage_2016 = data_region.iloc[idx_last, 7]
            count_2021 = data_region.iloc[idx_last, 8]
            percentage_2021 = data_region.iloc[idx_last, 9]
            merged_records.append({
                'ethnicity_cn': ethnicity_cn,
                'ethnicity_en': ethnicity_en,
                'count_2011': count_2011,
                'percentage_2011': percentage_2011,
                'count_2016': count_2016,
                'percentage_2016': percentage_2016,
                'count_2021': count_2021,
                'percentage_2021': percentage_2021
            })
            print("Processed an extra odd row at the end of data region.")
        except Exception as e:
            print("Error processing the extra odd row:", e)
    
    merged_df = pd.DataFrame(merged_records)
    print(f"After Step 2 (merge bilingual rows) shape: {merged_df.shape}")
    print("Merged records sample:")
    print(merged_df.head())
    
    # === Step 3: Reshape the wide data by unpivoting year-specific columns ===
    long_format_records = []
    for idx, row in merged_df.iterrows():
        for year, count_col, perc_col in [(2011, 'count_2011', 'percentage_2011'),
                                          (2016, 'count_2016', 'percentage_2016'),
                                          (2021, 'count_2021', 'percentage_2021')]:
            try:
                long_format_records.append({
                    'ethnicity_cn': row['ethnicity_cn'],
                    'ethnicity_en': row['ethnicity_en'],
                    'year': year,
                    'count': row[count_col],
                    'percentage': row[perc_col]
                })
            except Exception as e:
                print(f"Error processing row index {idx} for year {year}: {e}")
    
    long_df = pd.DataFrame(long_format_records)
    print(f"After Step 3 (reshape data) shape: {long_df.shape}")
    print("Long format sample:")
    print(long_df.head(10))
    
    # === Step 4: Clean, convert, and assign final target columns ===
    try:
        # Trim whitespace from ethnicity names
        long_df['ethnicity_cn'] = long_df['ethnicity_cn'].astype(str).str.strip()
        long_df['ethnicity_en'] = long_df['ethnicity_en'].astype(str).str.strip()
        
        # Convert count to integer (handle conversion errors gracefully)
        long_df['count'] = pd.to_numeric(long_df['count'], errors='coerce').fillna(0).astype(int)
        # Convert percentage to float
        long_df['percentage'] = pd.to_numeric(long_df['percentage'], errors='coerce').fillna(0.0).astype(float)
        # Ensure year is integer
        long_df['year'] = pd.to_numeric(long_df['year'], errors='coerce').fillna(0).astype(int)
        
        # Final target ordering: ['ethnicity_cn', 'ethnicity_en', 'year', 'count', 'percentage']
        result = long_df[['ethnicity_cn', 'ethnicity_en', 'year', 'count', 'percentage']].copy()
        print("After Step 4 (clean and convert) sample:")
        print(result.head(10))
    except Exception as e:
        print("Error in Step 4:", e)
        return None
    
    # === Step 5: Validate the transformation and handle any discrepancies ===
    try:
        # Expected rows = number of merged records * 3 (each record unpivoted into 3 years)
        expected_rows = (len(merged_df)) * 3
        actual_rows = result.shape[0]
        print(f"Validation: expected rows = {expected_rows}, actual rows = {actual_rows}")
        if actual_rows != expected_rows:
            print("Warning: Number of rows does not match expected count.")
        
        # Spot-check for the Filipino ethnic group for the year 2011.
        # Expected values:
        # ethnicity_cn: "菲律賓人", ethnicity_en: "Filipino", count: 133018, percentage: 29.5
        filipino_check = result[(result['ethnicity_cn'] == '菲律賓人') & (result['year'] == 2011)]
        if not filipino_check.empty:
            print("Filipino ethnic group spot-check (year 2011):")
            print(filipino_check)
            expected_count = 133018
            expected_percentage = 29.5
            actual_count = int(filipino_check.iloc[0]['count'])
            actual_percentage = float(filipino_check.iloc[0]['percentage'])
            if actual_count != expected_count or not np.isclose(actual_percentage, expected_percentage):
                print("Spot-check failed: Filipino group values do not match expected (count: {} vs {} and percentage: {} vs {}).".format(
                    actual_count, expected_count, actual_percentage, expected_percentage))
            else:
                print("Spot-check passed for Filipino group in 2011.")
        else:
            print("Spot-check failed: Filipino ethnic group with ethnicity_cn '菲律賓人' for year 2011 not found in the data.")
        
        # Check for any missing bilingual names or numeric values
        if result['ethnicity_cn'].isnull().any() or result['ethnicity_en'].isnull().any():
            print("Warning: There are missing bilingual names.")
        if result[['count', 'percentage']].isnull().any().any():
            print("Warning: There are missing numeric values.")
    except Exception as e:
        print("Error in Step 5 (validation):", e)
    
    print("Transformation completed successfully.")
    return result