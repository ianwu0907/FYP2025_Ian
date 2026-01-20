import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("=== Starting transformation ===")
    print(f"Input dataframe shape: {df.shape}")
    
    # === Step 1: Extract the main data region excluding extra header/footer rows ===
    try:
        # Based on strategy, the data region is assumed to start at row index 6 until row index 49 (inclusive of paired rows)
        # Do NOT drop rows with missing first column because percentage rows might have blank first cell.
        region_df = df.iloc[6:49].copy().reset_index(drop=True)
    except Exception as e:
        print("Error in slicing dataframe:", e)
        return pd.DataFrame()
    
    print(f"After Step 1: Extracted region shape: {region_df.shape}")
    print("Region sample (first 10 rows):")
    print(region_df.head(10))
    
    # === Step 2: Extract header for age group labels from row with index 3 in the original df ===
    try:
        header_row = df.iloc[3]
        age_groups = {}
        # Assume age group labels are in columns 3 to 9 (7 groups)
        for col in range(3, 10):
            label = str(header_row[col]).strip() if pd.notnull(header_row[col]) else f"col_{col}"
            age_groups[col] = label
        print("Extracted age group labels:")
        print(age_groups)
    except Exception as e:
        print("Error extracting age group header labels:", e)
        return pd.DataFrame()
    
    # === Step 3: Identify and pair bilingual rows (Chinese raw counts row and English percentage row) ===
    records = []
    current_year = None
    i = 0
    n_rows = len(region_df)
    
    # Iterate over the region rows with the expectation that section marker rows (year indicators)
    # appear and then come followed by paired rows (even if the first cell in percentage row is blank)
    while i < n_rows:
        row = region_df.iloc[i]
        first_cell = row.iloc[0]
        
        # Check if the row is a section marker for a new year (e.g., '2011')
        if pd.notnull(first_cell):
            year_str = str(first_cell).strip()
            if year_str.isdigit() and len(year_str) == 4:
                try:
                    current_year = int(year_str)
                    print(f"Found section marker with year: {current_year} at region_df index {i}")
                    i += 1
                    continue
                except Exception as e:
                    print(f"Error parsing year at index {i}: {e}")
        
        # Ensure there is a pair row available; if not, break the loop.
        if i + 1 >= n_rows:
            print(f"Insufficient rows to form a pair at index {i}. Breaking loop.")
            break
        
        # The current row is the raw count (Chinese) row and the next row is the percentage (English) row.
        raw_row = row
        perc_row = region_df.iloc[i+1]
        
        # Extract ethnicity names from column 1 in raw_row (Chinese) and from perc_row (English)
        ethnicity_cn = str(raw_row.iloc[1]).strip() if pd.notnull(raw_row.iloc[1]) else ""
        ethnicity_en = str(perc_row.iloc[1]).strip() if pd.notnull(perc_row.iloc[1]) else ""
        
        # Extract overall median age from raw_row column 11 if available
        overall_median = None
        if raw_row.shape[0] > 11:
            median_val = raw_row.iloc[11]
            try:
                overall_median = float(pd.to_numeric(median_val, errors='coerce')) if pd.notnull(median_val) else None
            except Exception as e:
                overall_median = None
        
        # For each age group defined by columns 3 to 9, form a tidy record.
        for col in range(3, 10):
            age_group_label = age_groups.get(col, f"col_{col}")
            # Convert raw (count) value from raw_row and percentage from perc_row
            raw_val = raw_row.iloc[col]
            perc_val = perc_row.iloc[col]
            try:
                count_val = int(pd.to_numeric(raw_val, errors='coerce')) if pd.notnull(raw_val) else None
            except Exception:
                count_val = None
            try:
                percentage_val = float(pd.to_numeric(perc_val, errors='coerce')) if pd.notnull(perc_val) else None
            except Exception:
                percentage_val = None
            
            record = {
                "year": current_year,
                "ethnicity_cn": ethnicity_cn,
                "ethnicity_en": ethnicity_en,
                "age_group": age_group_label,
                "count": count_val,
                "percentage": percentage_val,
                "median_age": overall_median
            }
            records.append(record)
        
        print(f"Processed ethnicity pair at region_df indexes {i} and {i+1}: year={current_year}, ethnicity_cn='{ethnicity_cn}', ethnicity_en='{ethnicity_en}', median_age={overall_median}")
        i += 2  # Move to the next potential pair
    
    # === Step 4: Assemble the DataFrame from the records ===
    result = pd.DataFrame(records)
    print(f"After Step 3: Resulting DataFrame shape: {result.shape}")
    print("Result sample data (first 10 rows):")
    print(result.head(10))
    
    # === Step 5: Finalize target columns and do type conversion ===
    target_columns = ['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']
    
    # Ensure all expected columns are present; if not, add them with default None
    for col in target_columns:
        if col not in result.columns:
            result[col] = None
            
    try:
        result['year'] = pd.to_numeric(result['year'], errors='coerce').astype('Int64')
    except Exception as e:
        print("Error converting 'year' column:", e)
    
    try:
        result['count'] = pd.to_numeric(result['count'], errors='coerce').astype('Int64')
    except Exception as e:
        print("Error converting 'count' column:", e)
    
    try:
        result['percentage'] = pd.to_numeric(result['percentage'], errors='coerce')
    except Exception as e:
        print("Error converting 'percentage' column:", e)
    
    try:
        result['median_age'] = pd.to_numeric(result['median_age'], errors='coerce')
    except Exception as e:
        print("Error converting 'median_age' column:", e)
    
    # Strip whitespace from string columns
    if 'ethnicity_cn' in result.columns:
        result['ethnicity_cn'] = result['ethnicity_cn'].astype(str).str.strip()
    if 'ethnicity_en' in result.columns:
        result['ethnicity_en'] = result['ethnicity_en'].astype(str).str.strip()
    if 'age_group' in result.columns:
        result['age_group'] = result['age_group'].astype(str).str.strip()
    
    # Reorder columns to match target schema
    result = result[target_columns]
    
    print("=== Final transformed DataFrame ===")
    print(result.head(10))
    print(f"Final shape: {result.shape}")
    
    return result

# Example usage:
# df = pd.read_excel("messy_data.xlsx")
# transformed_df = transform(df)
# transformed_df.to_csv("transformed.csv", index=False)