import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Step 0: Starting transformation")
    print("Input shape:", df.shape)
    
    # === Step 1: Extract the data region by skipping header and footer rows ===
    # We assume useful data starts at row index 6.
    # Convert the first column (year identifier) to numeric after stripping spaces;
    # filter out rows where conversion fails (likely footer rows or invalid data).
    data_region = df.iloc[6:].copy()
    # Convert first column to string, strip spaces, then to numeric
    data_region['year_temp'] = pd.to_numeric(data_region.iloc[:, 0].astype(str).str.strip(), errors='coerce')
    # Keep rows with valid numeric year
    data_region = data_region[data_region['year_temp'].notna()].copy()
    data_region = data_region.reset_index(drop=True)
    print("After Step 1 - Extracted data region shape:", data_region.shape)
    
    # Retrieve age group labels from the header row at index 3 (columns 3 to 9)
    # Ensure we strip any extra spaces in the labels.
    header_row = df.iloc[3]
    age_labels = [str(x).strip() for x in header_row.iloc[3:10].tolist()]
    print("Step 1: Age group labels extracted from header row:", age_labels)
    
    # === Step 2: Identify and segment bilingual paired rows ===
    # Each ethnicity should be represented by two consecutive rows:
    # the first (Chinese/counts) and the second (English/percentages).
    # If an extra row exists, drop the last row.
    if len(data_region) % 2 != 0:
        print("Warning: Number of rows in data region is odd, dropping the last row for proper pairing.")
        data_region = data_region.iloc[:-1]
    print("Step 2: Data region shape after ensuring paired rows:", data_region.shape)
    
    pairs = []
    # Pair the rows: first row of pair is count row, second is english row.
    for i in range(0, len(data_region), 2):
        count_row = data_region.iloc[i]
        english_row = data_region.iloc[i+1]
        pairs.append((count_row, english_row))
    print("Step 2: Number of paired rows (ethnicities):", len(pairs))
    
    # === Step 3: Reshape the wide age group columns into a long format ===
    # For each paired set, build records for each age group from columns 3 to 9.
    records = []
    for (count_row, english_row) in pairs:
        # Extract and clean 'year' from column index 0 using stripped value.
        try:
            year_val = int(str(count_row.iloc[0]).strip())
        except Exception as e:
            print("Error converting year from value:", count_row.iloc[0], e)
            year_val = np.nan
        
        # Clean ethnicity names from column index 1.
        ethnicity_cn = str(count_row.iloc[1]).strip()
        ethnicity_en = str(english_row.iloc[1]).strip()
        
        # Extract and convert median age from count row, column index 11.
        try:
            median_age_val = float(str(count_row.iloc[11]).strip())
        except Exception as e:
            print("Error converting median_age for ethnicity:", ethnicity_cn, "value:", count_row.iloc[11], e)
            median_age_val = np.nan
        
        # Iterate through each age group column (columns 3 to 9) using the extracted header labels.
        for idx, age_label in enumerate(age_labels):
            col_idx = 3 + idx  # Calculate the correct column index for the age group
            # Convert count value from the Chinese row.
            raw_count = count_row.iloc[col_idx]
            try:
                # Sometimes the count might be stored as float in string form; convert to integer if possible.
                count_val = int(float(str(raw_count).strip()))
            except Exception as e:
                print("Error converting count for ethnicity:", ethnicity_cn, 
                      "age group:", age_label, "value:", raw_count, e)
                count_val = np.nan
            
            # Convert percentage value from the English row.
            raw_percentage = english_row.iloc[col_idx]
            try:
                percentage_val = float(str(raw_percentage).strip())
            except Exception as e:
                print("Error converting percentage for ethnicity:", ethnicity_en, 
                      "age group:", age_label, "value:", raw_percentage, e)
                percentage_val = np.nan
            
            record = {
                "year": year_val,
                "ethnicity_cn": ethnicity_cn,
                "ethnicity_en": ethnicity_en,
                "age_group": age_label,
                "count": count_val,
                "percentage": percentage_val,
                "median_age": median_age_val
            }
            records.append(record)
    
    long_df = pd.DataFrame(records)
    print("Step 3: Reshaped long format DataFrame shape:", long_df.shape)
    print("Step 3: Sample data after reshaping:")
    print(long_df.head())
    
    # === Step 4: Create the target columns and perform data cleaning ===
    # At this point, values have been individually converted.
    # Any additional cleaning such as handling missing values can be done here.
    # For now, we assume that the conversion above is sufficient.
    
    # === Step 5: Final validation and reordering of columns ===
    # Reorder the columns to match the target schema.
    target_columns = ['year', 'ethnicity_cn', 'ethnicity_en', 'age_group', 'count', 'percentage', 'median_age']
    try:
        result = long_df[target_columns].copy()
    except KeyError as e:
        print("Error reordering columns: missing column", e)
        result = long_df.copy()
    print("Step 5: Final DataFrame columns:", result.columns.tolist())
    print("Step 5: Final DataFrame shape:", result.shape)
    
    if not result.empty:
        print("Step 5: Final DataFrame sample rows:")
        print(result.head())
    else:
        print("Warning: Final DataFrame is empty.")
    
    return result