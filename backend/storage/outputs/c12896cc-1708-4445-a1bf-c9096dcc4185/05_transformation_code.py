import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print(f"Input DataFrame shape: {df.shape}")
    
    # === Step 1: Extract the main data region and headers ===
    # Use the constant header row: row 3 (zero-index 2) for age group labels from columns 3 to 9 (slice 3:10)
    try:
        age_group_labels = df.iloc[2, 3:10].tolist()
        # Trim whitespaces from age group labels and ensure valid strings
        age_group_labels = [str(label).strip() for label in age_group_labels]
        print("Extracted age group labels from header (row 3, cols 3-9):")
        print(age_group_labels)
    except Exception as e:
        print("Error extracting header for age groups:", e)
        return None
    
    # Extract main data region: rows 7 to 48 (zero-index: rows 6 up to 48; row 7 inclusive, row 48 exclusive)
    try:
        data_region = df.iloc[6:48].copy()
        print(f"Extracted data region shape (rows 7-48): {data_region.shape}")
    except Exception as e:
        print("Error extracting main data region:", e)
        return None
        
    # === Step 2: Separate bilingual measurement rows into paired records ===
    # According to strategy: first row of the pair (odd index within data region) is Chinese row,
    # second row is English row.
    try:
        chinese_rows = data_region.iloc[0::2].reset_index(drop=True)
        english_rows = data_region.iloc[1::2].reset_index(drop=True)
        if len(english_rows) != len(chinese_rows):
            print("Warning: Number of English rows and Chinese rows do not match. Check data integrity.")
        print(f"Number of bilingual pairs: {len(chinese_rows)}")
    except Exception as e:
        print("Error separating bilingual rows:", e)
        return None
    
    # === Step 3: Unpivot age group columns into long format ===
    tidy_data = []
    num_pairs = len(chinese_rows)
    # Select age columns corresponding to age group data: columns with index 3 to 9 (7 age groups)
    age_col_indices = list(range(3, 10))
    
    for i in range(num_pairs):
        try:
            # For each pair, assign:
            # - Chinese row for count, median_age and chinese ethnicity name (col index 1)
            # - English row for percentage and english ethnicity name (col index 1)
            ch_row = chinese_rows.iloc[i]
            en_row = english_rows.iloc[i]
            
            # Extract common values:
            ethnicity_cn = str(ch_row.iloc[1]).strip() if pd.notnull(ch_row.iloc[1]) else ""
            ethnicity_en = str(en_row.iloc[1]).strip() if pd.notnull(en_row.iloc[1]) else ""
            
            # Extract median age from Chinese row column 11 (index 11)
            try:
                median_age = float(ch_row.iloc[11])
            except (ValueError, TypeError):
                median_age = np.nan
                
            # Iterate over each age group column
            for j, col_idx in enumerate(age_col_indices):
                # Map the age column index to the age group label from header extracted earlier
                age_group = age_group_labels[j] if j < len(age_group_labels) else f"col_{col_idx}"
                
                # Get count from Chinese row at col_idx (convert to int if possible)
                try:
                    count_val = ch_row.iloc[col_idx]
                    count = int(float(count_val)) if pd.notnull(count_val) and count_val != "" else np.nan
                except (ValueError, TypeError):
                    count = np.nan
                
                # Get percentage from English row at col_idx (convert to float)
                try:
                    percentage_val = en_row.iloc[col_idx]
                    percentage = float(percentage_val) if pd.notnull(percentage_val) and percentage_val != "" else np.nan
                except (ValueError, TypeError):
                    percentage = np.nan
                
                record = {
                    "year": 2011,  # constant as per instructions
                    "ethnicity_cn": ethnicity_cn,
                    "ethnicity_en": ethnicity_en,
                    "age_group": age_group,
                    "count": count,
                    "percentage": percentage,
                    "median_age": median_age
                }
                tidy_data.append(record)
        except Exception as e:
            print(f"Error processing bilingual pair index {i}:", e)
    
    # Create a DataFrame from the tidy data.
    try:
        result = pd.DataFrame(tidy_data)
    except Exception as e:
        print("Error creating tidy DataFrame:", e)
        return None
    
    # === Step 4: Create target columns and perform type conversion ===
    try:
        result["year"] = result["year"].astype(int)
        result["ethnicity_cn"] = result["ethnicity_cn"].astype(str).str.strip()
        result["ethnicity_en"] = result["ethnicity_en"].astype(str).str.strip()
        result["age_group"] = result["age_group"].astype(str).str.strip()
        result["count"] = pd.to_numeric(result["count"], errors='coerce').astype('Int64')
        result["percentage"] = pd.to_numeric(result["percentage"], errors='coerce')
        result["median_age"] = pd.to_numeric(result["median_age"], errors='coerce')
    except Exception as e:
        print("Error during type conversion:", e)
    
    print(f"After unpivoting and type conversion, result shape: {result.shape}")
    print("Sample data:")
    print(result.head())
    
    # === Step 5: Validation and cleanup ===
    try:
        expected_rows = num_pairs * len(age_col_indices)
        if result.shape[0] != expected_rows:
            print(f"Warning: Expected {expected_rows} rows, but got {result.shape[0]} rows.")
        else:
            print("Row count validation passed.")
    except Exception as e:
        print("Error during validation:", e)
    
    print("Transformation finished.")
    return result