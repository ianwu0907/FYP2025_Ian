import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation")
    print("Input shape:", df.shape)
    
    # === Step 1: Extract the data region ===
    # Skip the header rows (0-5) to extract only the data rows.
    try:
        data_region = df.iloc[6:].copy()
        print("After Step 1 - Data region shape:", data_region.shape)
    except Exception as e:
        print("Error in Step 1:", e)
        raise

    # === Step 2: Propagate and assign the reporting year ===
    # Use column 0 to get year markers, strip spaces, and forward-fill the year.
    try:
        # Convert first column to string, trim spaces, and replace empty strings with NaN.
        data_region['year_temp'] = data_region.iloc[:, 0].astype(str).str.strip()
        data_region['year_temp'].replace("", np.nan, inplace=True)
        # Forward-fill the year values to propagate the reporting year marker.
        data_region['year_temp'] = data_region['year_temp'].ffill()
        # Robust conversion: convert the forwarded year values to numeric (coerce errors)
        data_region['year'] = pd.to_numeric(data_region['year_temp'], errors='coerce')
        print("After Step 2 - Year propagation done. Sample years:\n", data_region['year'].head(10))
    except Exception as e:
        print("Error in Step 2:", e)
        raise

    # === Step 3: Pair bilingual rows for each ethnicity ===
    # Filter actual data rows where column 1 (ethnicity label) is non-null and non-empty.
    try:
        # Ensure column 1 is a string and strip whitespace.
        data_region.iloc[:, 1] = data_region.iloc[:, 1].astype(str).str.strip()
        data_rows = data_region[data_region.iloc[:, 1] != ""].copy().reset_index(drop=True)
        print("After filtering data rows (non-empty col1). Shape:", data_rows.shape)
        
        paired_records = []
        # Get age group headers from the original dataframe.
        # They are assumed to be in row 3, columns 3 to 9.
        age_group_headers = df.iloc[3, 3:10].tolist()
        # Clean headers: trim any extra whitespace. Also, for the '<' age group remove extra spaces.
        age_groups = []
        for header in age_group_headers:
            header_clean = str(header).strip()
            if header_clean.startswith("<"):
                header_clean = header_clean.replace(" ", "")
            age_groups.append(header_clean)
        print("Age group headers used:", age_groups)
        
        # Check that the number of data_rows is even (each pair: CN row and EN row)
        if len(data_rows) % 2 != 0:
            print("Warning: The number of filtered data rows is not even. Last row might be incomplete.")
        
        # Iterate by pairing subsequent rows: assume even-indexed row is CN, following row is EN.
        n_pairs = len(data_rows) // 2
        for i in range(0, n_pairs * 2, 2):
            cn_row = data_rows.iloc[i]
            en_row = data_rows.iloc[i+1]
            
            # Trim ethnicity labels from column 1.
            ethnicity_cn = str(cn_row.iloc[1]).strip()
            ethnicity_en = str(en_row.iloc[1]).strip()
            
            # Reporting year from the CN row.
            report_year = cn_row['year']
            
            # Extract median age from the CN row at column index 11, robust conversion.
            try:
                median_age = float(str(cn_row.iloc[11]).strip())
            except Exception as conv_err:
                print(f"Error converting median_age for ethnicity {ethnicity_cn}: {conv_err}")
                median_age = np.nan
            
            # Loop over the age group columns (columns 3 to 9) using the extracted age group headers.
            for col_idx, age_group in zip(range(3, 10), age_groups):
                # Process count value from CN row.
                try:
                    count_val = cn_row.iloc[col_idx]
                    if pd.isnull(count_val) or str(count_val).strip() == "":
                        count = np.nan
                    else:
                        count = int(str(count_val).replace(',', '').strip())
                except Exception as conv_err:
                    print(f"Error converting count for {ethnicity_cn}, age group {age_group}: {conv_err}")
                    count = np.nan
                
                # Process percentage value from EN row.
                try:
                    perc_val = en_row.iloc[col_idx]
                    if pd.isnull(perc_val) or str(perc_val).strip() == "":
                        percentage = np.nan
                    else:
                        # Remove any '%' signs and extra spaces.
                        percentage = float(str(perc_val).replace('%', '').strip())
                except Exception as conv_err:
                    print(f"Error converting percentage for {ethnicity_en}, age group {age_group}: {conv_err}")
                    percentage = np.nan
                
                record = {
                    'year': report_year,
                    'ethnicity_cn': ethnicity_cn,
                    'ethnicity_en': ethnicity_en,
                    'age_group': age_group,
                    'count': count,
                    'percentage': percentage,
                    'median_age': median_age
                }
                paired_records.append(record)
        paired_df = pd.DataFrame(paired_records)
        print("After Step 3 - Paired data shape:", paired_df.shape)
    except Exception as e:
        print("Error in Step 3:", e)
        raise

    # === Step 4: Unpivot of age group specific columns already achieved in pairing loop ===
    # Each record in paired_df now corresponds to a specific age group.
    
    # === Step 5: Assemble the final target dataframe (with type conversion) ===
    try:
        result = paired_df.copy()
        # Clean string columns by trimming whitespace.
        result['ethnicity_cn'] = result['ethnicity_cn'].astype(str).str.strip()
        result['ethnicity_en'] = result['ethnicity_en'].astype(str).str.strip()
        result['age_group'] = result['age_group'].astype(str).str.strip()
        
        # Convert the 'year' field to numeric and then to a nullable integer type.
        result['year'] = pd.to_numeric(result['year'], errors='coerce').astype("Int64")
        
        # Convert 'count' as nullable integer, 'percentage' and 'median_age' to float.
        result['count'] = pd.to_numeric(result['count'], errors='coerce').astype("Int64")
        result['percentage'] = pd.to_numeric(result['percentage'], errors='coerce')
        result['median_age'] = pd.to_numeric(result['median_age'], errors='coerce')
        print("After Step 5 - Final assembled dataframe shape:", result.shape)
    except Exception as e:
        print("Error in Step 5:", e)
        raise

    # === Step 6: Data cleaning and final validation ===
    try:
        # Final validation: Display the total number of rows.
        print("After Step 6 - Final row count:", result.shape[0])
        # Perform a spot-check for the first ethnicity in 2011 and the '<15' age group.
        # Note: If the stored age group header has extra space, we normalize it for the check.
        spot_check_age = "<15"
        spot = result[(result['year'] == 2011) & (result['age_group'] == spot_check_age)].head(1)
        if not spot.empty:
            print("Spot check for first ethnicity, age group '<15' in 2011:\n", spot.to_dict(orient='records')[0])
        else:
            print("Spot check record not found for year 2011 and age group", spot_check_age)
    except Exception as e:
        print("Error in Step 6:", e)
        raise

    return result

# Example usage:
# if __name__ == "__main__":
#     df = pd.read_excel("path_to_file.xlsx")
#     transformed_df = transform(df)
#     print(transformed_df.head())