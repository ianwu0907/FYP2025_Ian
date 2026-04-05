import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: SPARSE_ROW_FILL for year column (col 0)
    # Replace empty strings with NaN and forward-fill
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    print(f"After sparse fill of year column: {df.shape}")

    # Step 2: Extract multi-level headers for marital_status and sex mapping
    # Row 3 has marital status labels (Chinese/English), Row 4 has sex labels
    # We'll build column_map: {col_index: (marital_status_cn, marital_status_en, sex)}
    marital_status_cn_map = {}
    marital_status_en_map = {}
    sex_map = {}

    # Extract marital status from Row 3 (index 3) — columns 3,6,9,12
    # Map each column in group to the marital status label at its group start
    # Group starts: col 3 -> "從未結婚\nNever married", col 6 -> "已婚\nMarried", etc.
    marital_status_headers = df.iloc[3, [3, 6, 9, 12]].tolist()
    for idx, header in enumerate(marital_status_headers):
        if pd.notna(header):
            parts = str(header).split('\n')
            cn_part = parts[0].strip() if len(parts) > 0 else ""
            en_part = parts[1].strip() if len(parts) > 1 else ""
            # Assign this marital status to all 3 columns in the group: [3,4,5], [6,7,8], [9,10,11], [12,13,14]
            base_col = [3, 6, 9, 12][idx]
            for offset in [0, 1, 2]:
                col_idx = base_col + offset
                if col_idx < len(df.columns):
                    marital_status_cn_map[col_idx] = cn_part
                    marital_status_en_map[col_idx] = en_part

    # Extract sex from Row 4 (index 4) — columns 3,4,5,6,7,8,9,10,11,12,13,14
    # Row 4: [3]="男\nMale", [4]="女\nFemale", [5]="合計\nOverall", [6]="男\nMale", etc.
    sex_headers = df.iloc[4, [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]].tolist()
    for j, header in enumerate([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]):
        if j < len(sex_headers) and pd.notna(sex_headers[j]):
            parts = str(sex_headers[j]).split('\n')
            sex_part = parts[0].strip() if len(parts) > 0 else ""
            # Map Chinese sex label to English equivalent
            if sex_part == "男":
                sex_map[header] = "Male"
            elif sex_part == "女":
                sex_map[header] = "Female"
            elif sex_part == "合計":
                sex_map[header] = "Overall"
            else:
                sex_map[header] = sex_part

    print(f"Built marital_status and sex mappings")

    # Step 3: Identify rows to process (data region: rows 7 to 47 inclusive → indices 7..47)
    # Skip metadata rows (0-6), and exclude aggregation rows and section headers
    # IMPLICIT_AGGREGATION_ROWS: Rows 41, 44, 47 (by index)
    # SECTION_HEADER_ROWS: Rows 8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,42,45,47 (but 47 already excluded)
    # Also exclude blank rows (43, 46)
    # Also exclude "Total" marital status group (columns 12-14) per requirement → skip cols 12,13,14
    # And exclude "Overall" sub-columns (cols 5,8,11,14) per requirement → skip cols 5,8,11,14
    # So value columns to use: [3,4,6,7,9,10] only (6 columns: 3 marital groups × 2 sexes)

    # Build list of valid data rows (indices 7 to 47) excluding:
    # - IMPLICIT_AGGREGATION_ROWS: 41, 44, 47
    # - SECTION_HEADER_ROWS: 8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,42,45
    # - BLANK_ROWS: 43, 46
    # - Also row 40 is last valid ethnicity, so stop before 41
    exclude_rows = {41, 44, 47, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 42, 45, 43, 46}
    data_rows = [i for i in range(7, 48) if i not in exclude_rows]

    # Also need to pair Chinese rows with next English row for ethnicity_en
    # Chinese rows are at indices: 7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,40
    # English rows are at indices: 8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,42,45,47
    # But we only keep Chinese rows, and get English label from next row if it exists and is English label row
    # Build mapping: for each Chinese row i, if i+1 is in English rows and has non-empty label, use it
    en_label_map = {}
    english_row_indices = {8,10,12,14,16,18,20,22,24,26,28,30,32,34,36,38,42,45,47}
    for i in data_rows:
        if i in {7,9,11,13,15,17,19,21,23,25,27,29,31,33,35,37,39,40}:
            if i+1 in english_row_indices and pd.notna(df.iloc[i+1, 1]):
                en_label_map[i] = str(df.iloc[i+1, 1]).strip()
            else:
                en_label_map[i] = ""

    # Step 4: Record collection loop
    records = []
    # Value columns to use: 3,4,6,7,9,10 (Never married: 3,4; Married: 6,7; Widowed/divorced/separated: 9,10)
    value_cols = [3, 4, 6, 7, 9, 10]
    
    for i in data_rows:
        # Skip if row has no Chinese ethnicity label or is invalid
        if not pd.notna(df.iloc[i, 1]):
            continue
        ethnicity_cn = str(df.iloc[i, 1]).strip()
        
        # Get English label if available
        ethnicity_en = en_label_map.get(i, "")
        
        # Get year from column 0 (sparse-filled)
        year_val = df.iloc[i, 0]
        if pd.isna(year_val) or str(year_val).strip() == "":
            year = 2011
        else:
            try:
                year = int(str(year_val).strip())
            except:
                year = 2011
        
        # Extract values for each value column
        for j in value_cols:
            if j not in marital_status_cn_map or j not in sex_map:
                continue
                
            marital_status_cn = marital_status_cn_map[j]
            marital_status_en = marital_status_en_map[j]
            sex = sex_map[j]
            
            val = df.iloc[i, j]
            if pd.isna(val) or str(val).strip() == "" or str(val).strip() == "-":
                percentage = np.nan
            else:
                try:
                    # Remove any non-numeric characters except decimal point
                    clean_val = str(val).strip().replace(',', '').replace('%', '')
                    percentage = float(clean_val)
                except:
                    percentage = np.nan
            
            # Skip if percentage is NaN or if marital_status_cn is "總計" (Total) — but our value_cols exclude 12-14 so it's safe
            if pd.isna(percentage):
                continue
                
            records.append({
                "year": year,
                "ethnicity_cn": ethnicity_cn,
                "ethnicity_en": ethnicity_en,
                "marital_status_cn": marital_status_cn,
                "marital_status_en": marital_status_en,
                "sex": sex,
                "percentage": percentage
            })
    
    print(f"Collected {len(records)} records before DataFrame creation")

    # Step 5: Create result DataFrame with exact schema columns
    target_cols = ['year', 'ethnicity_cn', 'ethnicity_en', 'marital_status_cn', 'marital_status_en', 'sex', 'percentage']
    result = pd.DataFrame(records, columns=target_cols)
    
    # Drop rows where percentage is NaN
    result = result.dropna(subset=['percentage'])
    
    print(f"Final output: {result.shape}")
    return result