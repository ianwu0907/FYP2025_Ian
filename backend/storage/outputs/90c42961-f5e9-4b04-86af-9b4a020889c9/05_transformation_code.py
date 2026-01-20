import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the data region and remove non-data rows ===
    # Skip the header row (row 0) and drop rows that are completely blank (all values NaN or empty strings)
    data = df.iloc[1:].copy()
    # Apply strip to string cells and replace empty strings with NaN
    data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    data.replace("", np.nan, inplace=True)
    # Drop rows that are completely empty
    data = data.dropna(how='all')
    print("After Step 1 - Filtered data shape:", data.shape)
    print("Data sample after Step 1:\n", data.head(6))
    
    # === Step 2: Identify and propagate the division (section) marker ===
    # A division marker row is one where column 0 is non-empty and ALL other columns are empty (or NaN)
    def is_division_marker(row):
        # Check if col0 has non-empty string/number
        first = row.iloc[0]
        if pd.isnull(first) or (isinstance(first, str) and first.strip()==''):
            return False
        # Check that all other columns are NaN or empty string
        others = row.iloc[1:]
        return all((pd.isnull(x) or (isinstance(x, str) and x.strip()==''))
                   for x in others)
    
    # Create a boolean column for markers
    data['is_marker'] = data.apply(is_division_marker, axis=1)
    # Create a temporary division column: if marker row, use col0 (stripped), else NaN
    data['division_temp'] = data.iloc[:,0].apply(lambda x: x.strip() if isinstance(x, str) else x)
    data.loc[~data['is_marker'], 'division_temp'] = np.nan
    # Forward-fill the division based on markers
    data['division'] = data['division_temp'].ffill()
    print("After Step 2 - Data with Division Propagation:")
    # Show first two columns and division column for debugging
    print(data.iloc[:, [0, data.columns.get_loc('division')]].head(10))
    
    # === Step 3: Propagate employee names for multi-row records ===
    # Use column 'Name ' if it exists; otherwise, use the second column.
    if 'Name ' in data.columns:
        data['employee_name'] = data['Name '].ffill()
    else:
        data['employee_name'] = data.iloc[:,1].ffill()
    print("After Step 3 - Employee names propagated:")
    print(data[['employee_name']].head(10))
    
    # === Step 4: Map and reshape columns to target schema ===
    # We'll map columns based on position:
    #   division: Propagated in step 2 ('division')
    #   employee_name: from step 3 ('employee_name')
    #   rotation_date: column index 2
    #   group: column index 3
    #   supervisor: column index 4
    #   graduation_date: column index 5
    def safe_get(col_name, col_index):
        if col_name in data.columns:
            return data[col_name]
        elif data.shape[1] > col_index:
            return data.iloc[:, col_index]
        else:
            return pd.Series([np.nan]*len(data))
    
    mapped = pd.DataFrame({
        'division': data['division'],
        'employee_name': data['employee_name'],
        'rotation_date': safe_get('Rotation Date', 2) if 'Rotation Date' in data.columns else data.iloc[:,2],
        'group': safe_get('Group', 3) if 'Group' in data.columns else data.iloc[:,3],
        'supervisor': safe_get('Supervisor', 4) if 'Supervisor' in data.columns else data.iloc[:,4],
        'graduation_date': safe_get('Graduation Date:', 5) if 'Graduation Date:' in data.columns else data.iloc[:,5]
    })
    # Remove rows that are pure marker rows based on is_marker flag
    mapped = mapped[~data['is_marker']]
    # Drop rows that have no rotation_date (assume marker or incomplete rows)
    mapped = mapped[mapped['rotation_date'].notnull()]
    print("After Step 4 - Mapped and reshaped record sample:")
    print(mapped.head(10))
    
    # === Step 5: Clean data and convert data types ===
    # Convert Excel serial dates to proper datetime
    def convert_excel_date(value):
        try:
            if pd.notnull(value):
                # Sometimes value may be float or string
                days = int(float(value))
                return pd.to_datetime('1899-12-30') + pd.to_timedelta(days, unit='D')
        except Exception as e:
            print("Error converting date value:", value, e)
        return pd.NaT
    
    mapped['rotation_date'] = mapped['rotation_date'].apply(lambda x: convert_excel_date(x))
    mapped['graduation_date'] = mapped['graduation_date'].apply(lambda x: convert_excel_date(x) if pd.notnull(x) else pd.NaT)
    
    # Trim text fields. Use .str.strip() safely for objects.
    for col in ['employee_name', 'group', 'supervisor', 'division']:
        if mapped[col].dtype == object:
            mapped[col] = mapped[col].str.strip()
            # Replace empty strings with NaN
            mapped[col] = mapped[col].replace("", np.nan)
    print("After Step 5 - Data types conversion and cleaning:")
    print(mapped.head(10))
    
    # === Step 6: Finalize record selection and validate row count ===
    # Filter out rows with missing essential fields
    result = mapped[mapped['employee_name'].notnull() & mapped['rotation_date'].notnull()]
    print("After Step 6 - Final filtered data shape:", result.shape)
    
    rec_count = len(result)
    print("Final row count (employee records):", rec_count)
    if rec_count != 50:
        print("Warning: Expected 50 employee records, but got", rec_count)
    
    # Spot-check: Look for record for 'Steven Gim' in division 'Power'
    spot = result[(result['employee_name'] == 'Steven Gim') & (result['division'] == 'Power')]
    print("Spot-check record for 'Steven Gim' in 'Power':")
    print(spot)
    
    # Final DataFrame: reorder columns
    final_cols = ['division', 'employee_name', 'rotation_date', 'group', 'supervisor', 'graduation_date']
    result = result[final_cols].reset_index(drop=True)
    return result