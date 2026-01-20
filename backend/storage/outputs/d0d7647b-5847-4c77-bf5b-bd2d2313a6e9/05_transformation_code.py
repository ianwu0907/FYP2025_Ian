import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the data region and clean out non-data rows ===
    # Read rows 0 to 66 and remove rows that are completely blank or only whitespace.
    df = df.iloc[:67].copy()
    print("After slicing to 67 rows, shape:", df.shape)
    
    # Replace cells that contain only whitespace with NaN
    df = df.replace(r'^\s*$', np.nan, regex=True)
    # Drop rows where all values are NaN
    df = df.dropna(how='all')
    print("After dropping fully empty rows, shape:", df.shape)
    print("Data after step 1 (first 5 rows):")
    print(df.head())
    
    # Reset index to avoid alignment issues later
    df = df.reset_index(drop=True)
    
    # === Step 2: Identify and assign department markers ===
    # Use column 0 values as potential department markers and forward-fill for blank rows.
    df['department'] = df.iloc[:, 0]
    df['department'] = df['department'].ffill()
    print("After assigning department markers, sample 'department' column:")
    print(df['department'].head(10))
    
    # === Step 3: Propagate employee names for multi-row records ===
    # Assume column index 1 holds the employee names; forward-fill missing names.
    df['employee_name'] = df.iloc[:, 1]
    df['employee_name'] = df['employee_name'].ffill()
    print("After propagating employee names, sample 'employee_name' column:")
    print(df['employee_name'].head(10))
    
    # === Step 4: Extract rotation and personnel details and map to target columns ===
    # Filter only rows that have rotation details where column 2 (rotation_date) is not null.
    # Ensure that the boolean mask is aligned with df's index.
    mask = pd.Series(df.iloc[:, 2].notna().values, index=df.index)
    df_details = df.loc[mask].copy()
    print("After filtering rotation rows, shape:", df_details.shape)
    
    # Map and convert columns from the raw data.
    # Column 2 -> rotation_date: convert to numeric (int) if possible.
    df_details['rotation_date'] = pd.to_numeric(df_details.iloc[:, 2], errors='coerce')
    
    # Column 3 -> training_group: remove extra whitespace if string.
    df_details['training_group'] = df_details.iloc[:, 3].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Column 4 -> supervisor: remove extra whitespace if string.
    df_details['supervisor'] = df_details.iloc[:, 4].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Column 5 -> graduation_date: if empty set to None; else, keep as is after stripping whitespace.
    def process_grad_date(val):
        if pd.isna(val):
            return None
        if isinstance(val, str):
            v = val.strip()
            return v if v != "" else None
        return val
    df_details['graduation_date'] = df_details.iloc[:, 5].apply(process_grad_date)
    
    print("After mapping rotation details, sample data (first 5 rows):")
    print(df_details[['rotation_date', 'training_group', 'supervisor', 'graduation_date']].head())
    
    # === Step 5: Assign record type and finalize the schema ===
    # Create a new column 'record_type'. If graduation_date is non-null then 'status_update', else 'rotation_assignment'.
    df_details['record_type'] = df_details['graduation_date'].apply(lambda x: 'status_update' if x is not None else 'rotation_assignment')
    
    # Finalize the DataFrame with the target columns in order.
    result = df_details[['department', 'employee_name', 'rotation_date', 'training_group', 'supervisor', 'graduation_date', 'record_type']].copy()
    
    print("Final DataFrame shape:", result.shape)
    print("Final DataFrame sample (first 5 rows):")
    print(result.head())
    
    return result