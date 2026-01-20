import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print(f"Initial DataFrame shape: {df.shape}")
    
    # === Step 0: Standardize column names ===
    try:
        # Make a copy, strip whitespace from column names, and remove trailing colons.
        df = df.copy()
        df.columns = df.columns.str.strip().str.replace(":", "")
        # Debug: Show standardized columns
        print("After Step 0 (Standardize columns), columns are:")
        print(list(df.columns))
    except Exception as e:
        print("Error in Step 0:", e)
        return pd.DataFrame()
    
    # === Step 1: Extract the relevant data region and remove completely blank rows ===
    try:
        # Restrict to rows 0 through 66 (iloc end is exclusive) and make a copy.
        df_region = df.iloc[0:67].copy()
        # Replace cells that are just whitespace with NaN
        df_region = df_region.replace(r'^\s*$', np.nan, regex=True)
        # Drop rows where all values are NaN
        df_region = df_region.dropna(how='all')
        # Reset index for alignment in further processing
        df_region.reset_index(drop=True, inplace=True)
        print("After Step 1:")
        print(f"Shape after extracting region and dropping blank rows: {df_region.shape}")
        print(df_region.head(10))
    except Exception as e:
        print("Error in Step 1:", e)
        return pd.DataFrame()
    
    # === Step 2: Identify and propagate department markers ===
    try:
        # Assuming the first column is 'Unnamed: 0', forward-fill non-null department markers.
        # Using replace ensures that whitespace is treated as NaN.
        df_region['department'] = df_region['Unnamed: 0'].replace(r'^\s*$', np.nan, regex=True).ffill()
        print("After Step 2:")
        print("Data sample with department filled:")
        print(df_region[['Unnamed: 0', 'department']].head(10))
    except Exception as e:
        print("Error in Step 2:", e)
        return pd.DataFrame()
    
    # === Step 3: Identify employee records and propagate employee names ===
    try:
        # Employee names are in the column 'Name' (standardized from 'Name ')
        df_region['employee_name'] = df_region['Name'].replace(r'^\s*$', np.nan, regex=True).ffill()
        print("After Step 3:")
        print("Data sample with employee_name filled:")
        print(df_region[['Name', 'employee_name']].head(10))
    except Exception as e:
        print("Error in Step 3:", e)
        return pd.DataFrame()
    
    # === Step 4: Extract rotation record details from each row ===
    try:
        # Only consider rows that have a non-null rotation date. This excludes the pure marker rows.
        # Using the standardized column name 'Rotation Date'
        if "Rotation Date" not in df_region.columns:
            raise KeyError("Expected column 'Rotation Date' not found in DataFrame.")
        
        # Ensure we are filtering on the correct Series by reindexing if necessary:
        rotation_mask = df_region["Rotation Date"].notna()
        df_rotations = df_region.loc[rotation_mask].copy()
        
        # Map rotation record details.
        df_rotations['rotation_date'] = df_rotations['Rotation Date']
        df_rotations['rotation_group'] = df_rotations['Group']  if "Group" in df_rotations.columns else np.nan
        df_rotations['supervisor'] = df_rotations['Supervisor']  if "Supervisor" in df_rotations.columns else np.nan
        # Replace whitespace-only graduation date with NaN using standardized column name 'Graduation Date'
        if "Graduation Date" in df_rotations.columns:
            df_rotations['graduation_date'] = df_rotations['Graduation Date'].replace(r'^\s*$', np.nan, regex=True)
        else:
            df_rotations['graduation_date'] = np.nan
        print("After Step 4:")
        print(f"Shape of rotation records: {df_rotations.shape}")
        print(df_rotations[['department', 'employee_name', 'rotation_date', 'rotation_group', 'supervisor', 'graduation_date']].head(10))
    except Exception as e:
        print("Error in Step 4:", e)
        return pd.DataFrame()
    
    # === Step 5: Convert data types and perform final column selection and ordering ===
    try:
        # Convert rotation_date to numeric, coercing errors to NaN
        df_rotations['rotation_date'] = pd.to_numeric(df_rotations['rotation_date'], errors='coerce')
        # Drop rows where conversion failed (i.e. remains NaN)
        df_rotations = df_rotations.dropna(subset=['rotation_date'])
        # Convert rotation_date to integer (this will truncate decimals)
        df_rotations['rotation_date'] = df_rotations['rotation_date'].astype(int)
        
        # Final ordering of columns
        final_columns = ['department', 'employee_name', 'rotation_date', 'rotation_group', 'supervisor', 'graduation_date']
        result = df_rotations[final_columns].copy()
        
        print("After Step 5:")
        print("Final transformed DataFrame shape:", result.shape)
        print(result.head(10))
    except Exception as e:
        print("Error in Step 5:", e)
        return pd.DataFrame()
    
    print("Transformation completed successfully.")
    return result

# Example usage (for local testing):
# if __name__ == "__main__":
#     data = {
#         'Unnamed: 0': ['Power', np.nan, np.nan, np.nan, np.nan, np.nan, 'Gas'],
#         'Name ': [np.nan, 'Steven Gim', np.nan, np.nan, np.nan, 'Juan Padron', 'Alice'],
#         'Rotation Date': [np.nan, 36746, 36923, 37012, np.nan, 36724, 36950],
#         'Group': [np.nan, 'Transaction Devl (EIM)', 'Book Running', 'Fundamentals', np.nan, 'Fundamentals (power)', 'Research'],
#         'Supervisor': [np.nan, 'Amanda Colpean', 'Stacey White', 'Lloyd Will', np.nan, 'Doug Gilbert-Smith', 'Bob Manager'],
#         'Graduation Date:': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, '2021-12-31']
#     }
#     sample_df = pd.DataFrame(data)
#     transformed_df = transform(sample_df)
#     print(transformed_df)