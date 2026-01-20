import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the data region and remove extraneous header/footer rows ===
    # Limit to rows 0 through 66. Then filter rows where at least one key value column (or a section marker) is not null.
    try:
        df_region = df.iloc[:67].copy()  # extract rows 0 to 66
        # Reset index to ensure alignment for boolean indexing later (fixes unalignable boolean series issues)
        df_region.reset_index(drop=True, inplace=True)
    except Exception as e:
        print("Error slicing DataFrame rows:", e)
        return pd.DataFrame()
    
    # Define key columns that indicate a rotation record (employee name, rotation date, group assignment)
    raw_key_columns = ['Name ', 'Rotation Date', 'Group']
    # Use only those key columns that exist in df_region
    key_columns = [col for col in raw_key_columns if col in df_region.columns]
    
    # Replace empty strings with NaN in key columns (if they exist)
    for col in key_columns:
        try:
            df_region[col] = df_region[col].replace(r'^\s*$', np.nan, regex=True)
        except Exception as e:
            print(f"Error cleaning key column {col}:", e)
    
    # Also, check if the first column ('Unnamed: 0') contains section markers.
    try:
        if 'Unnamed: 0' in df_region.columns:
            # Replace empty strings with NaN to detect valid section marker
            col0 = df_region['Unnamed: 0'].replace(r'^\s*$', np.nan, regex=True)
            condition_marker = col0.notnull()
        else:
            condition_marker = pd.Series(False, index=df_region.index)
            print("Warning: 'Unnamed: 0' column not found for section markers.")
    except Exception as e:
        print("Error processing 'Unnamed: 0' for section markers:", e)
        condition_marker = pd.Series(False, index=df_region.index)
    
    # Create condition for rows that have at least one key field or a valid section marker.
    if key_columns:
        condition_key = df_region[key_columns].notnull().any(axis=1)
    else:
        condition_key = pd.Series(False, index=df_region.index)
        
    overall_condition = condition_key | condition_marker
    try:
        df_filtered = df_region[overall_condition].copy()
    except Exception as e:
        print("Error filtering DataFrame rows:", e)
        return pd.DataFrame()
    
    print("After Step 1 - Data region extraction and filtering, shape:", df_filtered.shape)
    print("Step 1 sample data:\n", df_filtered.head(5))
    
    # === Step 2: Identify and propagate business unit section markers ===
    # Use column 'Unnamed: 0' to detect business unit markers, then forward fill
    try:
        if 'Unnamed: 0' in df_filtered.columns:
            # Replace empty strings with NaN, then forward fill to get the current section marker
            df_filtered['business_unit_section'] = df_filtered['Unnamed: 0'].replace(r'^\s*$', np.nan, regex=True).ffill()
        else:
            df_filtered['business_unit_section'] = np.nan
            print("Warning: 'Unnamed: 0' column not found.")
    except Exception as e:
        print("Error in propagating business unit section markers:", e)
        df_filtered['business_unit_section'] = np.nan
    
    print("After Step 2 - Business unit section propagation, sample data:\n", 
          df_filtered[['Unnamed: 0', 'business_unit_section']].head(5))
    
    # === Step 3: Determine the employee name for each rotation record ===
    # Forward fill the 'Name ' column (if exists) to populate missing employee names in continuation rows.
    try:
        if 'Name ' in df_filtered.columns:
            df_filtered['employee_name'] = df_filtered['Name '].replace(r'^\s*$', np.nan, regex=True).ffill()
        else:
            df_filtered['employee_name'] = np.nan
            print("Warning: 'Name ' column not found.")
    except Exception as e:
        print("Error in employee name propagation:", e)
        df_filtered['employee_name'] = np.nan
    
    print("After Step 3 - Employee names after forward fill, sample data:\n", 
          df_filtered[['Name ', 'employee_name']].head(5))
    
    # === Step 4: Reshape and map data columns to target schema ===
    # Map and type convert: rotation_date from 'Rotation Date', group_training_assignment from 'Group',
    # supervisor from 'Supervisor', graduation_date from 'Graduation Date:'.
    try:
        if 'Rotation Date' in df_filtered.columns:
            # Convert rotation_date to numeric (using nullable integer type)
            df_filtered['rotation_date'] = pd.to_numeric(df_filtered['Rotation Date'], errors='coerce').astype('Int64')
        else:
            df_filtered['rotation_date'] = pd.NA
    except Exception as e:
        print("Error converting 'Rotation Date':", e)
        df_filtered['rotation_date'] = pd.NA
    
    try:
        if 'Group' in df_filtered.columns:
            # Clean group training assignment by stripping whitespace
            df_filtered['group_training_assignment'] = df_filtered['Group'].astype(str).str.strip()
        else:
            df_filtered['group_training_assignment'] = np.nan
            print("Warning: 'Group' column not found.")
    except Exception as e:
        print("Error processing 'Group':", e)
        df_filtered['group_training_assignment'] = np.nan
        
    try:
        if 'Supervisor' in df_filtered.columns:
            # Clean supervisor column by stripping whitespace, and treat string 'nan' as NaN
            df_filtered['supervisor'] = df_filtered['Supervisor'].astype(str).str.strip()
            df_filtered.loc[df_filtered['supervisor'].str.lower() == 'nan', 'supervisor'] = np.nan
        else:
            df_filtered['supervisor'] = np.nan
            print("Warning: 'Supervisor' column not found.")
    except Exception as e:
        print("Error processing 'Supervisor':", e)
        df_filtered['supervisor'] = np.nan
    
    try:
        if 'Graduation Date:' in df_filtered.columns:
            # Process graduation_date: attempt conversion to numeric using nullable integer type
            df_filtered['graduation_date'] = pd.to_numeric(df_filtered['Graduation Date:'], errors='coerce').astype('Int64')
        else:
            df_filtered['graduation_date'] = pd.NA
            print("Warning: 'Graduation Date:' column not found.")
    except Exception as e:
        print("Error converting 'Graduation Date:'", e)
        df_filtered['graduation_date'] = pd.NA
    
    print("After Step 4 - Column transformation, sample data:\n", 
          df_filtered[['rotation_date', 'group_training_assignment', 'supervisor', 'graduation_date']].head(5))
    
    # === Step 5: Drop rows that are markers only and not rotation records, then select and rename columns ===
    # We expect a valid rotation record to have an employee name; drop rows where employee_name is still missing.
    try:
        df_records = df_filtered[df_filtered['employee_name'].notnull()].copy()
    except Exception as e:
        print("Error filtering records by employee name:", e)
        return pd.DataFrame()
    print("After dropping marker-only rows (no employee name), shape:", df_records.shape)
    
    try:
        result = df_records[['business_unit_section', 'employee_name', 'rotation_date', 
                               'group_training_assignment', 'supervisor', 'graduation_date']].copy()
    except Exception as e:
        print("Error selecting final columns:", e)
        return pd.DataFrame()
    
    # Reset index in final result for neatness
    result.reset_index(drop=True, inplace=True)
    
    print("After Step 5 - Final tidy DataFrame shape:", result.shape)
    print("Final sample output:\n", result.head(10))
    
    return result

# For testing purposes:
if __name__ == "__main__":
    # Create a sample DataFrame similar to the source data
    sample_data = {
        'Unnamed: 0': ['Power', np.nan, np.nan, np.nan, np.nan, np.nan, 'Gas', np.nan],
        'Name ': [np.nan, 'Steven Gim', np.nan, np.nan, np.nan, 'Juan Padron', 'Alice Doe', np.nan],
        'Rotation Date': [np.nan, 36746, 36923, 37012, np.nan, 36724, 36800, 36900],
        'Group': [np.nan, 'Transaction Devl (EIM)', 'Book Running', 'Fundamentals', np.nan,
                  'Fundamentals (power)', 'Advanced', 'Basics'],
        'Supervisor': [np.nan, 'Amanda Colpean', 'Stacey White', 'Lloyd Will', np.nan,
                       'Doug Gilbert-Smith', 'Mark Twain', 'Susan Smith'],
        'Graduation Date:': [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 37000, np.nan]
    }
    test_df = pd.DataFrame(sample_data)
    transformed_df = transform(test_df)
    print("Transformed DataFrame:\n", transformed_df)