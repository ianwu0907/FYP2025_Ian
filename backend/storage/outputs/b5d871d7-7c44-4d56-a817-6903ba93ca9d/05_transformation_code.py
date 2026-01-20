import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation")
    print(f"Input DataFrame shape: {df.shape}")
    
    # === Preliminary: Rename columns by stripping whitespace ===
    # This ensures that we avoid issues due to trailing/leading spaces in column names.
    df = df.copy()
    df.columns = df.columns.str.strip()
    print("After stripping whitespace from column names:")
    print(df.columns.tolist())
    
    # === Step 1: Extract the main data region and remove irrelevant rows ===
    # Select rows 0 to 66 (inclusive) and create a copy
    df_subset = df.iloc[:67].copy()
    print(f"After selecting rows 0-66, shape: {df_subset.shape}")
    
    # Replace cells that are strings containing only whitespace with np.nan
    df_subset = df_subset.applymap(lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x)
    
    # Drop rows where all cells are NaN (across all columns)
    df_filtered = df_subset.dropna(how='all', subset=df_subset.columns.tolist())
    df_filtered.reset_index(drop=True, inplace=True)
    print(f"After dropping fully blank rows and resetting index, shape: {df_filtered.shape}")
    print("Sample rows after filtering:")
    print(df_filtered.head(5))
    
    # === Step 2: Identify department sections and forward-fill them ===
    # Department markers are expected in the first column (by position)
    # Use the first column for department markers and strip any extra whitespace
    if df_filtered.shape[1] < 1:
        raise KeyError("Expected at least 1 column for department markers")
    df_filtered['department_section'] = df_filtered.iloc[:,0].apply(lambda x: x.strip() if isinstance(x, str) else np.nan)
    df_filtered['department_section'] = df_filtered['department_section'].ffill()
    print("After forward-filling department sections (sample):")
    print(df_filtered[['department_section']].head(10))
    
    # === Step 3: Assign employee names and group rotation events ===
    # Employee names are expected in the second column (by position)
    if df_filtered.shape[1] < 2:
        raise KeyError("Expected at least 2 columns for employee names")
    df_filtered['employee_name'] = df_filtered.iloc[:,1].apply(lambda x: x.strip() if isinstance(x, str) else np.nan)
    df_filtered['employee_name'] = df_filtered['employee_name'].ffill()
    print("After assigning and forward-filling employee names (sample):")
    print(df_filtered[['employee_name']].head(10))
    
    # === Step 4: Reshape and map the rotation event fields ===
    # Expecting the following column positions:
    # Column 3 (index 2): rotation_id (Rotation Date)
    # Column 4 (index 3): group
    # Column 5 (index 4): supervisor
    # Column 6 (index 5): graduation_date
    if df_filtered.shape[1] < 6:
        raise KeyError("Expected at least 6 columns for rotation event fields")
    
    # Convert rotation_id from column 3 using pd.to_numeric with coercion
    df_filtered['rotation_id'] = pd.to_numeric(df_filtered.iloc[:,2].apply(
        lambda x: str(x).strip() if isinstance(x, str) else x), errors='coerce')
    
    # Map group from column 4
    df_filtered['group'] = df_filtered.iloc[:,3].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Map supervisor from column 5
    df_filtered['supervisor'] = df_filtered.iloc[:,4].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Map graduation_date from column 6; if blank or NaN, set to None
    df_filtered['graduation_date'] = df_filtered.iloc[:,5].apply(
        lambda x: x.strip() if isinstance(x, str) and x.strip() != '' else None)
    
    print("After mapping rotation event fields (sample):")
    print(df_filtered[['rotation_id', 'group', 'supervisor', 'graduation_date']].head(10))
    
    # === Step 5: Clean, type convert and finalize the dataset ===
    # Trim whitespace for department_section and employee_name
    df_filtered['department_section'] = df_filtered['department_section'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    df_filtered['employee_name'] = df_filtered['employee_name'].apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # Filter out rows where rotation_id is NaN (i.e. rows that are not valid rotation events)
    before_drop = df_filtered.shape[0]
    # Ensure that the boolean mask is aligned with df_filtered by using .loc
    mask = df_filtered['rotation_id'].notna()
    df_final = df_filtered.loc[mask].copy()
    after_drop = df_final.shape[0]
    print(f"Dropped {before_drop - after_drop} rows due to invalid rotation_id")
    
    # Select only the target columns in the specified order
    final_columns = ['department_section', 'employee_name', 'rotation_id', 'group', 'supervisor', 'graduation_date']
    result = df_final[final_columns].copy()
    
    # Final debug prints
    print("Final transformed data sample:")
    print(result.head(10))
    print(f"Final DataFrame shape: {result.shape}")
    
    return result