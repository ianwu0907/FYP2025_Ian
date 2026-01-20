import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Step 0: Starting transformation")
    print("Input shape:", df.shape)
    
    # === Step 1: Extract the data region and remove extraneous header/footer rows ===
    # Select rows 0 to 66, replace empty strings with NaN, drop rows completely empty, and reset index
    try:
        working_df = df.iloc[0:67].copy()
    except Exception as e:
        print("Error during row slicing:", e)
        return pd.DataFrame()
    
    # Replace empty strings (or strings with only whitespace) with NaN
    working_df = working_df.replace(r'^\s*$', np.nan, regex=True)
    working_df.dropna(how='all', inplace=True)
    # Reset index to ensure alignment for later operations
    working_df.reset_index(drop=True, inplace=True)
    # Strip whitespace from column names to avoid mismatches (e.g., "Name " -> "Name")
    working_df.columns = working_df.columns.str.strip()
    print("After Step 1 - working_df shape:", working_df.shape)
    print("Sample data after Step 1:\n", working_df.head(10))
    
    # === Step 2: Propagate department/section markers to subsequent employee records ===
    # Known department markers
    known_departments = {'Power', 'Gas'}
    working_df['department_section'] = np.nan
    current_department = None
    for idx, row in working_df.iterrows():
        # The department marker is expected to be in the first column ("Unnamed: 0")
        dept_val = row.get('Unnamed: 0')
        if pd.notna(dept_val):
            dept_clean = str(dept_val).strip()
            if dept_clean in known_departments:
                current_department = dept_clean
        working_df.at[idx, 'department_section'] = current_department
    print("After Step 2 - Added department_section column.")
    print("Sample data after Step 2:\n", working_df[['Unnamed: 0', 'department_section']].head(10))
    
    # === Step 3: Identify and propagate employee names across multiple rotation rows ===
    # Use the employee name from column 'Name' (stripped header) and forward-fill missing names
    if 'Name' not in working_df.columns:
        print("Error: Expected column 'Name' not found.")
        return pd.DataFrame()
    working_df['employee_name'] = working_df['Name'].replace(r'^\s*$', np.nan, regex=True)
    working_df['employee_name'] = working_df['employee_name'].ffill().str.strip()
    print("After Step 3 - Propagated employee names.")
    print("Sample data after Step 3:\n", working_df[['Name', 'employee_name']].head(10))
    
    # === Step 4: Extract and reshape rotation assignment records ===
    # Only rows with a non-null 'Rotation Date' are considered valid rotation records.
    if 'Rotation Date' not in working_df.columns:
        print("Error: Expected column 'Rotation Date' not found.")
        return pd.DataFrame()
    working_df['Rotation Date'] = working_df['Rotation Date'].replace(r'^\s*$', np.nan, regex=True)
    valid_df = working_df.loc[working_df['Rotation Date'].notna()].copy()
    valid_df.reset_index(drop=True, inplace=True)
    print("After filtering rows with valid rotation dates, shape:", valid_df.shape)
    
    # Define a safe conversion function to integer
    def safe_int(val):
        try:
            return int(float(val))
        except Exception:
            return np.nan

    # Map and convert columns as needed:
    # rotation_date from 'Rotation Date'
    valid_df['rotation_date'] = valid_df['Rotation Date'].apply(safe_int)
    
    # rotation_group from column 'Group'
    if 'Group' in valid_df.columns:
        valid_df['rotation_group'] = valid_df['Group'].astype(str).str.strip()
    else:
        valid_df['rotation_group'] = np.nan
        
    # supervisor from column 'Supervisor'
    if 'Supervisor' in valid_df.columns:
        valid_df['supervisor'] = valid_df['Supervisor'].astype(str).str.strip()
        valid_df['supervisor'].replace('', np.nan, inplace=True)
    else:
        valid_df['supervisor'] = np.nan
        
    # graduation_date from column 'Graduation Date:'
    if 'Graduation Date:' in valid_df.columns:
        valid_df['graduation_date'] = valid_df['Graduation Date:'].replace(r'^\s*$', np.nan, regex=True)
        valid_df['graduation_date'] = valid_df['graduation_date'].apply(lambda x: safe_int(x) if pd.notna(x) else np.nan)
    else:
        valid_df['graduation_date'] = np.nan

    print("After Step 4 - Rotation assignment records extracted:")
    print("Sample of rotation records:\n", valid_df[['department_section', 'employee_name', 'rotation_date', 
                                                      'rotation_group', 'supervisor', 'graduation_date']].head(10))
    
    # === Step 5: Clean data types and trim strings ===
    # Trim extra whitespace from text columns; numeric fields are already processed.
    valid_df['department_section'] = valid_df['department_section'].astype(str).str.strip()
    valid_df['employee_name'] = valid_df['employee_name'].astype(str).str.strip()
    valid_df['rotation_group'] = valid_df['rotation_group'].astype(str).str.strip()
    valid_df['supervisor'] = valid_df['supervisor'].astype(str).str.strip()
    
    # === Step 6: Final column selection and ordering ===
    final_cols = ['department_section', 'employee_name', 'rotation_date', 'rotation_group', 'supervisor', 'graduation_date']
    result = valid_df[final_cols].copy()
    print("After Step 6 - Final transformed DataFrame shape:", result.shape)
    print("Sample of final transformed data:\n", result.head(10))
    
    # Extra validation: check if final record count is 50; if not, print a warning
    final_count = result.shape[0]
    if final_count != 50:
        print(f"Warning: Expected 50 rotation records, but got {final_count}.")
    else:
        print("Final record count validation passed: 50 records.")
    
    return result