import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print("Input shape:", df.shape)
    
    try:
        # === Step 0: Clean column names (remove extraneous whitespace) ===
        df = df.copy()
        df.columns = [col.strip() for col in df.columns]
        print("Step 0: Cleaned column names. Columns are:", df.columns.tolist())
    except Exception as e:
        print("Error in Step 0:", e)
        return None

    # === Step 1: Extract the primary data region and remove extraneous header/footer rows ===
    try:
        # Select rows 0 to 66 (inclusive)
        df_region = df.iloc[0:67].copy()
        print("Step 1: Selected rows 0 to 66. Shape:", df_region.shape)
        
        # Only keep rows that have non-empty values in either the department marker column ('Unnamed: 0') 
        # or employee name column ('Name'). We rely on cleaned column names.
        cond_dept = df_region['Unnamed: 0'].notna() & (df_region['Unnamed: 0'].astype(str).str.strip() != "")
        cond_name = df_region['Name'].notna() & (df_region['Name'].astype(str).str.strip() != "")
        # Combining conditions to keep rows where at least one is True
        df_region = df_region[ cond_dept | cond_name ]
        
        # Reset index for proper boolean indexing later
        df_region = df_region.reset_index(drop=True)
        print("Step 1: After filtering blank rows and resetting index. Shape:", df_region.shape)
        print("Step 1 sample data:")
        print(df_region.head(10))
    except Exception as e:
        print("Error in Step 1:", e)
        return None

    # === Step 2: Assign the department context to each employee record ===
    try:
        # Forward-fill department markers in the 'Unnamed: 0' column
        # This will assign each row a new column 'department' based on the preceding marker.
        df_region['department'] = df_region['Unnamed: 0'].ffill()
        print("Step 2: After forward filling department markers. Shape:", df_region.shape)
        print("Step 2 sample data:")
        print(df_region[['Unnamed: 0', 'department']].head(10))
    except Exception as e:
        print("Error in Step 2:", e)
        return None

    # === Step 3: Filter for employee records only ===
    try:
        # Employee records are rows which have a non-empty value in 'Name'
        df_employees = df_region[ df_region['Name'].notna() & (df_region['Name'].astype(str).str.strip() != "") ].copy()
        
        # Reset index for clarity
        df_employees = df_employees.reset_index(drop=True)
        
        print("Step 3: After filtering for employee records only. Shape:", df_employees.shape)
        print("Step 3 sample data:")
        print(df_employees[['Name', 'department']].head(10))
    except Exception as e:
        print("Error in Step 3:", e)
        return None

    # === Step 4: Map and reshape the data to the target schema ===
    try:
        # Convert "Rotation Date" and "Graduation Date:" to numeric types if possible
        # For graduation_date, empty values will be converted to <NA>
        df_employees['rotation_date'] = pd.to_numeric(df_employees['Rotation Date'], errors='coerce').astype('Int64')
        df_employees['graduation_date'] = pd.to_numeric(df_employees['Graduation Date:'], errors='coerce').astype('Int64')
        
        # Create a new DataFrame with the target column mapping. Strip string values.
        result = pd.DataFrame({
            'department': df_employees['department'].astype(str).str.strip(),
            'employee_name': df_employees['Name'].astype(str).str.strip(),
            'rotation_date': df_employees['rotation_date'],
            'group_name': df_employees['Group'].astype(str).str.strip(),
            'supervisor': df_employees['Supervisor'].astype(str).str.strip(),
            'graduation_date': df_employees['graduation_date']
        })
        
        print("Step 4: After mapping columns to target schema. Shape:", result.shape)
        print("Step 4 sample data:")
        print(result.head(10))
    except Exception as e:
        print("Error in Step 4:", e)
        return None

    # === Step 5: Validation and final cleanup ===
    try:
        # Validate row count: expected 27 employee records.
        emp_count = result.shape[0]
        print("Step 5: Validation - total employee records:", emp_count)
        if emp_count != 27:
            print("Warning: Expected 27 employee records, but found", emp_count)
            
        # Spot-check for specific employee record: 'Steven Gim'
        spot_check = result[result['employee_name'] == 'Steven Gim']
        if not spot_check.empty:
            print("Step 5: Spot-check for 'Steven Gim':")
            print(spot_check)
        else:
            print("Step 5: 'Steven Gim' record not found in the data.")
            
        print("Step 5: Final DataFrame info:")
        print(result.info())
    except Exception as e:
        print("Error in Step 5:", e)
        return None

    print("Transformation completed successfully. Final shape:", result.shape)
    return result