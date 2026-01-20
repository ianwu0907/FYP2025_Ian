import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Starting transformation...")
    print(f"Initial input shape: {df.shape}")
    
    # === Step 1: Extract the raw data region and remove extraneous rows ===
    # Select rows 0 to 66 (i.e., first 67 rows) and remove rows that are entirely blank (NaN or empty strings)
    raw_data = df.iloc[0:67].copy()
    # Replace empty strings with np.nan to treat them as missing
    raw_data = raw_data.replace(r'^\s*$', np.nan, regex=True)
    # Drop rows where all values are NaN
    raw_data = raw_data.dropna(how='all')
    print("After Step 1 (raw data extraction):")
    print(f"Shape: {raw_data.shape}")
    print(raw_data.head(10))
    
    # === Step 2: Identify and handle section marker rows for departments ===
    # Known department markers (for example, 'Power', 'Gas'); add more markers if needed.
    known_markers = set(['Power', 'Gas'])
    current_department = None
    # We'll build a new DataFrame via list of rows, assigning a department based on preceding marker rows.
    cleaned_rows = []
    
    # First, iterate through raw_data sequentially, handling department marker rows.
    for index, row in raw_data.iterrows():
        col0_val = row['Unnamed: 0']
        # Check if col0_val is a string and if it is one of the known department markers
        if pd.notnull(col0_val) and str(col0_val).strip() in known_markers:
            current_department = str(col0_val).strip()
            # Debug: print encountered section marker and update variable; don't include this row
            print(f"Found department marker at row {index}: {current_department}")
            continue
        # For non-marker rows, we add a temporary column for department (to be used later)
        # We still keep the row, even if current_department is None (edge cases)
        row_copy = row.copy()
        row_copy['department'] = current_department
        cleaned_rows.append(row_copy)
    
    if not cleaned_rows:
        print("Warning: No valid rows left after removing section markers and blank rows.")
        return pd.DataFrame(columns=['department', 'employee_name', 'rotation_date', 'group', 'supervisor', 'graduation_date'])
    
    working_df = pd.DataFrame(cleaned_rows)
    print("After Step 2 (section marker processing):")
    print(f"Shape: {working_df.shape}")
    print(working_df.head(10))
    
    # === Step 3: Forward-fill the employee names to ensure each row is complete ===
    # The employee names are in column 'Name ' (note trailing space). We forward fill missing names.
    # First, trim whitespace from employee names to standardize the missing checks
    working_df['Name '] = working_df['Name '].apply(lambda x: x.strip() if isinstance(x, str) else x)
    working_df['employee_name'] = working_df['Name '].fillna(method='ffill')
    print("After Step 3 (forward-fill employee names):")
    print(working_df[['employee_name']].head(10))
    
    # === Step 4: Assign and reshape the remaining fields into structured rotation records ===
    # Map columns: rotation_date from 'Rotation Date'; group from 'Group'; supervisor from 'Supervisor'; graduation_date from 'Graduation Date:'
    # Define a helper function to convert numeric date values to our placeholder format.
    def convert_date(val):
        try:
            if pd.isna(val):
                return None
            # If it is numeric (or string that can be converted to int), do conversion. 
            # We check if it is a number using isinstance() or attempt conversion.
            if isinstance(val, (int, float, np.number)):
                return f"ConvertedDate({int(val)})"
            # Else if it's a string representing a number:
            val_str = str(val).strip()
            if val_str.isdigit():
                return f"ConvertedDate({int(val_str)})"
        except Exception as e:
            print(f"Error converting date value {val}: {e}")
        return None

    # Process each row and generate final structured records.
    records = []
    for idx, row in working_df.iterrows():
        # Prepare the record with available columns:
        record = {}
        record['department'] = row['department'] if pd.notnull(row['department']) else None
        record['employee_name'] = row['employee_name'] if pd.notnull(row['employee_name']) else None
        
        # Convert and assign rotation_date
        record['rotation_date'] = convert_date(row['Rotation Date'])
        
        # Group: trim whitespace, if string is provided.
        group_val = row['Group']
        record['group'] = group_val.strip() if isinstance(group_val, str) else group_val
        
        # Supervisor: trim whitespace if string.
        supervisor_val = row['Supervisor']
        record['supervisor'] = supervisor_val.strip() if isinstance(supervisor_val, str) else supervisor_val
        
        # Graduation date: convert numeric value if available.
        record['graduation_date'] = convert_date(row['Graduation Date:'])
        
        records.append(record)
    
    result = pd.DataFrame(records)
    print("After Step 4 (reshape into rotation records):")
    print(f"Shape: {result.shape}")
    print(result.head(10))
    
    # === Step 5: Clean data and perform type conversion on fields ===
    # Trim whitespace from all text fields where applicable (department, employee_name, group, supervisor)
    for col in ['department', 'employee_name', 'group', 'supervisor']:
        result[col] = result[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
    # (Dates are already converted using the helper function, and missing numeric dates are set to None)
    print("After Step 5 (data cleaning and type conversion):")
    print(result.head(10))
    
    # === Step 6: Select final columns and validate dataset ===
    final_columns = ['department', 'employee_name', 'rotation_date', 'group', 'supervisor', 'graduation_date']
    result = result[final_columns]
    print("After Step 6 (select final columns):")
    print(f"Final shape: {result.shape}")
    print(result.head(10))
    
    # Validate row count. Expected row count is 50.
    expected_row_count = 50
    actual_row_count = result.shape[0]
    if actual_row_count != expected_row_count:
        print(f"Validation Warning: Expected {expected_row_count} rows but found {actual_row_count} rows.")
    else:
        print("Row count validated: 50 rows.")
    
    # Spot-check validation for a sample record (e.g., "Steven Gim")
    sample_record = result[result['employee_name'] == "Steven Gim"]
    if not sample_record.empty:
        print("Spot-check for 'Steven Gim':")
        print(sample_record.head(1))
    else:
        print("Spot-check for 'Steven Gim' not found.")
    
    print("Transformation complete.")
    return result

# Example usage:
# if __name__ == "__main__":
#     # Read your input data into a DataFrame (e.g., via pd.read_excel or pd.read_csv)
#     df_input = pd.read_csv("input_file.csv")
#     transformed_df = transform(df_input)
#     transformed_df.to_csv("transformed_output.csv", index=False)