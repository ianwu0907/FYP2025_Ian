import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the data region and remove irrelevant header/footer rows ===
    # Restrict rows 0 to 66 (inclusive) and remove fully blank rows
    df = df.iloc[:67].copy()
    print("After slicing rows 0-66, shape:", df.shape)
    
    # Remove rows that are completely blank (all cells are NaN or empty after strip)
    # Replace empty strings with np.nan for proper dropna handling
    df = df.replace(r'^\s*$', np.nan, regex=True)
    df.dropna(how='all', inplace=True)
    df.reset_index(drop=True, inplace=True)
    print("After dropping fully blank rows, shape:", df.shape)
    print("Data sample after Step 1:\n", df.head(10))
    
    # Known section markers list (adjust as needed)
    section_markers = ['Power', 'Gas', '** Offer outstanding', 'New Hires', 'Graduates:']
    
    # === Step 2: Identify and assign section markers ===
    # Create a new column 'section_marker' that uses the value in the first column if it matches known markers
    # Otherwise, it remains NaN. Then forward-fill it.
    df['section_marker'] = np.where(df['Unnamed: 0'].isin(section_markers), df['Unnamed: 0'], np.nan)
    df['section_marker'] = df['section_marker'].ffill()
    print("After assigning and forward-filling section markers, sample:\n", df[['Unnamed: 0', 'section_marker']].head(10))
    
    # === Step 3: Forward fill missing employee names ===
    # Employee names are in column 'Name ' so we trim spaces and forward fill
    df['employee_name'] = df['Name '].astype(str).str.strip()
    # Replace 'nan' string with np.nan
    df['employee_name'] = df['employee_name'].replace('nan', np.nan)
    df['employee_name'] = df['employee_name'].ffill()
    print("After forward-filling employee names, sample:\n", df[['Name ', 'employee_name']].head(10))
    
    # === Step 4: Reshape the data and map source columns to target fields ===
    # Filter out rows that are solely section markers: these rows will likely have missing rotation details.
    # We check if 'Rotation Date' and 'Group' are not empty (at least one should be non-null)
    # According to strategy, we filter in rows that have data in Rotation Date and Group.
    rotation_mask = df['Rotation Date'].notna() & df['Group'].notna()
    df_rotation = df[rotation_mask].copy()
    df_rotation.reset_index(drop=True, inplace=True)
    print("After filtering out section marker-only rows, shape:", df_rotation.shape)
    print("Rotation records sample:\n", df_rotation.head(10))
    
    # Map source columns to new columns for target schema:
    df_rotation['rotation_date'] = df_rotation['Rotation Date']
    df_rotation['group'] = df_rotation['Group']
    df_rotation['supervisor'] = df_rotation['Supervisor']
    # Graduation date column: if NaN, set as empty string
    df_rotation['graduation_date'] = df_rotation['Graduation Date:'].fillna('')
    
    # === Step 5: Clean data and perform type conversion ===
    # Trim spaces from string fields: employee_name, group, supervisor
    df_rotation['employee_name'] = df_rotation['employee_name'].astype(str).str.strip()
    df_rotation['group'] = df_rotation['group'].astype(str).str.strip()
    df_rotation['supervisor'] = df_rotation['supervisor'].astype(str).str.strip()
    
    # Convert rotation_date to numeric (integer) if possible; flag non-numeric by converting them to NaN
    df_rotation['rotation_date'] = pd.to_numeric(df_rotation['rotation_date'], errors='coerce')
    
    # Check for any rows where rotation_date could not be converted
    invalid_dates = df_rotation[df_rotation['rotation_date'].isna()]
    if not invalid_dates.empty:
        print("Warning: Some rotation_date values are not numeric and have been set to NaN. Check the following rows:")
        print(invalid_dates[['rotation_date']])
    
    # Convert rotation_date to integer type if there are no NaN values; if there are, leave as float for now.
    if df_rotation['rotation_date'].isna().sum() == 0:
        df_rotation['rotation_date'] = df_rotation['rotation_date'].astype(int)
    else:
        # Optionally, fill NaN with a default value or keep as is
        df_rotation['rotation_date'] = df_rotation['rotation_date'].fillna(0).astype(int)
    
    # Graduation_date remains as string; ensure string type and trim spaces
    df_rotation['graduation_date'] = df_rotation['graduation_date'].astype(str).str.strip()
    
    print("After cleaning and type conversion, sample data:\n", df_rotation[['section_marker', 'employee_name', 'rotation_date', 'group', 'supervisor', 'graduation_date']].head(10))
    
    # === Step 6: Final column selection, naming, and ordering ===
    # Select exactly the intended columns in the proper order.
    result = df_rotation[['section_marker', 'employee_name', 'rotation_date', 'group', 'supervisor', 'graduation_date']].copy()
    print("Final output shape:", result.shape)
    print("Final data sample:\n", result.head(10))
    
    return result