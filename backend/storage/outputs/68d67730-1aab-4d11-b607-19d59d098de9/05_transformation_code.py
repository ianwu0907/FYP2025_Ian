import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial input shape:", df.shape)
    
    # === Step 1: Extract the primary data region ===
    try:
        # Slice rows 0 to 66 (inclusive of header row) and drop rows that are completely empty
        df = df.iloc[0:67].dropna(how='all')
        # Reset the index so that further operations align properly
        df = df.reset_index(drop=True)
        print("After Step 1 (extract primary data region), shape:", df.shape)
        print("Sample data after Step 1:")
        print(df.head(5))
    except Exception as e:
        print("Error in Step 1:", e)
        return df
    
    # === Step 2: Rename columns and propagate department markers ===
    try:
        # Rename columns for easier referencing based on the provided mapping.
        # If a mapping key is not present, the column will be left unchanged.
        col_mapping = {
            'Unnamed: 0': 'col0',
            'Name ': 'col1',
            'Rotation Date': 'col2',
            'Group': 'col3',
            'Supervisor': 'col4',
            'Graduation Date:': 'col5'
        }
        df.rename(columns=col_mapping, inplace=True)
        
        # Ensure that 'col0' exists; if not, create an empty column to avoid key errors.
        if 'col0' not in df.columns:
            df['col0'] = ""
        # Convert col0 to string and trim whitespace.
        df['col0'] = df['col0'].astype(str).str.strip()

        # Known department markers
        known_departments = ['Power', 'Gas']
        # Create a new 'department' column: if col0 is one of the markers, use it; else leave as NaN.
        df['department'] = df['col0'].where(df['col0'].isin(known_departments))
        # Forward-fill department markers for subsequent rows.
        df['department'] = df['department'].ffill()

        # Identify marker-only rows. These are rows where the employee name (col1) is missing or empty.
        if 'col1' not in df.columns:
            df['col1'] = ""
        df['is_marker'] = df['col1'].apply(lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == ''))
        
        print("After Step 2 (rename columns and propagate department markers), shape:", df.shape)
        print("Sample data after Step 2:")
        print(df[['col0', 'col1', 'department', 'is_marker']].head(10))
    except Exception as e:
        print("Error in Step 2:", e)
        return df

    # === Step 3: Fill down missing employee names ===
    try:
        # Use forward fill on col1 to create a new 'name' column.
        df['name'] = df['col1'].ffill()
        print("After Step 3 (fill down missing employee names), shape:", df.shape)
        print("Sample data after Step 3:")
        print(df[['col1', 'name']].head(10))
    except Exception as e:
        print("Error in Step 3:", e)
        return df

    # === Step 4: Map and convert rotation details ===
    try:
        # Convert rotation_date from col2 to numeric (integer) with errors set to NaN.
        df['rotation_date'] = pd.to_numeric(df['col2'], errors='coerce')
        # Process group (col3) by stripping any extra whitespace.
        df['group'] = df['col3'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        # Process supervisor (col4) by stripping any extra whitespace.
        df['supervisor'] = df['col4'].apply(lambda x: x.strip() if isinstance(x, str) else x)
        # Process graduation_date from col5: convert to numeric (errors become NaN)
        df['graduation_date'] = pd.to_numeric(df['col5'], errors='coerce')
        print("After Step 4 (map and convert rotation details), shape:", df.shape)
        print("Sample data after Step 4:")
        print(df[['rotation_date', 'group', 'supervisor', 'graduation_date']].head(10))
    except Exception as e:
        print("Error in Step 4:", e)
        return df

    # === Step 5: Assign record type and filter out non-data rows ===
    try:
        # Create a combined mask to filter out marker-only rows and rows missing essential rotation record values.
        valid_mask = (~df['is_marker']) & (df['name'].notna()) & (df['rotation_date'].notna())
        # Use .loc with the valid_mask to ensure index alignment, then make a copy.
        df = df.loc[valid_mask].copy()
        # Add new column 'record_type' with constant value 'rotation'.
        df['record_type'] = 'rotation'
        print("After Step 5 (filter invalid markers and assign record_type), shape:", df.shape)
        print("Sample data after Step 5:")
        print(df[['department', 'name', 'rotation_date', 'record_type']].head(10))
    except Exception as e:
        print("Error in Step 5:", e)
        return df

    # === Step 6: Select final columns and ordering ===
    try:
        # Select only the required columns and set the order as specified.
        result = df[['department', 'name', 'rotation_date', 'group', 'supervisor', 'graduation_date', 'record_type']]
        print("After Step 6 (select final columns), final shape:", result.shape)
        print("Final sample data:")
        print(result.head(10))
    except Exception as e:
        print("Error in Step 6:", e)
        return df

    return result