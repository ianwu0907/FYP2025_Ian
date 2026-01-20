import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the valid data region ===
    # Skip the header row (row 0) and any footer rows; select rows 1 to 12 and columns 1 to 6.
    try:
        # Using .iloc[1:13, 1:7] to get rows 1 to 12 (inclusive of row 1, exclusive of row 13)
        # and columns 1 through 6 (zero-indexed). Then reset the index to ensure proper alignment.
        extracted = df.iloc[1:13, 1:7].copy().reset_index(drop=True)
        print("After Step 1 - Extracted data shape:", extracted.shape)
        print("Sample data after Step 1:\n", extracted.head())
    except Exception as e:
        print("Error in Step 1:", e)
        return None

    # === Step 2: Assign header names and validate data alignment ===
    # Assign the expected header names to these 6 columns.
    expected_headers = ['Month', 'Strike', 'Bid', 'Offer', 'Mid-Pt. Vol', 'Net Change']
    try:
        if extracted.shape[1] == len(expected_headers):
            extracted.columns = expected_headers
        else:
            raise ValueError("Number of extracted columns does not match expected header count.")
        print("After Step 2 - Columns assigned:", extracted.columns.tolist())
        print("Sample data after Step 2:\n", extracted.head())
    except Exception as e:
        print("Error in Step 2:", e)
        return None

    # === Step 3: Clean and convert data types ===
    # For columns that should be numeric, strip extra spaces and convert values to float.
    numeric_cols = ['Strike', 'Bid', 'Offer', 'Mid-Pt. Vol', 'Net Change']
    try:
        for col in numeric_cols:
            # Convert to string and strip spaces (handles numbers stored as strings).
            extracted[col] = extracted[col].astype(str).str.strip()
            # Convert processed strings to numeric, coercing errors to NaN.
            extracted[col] = pd.to_numeric(extracted[col], errors='coerce')
    except Exception as e:
        print("Error in Step 3 while converting numeric columns:", e)
        return None
    print("After Step 3 - Data types:\n", extracted.dtypes)
    print("Sample data after Step 3:\n", extracted.head())
    
    # === Step 4: Map columns to the target schema and rename where necessary ===
    # Rename columns to match target schema: 'month', 'strike', 'bid', 'offer', 'mid_pt_vol', 'net_change'
    try:
        result = extracted.rename(columns={
            'Month': 'month', 
            'Strike': 'strike', 
            'Bid': 'bid', 
            'Offer': 'offer', 
            'Mid-Pt. Vol': 'mid_pt_vol', 
            'Net Change': 'net_change'
        })
        print("After Step 4 - Final columns:", result.columns.tolist())
        print("Final data sample:\n", result.head())
        print("Final DataFrame shape:", result.shape)
    except Exception as e:
        print("Error in Step 4:", e)
        return None

    return result