import pandas as pd
import numpy as np

def transform(df):
    """
    Transform messy spreadsheet to tidy format.
    """
    print("Initial DataFrame shape:", df.shape)
    
    # === Step 1: Extract the data region containing trading observations ===
    # Data exists in rows 1 to 12 (inclusive) -> use .iloc[1:13]
    try:
        df_data = df.iloc[1:13].copy()
    except Exception as e:
        print("Error extracting data rows:", e)
        return pd.DataFrame()
    print("After Step 1, extracted DataFrame shape:", df_data.shape)
    print("Data sample after Step 1:")
    print(df_data.head())
    
    # === Step 2: Rename and subset columns according to the target schema ===
    # We assume that the original data region has 7 columns. We ignore the first column (index 0)
    # and extract columns with positions 1 to 6, then rename them.
    try:
        df_subset = df_data.iloc[:, 1:7].copy()
    except Exception as e:
        print("Error subsetting columns:", e)
        return pd.DataFrame()
    
    # Rename columns to match target schema
    new_columns = ['month', 'strike', 'bid', 'offer', 'mid_pt_vol', 'net_change']
    df_subset.columns = new_columns
    print("After Step 2, DataFrame shape:", df_subset.shape)
    print("Data sample after Step 2:")
    print(df_subset.head())
    
    # === Step 3: Convert data types for proper numeric operations ===
    numeric_columns = ['strike', 'bid', 'offer', 'mid_pt_vol', 'net_change']
    for col in numeric_columns:
        try:
            # Remove any extraneous spaces and ensure conversion to numeric
            df_subset[col] = pd.to_numeric(df_subset[col].astype(str).str.strip(), errors='coerce')
        except Exception as e:
            print(f"Error converting column {col} to numeric:", e)
    print("After Step 3, DataFrame dtypes:")
    print(df_subset.dtypes)
    print("Data sample after Step 3:")
    print(df_subset.head())
    
    # === Step 4: Reshape data if necessary and perform final cleaning ===
    # Drop any rows with missing values in the required columns.
    required_columns = ['month', 'strike', 'bid', 'offer', 'mid_pt_vol', 'net_change']
    df_clean = df_subset.dropna(subset=required_columns).copy()
    print("After Step 4, final DataFrame shape (after dropping missing values):", df_clean.shape)
    print("Final data sample after Step 4:")
    print(df_clean.head())
    
    # === Step 5: Spot-check specific observation for semantic validation ===
    # Spot-check for observation with month 'M' and strike 4.75.
    # We expect the observation to have specific numeric values.
    try:
        # Create boolean conditions directly using pandas Series operations.
        condition = (df_clean['month'] == 'M') & (np.isclose(df_clean['strike'], 4.75))
        spot_check = df_clean[condition]
        if not spot_check.empty:
            print("Spot-check sample found:")
            print(spot_check)
            # Additionally, check if the found record matches expected numeric values.
            # Here we compare each value with a tolerance where needed.
            expected = {
                "month": "M",
                "strike": 4.75,
                "bid": 0.45,
                "offer": 0.5,
                "mid_pt_vol": 0.4601000000000031,
                "net_change": 0.009299999999998976
            }
            row = spot_check.iloc[0]
            errors = []
            if row['month'] != expected['month']:
                errors.append(f"month {row['month']} != {expected['month']}")
            for key in ['strike', 'bid', 'offer', 'mid_pt_vol', 'net_change']:
                if not np.isclose(row[key], expected[key]):
                    errors.append(f"{key} {row[key]} != {expected[key]}")
            if errors:
                print("Spot-check values do not match expected values:", errors)
            else:
                print("Spot-check passed: values match expected data.")
        else:
            print("Spot-check sample for month 'M' and strike 4.75 not found.")
    except Exception as e:
        print("Error during spot-check validation:", e)
    
    return df_clean