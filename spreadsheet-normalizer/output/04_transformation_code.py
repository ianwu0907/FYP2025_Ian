import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    result = df.iloc[10:50].copy()  # Data starts from row 10 to row 49
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {result.shape}")

    # Step 3: Forward-fill the headers
    headers = df.iloc[3:5].ffill(axis=1)
    economy_types = [str(headers.iloc[1, j]).strip() for j in range(2, 8)]
    print(f"Extracted economy types: {economy_types}")

    # Step 4: Unpivot wide columns into long format using record loop
    records = []
    for i in range(len(result)):
        year = int(result.iloc[i, 0])  # Year from column 0
        for j in range(1, 7):  # Investment amounts from columns 1 to 6
            investment_amount = pd.to_numeric(result.iloc[i, j], errors='coerce')
            economy_type = economy_types[j - 1]  # Adjust index for economy types
            if investment_amount is not np.nan:
                records.append({"year": year, "economy_type": economy_type, "investment_amount": investment_amount})

    result_df = pd.DataFrame(records)
    print(f"Final output: {result_df.shape}")
    return result_df