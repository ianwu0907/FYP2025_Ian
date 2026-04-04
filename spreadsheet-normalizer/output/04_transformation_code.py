import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    result = df.iloc[6:47].copy()
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {result.shape}")

    # Step 3: Clean up and forward fill ethnicity and marital status
    result.iloc[:, 1] = result.iloc[:, 1].replace(r'^\s*$', np.nan, regex=True).ffill()
    
    marital_statuses = [str(result.iloc[2, j]).strip() if pd.notna(result.iloc[2, j]) else "" for j in range(3, 14, 3)]
    sexes = [str(result.iloc[4, j]).strip() if pd.notna(result.iloc[4, j]) else "" for j in range(3, 14, 3)]

    records = []
    for i in range(len(result)):
        if i < 2 or "Total" in str(result.iloc[i, 1]):
            continue
        
        year = 2011
        ethnicity = str(result.iloc[i, 1]).strip()

        for j in range(3, 14, 3):
            marital_status = marital_statuses[(j - 3) // 3]
            for k in range(3):
                sex = sexes[k]
                percentage = pd.to_numeric(result.iloc[i, j + k], errors='coerce')
                if pd.notna(percentage):
                    records.append({
                        "year": year,
                        "ethnicity": ethnicity,
                        "marital_status": marital_status,
                        "sex": sex,
                        "percentage": percentage
                    })

    output = pd.DataFrame(records, columns=['year', 'ethnicity', 'marital_status', 'sex', 'percentage'])
    print(f"Final output: {output.shape}")
    return output