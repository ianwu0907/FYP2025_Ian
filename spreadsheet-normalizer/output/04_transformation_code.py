import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region
    result = df.iloc[6:47].copy()  # 6 = data start row, 47 = data end row + 1
    result = result.reset_index(drop=True)
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Remove blank rows
    result = result.dropna(how="all")
    print(f"After removing blank rows: {result.shape}")

    # Step 3: Forward-fill 'Year' column to handle sparse rows
    result.iloc[:, 0] = result.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    print("Forward-filled the 'Year' column.")

    marital_statuses = ["從未結婚\nNever married", "已婚\nMarried", "喪偶／離婚／分居\nWidowed/divorced/\nseparated", "總計\nTotal"]
    sexes = ["男\nMale", "女\nFemale", "合計\nOverall"]

    records = []
    for i in range(len(result)):
        year = int(result.iloc[i, 0])  # Year is always in the first column
        ethnicity = str(result.iloc[i, 1]).strip()  # Ethnicity in the second column

        for j, marital_status in enumerate(marital_statuses):
            for k, sex in enumerate(sexes):
                value_index = 3 + j * 3 + k  # Calculate correct column index
                value = result.iloc[i, value_index]
                
                # Handle invalid values
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.notna(numeric_value):  # Only add if value is present
                    records.append({
                        "year": year,
                        "ethnicity": ethnicity,
                        "marital_status": marital_status,
                        "sex": sex,
                        "percentage": float(numeric_value)
                    })

    target_cols = ['year', 'ethnicity', 'marital_status', 'sex', 'percentage']
    output = pd.DataFrame(records, columns=target_cols)
    print(f"Final output: {output.shape}")
    return output