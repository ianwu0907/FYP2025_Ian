def transform(df):
    import pandas as pd
    import numpy as np

    # Step 1: Forward-fill the year and ethnicity headers
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    df.iloc[:, 1] = df.iloc[:, 1].replace(r'^\s*$', np.nan, regex=True).ffill()

    # Step 2: Create multi-level headers from the given rows
    headers = df.iloc[2:4].ffill(axis=1)

    # Step 3: Delete rows that contain Total or Average rows based on ethnicity
    exclude_ethnicities = ["合計", "總計", "全部", "全部少數族裔人士", "撇除外籍家庭傭工後的所有少數族裔人士"]
    mask = ~df.iloc[:, 1].isin(exclude_ethnicities)
    df = df[mask].reset_index(drop=True)

    # Step 4: Drop aggregate columns
    agg_col_indices = [5, 8, 11, 14]
    df = df.drop(columns=[df.columns[i] for i in agg_col_indices if i < len(df.columns)], errors="ignore")

    # Step 5: Reshape the DataFrame to long format using lists
    records = []
    for i in range(len(df)):
        year = df.iloc[i, 0]
        ethnicity = df.iloc[i, 1]
        for j in range(3):  # Marital status index
            marital_status = headers.iloc[1, 3 * j + 0]  # Always starting with Male index
            for k in range(3):  # Sex index (0: Male, 1: Female, 2: Overall)
                sex = headers.iloc[1, 3 * j + k]
                percentage = df.iloc[i, 3 + 3 * j + k]  # Start from column index 3
                records.append({'year': year, 'ethnicity': ethnicity, 'marital_status': marital_status, 'sex': sex, 'percentage': percentage})

    # Step 6: Create DataFrame from the records list
    long_df = pd.DataFrame(records)

    # Step 7: Clean and convert percentage to numeric
    long_df['percentage'] = pd.to_numeric(long_df['percentage'], errors='coerce')
    long_df.dropna(subset=['percentage'], inplace=True)

    return long_df[['year', 'ethnicity', 'marital_status', 'sex', 'percentage']]