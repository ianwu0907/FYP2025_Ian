def transform(df):
    import pandas as pd
    import numpy as np

    # Step 1: Drop rows where income_range IN ("total", "dollars", "average", "median")
    df = df[~df.iloc[:, 0].str.lower().isin(["total", "dollars", "average", "median"])]

    print("After dropping total, dollars, average, and median:")
    print(df)

    # Step 2: Replace empty strings with NaN and forward fill the income_range column
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()

    print("After forward filling income_range:")
    print(df)

    # Step 3: Drop rows that are now empty in the income_range column
    df = df.dropna(subset=[df.columns[0]])

    print("After dropping empty income_range rows:")
    print(df)

    # Step 4: Rename the columns to the tidy schema
    df = df.rename(columns={df.columns[0]: 'income_range', 
                            df.columns[1]: 'women_percentage', 
                            df.columns[2]: 'men_percentage', 
                            df.columns[3]: 'difference'})

    print("After renaming columns:")
    print(df)

    # Step 5: Return only the tidy DataFrame
    tidy_df = df[['income_range', 'women_percentage', 'men_percentage', 'difference']]

    print("Final tidy DataFrame:")
    print(tidy_df)

    return tidy_df.reset_index(drop=True)