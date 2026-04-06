def transform(df):
    import pandas as pd
    import numpy as np
    
    # Step 1: Remove total rows
    df = df[~df.iloc[:, 0].isin(["total"])]
    print(df)

    # Step 2: Forward fill behaviour column
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    print(df)

    # Step 3: Reset index and drop previous index
    df.reset_index(drop=True, inplace=True)
    print(df)

    # Step 4: Rename columns to match target schema
    df.columns = ['behaviour', 'beta', 'p_value']
    print(df)

    # Step 5: Filter for weekday and weekend observations
    df = df[df['behaviour'].isin(["screen time", "sedentary time", "lpa", "mvpa"])]
    print(df)

    return df.reset_index(drop=True)