import pandas as pd
import numpy as np

def transform(df):
<<<<<<< HEAD
    # Exclude summary labels
    exclude_labels = [
        "所有少數族裔人士[3]",
        "All ethnic minorities[3]",
        "撇除外籍家庭傭工後的所有少數族裔人士[3]",
        "All ethnic minorities, excluding foreign domestic ...",
        "全港人口",
        "Whole population"
    ]
    
    # Data rows start at row 5, ethnicity labels in column 0
    # Number and percentage columns: 2011 (3,4), 2016 (5,6), 2021 (7,8)
    records = []
    
    for i in range(5, 38):
        ethnicity = str(df.iloc[i, 0]).strip() if pd.notna(df.iloc[i, 0]) else ""
        
        # Skip excluded or empty rows
        if ethnicity in exclude_labels or not ethnicity:
            continue
        
        # Extract data for each year
        for year, num_col, pct_col in [(2011, 3, 4), (2016, 5, 6), (2021, 7, 8)]:
            number = pd.to_numeric(df.iloc[i, num_col], errors='coerce')
            percentage = pd.to_numeric(df.iloc[i, pct_col], errors='coerce')
            
            records.append({
                'ethnicity': ethnicity,
                'year': year,
                'number': number,
                'percentage': percentage
            })
    
    result = pd.DataFrame(records, columns=['ethnicity', 'year', 'number', 'percentage'])
    return result
=======
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
>>>>>>> 5b4c511b7694d4069a5093a49002e2550c7097b3
