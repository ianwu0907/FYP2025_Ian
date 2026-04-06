import pandas as pd
import numpy as np

def transform(df):
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