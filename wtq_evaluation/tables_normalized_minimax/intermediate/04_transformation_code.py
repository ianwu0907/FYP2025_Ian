import pandas as pd
import numpy as np

def transform(df):
    # Work on a copy to avoid mutating the input
    result = df.copy()
    
    # Ensure column names are strings
    result.columns = [str(c) for c in result.columns]
    
    # Identify the category column (first column) and year columns (all other columns)
    id_vars = [result.columns[0]]  # 'Unnamed: 0' is the category column
    value_vars = [c for c in result.columns if c != result.columns[0]]  # Year columns
    
    # Melt from wide to long format
    result = pd.melt(
        result,
        id_vars=id_vars,
        value_vars=value_vars,
        var_name='year',
        value_name='value'
    )
    
    # Rename the id column to 'category'
    result = result.rename(columns={id_vars[0]: 'category'})
    
    # Convert types safely
    result['year'] = result['year'].astype(int)
    result['value'] = pd.to_numeric(result['value'], errors='coerce')
    
    # Drop rows with null values in 'value'
    result = result.dropna(subset=['value'])
    
    # Ensure correct column order
    result = result[['category', 'year', 'value']]
    
    return result