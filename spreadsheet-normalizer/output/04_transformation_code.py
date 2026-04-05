import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    result = df.copy()
    result.columns = [str(c) for c in result.columns]
    result = result.dropna(how="all")
    print(f"After cleaning: {result.shape}")

    id_vars = [result.columns[0]]  # Ethnicity (column 0)
    value_vars = result.columns[1:]  # Year columns (starting from index 1)
    result = pd.melt(result,
                     id_vars=id_vars,
                     value_vars=value_vars,
                     var_name="year",
                     value_name="population_count")
    
    result = result.dropna(subset=["population_count"])
    result.iloc[:, 0] = result.iloc[:, 0].str.strip()  # Clean 'ethnicity'
    result = result[~result.iloc[:, 0].isin(["Exact", "Values", "To", "Drop"])]
    
    result["population_count"] = result["population_count"].str.replace(r'\s*\(.*\)', '', regex=True)
    result["population_count"] = result["population_count"].str.replace(',', '')
    result["population_count"] = pd.to_numeric(result["population_count"], errors="coerce")
    result = result.dropna(subset=["population_count"])

    result = result.rename(columns={result.columns[0]: "ethnicity",
                                     result.columns[1]: "year",
                                     result.columns[2]: "population_count"})
    
    result = result[["ethnicity", "year", "population_count"]]
    print(f"Final output: {result.shape}")
    return result