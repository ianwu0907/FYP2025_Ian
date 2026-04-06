import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Slice to data region (all 8 rows)
    result = df.iloc[0:8].copy()
    result.columns = [str(c) for c in result.columns]
    result = result.dropna(how='all')
    print(f"After slicing to data region: {result.shape}")

    # Step 2: Extract source_name column before any operations
    source_name_col = result.columns[0]
    result['source_name'] = result[source_name_col].astype(str)

    # Step 3: Exclude aggregation rows
    semantic_agg_labels = ["Total availability", "Total production"]
    result = result[~result['source_name'].isin(semantic_agg_labels)]
    print(f"After excluding aggregate rows: {result.shape}")

    # Step 4: Derive metric_type from row grouping (before melt to preserve position info)
    # Rows 0-2 in filtered data = availability, Rows 3-5 = production
    total_availability_idx = 3
    result['metric_type'] = np.where(
        result.index < total_availability_idx,
        'availability',
        'production'
    )
    print(f"After adding metric_type: {result.shape}")

    # Step 5: Melt wide format (columns 1-13 are year values 2000-2012)
    id_vars = [result.columns[0], 'metric_type']
    value_vars = [result.columns[i] for i in range(1, 14)]
    result = pd.melt(result,
                     id_vars=id_vars,
                     value_vars=value_vars,
                     var_name='year',
                     value_name='value')
    print(f"After melt: {result.shape}")

    # Step 6: Rename and type conversions
    result = result.rename(columns={result.columns[0]: 'source_name'})
    result['year'] = result['year'].astype(int)
    result['value'] = pd.to_numeric(result['value'], errors='coerce')
    result = result.dropna(subset=['value'])

    # Final column order to match schema
    result = result[['source_name', 'metric_type', 'year', 'value']]
    print(f"Final output: {result.shape}")
    return result