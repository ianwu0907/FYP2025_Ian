import pandas as pd
import numpy as np

def transform(df):
    print(f"Input shape: {df.shape}")

    # Step 1: Forward-fill sparse rows for the self-rated mental health indicators
    df.iloc[:, 0] = df.iloc[:, 0].replace(r'^\s*$', np.nan, regex=True).ffill()
    print(f"After forward-fill on self-rated mental health: {df.iloc[:, 0].unique()}")

    # Step 2: Slice to data region
    data_region = df.iloc[3:9].copy()  # Selecting rows 3 to 8
    print(f"After slicing to data region: {data_region.shape}")

    # Step 3: Remove rows where self-rated mental health is "Drop"
    data_region = data_region[data_region.iloc[:, 0] != "Drop"]
    print(f"After dropping excluded rows: {data_region.shape}")

    # Step 4: Retrieve multi-level headers
    headers = df.iloc[[0, 1, 2]].ffill(axis=1)
    sexual_orientations = headers.iloc[0, [1, 4, 7, 10, 13]].values
    print(f"Sexual orientations: {sexual_orientations}")

    # Step 5: Unpivot wide columns into long format using record-collection loop
    records = []
    for i in range(len(data_region)):
        mental_health_indicator = str(data_region.iloc[i, 0]).strip()
        for j, sexual_orientation in enumerate(sexual_orientations):
            percent = pd.to_numeric(data_region.iloc[i, 1 + j * 3], errors='coerce')
            confidence_interval_from = pd.to_numeric(data_region.iloc[i, 2 + j * 3], errors='coerce')
            confidence_interval_to = pd.to_numeric(data_region.iloc[i, 3 + j * 3], errors='coerce')
            records.append({
                "self_rated_mental_health": mental_health_indicator,
                "sexual_orientation": str(sexual_orientation).strip(),
                "percent": percent,
                "confidence_interval_from": confidence_interval_from,
                "confidence_interval_to": confidence_interval_to
            })

    # Step 6: Create a DataFrame and reorder to the target schema
    result = pd.DataFrame(records)
    target_cols = ['self_rated_mental_health', 'sexual_orientation', 'percent', 'confidence_interval_from', 'confidence_interval_to']
    result = result[target_cols]
    print(f"Final output: {result.shape}")
    return result