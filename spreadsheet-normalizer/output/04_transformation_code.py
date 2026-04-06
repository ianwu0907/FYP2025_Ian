import pandas as pd

def transform(df):
    # Skip metadata rows
    data = df.iloc[3:]

    # Forward-fill the header rows horizontally
    headers = df.iloc[[0, 1]].ffill(axis=1)
    header_row = headers.iloc[0]
    
    # Build a column mapping based on the specified structure
    column_map = {
        1: ('cisgender', 'percent'), 
        2: ('cisgender', 'confidence_interval_from'), 
        3: ('cisgender', 'confidence_interval_to'), 
        4: ('transgender', 'percent'), 
        5: ('transgender', 'confidence_interval_from'), 
        6: ('transgender', 'confidence_interval_to')
    }

    # Unpivot using the mapping
    records = []
    for i in range(len(data)):
        if i < 3: 
            continue
        
        row = data.iloc[i]
        self_rated_mental_health = row.iloc[0]

        if self_rated_mental_health in ["Total", "Drop"]:
            continue

        for col_idx, (group_val, var_name) in column_map.items():
            value = pd.to_numeric(row.iloc[col_idx], errors='coerce')
            
            if var_name == 'percent':
                percent = value
            elif var_name == 'confidence_interval_from':
                confidence_interval_from = value
            elif var_name == 'confidence_interval_to':
                confidence_interval_to = value
            
            if col_idx == 6:  # Ensure we append after the last column is processed
                records.append({
                    'self_rated_mental_health': self_rated_mental_health,
                    'group': group_val,
                    'percent': percent if group_val == 'cisgender' else row.iloc[1],
                    'confidence_interval_from': confidence_interval_from,
                    'confidence_interval_to': confidence_interval_to
                })

    result = pd.DataFrame(records)
    print(result)
    
    return result