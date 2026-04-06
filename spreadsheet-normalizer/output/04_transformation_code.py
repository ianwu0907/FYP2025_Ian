def transform(df):
    import pandas as pd
    import numpy as np
    
    print("Initial shape:", df.shape)
    print("Initial columns:", df.columns.tolist())
    
    # Drop the total row (last row where sector = "total")
    df = df[df.iloc[:, 0] != "total"].copy()
    print("After dropping total row:", df.shape)
    
    # Generation statuses (excluding 'total')
    generation_statuses = ['immigrant', 'second_generation', 'third_generation_or_more']
    
    records = []
    
    # Iterate through each row (sector)
    for i in range(len(df)):
        sector = df.iloc[i, 0]
        
        # Black female workers: columns 1, 2, 3 (immigrant, second_gen, third_gen)
        for j, gen_status in enumerate(generation_statuses):
            records.append({
                'sector': sector,
                'worker_group': 'black_female_workers',
                'generation_status': gen_status,
                'percentage': df.iloc[i, 1 + j]
            })
        
        # Other female workers: columns 5, 6, 7 (immigrant, second_gen, third_gen)
        for j, gen_status in enumerate(generation_statuses):
            records.append({
                'sector': sector,
                'worker_group': 'other_female_workers',
                'generation_status': gen_status,
                'percentage': df.iloc[i, 5 + j]
            })
    
    print("Number of records created:", len(records))
    
    result = pd.DataFrame(records)
    print("Final shape:", result.shape)
    print("Sample output:")
    print(result.head(10))
    
    return result